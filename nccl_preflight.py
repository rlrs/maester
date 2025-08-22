# ================= NCCL PREFLIGHT SANITY & STRESS =================
# Drop this near the top of your train.py and call run_nccl_preflight()
# AFTER dist.init_process_group(...) and torch.cuda.set_device(local_rank).

import os, json, time, math, socket, statistics, traceback
from typing import Dict, Any, List, Tuple
import torch
import torch.distributed as dist

def _dtype_from_str(s: str):
    s = s.lower().strip()
    if s in ("fp32","float","float32","torch.float32"): return torch.float32
    if s in ("fp16","float16","torch.float16","half"): return torch.float16
    if s in ("bf16","bfloat16","torch.bfloat16"): return torch.bfloat16
    raise ValueError(f"Unknown dtype '{s}'")

def _dtype_name(dt):
    return {torch.float32:"fp32", torch.float16:"fp16", torch.bfloat16:"bf16"}.get(dt, str(dt))

def _elem_size(dt: torch.dtype) -> int:
    return torch.tensor([], dtype=dt).element_size()

def _now(): return time.time()

def _ring_bytes_allreduce(nbytes: int, world: int) -> float:
    # Approximate bytes moved per rank by ring allreduce
    # ~ 2 * (world-1)/world * nbytes
    return 2.0 * (world - 1) / world * nbytes

def _ring_bytes_allgather(nbytes_each: int, world: int) -> float:
    # Each rank contributes nbytes_each; total out = nbytes_each * world.
    # Approx ring bytes per rank ~ (world-1)/world * total
    return (world - 1) / world * (nbytes_each * world)

def _ring_bytes_reduce_scatter(nbytes_total: int, world: int) -> float:
    # Reduce-scatter is roughly symmetric to allgather.
    return (world - 1) / world * nbytes_total

def _safe_scale_for(dtype: torch.dtype, world: int) -> float:
    # Keep sums comfortably finite across ranks for each dtype
    if dtype is torch.float16:
        return 1e-3 / max(1, world)  # very conservative
    if dtype is torch.bfloat16:
        return 1e-2 / max(1, world)
    return 1.0 / max(1, world)

def _tol_for(dtype: torch.dtype) -> float:
    return 1e-5 if dtype is torch.float32 else (5e-3 if dtype is torch.bfloat16 else 1e-2)

def _finite_stats(x: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        finite = torch.isfinite(x)
        nonfinite = int((~finite).sum().item())
        xm = x.float()
        return dict(
            nonfinite=nonfinite,
            min=float(xm.min().item()),
            max=float(xm.max().item()),
            mean=float(xm.mean().item()),
            sum=float(xm.sum().item()),
        )

def _sizes_list(max_mb: int, world: int) -> List[int]:
    # test a spread of sizes (per-rank sizes for allreduce; for allgather, same per-rank)
    # from 1KB up to max_mb, including some larger steps
    sizes = [1<<10, 4<<10, 64<<10, 256<<10, 1<<20, 8<<20, 32<<20, max_mb<<20]
    # remove duplicates and sizes > max_mb
    sizes = sorted(set([s for s in sizes if s <= (max_mb<<20)]))
    return sizes

def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

def run_nccl_preflight() -> None:
    if os.environ.get("NCCL_PREFLIGHT", "1") != "1":
        return
    assert dist.is_initialized(), "run_nccl_preflight must be called after dist.init_process_group()"
    assert torch.cuda.is_available(), "CUDA required for NCCL preflight"

    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = torch.device("cuda", torch.cuda.current_device())
    host = socket.gethostname()

    report_dir = os.environ.get("NCCL_PREFLIGHT_REPORT_DIR", "/scratch/nccl_preflight")
    os.makedirs(report_dir, exist_ok=True)
    rank_log_path = os.path.join(report_dir, f"preflight_rank{rank}.log")
    rank_log = open(rank_log_path, "a", buffering=1)

    def log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        rank_log.write(f"{ts} R{rank}/{world} {host}: {msg}\n"); rank_log.flush()

    max_mb = int(os.environ.get("NCCL_PREFLIGHT_MAX_MB", "128"))
    iters = int(os.environ.get("NCCL_PREFLIGHT_ITERS", "50"))
    warmup = int(os.environ.get("NCCL_PREFLIGHT_WARMUP", "10"))
    try_rs = os.environ.get("NCCL_PREFLIGHT_TRY_REDUCE_SCATTER", "1") == "1"
    dtypes_req = [s.strip() for s in os.environ.get("NCCL_PREFLIGHT_DTYPES", "fp32,fp16,bf16").split(",") if s.strip()]
    dtypes: List[torch.dtype] = []
    for s in dtypes_req:
        try:
            dt = _dtype_from_str(s)
            if dt is torch.bfloat16 and not _supports_bf16():
                if rank == 0: print("[preflight] skipping bf16 (not supported)", flush=True)
                continue
            dtypes.append(dt)
        except Exception as e:
            if rank == 0: print(f"[preflight] ignoring dtype '{s}': {e}", flush=True)

    # Base metadata
    meta = dict(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        device_name=torch.cuda.get_device_name(dev),
        capability=torch.cuda.get_device_capability(dev),
        world_size=world,
        hostnames=None,  # filled by gather
        params=dict(max_mb=max_mb, iters=iters, warmup=warmup, dtypes=[_dtype_name(d) for d in dtypes]),
        started=_now(),
    )

    # Gather hostnames
    try:
        h = [None]
        if rank == 0:
            hosts = [None for _ in range(world)]
        else:
            hosts = None
        dist.gather_object(host, dst=0, obj=host if rank != 0 else None)  # type: ignore
    except Exception:
        # some older torch versions don't have gather_object in this form—fallback with all_gather_object
        hs = [host]
        obj_list = [None for _ in range(world)]
        dist.all_gather_object(obj_list, hs)
        if rank == 0:
            meta["hostnames"] = [o[0] for o in obj_list]
    else:
        if rank == 0:
            # Some torch versions require manual gather; if missing, meta["hostnames"] may be None
            pass

    # sizes (per-rank byte counts for allreduce/allgather)
    sizes = _sizes_list(max_mb, world)
    results: Dict[str, Any] = {"allreduce": [], "allgather": [], "reduce_scatter": []}

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Core test routine
    def time_op(fn, sync=True) -> float:
        t0 = _now()
        fn()
        if sync:
            torch.cuda.synchronize()
        return (_now() - t0)

    # ===================== ALLREDUCE (SUM) ======================
    for dt in dtypes:
        scale = _safe_scale_for(dt, world)
        for nbytes in sizes:
            nelems = max(1, nbytes // _elem_size(dt))
            # pattern: per-rank constant value so we can predict the sum exactly
            x = torch.full((nelems,), fill_value=(rank+1)*scale, dtype=dt, device=dev)
            # warmup
            for _ in range(warmup):
                dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)
            torch.cuda.synchronize()

            # timed
            lat_us = []
            nonfinite_seen = 0
            for _ in range(iters):
                # reset input each iter to avoid accumulating in-place
                x.fill_((rank+1)*scale)
                dt_s = time_op(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False))
                lat_us.append(dt_s * 1e6)
                # quick numerics check
                if not torch.isfinite(x).all():
                    nonfinite_seen += 1

            # correctness: expected sum value
            expected = scale * (world * (world + 1) / 2.0)
            err = (x.float() - expected).abs()
            ok = bool((err.max().item() <= _tol_for(dt)) and (nonfinite_seen == 0))

            stats = dict(
                op="allreduce",
                dtype=_dtype_name(dt),
                nbytes=int(nbytes),
                nelems=int(nelems),
                latency_us_min=float(min(lat_us)),
                latency_us_p50=float(statistics.median(lat_us)),
                latency_us_p99=float(sorted(lat_us)[max(0, int(len(lat_us)*0.99)-1)]),
                nonfinite_iters=int(nonfinite_seen),
                finite_stats=_finite_stats(x),
                expected=float(expected),
                max_abs_err=float(err.max().item()),
                ok=ok,
                approx_bytes_per_rank=float(_ring_bytes_allreduce(nbytes, world)),
            )
            results["allreduce"].append(stats)

    # ===================== ALLGATHER ============================
    for dt in dtypes:
        for nbytes_each in sizes:
            # target per-rank elements
            nelems_each_target = max(1, int(nbytes_each // _elem_size(dt)))
            elem_size = _elem_size(dt)

            # decide how many elements we can attempt (respect free memory heuristics if available)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
            except Exception:
                free_bytes = None
                total_bytes = None

            headroom_bytes = int(max(128 << 20, (0.02 * total_bytes) if total_bytes else (128 << 20)))
            nelems_each = nelems_each_target
            if free_bytes is not None:
                max_possible_nelems_each = max(1, (free_bytes - headroom_bytes) // (elem_size * world))
                if max_possible_nelems_each < nelems_each_target:
                    nelems_each = max_possible_nelems_each
                    if rank == 0:
                        print(f"[preflight] reducing target allgather size to {nelems_each*elem_size} B/rank to fit free memory", flush=True)

            total_elems = nelems_each * world
            alloc_ok = False
            out = src = None
            last_err = None
            # conservative loop to attempt allocation, halving on failure
            while nelems_each >= 1 and not alloc_ok:
                try:
                    out = torch.empty((total_elems,), dtype=dt, device=dev)
                    src = torch.full((nelems_each,), float(rank), dtype=dt, device=dev)
                    alloc_ok = True
                except RuntimeError as e:
                    last_err = repr(e)
                    # reduce size and retry
                    nelems_each = nelems_each // 2
                    total_elems = nelems_each * world
                    if nelems_each < 1:
                        break

            if not alloc_ok:
                if rank == 0:
                    print(f"[preflight] skipping allgather test for dtype={_dtype_name(dt)} nbytes_each={nbytes_each} (alloc failed); last_err={last_err}", flush=True)
                results["allgather"].append(dict(
                    op="allgather",
                    dtype=_dtype_name(dt),
                    nbytes_each=int(nbytes_each),
                    nelems_each=int(nelems_each),
                    allocated=False,
                    ok=False,
                    last_err=last_err,
                ))
                continue

            # reshape to per-rank view; this is a view, not a copy
            try:
                out = out.view(world, nelems_each)
            except Exception as e:
                # if reshape fails, record and continue
                last_err = repr(e)
                results["allgather"].append(dict(
                    op="allgather", dtype=_dtype_name(dt), nbytes_each=int(nbytes_each),
                    nelems_each=int(nelems_each), allocated=True, ok=False, last_err=last_err
                ))
                del out, src
                torch.cuda.empty_cache()
                continue

            # warmup single-call gathers
            try:
                for _ in range(warmup):
                    src.fill_(float(rank))
                    dist.all_gather_into_tensor(out, src)
                torch.cuda.synchronize()
            except Exception as e:
                last_err = repr(e)
                results["allgather"].append(dict(
                    op="allgather", dtype=_dtype_name(dt), nbytes_each=int(nbytes_each),
                    nelems_each=int(nelems_each), allocated=True, ok=False, last_err=last_err
                ))
                del out, src
                torch.cuda.empty_cache()
                continue

            # timed iterations for the single large allgather call
            lat_us = []
            nonfinite_iters = 0
            ok = True
            for _ in range(iters):
                try:
                    src.fill_(float(rank))
                    t0 = _now()
                    dist.all_gather_into_tensor(out, src)
                    torch.cuda.synchronize()
                    lat_us.append((_now() - t0) * 1e6)
                except Exception as e:
                    last_err = repr(e)
                    ok = False
                    break
                # cheap global numerics check using scalar accumulations only
                try:
                    total_sum_scalar = out.sum(dtype=torch.float64).item()
                    if not math.isfinite(total_sum_scalar):
                        nonfinite_iters += 1
                except Exception:
                    nonfinite_iters += 1

            # final per-segment verification (use scalar min/max/sum; avoid creating big temporaries)
            final_ok = True
            try:
                for r in range(world):
                    seg = out[r]  # view
                    seg_min = float(seg.min().item())
                    seg_max = float(seg.max().item())
                    seg_sum = float(seg.sum(dtype=torch.float64).item())
                    r_val = float(r)
                    if not (math.isfinite(seg_min) and math.isfinite(seg_max) and math.isfinite(seg_sum)):
                        final_ok = False
                        last_err = "non-finite in segment reduction"
                        break
                    if abs(seg_min - r_val) > _tol_for(dt) or abs(seg_max - r_val) > _tol_for(dt):
                        final_ok = False
                        last_err = f"segment min/max mismatch rank {r}: got min={seg_min} max={seg_max} expected={r_val}"
                        break
                    if abs(seg_sum - (r_val * nelems_each)) > max(1e-3 * abs(r_val * nelems_each), 1e-6):
                        final_ok = False
                        last_err = f"segment sum mismatch rank {r}: got {seg_sum} expected {r_val * nelems_each}"
                        break
            except Exception as e:
                final_ok = False
                last_err = repr(e)

            # compute aggregate scalar stats robustly (each in its own try so we don't leave things undefined)
            overall_min = overall_max = overall_sum = overall_count = overall_mean = None
            try:
                overall_min = float(out.min().item())
            except Exception:
                overall_min = None
            try:
                overall_max = float(out.max().item())
            except Exception:
                overall_max = None
            try:
                overall_sum = float(out.sum(dtype=torch.float64).item())
                overall_count = float(total_elems)
                overall_mean = overall_sum / overall_count if overall_count > 0 else None
            except Exception:
                overall_sum = None
                overall_count = None
                overall_mean = None

            # final ok considers timed-phase ok, nonfinite counts, and final verification
            final_status = bool(ok and (nonfinite_iters == 0) and final_ok)

            stats = dict(
                op="allgather",
                dtype=_dtype_name(dt),
                nbytes_each=int(nbytes_each),
                nelems_each=int(nelems_each),
                allocated=True,
                latency_us_min=float(min(lat_us)) if lat_us else 0.0,
                latency_us_p50=float(statistics.median(lat_us)) if lat_us else 0.0,
                latency_us_p99=float(sorted(lat_us)[max(0, int(len(lat_us)*0.99)-1)]) if lat_us else 0.0,
                nonfinite_iters=int(nonfinite_iters),
                finite_stats=dict(min=overall_min, max=overall_max, mean=overall_mean, sum=overall_sum),
                ok=final_status,
                last_err=last_err,
                approx_bytes_per_rank=float(_ring_bytes_allgather(nelems_each*elem_size, world)),
            )
            results["allgather"].append(stats)

            # cleanup
            try:
                del out, src
            except Exception:
                pass
            torch.cuda.empty_cache()

    # =============== REDUCE_SCATTER (optional) ==================
    if try_rs and hasattr(dist, "reduce_scatter_tensor"):
        for dt in dtypes:
            for nbytes_chunk in sizes:
                # Need total elements divisible by world: make total = chunk * world
                nelems_chunk = max(1, nbytes_chunk // _elem_size(dt))
                nelems_total = nelems_chunk * world
                x = torch.full((nelems_total,), fill_value=(rank+1)*_safe_scale_for(dt, world), dtype=dt, device=dev)
                out = torch.empty((nelems_chunk,), dtype=dt, device=dev)
                # warmup
                for _ in range(warmup):
                    dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                lat_us = []
                nonfinite_seen = 0
                for _ in range(iters):
                    x.fill_((rank+1)*_safe_scale_for(dt, world))
                    dt_s = time_op(lambda: dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM))
                    lat_us.append(dt_s * 1e6)
                    if not torch.isfinite(out).all():
                        nonfinite_seen += 1
                # correctness: each element in out should equal sum_{r}( (r+1)*scale )
                expected = _safe_scale_for(dt, world) * (world * (world + 1) / 2.0)
                ok = torch.allclose(out.float(), torch.full_like(out.float(), expected), rtol=_tol_for(dt), atol=_tol_for(dt)) and (nonfinite_seen == 0)

                stats = dict(
                    op="reduce_scatter",
                    dtype=_dtype_name(dt),
                    nbytes_chunk=int(nbytes_chunk),
                    nelems_chunk=int(nelems_chunk),
                    latency_us_min=float(min(lat_us)),
                    latency_us_p50=float(statistics.median(lat_us)),
                    latency_us_p99=float(sorted(lat_us)[max(0, int(len(lat_us)*0.99)-1)]),
                    nonfinite_iters=int(nonfinite_seen),
                    finite_stats=_finite_stats(out),
                    expected=float(expected),
                    ok=ok,
                    approx_bytes_per_rank=float(_ring_bytes_reduce_scatter(nelems_total*_elem_size(dt), world)),
                )
                results["reduce_scatter"].append(stats)

    # Gather results on rank 0
    my_result = dict(
        rank=rank,
        host=host,
        metrics=results,
        finished=_now(),
        env=dict(
            NCCL_DEBUG=os.environ.get("NCCL_DEBUG"),
            NCCL_DEBUG_SUBSYS=os.environ.get("NCCL_DEBUG_SUBSYS"),
            NCCL_IB_DISABLE=os.environ.get("NCCL_IB_DISABLE"),
            NCCL_SOCKET_IFNAME=os.environ.get("NCCL_SOCKET_IFNAME"),
            TORCH_DISTRIBUTED_DEBUG=os.environ.get("TORCH_DISTRIBUTED_DEBUG"),
        ),
    )

    gathered: List[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, my_result)

    if rank == 0:
        # Build a concise verdict + write JSON report
        verdicts: List[str] = []
        def summarize(op_name: str, records: List[Dict[str, Any]]) -> str:
            if not records: return f"{op_name}: skipped"
            oks = sum(1 for r in records if r["ok"])
            total = len(records)
            # crude throughput: median approx_bytes / median latency
            lat_p50 = statistics.median([r["latency_us_p50"] for r in records])
            bytes_p50 = statistics.median([r.get("approx_bytes_per_rank", 0.0) for r in records])
            gbps = (bytes_p50 / (lat_p50/1e6)) / (1<<30) if lat_p50 > 0 else 0.0
            return f"{op_name}: {oks}/{total} OK, ~{gbps:.2f} GB/s per-rank (p50 size)"

        flat = [summarize(k, v) for k, v in dict(
            allreduce=sum((g["metrics"]["allreduce"] for g in gathered if g), []),
            allgather=sum((g["metrics"]["allgather"] for g in gathered if g), []),
            reduce_scatter=sum((g["metrics"].get("reduce_scatter", []) for g in gathered if g), []),
        ).items()]

        combined = dict(
            meta=meta,
            ranks=gathered,
            verdicts=flat,
            ended=_now(),
        )
        out_path = os.path.join(report_dir, "nccl_preflight_report.json")
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        print("[preflight] NCCL preflight done. " + " | ".join(flat) + f" | report: {out_path}", flush=True)

    # small barrier so logs are flushed in order
    dist.barrier()
# ================= END NCCL PREFLIGHT =================
