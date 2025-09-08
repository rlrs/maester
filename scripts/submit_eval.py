#!/usr/bin/env python3
"""
Submit a model to the eval service API.
Handles conversion to HF format, upload, and job submission.

Usage:
    python submit_model.py model.pt --name "my-model-v1"
    python submit_model.py /path/to/model/dir --name "my-model" --token $MY_TOKEN

Environment Variables (.env file supported):
    EVAL_API_URL: API endpoint URL (default: http://localhost:8080)
    EVAL_TOKEN: API authentication token
"""

import os
import sys
import json
import time
import tarfile
import tempfile
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

# Try to load .env file if it exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in ["EVAL_API_URL", "EVAL_TOKEN"]:
                    os.environ[key] = value

# Default API endpoint
DEFAULT_API_URL = os.environ.get("EVAL_API_URL", "http://localhost:8080")
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks for upload


def create_session():
    """Create requests session with retries"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def convert_to_hf(model_path: str, output_dir: str, converter: str = None, 
                  converter_args: List[str] = None) -> bool:
    """
    Convert model to HuggingFace format using specified converter script.
    
    Args:
        model_path: Path to input model
        output_dir: Directory to save converted model
        converter: Name of converter script (e.g., 'convert_dcp_to_hf.py')
        converter_args: Additional arguments to pass to converter
    
    Converter Interface:
        All converters should accept:
        - First arg: input path (model/checkpoint)
        - Second arg: output directory
        - Additional args specific to converter type
    """
    print(f"Converting {model_path} to HF format...")
    
    if not converter:
        # Default fallback behavior
        print("⚠ No converter specified, assuming model is already in HF format")
        import shutil
        if os.path.isdir(model_path):
            shutil.copytree(model_path, output_dir, dirs_exist_ok=True)
        else:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(model_path, output_dir)
        return True
    
    # Build command based on converter type
    scripts_dir = Path(__file__).parent
    converter_path = scripts_dir / converter
    
    if not converter_path.exists():
        print(f"✗ Converter script not found: {converter_path}")
        return False
    
    # Standard interface: script input_path output_path [additional_args]
    # Use venv python if available, otherwise system python
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else "python"
    
    cmd = [python_exe, str(converter_path), model_path, output_dir]
    
    # Add any additional converter-specific arguments
    if converter_args:
        cmd.extend(converter_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Model converted successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e.stderr}")
        return False


def create_tarball(source_dir: str, output_path: str, compression: str = "zstd") -> tuple[str, int]:
    """
    Create tarball from directory. Returns (actual_path, size_in_bytes).
    
    Args:
        source_dir: Directory to archive
        output_path: Output tar file path
        compression: Compression type ('none', 'zstd', 'zstd-fast', 'zstd-best')
    """
    import io
    try:
        import zstandard as zstd
        has_zstd = True
    except ImportError:
        has_zstd = False
        if compression.startswith("zstd"):
            print("Warning: zstandard not installed, falling back to no compression")
            print("Install with: pip install zstandard")
            compression = "none"
    
    print(f"Creating tarball from {source_dir}...")
    
    # Determine actual path and compression settings
    if compression == "none":
        actual_path = output_path.replace(".zst", "").replace(".gz", "")
        if not actual_path.endswith(".tar"):
            actual_path += ".tar"
        mode = "w"
    elif compression.startswith("zstd") and has_zstd:
        actual_path = output_path.replace(".gz", ".zst")
        if not actual_path.endswith(".zst"):
            actual_path = actual_path.replace(".tar", ".tar.zst")
        mode = "w"  # We'll handle compression separately
        
        # Determine compression level
        if compression == "zstd-fast":
            level = 1  # Fastest
        elif compression == "zstd-best":
            level = 19  # Best compression
        else:  # Default zstd
            level = 3  # Good balance
    else:
        actual_path = output_path
        mode = "w"
    
    # Create tarball with optional zstd compression
    if compression.startswith("zstd") and has_zstd:
        print(f"  Creating tar.zst with compression level {level}...")
        
        # Open zstd compressed file for writing
        cctx = zstd.ZstdCompressor(level=level, threads=-1)  # Use all CPU threads
        with open(actual_path, 'wb') as zstd_file:
            with cctx.stream_writer(zstd_file) as compressor:
                # Create tar stream into the compressor
                with tarfile.open(fileobj=compressor, mode='w|') as tar:
                    tar.add(source_dir, arcname=os.path.basename(source_dir))
    else:
        # Create uncompressed tar
        with tarfile.open(actual_path, mode) as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
    
    size = os.path.getsize(actual_path)
    if compression == "none":
        comp_type = "uncompressed"
    elif compression.startswith("zstd") and has_zstd:
        comp_type = f"zstd level {level}"
    else:
        comp_type = compression
    
    print(f"✓ Tarball created ({comp_type}): {size / (1024*1024):.1f} MB")
    return actual_path, size


def upload_model(session: requests.Session, api_url: str, token: str, 
                 tarball_path: str, model_name: str) -> str:
    """
    Upload model using TUS resumable upload protocol.
    Supports automatic resume on connection failure and across script runs.
    Returns upload_id.
    """
    import base64
    import hashlib
    
    file_size = os.path.getsize(tarball_path)
    filename = os.path.basename(tarball_path)
    
    print(f"Uploading {filename} ({file_size / (1024*1024):.1f} MB)...")
    
    # Create a resume file based on tarball path and size
    # This helps identify the same upload across runs
    file_hash = hashlib.md5(f"{tarball_path}:{file_size}:{model_name}".encode()).hexdigest()[:8]
    resume_file = Path(tempfile.gettempdir()) / f".eval_upload_{file_hash}.json"
    
    upload_id = None
    
    # Check if we have a previous upload to resume
    if resume_file.exists():
        try:
            with open(resume_file) as f:
                resume_data = json.load(f)
            
            # Verify the file hasn't changed
            if (resume_data.get("file_path") == tarball_path and 
                resume_data.get("file_size") == file_size and
                resume_data.get("api_url") == api_url):
                
                upload_id = resume_data.get("upload_id")
                print(f"Found previous upload: {upload_id}")
                
                # Verify upload still exists on server
                try:
                    head_resp = session.head(
                        f"{api_url}/api/tus/{upload_id}",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Tus-Resumable": "1.0.0"
                        }
                    )
                    if head_resp.status_code == 200:
                        print("  Resuming previous upload...")
                    else:
                        # Upload doesn't exist anymore
                        upload_id = None
                        resume_file.unlink()
                except:
                    # Can't reach upload, start fresh
                    upload_id = None
                    resume_file.unlink()
        except Exception as e:
            # Invalid resume file, remove it
            resume_file.unlink()
            upload_id = None
    
    # Create new upload if needed
    if not upload_id:
        # Encode metadata for TUS protocol (base64 encoded key-value pairs)
        metadata_dict = {
            "filename": filename,
            "model_name": model_name
        }
        metadata_pairs = []
        for key, value in metadata_dict.items():
            value_b64 = base64.b64encode(value.encode('utf-8')).decode('ascii')
            metadata_pairs.append(f"{key} {value_b64}")
        metadata_header = ",".join(metadata_pairs)
        
        # Create TUS upload
        create_resp = session.post(
            f"{api_url}/api/tus/",
            headers={
                "Authorization": f"Bearer {token}",
                "Upload-Length": str(file_size),
                "Upload-Metadata": metadata_header,
                "Tus-Resumable": "1.0.0"
            }
        )
        
        if create_resp.status_code == 401:
            raise Exception("Authentication failed. Check your token.")
        
        # Debug response if something goes wrong
        if create_resp.status_code != 201:
            print(f"Unexpected response status: {create_resp.status_code}")
            print(f"Response headers: {dict(create_resp.headers)}")
            print(f"Response body: {create_resp.text}")
        
        create_resp.raise_for_status()
        
        # Extract upload ID from Location header
        location = create_resp.headers.get("Location")
        if not location:
            print(f"Error: No Location header in response")
            print(f"Status code: {create_resp.status_code}")
            print(f"Response headers: {dict(create_resp.headers)}")
            print(f"Response body: {create_resp.text[:500] if create_resp.text else 'No body'}")
            raise ValueError("No Location header in response - server may not be implementing TUS protocol correctly")
        upload_id = location.rstrip('/').split('/')[-1]
        print(f"Upload ID: {upload_id}")
        
        # Save resume info
        with open(resume_file, 'w') as f:
            json.dump({
                "upload_id": upload_id,
                "file_path": tarball_path,
                "file_size": file_size,
                "model_name": model_name,
                "api_url": api_url,
                "created_at": datetime.now().isoformat()
            }, f)
    
    # 2. Check current offset (for resume support)
    head_resp = session.head(
        f"{api_url}/api/tus/{upload_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Tus-Resumable": "1.0.0"
        }
    )
    head_resp.raise_for_status()
    current_offset = int(head_resp.headers.get("Upload-Offset", "0"))
    
    if current_offset > 0:
        print(f"  Resuming from {current_offset / (1024*1024):.1f} MB...")
    elif current_offset >= file_size:
        print("  Upload already complete!")
        return upload_id
    
    # 3. Upload chunks (starting from current offset)
    uploaded = current_offset
    retry_count = 0
    max_retries = 3
    
    # Setup progress bar
    if has_tqdm:
        progress_bar = tqdm(
            total=file_size,
            initial=current_offset,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="  Uploading",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    with open(tarball_path, "rb") as f:
        # Seek to the resume position
        if current_offset > 0:
            f.seek(current_offset)
        
        while uploaded < file_size:
            # Read chunk
            chunk_size = min(CHUNK_SIZE, file_size - uploaded)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            try:
                # Upload chunk using TUS PATCH
                chunk_resp = session.patch(
                    f"{api_url}/api/tus/{upload_id}",
                    data=chunk,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Upload-Offset": str(uploaded),
                        "Content-Type": "application/offset+octet-stream",
                        "Tus-Resumable": "1.0.0"
                    }
                )
                
                # Handle offset conflict (409)
                if chunk_resp.status_code == 409:
                    # Get correct offset and retry
                    if has_tqdm:
                        progress_bar.write("  Offset mismatch, resuming...")
                    else:
                        print("\n  Offset mismatch, resuming...")
                    
                    head_resp = session.head(
                        f"{api_url}/api/tus/{upload_id}",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Tus-Resumable": "1.0.0"
                        }
                    )
                    head_resp.raise_for_status()
                    new_offset = int(head_resp.headers.get("Upload-Offset", "0"))
                    if new_offset != uploaded:
                        if has_tqdm:
                            # Update progress bar to new position
                            progress_bar.n = new_offset
                            progress_bar.refresh()
                        uploaded = new_offset
                        f.seek(new_offset)
                        retry_count = 0
                        continue
                
                chunk_resp.raise_for_status()
                
                # Update progress
                uploaded += len(chunk)
                if has_tqdm:
                    progress_bar.update(len(chunk))
                else:
                    progress = (uploaded / file_size) * 100
                    mb_uploaded = uploaded / (1024 * 1024)
                    mb_total = file_size / (1024 * 1024)
                    print(f"  Uploading... {progress:.1f}% ({mb_uploaded:.1f}/{mb_total:.1f} MB)", end='\r', flush=True)
                
                # Reset retry count on success
                retry_count = 0
                
            except requests.exceptions.RequestException as e:
                # Network error - retry with resume
                retry_count += 1
                if retry_count > max_retries:
                    if has_tqdm:
                        progress_bar.close()
                    print(f"\n✗ Upload failed after {max_retries} retries: {e}")
                    raise
                
                if has_tqdm:
                    progress_bar.write(f"  Connection error, retrying ({retry_count}/{max_retries})...")
                else:
                    print(f"\n  Connection error, retrying ({retry_count}/{max_retries})...")
                
                time.sleep(2 ** retry_count)  # Exponential backoff
                
                # Get current offset from server
                try:
                    head_resp = session.head(
                        f"{api_url}/api/tus/{upload_id}",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Tus-Resumable": "1.0.0"
                        }
                    )
                    head_resp.raise_for_status()
                    server_offset = int(head_resp.headers.get("Upload-Offset", "0"))
                    if server_offset != uploaded:
                        if has_tqdm:
                            # Update progress bar to server position
                            progress_bar.n = server_offset
                            progress_bar.refresh()
                        uploaded = server_offset
                        f.seek(server_offset)
                except:
                    pass  # Will retry from last position
    
    if has_tqdm:
        progress_bar.close()
    
    print("\n✓ Upload complete" if not has_tqdm else "✓ Upload complete")
    
    # 4. Finalize upload (bridge TUS with existing system)
    finalize_resp = session.post(
        f"{api_url}/api/uploads/{upload_id}/finalize",
        headers={"Authorization": f"Bearer {token}"}
    )
    finalize_resp.raise_for_status()
    
    # Clean up resume file on successful completion
    if resume_file.exists():
        resume_file.unlink()
        print("  Cleaned up resume file")
    
    return upload_id


def submit_job(session: requests.Session, api_url: str, token: str,
               upload_id: str, name: str) -> str:
    """
    Submit evaluation job for uploaded model.
    Returns job_id.
    """
    print("Submitting evaluation job...")
    
    payload = {
        "name": name,
        "model": {
            "type": "upload",
            "upload_id": upload_id
        }
    }
    
    resp = session.post(
        f"{api_url}/api/jobs",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    
    job_data = resp.json()
    job_id = job_data["job_id"]
    print(f"✓ Job submitted: {job_id}")
    
    return job_id


def monitor_job(session: requests.Session, api_url: str, token: str, 
                job_id: str, follow: bool = True):
    """Monitor job status."""
    print(f"\nJob status: {api_url}/api/jobs/{job_id}")
    
    if not follow:
        return
    
    print("Monitoring job progress...")
    last_status = None
    
    while True:
        try:
            resp = session.get(
                f"{api_url}/api/jobs/{job_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            resp.raise_for_status()
            
            job = resp.json()
            status = job.get("status", "UNKNOWN")
            
            if status != last_status:
                print(f"  Status: {status}")
                last_status = status
            
            # Show eval progress
            if "eval_results" in job and job["eval_results"]:
                for eval_name, eval_data in job["eval_results"].items():
                    if eval_data and "status" in eval_data:
                        eval_status = eval_data["status"]
                        metrics = eval_data.get("metrics", {})
                        if eval_status == "COMPLETED" and metrics:
                            # Show first metric as summary
                            first_metric = next(iter(metrics.items()), (None, None))
                            if first_metric[0]:
                                print(f"    {eval_name}: {first_metric[0]}={first_metric[1]}")
                        elif eval_status == "RUNNING":
                            print(f"    {eval_name}: Running...")
                        elif eval_status == "FAILED":
                            error = metrics.get("error", "Unknown error")
                            print(f"    {eval_name}: Failed - {error[:100]}")
            
            if status in ["COMPLETED", "FAILED"]:
                print(f"\n{'✓' if status == 'COMPLETED' else '✗'} Job {status.lower()}")
                break
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nStopped monitoring (job still running)")
            break
        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(
        description="Submit a model to the eval service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Converter Examples:
  DCP to HF (for mistral/llama models):
    python submit_eval.py checkpoints/step_1000 --converter convert_dcp_to_hf.py \\
      --converter-args "--name step-1000 --base mistralai/Mistral-7B-v0.1"
  
  Gemma DCP to HF:
    python submit_eval.py checkpoints/step_1000 --converter convert_gemma_from_dcp.py
  
  MuP to HF:
    python submit_eval.py model_dir --converter convert_mup_to_hf.py
  
  No conversion (already HF format):
    python submit_eval.py hf_model_dir --no-convert
        """
    )
    parser.add_argument("model", help="Path to model file or directory")
    parser.add_argument("--name", help="Name for the evaluation job", default=None)
    parser.add_argument("--token", help="API token (or set EVAL_TOKEN in .env file)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API endpoint URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--no-convert", action="store_true", help="Skip conversion (model already in HF format)")
    parser.add_argument("--converter", help="Converter script to use (e.g., convert_dcp_to_hf.py)")
    parser.add_argument("--converter-args", nargs=argparse.REMAINDER, 
                        help="Additional arguments to pass to converter (use -- before args)")
    parser.add_argument("--no-follow", action="store_true", help="Don't monitor job after submission")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files after upload")
    parser.add_argument("--compression", choices=["none", "zstd", "zstd-fast", "zstd-best"], default="zstd",
                        help="Compression type: none (no compression), zstd (balanced), zstd-fast (level 1), zstd-best (level 19)")
    
    args = parser.parse_args()
    
    # Get token (check EVAL_TOKEN first, then USER_TOKEN for backward compatibility)
    token = args.token or os.environ.get("EVAL_TOKEN") or os.environ.get("USER_TOKEN")
    if not token:
        print("Error: No token provided. Set EVAL_TOKEN in .env file or use --token")
        sys.exit(1)
    
    # Generate name if not provided
    if not args.name:
        model_basename = Path(args.model).stem
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.name = f"{model_basename}-{timestamp}"
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    session = create_session()
    
    # Create temp directory for conversion and tarball
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.keep_temp:
            # Use a persistent temp directory
            tmpdir = tempfile.mkdtemp(prefix="eval_submit_")
            print(f"Temp directory: {tmpdir}")
        
        # Convert model to HF format
        hf_model_dir = os.path.join(tmpdir, "hf_model")
        if not args.no_convert:
            # Use specified converter or let convert_to_hf handle the default
            if not convert_to_hf(args.model, hf_model_dir, 
                                 converter=args.converter,
                                 converter_args=args.converter_args):
                print("Error: Model conversion failed")
                sys.exit(1)
        else:
            # Just use the model as-is
            import shutil
            if os.path.isdir(args.model):
                shutil.copytree(args.model, hf_model_dir)
            else:
                os.makedirs(hf_model_dir, exist_ok=True)
                shutil.copy2(args.model, hf_model_dir)
        
        # Create tarball
        tarball_name = os.path.join(tmpdir, f"{args.name}.tar.zst")
        tarball_path, file_size = create_tarball(hf_model_dir, tarball_name, compression=args.compression)
        
        try:
            # Upload model
            upload_id = upload_model(session, args.api_url, token, tarball_path, args.name)
            
            # Submit job
            job_id = submit_job(session, args.api_url, token, upload_id, args.name)
            
            # Monitor job
            monitor_job(session, args.api_url, token, job_id, follow=not args.no_follow)
            
            print(f"\nView results at: {args.api_url}/")
            
        except requests.exceptions.HTTPError as e:
            print(f"Error: {e}")
            if e.response.text:
                print(f"Details: {e.response.text}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        if args.keep_temp:
            print(f"\nTemp files kept at: {tmpdir}")


if __name__ == "__main__":
    main()