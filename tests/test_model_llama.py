import os
import torch
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM
from maester.models.llama.model import Transformer, ModelArgs, precompute_freqs_cis
from maester.checkpoint import CheckpointManager
from maester.log_utils import init_logger, logger
from maester.models import model_name_to_cls, models_config, model_name_to_tokenizer

class Config(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=(), arbitrary_types_allowed=True)

    model_name: str = "llama3"
    flavor: str = "8B"
    seq_len: int = 256
    norm_type: str = "rmsnorm"
    batch_size: int = 1
    
    job_folder: str = "job/"
    checkpoint_folder: str = "checkpoints"
    enable_checkpoint: bool = True
    checkpoint_interval: int = 50
    model_weights_only: bool = True
    export_dtype: str = "bfloat16"
    

def main():
    init_logger()
    cfg = Config()
    model_cls = model_name_to_cls[cfg.model_name]
    model_config = models_config[cfg.model_name][cfg.flavor]
    model_config.norm_type = cfg.norm_type
    model_config.vocab_size = 128256
    model_config.max_seq_len = cfg.seq_len

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "32145"
    torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)

    # Initialize Hugging Face model
    hf_model_name = "meta-llama/Meta-Llama-3-8B"
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)

    # Initialize Maester model
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)
    model.to_empty(device="cpu")
    model.init_weights()

    checkpoint = CheckpointManager(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        dataloader=None,
        states={},
        cfg=cfg,
    )
    assert checkpoint.load(step=0)

    # # 1. Testing that weights in all layers are the same
    # # Compare embedding layers
    # maester_embedding = model.tok_embeddings.weight
    # hf_embedding = hf_model.get_input_embeddings().weight
    # assert torch.allclose(maester_embedding, hf_embedding, atol=1e-5), "Embedding layers do not match"

    # # Compare attention layers
    # for i, (maester_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
    #     maester_attention = maester_layer.attention
    #     hf_attention = hf_layer.self_attn

    #     assert torch.allclose(maester_attention.wq.weight, hf_attention.q_proj.weight, atol=1e-5), f"Attention layer {i} WQ weights do not match"
    #     assert torch.allclose(maester_attention.wk.weight, hf_attention.k_proj.weight, atol=1e-5), f"Attention layer {i} WK weights do not match"
    #     assert torch.allclose(maester_attention.wv.weight, hf_attention.v_proj.weight, atol=1e-5), f"Attention layer {i} WV weights do not match"
    #     assert torch.allclose(maester_attention.wo.weight, hf_attention.o_proj.weight, atol=1e-5), f"Attention layer {i} WO weights do not match"

    # # Compare feedforward layers
    # for i, (maester_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
    #     maester_ffn = maester_layer.feed_forward
    #     hf_ffn = hf_layer.mlp

    #     assert torch.allclose(maester_ffn.w1.weight, hf_ffn.gate_proj.weight, atol=1e-5), f"Feedforward layer {i} W1 weights do not match"
    #     assert torch.allclose(maester_ffn.w2.weight, hf_ffn.down_proj.weight, atol=1e-5), f"Feedforward layer {i} W2 weights do not match"
    #     assert torch.allclose(maester_ffn.w3.weight, hf_ffn.up_proj.weight, atol=1e-5), f"Feedforward layer {i} W3 weights do not match"

    # # Compare output layers
    # maester_output = model.output.weight
    # hf_output = hf_model.get_output_embeddings().weight
    # assert torch.allclose(maester_output, hf_output, atol=1e-5), "Output layers do not match"

    # # 2. Testing that layer outputs match
    # Compare embedding layers
    input_tensor = torch.randint(0, model_config.vocab_size, (cfg.batch_size, cfg.seq_len), dtype=torch.long)
    maester_embedding_output = model.tok_embeddings(input_tensor)
    hf_embedding_output = hf_model.get_input_embeddings()(input_tensor)
    assert torch.allclose(maester_embedding_output, hf_embedding_output, atol=1e-5), "Embedding layer outputs do not match"

    # Compare feedforward layers
    input_tensor = torch.randn(cfg.batch_size, cfg.seq_len, model_config.dim)
    for i, (maester_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
        maester_ffn_output = maester_layer.feed_forward(maester_layer.ffn_norm(input_tensor))
        hf_ffn_output = hf_layer.mlp(hf_layer.post_attention_layernorm(input_tensor))

        assert torch.allclose(maester_ffn_output, hf_ffn_output, atol=1e-5), f"Feedforward layer {i} outputs do not match"

    # Compare attention layers
    input_tensor = torch.randn(cfg.batch_size, cfg.seq_len, model_config.dim)
    freqs_cis = model.freqs_cis[0:cfg.seq_len]
    
    for i, (maester_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
        maester_attention_output = maester_layer.attention(maester_layer.attention_norm(input_tensor), freqs_cis)
        hf_attention_output, _, _ = hf_layer.self_attn(
            hf_layer.input_layernorm(input_tensor), 
            position_ids=torch.arange(cfg.seq_len, dtype=torch.long).unsqueeze(0).expand(cfg.batch_size, -1)
        )

        assert torch.allclose(maester_attention_output, hf_attention_output, atol=1e-5), f"Attention layer {i} outputs do not match"

    # Compare output layers
    input_tensor = torch.randn(cfg.batch_size, cfg.seq_len, model_config.dim)
    maester_output_layer_output = model.output(input_tensor)
    hf_output_layer_output = hf_model.get_output_embeddings()(input_tensor)
    assert torch.allclose(maester_output_layer_output, hf_output_layer_output, atol=1e-5), "Output layer outputs do not match"




if __name__ == "__main__":
    main()
