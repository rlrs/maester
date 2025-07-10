import os
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from maester.models.gemma.model import GemmaTextModel
from maester.models.gemma import gemma3_configs
from maester.models import models_config
import pytest


def test_gemma_dcp_checkpoints_exist():
    """Test that DCP checkpoints exist for Gemma models."""
    checkpoints = {
        "1B": Path("google-gemma-1b-pt-dcp"),
        "4B": Path("google-gemma-4b-pt-dcp"),
    }
    
    available = {}
    for size, path in checkpoints.items():
        available[size] = path.exists()
        if path.exists():
            print(f"✓ Found Gemma {size} checkpoint at {path}")
        else:
            print(f"✗ Gemma {size} checkpoint not found at {path}")
    
    return available


@pytest.mark.parametrize("model_size,dcp_path,hf_model_name,port", [
    ("1B", "google-gemma-1b-pt-dcp", "google/gemma-3-1b-pt", "12355"),
    ("4B", "google-gemma-4b-pt-dcp", "google/gemma-3-4b-pt", "12356"),
])
def test_gemma3_logits_comparison(model_size, dcp_path, hf_model_name, port):
    """Test Gemma3 model logits against HuggingFace implementation."""
    
    # Skip if checkpoint doesn't exist
    if not Path(dcp_path).exists():
        pytest.skip(f"Gemma {model_size} DCP checkpoint not available")
    
    # Use the actual config from models_config
    config = models_config["gemma3"][model_size]
    dcp_path = Path(dcp_path)
    
    # Initialize process group for DCP
    if not torch.distributed.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
    
    try:
        import torch.distributed.checkpoint as dcp
        
        # Initialize model on CPU directly (not meta device)
        model = GemmaTextModel.from_model_args(config)
        
        # Create state dict wrapper for DCP
        state_dict = {"model": model}
        
        # Load checkpoint directly into model
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(str(dcp_path)),
        )
        
        model.eval()
        our_model = model
        
        print(f"✓ Loaded our Gemma 3 {model_size} model from DCP checkpoint")
        
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    
    print(f"Loading HF model: {hf_model_name}")
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    
    # Test with multiple inputs
    test_texts = [
        "The capital of France is",
        "Machine learning is",
        "2 + 2 =",
    ]
    
    for test_text in test_texts:
        print(f"\nTesting: '{test_text}'")
        input_ids = tokenizer(test_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            # Get logits from both models
            our_logits = our_model(input_ids)
            hf_outputs = hf_model(input_ids)
            hf_logits = hf_outputs.logits
            
            # Handle potential vocab size mismatch
            # HF model might have vision tokens that we don't use for text-only training
            if hf_logits.shape[-1] > our_logits.shape[-1]:
                print(f"  Note: HF model has {hf_logits.shape[-1] - our_logits.shape[-1]} extra tokens (likely vision tokens)")
                print(f"  Comparing only text vocabulary ({our_logits.shape[-1]} tokens)")
                hf_logits = hf_logits[:, :, :our_logits.shape[-1]]
        
        # Compare logits
        max_diff = torch.max(torch.abs(hf_logits - our_logits)).item()
        mean_diff = torch.mean(torch.abs(hf_logits - our_logits)).item()
        
        print(f"  Max logit difference: {max_diff:.6f}")
        print(f"  Mean logit difference: {mean_diff:.6f}")
        
        # Check top predictions
        hf_top5 = torch.topk(hf_logits[0, -1], k=5)
        our_top5 = torch.topk(our_logits[0, -1], k=5)
        
        print(f"  HF top 5: {tokenizer.batch_decode(hf_top5.indices.tolist())}")
        print(f"  Our top 5: {tokenizer.batch_decode(our_top5.indices.tolist())}")
        
        # Stricter tolerance for logit comparison
        assert max_diff < 1e-3, f"Max logit difference too large: {max_diff}"
        assert mean_diff < 1e-4, f"Mean logit difference too large: {mean_diff}"
        
        # Top prediction should match
        assert hf_top5.indices[0] == our_top5.indices[0], f"Top prediction mismatch for '{test_text}'"
    
    print(f"\n✓ Gemma {model_size} logits comparison test passed!")



def test_gemma3_config_structure():
    """Test that Gemma3 configs are properly structured."""
    
    # Test 1B config
    config_1b = gemma3_configs["1B"]
    assert config_1b.vocab_size == 262_144
    assert config_1b.dim == 1152
    assert config_1b.n_layers == 26
    assert config_1b.sliding_window_size == 512
    assert config_1b.use_qk_norm == True
    assert len(config_1b.attn_types) == 6
    
    # Test 4B config  
    config_4b = gemma3_configs["4B"]
    # NON-STANDARD: We use text-only vocab size, not the full multimodal vocab
    assert config_4b.vocab_size == 262_144  # Text-only tokens (vision tokens removed)
    assert config_4b.dim == 2560
    assert config_4b.n_layers == 34
    assert config_4b.sliding_window_size == 1024
    assert config_4b.num_key_value_heads == 4
    
    print("✓ Config structure test passed!")


if __name__ == "__main__":
    # Run tests
    print("Testing Gemma3 implementation...")
    
    # Test configs
    test_gemma3_config_structure()
    
    # Check which checkpoints are available
    available = test_gemma_dcp_checkpoints_exist()
    
    # Test models with DCP weights if available
    for model_size, dcp_path, hf_model_name, port in [
        ("1B", "google-gemma-1b-pt-dcp", "google/gemma-3-1b-pt", "12355"),
        ("4B", "google-gemma-4b-pt-dcp", "google/gemma-3-4b-pt", "12356"),
    ]:
        if available.get(model_size):
            test_gemma3_logits_comparison(model_size, dcp_path, hf_model_name, port)
        else:
            print(f"\nSkipping {model_size} logits test - checkpoint not available")
    
    print("\nAll available tests passed!")