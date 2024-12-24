import unit_scaling as uu
import torch
from maester.models.umup.model import Transformer, TransformerBlock, ModelArgs, precompute_freqs_cis

seqlen = 512
model_config = ModelArgs(dim=512, n_layers=3, norm_type="umup_rmsnorm", vocab_size=1024, n_kv_heads=32)

# with torch.device("meta"):
#     layer = Transformer(model_config)
# layer.to_empty(device="cpu")
# layer.init_weights()
# input = torch.randint(0, 1024, size=(128, seqlen))
# output = layer(input)

# with torch.device("meta"):
#     layer = TransformerBlock(0, model_config)
# layer.to_empty(device="cpu")
# layer.init_weights()

# freqs_cis = precompute_freqs_cis(layer.dim // layer.n_heads, seqlen)
# output = layer(input, freqs_cis)

# mhsa_tau = uu.transformer_residual_scaling_rule(1, 1)(0, 2)
# print(f"mhsa_tau = {mhsa_tau}")
# mlp_tau = uu.transformer_residual_scaling_rule(1, 1)(1, 2)
# print(f"mlp_tau = {mlp_tau}")
# layer = uu.TransformerLayer(model_config.dim, model_config.n_heads, mhsa_tau, mlp_tau, is_causal=True)
# input = torch.randn((128, seqlen, model_config.dim), requires_grad=True)
# output = layer(input)
layer = uu.TransformerDecoder(model_config.dim, 1024, layers=6, heads=16)
input = torch.randint(0, 1024, (128, seqlen))
output = layer(input)


output.backward(torch.randn_like(output))
print(f"# {type(layer).__name__}:")
for k, v in {
    "output": output.std(),
    # "input.grad": input.grad.std(),
    **{f"{name}": param.std() for name, param in layer.named_parameters()},
    **{f"{name}.grad": param.grad.std() for name, param in layer.named_parameters()},
}.items():
    print(f"{k:>20}.std = {v.item():.2f}")