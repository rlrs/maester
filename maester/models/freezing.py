from fnmatch import fnmatch
import torch

def freeze_model_params(
    model: torch.nn.Module,
    freeze_patterns: list[str],
    unfreeze_patterns: list[str],
):
    frozen = {}
    for name, param in model.named_parameters():
        name = '.'.join(name.split('.')[:-1]) if name.endswith('weight') or name.endswith('bias') else name
        print(f"Checking parameter: {name} for freezing/unfreezing with patterns {freeze_patterns} / {unfreeze_patterns}")
        if any(fnmatch(name, pattern) for pattern in freeze_patterns):
            print(f"Freezing parameter: {name} due to matching freeze pattern {[pattern for pattern in freeze_patterns if fnmatch(name, pattern)]}")
            param.requires_grad = False
            frozen[name] = [pattern for pattern in freeze_patterns if fnmatch(name, pattern)][0]
        if any(fnmatch(name, pattern) for pattern in unfreeze_patterns):
            print(f"Unfreezing parameter: {name} due to matching unfreeze pattern {[pattern for pattern in unfreeze_patterns if fnmatch(name, pattern)]}")
            param.requires_grad = True
            if name in frozen:
                del frozen[name]
    return frozen