import torch
from torch._six import inf

def move(batch, device):
    moved = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return moved


def convert_activations(model, from_class, to_class):
    for child_name, child in model.named_children():
        if isinstance(child, from_class):
            setattr(model, child_name, to_class())
        else:
            convert_activations(child, from_class, to_class)


def no_grad(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def clip_grad_norm_(
    parameters, max_norm, norm_type=2.0
):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        is_non_finite = True
        clip_coef = torch.tensor(0.0)
    else:
        is_non_finite = False
        clip_coef = max_norm / (total_norm + 1e-6)

    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return is_non_finite, total_norm