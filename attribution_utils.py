import torch

def lrp_rule_ratio(nom, denom, eps) -> torch.Tensor:
    # Remark: for some reason, torch automatically remove the batch axis of context
    # could this be PyTorch's bug?
    if nom.shape[0] == 1 and len(nom.shape) == 4 and len(nom.shape) == 3:
        output = output.unsqueeze(0)
    # this trick combats getting nan from backprop of x/0.
    # see https://github.com/pytorch/pytorch/issues/4132
    nonzero_ix = denom.abs() > eps
    new_output = torch.zeros_like(nom)
    new_output[nonzero_ix] = nom[nonzero_ix] / denom[nonzero_ix]
    return new_output