import zennit
from zennit.attribution import Gradient
from zennit.rules import Epsilon
import torch

from copy import deepcopy

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from zennit.core import (
    BasicHook,
    Stabilizer,
    expand,
    Hook,
)

from zennit.rules import (
    ZBox,
    Pass,
    Gamma,
    GammaMod,
    NoMod,
    zero_bias,
    ClampMod,
    Flat
)
from torchvision.models import vision_transformer
from attribution_utils import lrp_rule_ratio

import zennit

class SafeGamma(BasicHook):
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):
        mod_kwargs = {"zero_params": zero_params}
        mod_kwargs_nobias = {"zero_params": zero_bias(zero_params)}
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input,
            ],
            param_modifiers=[
                GammaMod(gamma, min=0.0, **mod_kwargs),
                GammaMod(gamma, max=0.0, **mod_kwargs_nobias),
                GammaMod(gamma, max=0.0, **mod_kwargs),
                GammaMod(gamma, min=0.0, **mod_kwargs_nobias),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * lrp_rule_ratio(nom=out_grad, denom=denom, eps=stabilizer)
                    for output, denom in (
                        [(outputs[4] > 0.0, sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0.0, sum(outputs[2:4]))] * 2
                    )
                ]
                + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(
                    input * gradient
                    for input, gradient in zip(inputs[:4], gradients[:4])
                )
            ),
        )


class SafeZBox(BasicHook):
    def __init__(self, low, high, stabilizer=1e-6, zero_params=None):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        mod_kwargs = {"zero_params": zero_params}

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(low, input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(high, input.shape, cut_batch_dim=True).to(input),
            ],
            param_modifiers=[
                NoMod(**mod_kwargs),
                ClampMod(min=0.0, **mod_kwargs),
                ClampMod(max=0.0, **mod_kwargs),
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(
                lambda out_grad, outputs: (
                    lrp_rule_ratio(out_grad, sub(*outputs), eps=stabilizer),
                )
                * 3
            ),
            reducer=(
                lambda inputs, gradients: sub(
                    *(input * gradient for input, gradient in zip(inputs, gradients))
                )
            ),
        )