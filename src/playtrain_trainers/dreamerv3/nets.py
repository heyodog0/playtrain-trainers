"""Layers, initializers and the symlog pair — a port of ``embodied/jax/nets.py``.

Only what the frozen atari100k config reaches is ported: ``Initializer``,
``Linear``, ``BlockLinear``, ``Conv2D``, ``Norm`` (rms and layer), ``MLP``,
``symlog``/``symexp`` and the activation lookup. ``Attention``, ``Transformer``,
``Conv3D``, ``Embed``, ``DictEmbed``, ``rope`` and the jax dtype-assertion plumbing
are not ported; the fidelity row D-016 says so per item.

Two structural differences from the original, both forced by the framework and both
chosen to keep the NUMBERS identical (D-012):

* ninjax creates parameters lazily on first call (``self.value(name, init, shape)``),
  so the official layers take only their output size. ``torch.nn.Module`` wants the
  shapes up front, so every layer here takes its input size too. The initializer sees
  exactly the same shape tuple it would see in JAX, so the fans -- and therefore the
  scales -- are unchanged.
* Weights are stored in the JAX layout (``Linear``: ``(in, out)``; ``Conv2D``: HWIO)
  and applied with ``x @ w`` / a permute into torch's OIHW. Storing them transposed
  would change which axis ``compute_fans`` reads and silently rescale every layer.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import nn

COMPUTE_DTYPE = torch.bfloat16


def set_compute_dtype(name: str) -> None:
    """``jax.compute_dtype``: the dtype ``cast`` converts activations to (D-012).

    The official code does not use a framework autocast. It casts explicitly, at
    fixed sites (``nn.cast`` on the RSSM inputs/carry/samples, the encoder's image,
    the decoder's inputs and ``feat2tensor``), every kernel follows the activation
    dtype, norms compute in f32 and cast back, and every output distribution takes f32
    logits. The port mirrors those sites, so this one setting governs the whole model.
    """
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def cast(xs, force: bool = False):
    """``nets.cast``: floating tensors (all tensors when ``force``) to ``COMPUTE_DTYPE``,
    through dicts, lists and tuples."""
    if isinstance(xs, dict):
        return {k: cast(v, force) for k, v in xs.items()}
    if isinstance(xs, (list, tuple)):
        return type(xs)(cast(v, force) for v in xs)
    if torch.is_tensor(xs) and (force or xs.is_floating_point()):
        return xs.to(COMPUTE_DTYPE)
    return xs


# ----------------------------------------------------------------------
# Activations, symlog (nets.py `act`, `symlog`, `symexp`)
# ----------------------------------------------------------------------
def act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "none":
        return lambda x: x
    if name == "mish":
        return lambda x: x * torch.tanh(nn.functional.softplus(x))
    if name == "relu2":
        return lambda x: torch.square(torch.relu(x))
    if name == "silu":
        return nn.functional.silu
    if name == "gelu":
        # jax.nn.gelu defaults to the tanh approximation, which torch spells
        # approximate="tanh"; the exact one is a different function.
        return lambda x: nn.functional.gelu(x, approximate="tanh")
    if name == "relu":
        return torch.relu
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    raise NotImplementedError(f"activation {name!r}")


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ----------------------------------------------------------------------
# Initializer (nets.py `Initializer`, `init`)
# ----------------------------------------------------------------------
def compute_fans(shape: Sequence[int]) -> tuple[float, float]:
    """``Initializer.compute_fans``, verbatim.

    Note what this means for ``BlockLinear``, whose kernel is
    ``(blocks, in/blocks, out/blocks)``: ``space = blocks``, so ``fanin`` comes out as
    the FULL input width, not the per-block one. Copied deliberately.
    """
    if len(shape) == 0:
        return (1, 1)
    if len(shape) == 1:
        return (1, shape[0])
    if len(shape) == 2:
        return (shape[0], shape[1])
    space = math.prod(shape[:-2])
    return (shape[-2] * space, shape[-1] * space)


class Initializer:
    """``nets.Initializer``: the five distributions, the three fan modes."""

    def __init__(self, dist: str = "trunc_normal", fan: str = "in", scale: float = 1.0):
        self.dist = dist
        self.fan = fan
        self.scale = scale

    def __call__(
        self,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
        fshape: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        assert all(isinstance(x, int) for x in shape), shape
        assert all(x > 0 for x in shape), shape
        fanin, fanout = compute_fans(shape if fshape is None else fshape)
        fan = {"avg": (fanin + fanout) / 2, "in": fanin, "out": fanout, "none": 1}[self.fan]
        if self.dist == "zeros":
            x = torch.zeros(shape, dtype=torch.float32)
        elif self.dist == "uniform":
            limit = math.sqrt(1 / fan)
            x = torch.empty(shape, dtype=torch.float32).uniform_(-limit, limit, generator=generator)
        elif self.dist == "normal":
            x = torch.randn(shape, dtype=torch.float32, generator=generator)
            x *= math.sqrt(1 / fan)
        elif self.dist == "trunc_normal":
            # jax.random.truncated_normal(-2, 2) is a standard normal truncated to
            # [-2, 2]; torch's nn.init.trunc_normal_ with std=1 is the same draw. The
            # 1.1368 factor undoes the variance the truncation removes.
            x = torch.empty(shape, dtype=torch.float32)
            nn.init.trunc_normal_(x, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=generator)
            x *= 1.1368 * math.sqrt(1 / fan)
        elif self.dist == "normed":
            x = torch.empty(shape, dtype=torch.float32).uniform_(-1, 1, generator=generator)
            x = x / torch.linalg.norm(x.reshape(-1, shape[-1]), 2, dim=0)
        else:
            raise NotImplementedError(self.dist)
        x = x * self.scale
        return x.to(dtype)

    def __repr__(self) -> str:
        return f"Initializer({self.dist}, {self.fan}, {self.scale})"

    def __eq__(self, other: object) -> bool:
        return all(
            getattr(self, k) == getattr(other, k, None) for k in ("dist", "fan", "scale")
        )


def init(name: str | Initializer | Callable) -> Initializer | Callable:
    """``nets.init``: ``'trunc_normal_in'`` -> ``Initializer('trunc_normal', 'in')``."""
    if callable(name):
        return name
    if name.endswith(("_in", "_out", "_avg")):
        dist, fan = name.rsplit("_", 1)
    else:
        dist, fan = name, "in"
    return Initializer(dist, fan, 1.0)


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------
class Linear(nn.Module):
    """``nets.Linear``: ``x @ kernel + bias``, kernel stored ``(in, out)``.

    ``units`` may be a tuple, in which case the output is reshaped to it (the official
    layer does the same, e.g. the decoder's ``sp2`` producing a ``(4, 4, 256)`` block).
    """

    def __init__(
        self,
        insize: int,
        units: int | Sequence[int],
        bias: bool = True,
        winit: str | Initializer | Callable = "trunc_normal",
        binit: str | Initializer | Callable = "zeros",
        outscale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.units = (units,) if isinstance(units, int) else tuple(units)
        self.insize = insize
        size = math.prod(self.units)
        kernel = init(winit)((insize, size), generator=generator) * outscale
        self.kernel = nn.Parameter(kernel)
        if bias:
            self.bias = nn.Parameter(init(binit)((size,), generator=generator))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.kernel.to(x.dtype)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x.reshape(*x.shape[:-1], *self.units)


class BlockLinear(nn.Module):
    """``nets.BlockLinear``: a block-diagonal matmul, ``einsum('...ki,kio->...ko')``.

    The input is split into ``blocks`` groups along the last axis and each group gets
    its own weight block, so the layer costs ``1/blocks`` of a dense one. This is what
    makes the 8192-unit GRU affordable; it is also why the RSSM's recurrence mixes
    across blocks only through the three ``dynin`` projections.
    """

    def __init__(
        self,
        insize: int,
        units: int,
        blocks: int,
        bias: bool = True,
        winit: str | Initializer | Callable = "trunc_normal",
        binit: str | Initializer | Callable = "zeros",
        outscale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        assert isinstance(units, int), units
        assert blocks <= units and units % blocks == 0, (blocks, units)
        assert insize % blocks == 0, (insize, blocks)
        self.units = units
        self.blocks = blocks
        self.insize = insize
        shape = (blocks, insize // blocks, units // blocks)
        self.kernel = nn.Parameter(init(winit)(shape, generator=generator) * outscale)
        if bias:
            self.bias = nn.Parameter(init(binit)((units,), generator=generator))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.insize, (x.shape, self.insize)
        lead = x.shape[:-1]
        x = x.reshape(*lead, self.blocks, self.insize // self.blocks)
        x = torch.einsum("...ki,kio->...ko", x, self.kernel.to(x.dtype))
        x = x.reshape(*lead, self.units)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x


class Conv2D(nn.Module):
    """``nets.Conv2D`` for the strided=False, transp=False, groups=1 path.

    The kernel is kept in JAX's HWIO layout so ``compute_fans`` reads the same axes;
    it is permuted to torch's OIHW at call time. Inputs are NHWC, as in the original,
    and are permuted to NCHW around the convolution. ``pad='same'`` with an odd kernel
    and stride 1 is symmetric padding of ``(K - 1) // 2``, which is what XLA's SAME
    does in that case.
    """

    def __init__(
        self,
        insize: int,
        depth: int,
        kernel: int | tuple[int, int],
        stride: int = 1,
        bias: bool = True,
        winit: str | Initializer | Callable = "trunc_normal",
        binit: str | Initializer | Callable = "zeros",
        outscale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.depth = depth
        self.kernel_size = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        self.stride = stride
        self.insize = insize
        if any(k % 2 == 0 for k in self.kernel_size):
            raise NotImplementedError("only odd kernels have symmetric SAME padding")
        shape = (*self.kernel_size, insize, depth)  # HWIO
        self.kernel = nn.Parameter(init(winit)(shape, generator=generator) * outscale)
        if bias:
            self.bias = nn.Parameter(init(binit)((depth,), generator=generator))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NHWC -> NCHW, HWIO -> OIHW
        weight = self.kernel.permute(3, 2, 0, 1).to(x.dtype)
        pad = tuple((k - 1) // 2 for k in self.kernel_size)
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.conv2d(x, weight, None, self.stride, pad)
        x = x.permute(0, 2, 3, 1)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x


class Norm(nn.Module):
    """``nets.Norm``: ``rms``, ``layer`` and ``none``, computed in float32.

    ``eps`` is 1e-4, not the 1e-5/1e-6 torch defaults, and the normalization runs in
    f32 even when the activations are bf16 -- both matter at this width.
    """

    def __init__(self, impl: str, shape: int | Sequence[int], eps: float = 1e-4):
        super().__init__()
        if "1em" in impl:
            impl, exp = impl.split("1em")
            eps = 10 ** -int(exp)
        self.impl = impl
        self.eps = eps
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if impl == "none":
            self.register_parameter("scale", None)
            self.register_parameter("shift", None)
        elif impl == "rms":
            self.scale = nn.Parameter(torch.ones(shape, dtype=torch.float32))
            self.register_parameter("shift", None)
        elif impl == "layer":
            self.scale = nn.Parameter(torch.ones(shape, dtype=torch.float32))
            self.shift = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        else:
            raise NotImplementedError(impl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        if self.impl == "none":
            return x.to(dtype)
        if self.impl == "rms":
            mean2 = torch.square(x).mean(-1, keepdim=True)
            x = x * (torch.rsqrt(mean2 + self.eps) * self.scale)
        else:
            mean = x.mean(-1, keepdim=True)
            mean2 = torch.square(x).mean(-1, keepdim=True)
            var = torch.clamp(mean2 - torch.square(mean), min=0)
            x = (x - mean) * (torch.rsqrt(var + self.eps) * self.scale) + self.shift
        return x.to(dtype)


class MLP(nn.Module):
    """``nets.MLP``: ``layers`` x (Linear -> Norm -> act), no output layer."""

    def __init__(
        self,
        insize: int,
        layers: int = 5,
        units: int = 1024,
        act_name: str = "silu",
        norm: str = "rms",
        bias: bool = True,
        winit: str | Initializer | Callable = "trunc_normal",
        binit: str | Initializer | Callable = "zeros",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.act = act(act_name)
        self.layers = layers
        self.units = units
        lins, norms = [], []
        size = insize
        for _ in range(layers):
            lins.append(
                Linear(size, units, bias=bias, winit=winit, binit=binit, generator=generator)
            )
            norms.append(Norm(norm, units))
            size = units
        self.lins = nn.ModuleList(lins)
        self.norms = nn.ModuleList(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        for lin, norm in zip(self.lins, self.norms):
            x = self.act(norm(lin(x)))
        return x.reshape(*shape, x.shape[-1])
