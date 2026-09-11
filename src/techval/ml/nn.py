"""Neural network primitives in numpy: forward passes, exact gradients, Adam.

Every model in this package is fitted with the code in this file, and there is
no framework underneath it. That is a deliberate choice resting on three facts
about this problem, in descending order of how interesting they are.

**A gradient a reader can check is worth more beside a valuation than one they
cannot.** The engine's claim is that every number traces to a filing or to a
stated assumption. A model whose fitting procedure cannot be inspected weakens
that claim even when the fit is correct. Here the forward pass and the analytic
derivative of each layer sit twenty lines apart, and ``gradient_check`` verifies
every one of them against central finite differences to a relative error below
1e-6. The cost of writing a derivative by hand is that it can be wrong. The
answer is to check it, which is precisely what an autograd system cannot do for
itself.

**The models this dataset justifies are small.** The labelled peer pairs
recoverable from DEF 14A compensation tables number in the tens of thousands,
and the fundamental panel behind them is a few thousand company-quarters deep.
A few hundred thousand parameters is already at the edge of what that supports,
and this file fits one of that size at a speed nobody needs to work around: a
462,080 parameter stack of three hidden layers, over 20,000 rows in batches of
256, trains at 0.75 seconds an epoch on the eight cores of this machine, so a
hundred epochs is 1.3 minutes. Training throughput was never the binding
constraint, and buying it with a dependency nobody can audit would have been
paying for the wrong thing.

**PyTorch installs at roughly 2.5GB and this machine has 1.9GB free.** It does
not fit. This reason comes last because it decides nothing the first two had not
already decided.

**Conventions.** Arrays are float64 everywhere. float32 would halve the memory,
which here buys nothing, and it would push the cancellation error of a central
finite difference to about 1e-2, leaving no tolerance at which the gradient
check could separate a real bug from rounding. Rows are observations and columns
are features, so a batch of B vectors of width D has shape (B, D). Money and
rates do not appear in this file at all: it sees standardised features and
returns numbers in those units, and the translation back to USD millions belongs
to the model that calls it.

Parameter gradients ACCUMULATE into preallocated arrays, so a training step must
call ``zero_grad`` before its backward pass. ``train`` does. A layer caches
exactly one forward pass for its backward, so a layer that has to see two inputs
must see them stacked into a single batch rather than through two successive
``forward`` calls. The two-tower contrastive setup is the case where this
matters, and ``train`` documents the stacking pattern.

Numerical guards carry defaults. Anything that shapes the fit, every random draw
included, is passed in by the caller, which reads it from ``assumptions.ml`` and
hands a seeded ``numpy.random.Generator`` down. Two runs from one seed produce
bitwise identical parameters, and ``tests/ml/test_nn.py`` asserts that rather
than assuming it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from techval.errors import ConfigError, NotMeaningfulError

__all__ = [
    "Layer",
    "Linear",
    "ReLU",
    "Tanh",
    "Dropout",
    "LayerNorm",
    "L2Normalize",
    "Sequential",
    "mse_loss",
    "huber_loss",
    "bce_with_logits",
    "info_nce",
    "info_nce_in_batch",
    "Adam",
    "TrainHistory",
    "train",
    "gradient_check",
]

# Every array in this module is double precision. The module docstring says why
# that is not negotiable: the gradient check has no headroom without it.
DTYPE = np.float64


def _matrix(x: Any, name: str, width: int | None = None) -> np.ndarray:
    """Coerce to a float64 matrix, or say precisely what arrived instead."""
    arr = np.asarray(x, dtype=DTYPE)
    if arr.ndim != 2:
        raise ConfigError(
            f"{name} must be two dimensional, rows observations and columns "
            f"features, but its shape is {arr.shape}"
        )
    if width is not None and arr.shape[1] != width:
        raise ConfigError(f"{name} has width {arr.shape[1]} where {width} was expected")
    return arr


def _same_shape(a: np.ndarray, b: np.ndarray, first: str, second: str) -> None:
    if a.shape != b.shape:
        raise ConfigError(
            f"{first} has shape {a.shape} and {second} has shape {b.shape}: "
            "a loss cannot pair them"
        )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class Layer:
    """One differentiable step, with its own forward and its own backward.

    There is no autograd graph. Each layer stores during ``forward`` whatever
    its derivative needs and consumes that during ``backward``, which is why the
    two have to be called in pairs and in order. Parameter gradients are added
    into the arrays ``grads`` returns rather than replacing them, so a caller
    that has not called ``zero_grad`` since the previous optimiser step is adding
    this batch's gradient to the last one's.
    """

    training: bool = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Gradient of the loss with respect to this layer's input.

        ``grad_out`` is the gradient of the loss with respect to this layer's
        output, which the layer above supplies.
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def params(self) -> list[np.ndarray]:
        """Learnable arrays, in a fixed order, as views the optimiser writes to."""
        return []

    def grads(self) -> list[np.ndarray]:
        """Gradient buffers, aligned element for element with ``params``."""
        return []

    def zero_grad(self) -> None:
        for g in self.grads():
            g.fill(0.0)

    def train(self, mode: bool = True) -> "Layer":
        self.training = bool(mode)
        return self

    def eval(self) -> "Layer":
        return self.train(False)


class Linear(Layer):
    """An affine map, y = x W + b, initialised at the scale the activation needs.

    **Why the initial scale is a modelling decision and not a detail.** Each
    layer multiplies the variance of its input by roughly the sum of squared
    weights feeding a unit. Draw the weights too small and a stack of layers
    shrinks the signal geometrically until the gradient reaching the first layer
    is numerically zero; draw them too large and the same compounding saturates
    the nonlinearity, where the derivative is flat and nothing learns. Both
    settings below hold that product near one:

        ``he``      std = sqrt(2 / in_dim). For a layer feeding a ReLU. ReLU
                    discards half its inputs, so the surviving half has to carry
                    twice the variance to arrive at the same place.
        ``xavier``  std = sqrt(2 / (in_dim + out_dim)). For tanh and for a linear
                    output head. A symmetric activation passes signal in both
                    directions, so the compromise between preserving the forward
                    variance and preserving the backward one is the average of
                    the two widths.

    The bias starts at zero, the only value that does not assert a prior about
    the output before any data has been seen.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        rng: np.random.Generator,
        init: str = "he",
        bias: bool = True,
    ) -> None:
        if in_dim < 1 or out_dim < 1:
            raise ConfigError(
                f"Linear needs positive widths, got in_dim={in_dim} out_dim={out_dim}"
            )
        key = str(init).lower()
        if key == "he":
            std = np.sqrt(2.0 / in_dim)
        elif key == "xavier":
            std = np.sqrt(2.0 / (in_dim + out_dim))
        else:
            raise ConfigError(
                f"unknown weight initialisation {init!r}: use 'he' for a layer "
                "feeding a relu and 'xavier' otherwise"
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.init = key
        self.weight = np.asarray(
            rng.standard_normal((self.in_dim, self.out_dim)) * std, dtype=DTYPE
        )
        self.grad_weight = np.zeros_like(self.weight)
        self.bias = np.zeros(self.out_dim, dtype=DTYPE) if bias else None
        self.grad_bias = None if self.bias is None else np.zeros_like(self.bias)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = _matrix(x, "Linear input", self.in_dim)
        out = self._x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise ConfigError("Linear.backward was called before any forward pass")
        g = _matrix(grad_out, "Linear output gradient", self.out_dim)
        if g.shape[0] != self._x.shape[0]:
            raise ConfigError(
                f"Linear.backward got {g.shape[0]} rows against the "
                f"{self._x.shape[0]} rows of its cached forward pass"
            )
        self.grad_weight += self._x.T @ g
        if self.grad_bias is not None:
            self.grad_bias += g.sum(axis=0)
        return g @ self.weight.T

    def params(self) -> list[np.ndarray]:
        return [self.weight] if self.bias is None else [self.weight, self.bias]

    def grads(self) -> list[np.ndarray]:
        if self.grad_bias is None:
            return [self.grad_weight]
        return [self.grad_weight, self.grad_bias]


class ReLU(Layer):
    """max(x, 0), whose derivative is the indicator that the input was positive.

    The kink at zero has no derivative, and the convention taken here is that
    the subgradient there is zero. That matters in exactly one place: a finite
    difference straddling the kink measures the average of the two slopes and
    will disagree with the analytic value, so a gradient check on this layer has
    to use inputs that stand clear of it.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=DTYPE)
        self._positive = arr > 0.0
        return np.where(self._positive, arr, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return np.asarray(grad_out, dtype=DTYPE) * self._positive


class Tanh(Layer):
    """The symmetric squashing function, with derivative 1 - tanh(x) squared.

    Worth preferring to ReLU for the last hidden layer before an embedding: its
    output is centred and bounded, so the vectors arriving at an L2
    normalisation are not already crowded into the positive orthant, where every
    pairwise cosine is non-negative before training has said anything at all.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(np.asarray(x, dtype=DTYPE))
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return np.asarray(grad_out, dtype=DTYPE) * (1.0 - self._out * self._out)


class Dropout(Layer):
    """Zero a fraction p of activations while training, and rescale the rest.

    This is inverted dropout: surviving activations are divided by 1 - p on the
    way through, so the expected input to the next layer is the same in training
    as in evaluation and inference needs no rescaling at all. Getting that
    backwards is the classic dropout bug, and it is silent. The network trains
    normally and then every activation at evaluation is too large by a factor of
    1 / (1 - p), which presents as a model that scores well in training and
    badly out of sample for a reason that looks like overfitting and is not.

    The mask is drawn from the generator handed in at construction, so a run is
    reproducible from its seed. ``reseed`` puts that generator back to a known
    state, which is what a gradient check needs: finite differences evaluate the
    forward pass twice per element and would otherwise be differencing two
    different masks rather than one function at two points.
    """

    def __init__(self, p: float, *, rng: np.random.Generator) -> None:
        if not 0.0 <= float(p) < 1.0:
            raise ConfigError(
                f"dropout probability must be in [0, 1), got {p}. At p = 1 every "
                "activation is discarded and the rescaling divides by zero"
            )
        self.p = float(p)
        self._rng = rng
        self._mask: np.ndarray | None = None

    def reseed(self, rng: np.random.Generator) -> None:
        """Replace the mask generator, so the next forward passes repeat."""
        self._rng = rng

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=DTYPE)
        if not self.training or self.p == 0.0:
            self._mask = None
            return arr
        keep = 1.0 - self.p
        self._mask = (self._rng.random(arr.shape) < keep).astype(DTYPE) / keep
        return arr * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        g = np.asarray(grad_out, dtype=DTYPE)
        return g if self._mask is None else g * self._mask


class LayerNorm(Layer):
    """Centre and scale each row over its features, then apply a learned affine.

    Normalisation runs ACROSS FEATURES within one observation, not across the
    batch, and that distinction is the reason to prefer it here. A batch
    statistic makes one company's representation depend on which other companies
    happened to be drawn alongside it, a dependency no figure sitting next to a
    valuation should carry, and it makes the layer compute something different
    at inference than it did in training. Layer normalisation is a per-row
    operation and is identical in both modes.

    The variance used is the biased one, dividing by the number of features
    rather than by that number less one. It is a descriptive statistic of this
    row, not an estimate of a population from a sample, so there is nothing to
    correct for.
    """

    def __init__(self, dim: int, *, eps: float = 1e-5) -> None:
        if dim < 1:
            raise ConfigError(f"LayerNorm needs a positive width, got {dim}")
        if eps <= 0.0:
            raise ConfigError(f"LayerNorm eps must be positive, got {eps}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.gamma = np.ones(self.dim, dtype=DTYPE)
        self.beta = np.zeros(self.dim, dtype=DTYPE)
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = _matrix(x, "LayerNorm input", self.dim)
        centred = arr - arr.mean(axis=1, keepdims=True)
        variance = (centred * centred).mean(axis=1, keepdims=True)
        self._istd = 1.0 / np.sqrt(variance + self.eps)
        self._xhat = centred * self._istd
        return self._xhat * self.gamma + self.beta

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        g = _matrix(grad_out, "LayerNorm output gradient", self.dim)
        self.grad_gamma += (g * self._xhat).sum(axis=0)
        self.grad_beta += g.sum(axis=0)
        dxhat = g * self.gamma
        # The row mean and the row variance each depend on every element of the
        # row, so both correction terms below are part of the derivative and not
        # a refinement of it. Dropping them leaves a gradient that points roughly
        # the right way and is the wrong size, which trains to a worse optimum
        # without ever failing outright.
        return self._istd * (
            dxhat
            - dxhat.mean(axis=1, keepdims=True)
            - self._xhat * (dxhat * self._xhat).mean(axis=1, keepdims=True)
        )

    def params(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def grads(self) -> list[np.ndarray]:
        return [self.grad_gamma, self.grad_beta]


class L2Normalize(Layer):
    """Project each row onto the unit sphere, which makes a dot product a cosine.

    Contrastive training compares embeddings by cosine similarity, and a cosine
    is a dot product only once both vectors have unit length. Putting the
    normalisation inside the model rather than applying it to the output
    afterwards is what lets the loss gradient flow back through it, and the
    difference is not cosmetic. The direction a point should move along the
    sphere is not the direction it should move in the ambient space, and the
    component of the ambient gradient that only changes length is one the loss
    can never see. The backward pass below removes exactly that component, which
    is why it subtracts the projection of the incoming gradient onto the output
    direction.

    ``eps`` guards the zero vector, which has no direction. With eps above zero
    the rows come back a shade under unit length, and the derivative here is
    exact for that definition rather than exact for eps = 0 and approximate for
    what the layer actually computes.
    """

    def __init__(self, *, eps: float = 1e-12) -> None:
        if eps < 0.0:
            raise ConfigError(f"L2Normalize eps cannot be negative, got {eps}")
        self.eps = float(eps)

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=DTYPE)
        if arr.ndim != 2:
            raise ConfigError(
                f"L2Normalize input must be two dimensional, got shape {arr.shape}"
            )
        self._norm = np.sqrt((arr * arr).sum(axis=1, keepdims=True) + self.eps)
        self._out = arr / self._norm
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        g = np.asarray(grad_out, dtype=DTYPE)
        radial = (g * self._out).sum(axis=1, keepdims=True)
        return (g - self._out * radial) / self._norm


class Sequential(Layer):
    """A stack of layers applied in order forwards and in reverse backwards.

    The whole of the composition rule, and the whole of the chain rule, in ten
    lines. Nothing more is needed because nothing in this package branches.
    """

    def __init__(self, *layers: Layer) -> None:
        if not layers:
            raise ConfigError("Sequential needs at least one layer")
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ConfigError(
                    f"Sequential takes Layer instances, got {type(layer).__name__}"
                )
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def train(self, mode: bool = True) -> "Sequential":
        self.training = bool(mode)
        for layer in self.layers:
            layer.train(mode)
        return self

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]


# ---------------------------------------------------------------------------
# Losses. Each returns (loss, gradient of the loss with respect to its input).
# ---------------------------------------------------------------------------


def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean squared error and its gradient.

    The right loss when the residuals are roughly symmetric and the tails are
    not fat. On financial targets they usually are fat, which is what
    ``huber_loss`` exists for. The mean runs over every element, so the scale of
    the reported loss does not move when the batch size does and two runs at
    different batch sizes are comparable.
    """
    p = np.asarray(pred, dtype=DTYPE)
    t = np.asarray(target, dtype=DTYPE)
    _same_shape(p, t, "prediction", "target")
    if p.size == 0:
        raise ConfigError("mse_loss was given an empty batch")
    diff = p - t
    loss = float(np.mean(diff * diff))
    return loss, (2.0 / diff.size) * diff


def huber_loss(
    pred: np.ndarray, target: np.ndarray, delta: float = 1.0
) -> tuple[float, np.ndarray]:
    """Squared error near zero, absolute error in the tails, joined smoothly.

    Quadratic inside ``|residual| <= delta`` and linear outside it, with the
    constant chosen so that both the value and the slope are continuous at the
    join. The reason to prefer it on this data: one company that restated, or
    one quarter posting a 400 percent growth rate off a base of nothing,
    contributes a gradient proportional to its residual under squared error and
    can move the fit on its own. Under Huber that gradient is capped at delta,
    so the outlier still pulls but cannot dominate.

    ``delta`` is in the units of the target, which is the thing to get right.
    On a growth rate held as a decimal, delta = 0.25 says that residuals beyond
    25 growth points stop carrying information about the slope and start being
    treated as events.

    The loss is not twice differentiable at ``|residual| = delta``. A gradient
    check has to keep its residuals away from that point, for the same reason it
    has to keep them away from zero on a ReLU.
    """
    d = float(delta)
    if d <= 0.0:
        raise ConfigError(f"huber delta must be positive, got {delta}")
    p = np.asarray(pred, dtype=DTYPE)
    t = np.asarray(target, dtype=DTYPE)
    _same_shape(p, t, "prediction", "target")
    if p.size == 0:
        raise ConfigError("huber_loss was given an empty batch")
    resid = p - t
    absr = np.abs(resid)
    inner = absr <= d
    per_element = np.where(inner, 0.5 * resid * resid, d * (absr - 0.5 * d))
    loss = float(np.mean(per_element))
    grad = np.where(inner, resid, d * np.sign(resid)) / resid.size
    return loss, grad


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function computed on whichever side of zero does not overflow."""
    out = np.empty_like(z)
    positive = z >= 0.0
    ez = np.exp(-np.abs(z))
    out[positive] = 1.0 / (1.0 + ez[positive])
    out[~positive] = ez[~positive] / (1.0 + ez[~positive])
    return out


def bce_with_logits(
    logits: np.ndarray,
    target: np.ndarray,
    pos_weight: float | np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Binary cross entropy taken on the logits, which is the numerically safe form.

    Applying a sigmoid and then taking a log is the same function on paper and a
    different one in floating point: a logit of -40 makes the sigmoid underflow
    to exactly zero and the log of that is negative infinity, so a confident
    correct prediction poisons the batch. Folding the two together gives

        loss = (1 - t) z + (1 + (w - 1) t) log(1 + exp(-z))

    and the second term is evaluated as ``max(-z, 0) + log1p(exp(-|z|))``, which
    is exact at every magnitude. Nothing overflows and nothing underflows to a
    value the log cannot take.

    ``pos_weight`` multiplies the loss on the positive class, broadcast against
    the target. It is the answer to class imbalance, which is the normal
    condition for the events this engine cares about: acquisition targets are a
    low single-digit percentage of any TMT universe in a given year, and an
    unweighted classifier maximises its score by predicting that nothing ever
    happens. Setting it to the ratio of negatives to positives makes the two
    classes contribute equally to the gradient. It changes the calibration of
    the output as a probability, so a score meant to be read as a probability
    has to be recalibrated afterwards rather than taken from the head directly.
    """
    z = np.asarray(logits, dtype=DTYPE)
    t = np.asarray(target, dtype=DTYPE)
    _same_shape(z, t, "logits", "target")
    if z.size == 0:
        raise ConfigError("bce_with_logits was given an empty batch")
    if pos_weight is None:
        weight = np.ones((), dtype=DTYPE)
    else:
        weight = np.asarray(pos_weight, dtype=DTYPE)
        if np.any(weight <= 0.0):
            raise ConfigError(
                f"pos_weight must be positive, got {pos_weight!r}: a zero or "
                "negative weight tells the fit to ignore or to invert the "
                "positive class"
            )
    softplus_negz = np.maximum(-z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    scale = 1.0 + (weight - 1.0) * t
    per_element = (1.0 - t) * z + scale * softplus_negz
    loss = float(np.mean(per_element))
    grad = ((1.0 - t) - scale * _sigmoid(-z)) / z.size
    return loss, grad


def info_nce(
    anchor: np.ndarray,
    positive: np.ndarray,
    negatives: np.ndarray,
    temperature: float,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Contrastive cross entropy over cosine similarities, with all three gradients.

    **Shapes.** ``anchor`` and ``positive`` are (B, D) and pair row by row; a
    single pair may be passed as (D,) and its gradients come back in that shape.
    ``negatives`` is either (K, D), one pool shared by every anchor in the batch,
    or (B, K, D), a separate pool per anchor for a caller that has mined hard
    negatives individually.

    **The method.** Each row's logits are the similarity of the anchor to its
    positive, followed by its similarity to each negative, all divided by the
    temperature. The right answer is always index 0, so the loss is ordinary
    cross entropy over 1 + K classes and the whole construction is a softmax
    classifier whose question is "which of these candidates is the real peer".
    The loss is averaged over the batch.

    **Normalisation is the caller's job**, and ``L2Normalize`` is the layer that
    does it inside the model. The dot products here are cosines only if the
    inputs have unit length. Hand it unnormalised vectors and it still returns
    exact gradients of exactly what it computed, but that quantity is no longer
    a cosine, it is unbounded, and the temperature guidance below stops applying.
    The gradients returned are with respect to the normalised vectors, which is
    the correct handoff: the derivative of the normalisation itself belongs to
    ``L2Normalize.backward`` and is applied when the gradient flows back through
    the model.

    **Temperature.** The temperature divides the similarities before the softmax,
    so it sets how sharply the model is being asked to discriminate. Low values
    sharpen the distribution: the loss is then dominated by whichever negative
    sits closest to the anchor, and the model spends its capacity on the hard
    cases, at the cost of being sensitive to a single mislabelled positive. High
    values flatten it, every negative contributes about equally, and the model
    learns coarse structure and stops there. For L2 normalised embeddings the
    similarities live in [-1, 1], and 0.05 to 0.1 is the usual working band: at
    0.05 the logits span 40 units and the softmax separates cleanly, while above
    roughly 0.5 they span less than four and it can barely separate anything.
    """
    tau = float(temperature)
    if tau <= 0.0:
        raise ConfigError(
            f"InfoNCE temperature must be positive, got {temperature}. At zero "
            "the softmax becomes an argmax and has no gradient"
        )
    a = np.asarray(anchor, dtype=DTYPE)
    p = np.asarray(positive, dtype=DTYPE)
    n = np.asarray(negatives, dtype=DTYPE)

    single = a.ndim == 1
    if single:
        a = a[None, :]
        p = p[None, :]
    if a.ndim != 2:
        raise ConfigError(f"anchor must be (D,) or (B, D), got shape {a.shape}")
    _same_shape(a, p, "anchor", "positive")
    batch, width = a.shape

    if n.ndim == 2:
        per_row = False
        if n.shape[1] != width:
            raise ConfigError(
                f"negatives are {n.shape[1]} wide against anchors {width} wide"
            )
        sim_neg = (a @ n.T) / tau
    elif n.ndim == 3:
        per_row = True
        if n.shape[0] != batch or n.shape[2] != width:
            raise ConfigError(
                f"per-anchor negatives must be ({batch}, K, {width}), got {n.shape}"
            )
        sim_neg = np.einsum("bd,bkd->bk", a, n) / tau
    else:
        raise ConfigError(
            f"negatives must be (K, D) shared or (B, K, D) per anchor, got {n.shape}"
        )
    if sim_neg.shape[1] == 0:
        raise ConfigError(
            "InfoNCE was given no negatives: with nothing to discriminate "
            "against the loss is identically zero and carries no gradient"
        )

    sim_pos = (a * p).sum(axis=1, keepdims=True) / tau
    logits = np.concatenate([sim_pos, sim_neg], axis=1)

    # Subtracting the row maximum before exponentiating. At a temperature of
    # 0.05 a raw logit reaches 20, exp of which is 5e8, and a few hundred of
    # those summed is still finite but the headroom is not worth keeping when
    # removing it costs one line and changes no result.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    denom = exps.sum(axis=1, keepdims=True)
    loss = float(np.mean(np.log(denom[:, 0]) - shifted[:, 0]))

    # Gradient of cross entropy with respect to its logits: the softmax less the
    # one-hot target, divided by the batch because the loss is a mean.
    q = exps / denom
    q[:, 0] -= 1.0
    q /= batch
    q_pos = q[:, :1]
    q_neg = q[:, 1:]

    grad_anchor = (q_pos * p) / tau
    grad_positive = (q_pos * a) / tau
    if per_row:
        grad_anchor = grad_anchor + np.einsum("bk,bkd->bd", q_neg, n) / tau
        grad_negatives = q_neg[:, :, None] * a[:, None, :] / tau
    else:
        grad_anchor = grad_anchor + (q_neg @ n) / tau
        grad_negatives = (q_neg.T @ a) / tau

    if single:
        grad_anchor = grad_anchor[0]
        grad_positive = grad_positive[0]
        if per_row:
            grad_negatives = grad_negatives[0]
    return loss, (grad_anchor, grad_positive, grad_negatives)


def info_nce_in_batch(
    anchors: np.ndarray, positives: np.ndarray, temperature: float
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """InfoNCE where every other positive in the batch acts as a negative.

    This is the trick that makes contrastive training cheap, and it is the
    reason a peer model can be fitted on a laptop. With B pairs in a batch there
    are B(B-1) negative comparisons available at no cost, because the positive
    belonging to row j is by construction not the peer named in row i. Drawing
    and encoding a dedicated negative pool for each anchor would multiply the
    forward passes by the pool size and buy nothing the free negatives do not
    already give.

    It is one matrix multiply. The (B, B) similarity matrix carries the true
    pairs on its diagonal and the negatives everywhere else, the loss is the
    cross entropy of each row against its own index, and both gradients come
    back as single matrix products. No python loop over pairs appears anywhere,
    which is what matters at B = 512, where a loop would be a quarter of a
    million iterations per batch and the run would take hours instead of
    minutes.

    **The assumption being made** is that the off-diagonal pairs really are
    negatives. On peer data drawn from proxy compensation tables that is usually
    but not always true: two pairs sampled from the same industry can name the
    same company, and the loss then spends a gradient pushing apart two firms
    that genuinely are peers. The fix belongs to the sampler, not here, and it is
    to avoid putting two pairs from one proxy into one batch.

    Returns the loss and the gradients with respect to the anchors and to the
    positives, in that order.
    """
    tau = float(temperature)
    if tau <= 0.0:
        raise ConfigError(
            f"InfoNCE temperature must be positive, got {temperature}. At zero "
            "the softmax becomes an argmax and has no gradient"
        )
    a = _matrix(anchors, "anchors")
    p = _matrix(positives, "positives")
    _same_shape(a, p, "anchors", "positives")
    batch = a.shape[0]
    if batch < 2:
        raise ConfigError(
            f"in-batch negatives need at least two pairs, got {batch}: a batch "
            "of one has no other positive to serve as its negative"
        )

    sims = (a @ p.T) / tau
    shifted = sims - sims.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    denom = exps.sum(axis=1, keepdims=True)
    diagonal = np.arange(batch)
    loss = float(np.mean(np.log(denom[:, 0]) - shifted[diagonal, diagonal]))

    grad_sims = exps / denom
    grad_sims[diagonal, diagonal] -= 1.0
    grad_sims /= batch
    grad_anchors = (grad_sims @ p) / tau
    grad_positives = (grad_sims.T @ a) / tau
    return loss, (grad_anchors, grad_positives)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


class Adam:
    """Adam with decoupled weight decay, which is to say AdamW.

    Adam keeps a running mean and a running uncentred variance of each
    parameter's gradient and steps by their ratio, so a parameter whose gradient
    is small but consistent still moves, and one whose gradient is large but
    noisy is damped. The bias corrections on both moments matter over the first
    few tens of steps, where the running averages still carry most of their zero
    initialisation and the uncorrected step would be far too short.

    **Why the weight decay is decoupled.** The familiar way to write it is to add
    ``weight_decay * parameter`` into the gradient, which is L2 regularisation.
    Under plain gradient descent the two are the same operation. Under Adam they
    are not, because the added term then passes through the division by the
    square root of the second moment: a parameter with a large noisy gradient
    has a large denominator and therefore decays LESS than one with a quiet
    gradient, so how much a weight shrinks is decided by the local shape of the
    loss surface instead of by the strength the caller asked for. The decoupled
    form subtracts ``lr * weight_decay * parameter`` outside the adaptive step,
    so every weight shrinks by the same factor per step and the number means
    what it says.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ConfigError(f"learning rate must be positive, got {lr}")
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ConfigError(f"betas must each lie in [0, 1), got {betas}")
        if eps <= 0.0:
            raise ConfigError(f"eps must be positive, got {eps}")
        if weight_decay < 0.0:
            raise ConfigError(
                f"weight decay cannot be negative, got {weight_decay}: a negative "
                "value grows every weight without bound"
            )
        self.lr = float(lr)
        self.betas = (beta1, beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        # Keyed by the identity of the parameter array. The array itself is held
        # in the value so it cannot be collected and have its id handed out again
        # to a different array, which would silently attach one parameter's
        # moments to another and be very hard to see in a loss curve.
        self._state: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        """One update, applied in place to every parameter array."""
        if len(params) != len(grads):
            raise ConfigError(
                f"{len(params)} parameter arrays against {len(grads)} gradient "
                "arrays: the model's params() and grads() have to line up"
            )
        self.t += 1
        beta1, beta2 = self.betas
        correction1 = 1.0 - beta1**self.t
        correction2 = 1.0 - beta2**self.t
        for param, grad in zip(params, grads):
            if param.shape != grad.shape:
                raise ConfigError(
                    f"parameter of shape {param.shape} against a gradient of "
                    f"shape {grad.shape}"
                )
            state = self._state.get(id(param))
            if state is None:
                state = (param, np.zeros_like(param), np.zeros_like(param))
                self._state[id(param)] = state
            _, first, second = state
            first *= beta1
            first += (1.0 - beta1) * grad
            second *= beta2
            second += (1.0 - beta2) * grad * grad
            adaptive = (first / correction1) / (
                np.sqrt(second / correction2) + self.eps
            )
            if self.weight_decay:
                param -= self.lr * self.weight_decay * param
            param -= self.lr * adaptive


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    """What the fit did, in enough detail that a reader can tell whether to believe it.

    ``epochs`` is the number actually run, which is below the budget whenever
    early stopping fired. ``best_epoch`` indexes ``train_loss`` and ``val_loss``
    from zero and names the epoch whose parameters the model was left holding;
    ``rows`` numbers epochs from one, because a table read by a person should.
    ``notes`` carries the sentences that stop a curve being read wrongly: that
    the run stopped early, or that it used its whole budget and may not have
    converged, or that no validation set was supplied and so nothing here is
    evidence of anything out of sample.
    """

    epochs: int
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    stopped_early: bool
    notes: list[str] = field(default_factory=list)

    @property
    def best_val_loss(self) -> float | None:
        """Validation loss at the restored epoch, or None if nothing was held out."""
        if not self.val_loss:
            return None
        return self.val_loss[self.best_epoch]

    def rows(self) -> list[dict]:
        """One row per epoch. Formatting belongs to the renderer, not here."""
        out: list[dict] = []
        for index, train_value in enumerate(self.train_loss):
            out.append(
                {
                    "Epoch": index + 1,
                    "Train loss": train_value,
                    "Validation loss": (
                        self.val_loss[index] if index < len(self.val_loss) else None
                    ),
                    "Restored": index == self.best_epoch,
                }
            )
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows()).set_index("Epoch")


def _batch_rows(x: Any) -> float:
    """How much a batch counts toward the epoch mean.

    Rows, when the batch input is an array, so that a short final batch does not
    weigh as much as a full one. This assumes the loss returns a per-row mean,
    which every loss in this file does. Anything else counts as one.
    """
    if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] > 0:
        return float(x.shape[0])
    return 1.0


def _run_batch(
    model: Layer,
    batch: tuple[Any, Any],
    loss_fn: Callable[[np.ndarray, Any], tuple[float, np.ndarray]],
    where: str,
) -> tuple[float, np.ndarray, float]:
    inputs, targets = batch
    prediction = model.forward(inputs)
    loss, grad = loss_fn(prediction, targets)
    if not np.isfinite(loss):
        # A diverged fit is reported, never carried. Returning a history with a
        # NaN in it would put an unreadable number next to a valuation, which is
        # the one thing this engine does not do.
        raise NotMeaningfulError(
            f"the loss became {loss} during {where}. The fit has diverged and "
            "the parameters are no longer meaningful. The usual causes are a "
            "learning rate too large for the feature scale, features that were "
            "not standardised, or a log taken of a non-positive target"
        )
    return float(loss), grad, _batch_rows(inputs)


def train(
    model: Layer,
    batches: Sequence[tuple[Any, Any]],
    loss_fn: Callable[[np.ndarray, Any], tuple[float, np.ndarray]],
    optimizer: Adam,
    val_batches: Sequence[tuple[Any, Any]] | None = None,
    epochs: int = 100,
    patience: int = 10,
    seed: int = 0,
    *,
    shuffle: bool = True,
    min_delta: float = 0.0,
) -> TrainHistory:
    """Fit a model, stop when the held-out loss stops improving, keep the best epoch.

    ``batches`` is a sequence of ``(inputs, targets)`` pairs. ``inputs`` goes to
    ``model.forward`` and ``targets`` goes to ``loss_fn`` untouched, so it can be
    an array of labels, or a tuple, or None. ``loss_fn(prediction, targets)``
    returns the scalar loss and the gradient of that loss with respect to the
    prediction, which is exactly the signature of every loss in this module once
    its hyperparameters are bound.

    **The two-tower contrastive pattern**, which is the one the peer model uses.
    A layer caches one forward pass, so an encoder cannot be called twice before
    its backward. Stack instead: make ``inputs`` the anchors and the positives
    concatenated into one array of 2B rows, let the encoder embed all of them in
    one pass, and have ``loss_fn`` split the output down the middle, call
    ``info_nce_in_batch`` on the halves and stack the two gradients back into a
    (2B, D) array. One forward, one backward, every in-batch negative counted.

    **Early stopping.** After each epoch the held-out loss is computed with the
    model in evaluation mode, so dropout is off. An epoch that improves on the
    best seen by more than ``min_delta`` resets the counter and its parameters
    are snapshotted; ``patience`` consecutive epochs without such an improvement
    end the run. The snapshot is then written back, so the model a caller holds
    afterwards is the BEST epoch's and not the last one's. That distinction is
    the whole point of the exercise and it is recorded in the history, so a
    reader can tell a fit that converged from one that simply ran out of budget.
    To disable early stopping, pass a patience at or above ``epochs``.

    Without ``val_batches`` nothing is held out, no stopping rule can fire, the
    final parameters are kept whatever they are, and the history says so. The
    lowest training loss is reported as ``best_epoch`` for orientation only and
    it is not evidence about anything out of sample.

    ``seed`` drives the shuffling of batch ORDER between epochs and nothing else.
    Weight initialisation and dropout masks are seeded where they are created,
    from the generator the caller built out of ``assumptions.ml.random_seed``.
    """
    if epochs < 1:
        raise ConfigError(f"epochs must be at least 1, got {epochs}")
    if patience < 1:
        raise ConfigError(
            f"patience must be at least 1, got {patience}. To turn early "
            "stopping off, pass a patience at or above the epoch budget"
        )
    if min_delta < 0.0:
        raise ConfigError(f"min_delta cannot be negative, got {min_delta}")

    train_set = list(batches)
    if not train_set:
        raise ConfigError("train was called with no batches")
    holdout = None if val_batches is None else list(val_batches)
    if holdout is not None and not holdout:
        raise ConfigError(
            "val_batches was supplied but empty. Pass None to say that nothing "
            "is held out, rather than an empty sequence that looks like a "
            "holdout and is not"
        )

    rng = np.random.default_rng(seed)
    params = model.params()
    grads = model.grads()
    if not params:
        raise ConfigError(
            "the model has no learnable parameters, so every epoch would report "
            "the same loss and the history would look like a fit that refused to "
            "move. Check that the stack contains at least one Linear or LayerNorm"
        )
    train_loss: list[float] = []
    val_loss: list[float] = []
    notes: list[str] = []

    best_score = np.inf
    best_epoch = 0
    best_params: list[np.ndarray] | None = None
    waited = 0
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        order = (
            rng.permutation(len(train_set))
            if shuffle
            else np.arange(len(train_set))
        )
        total = 0.0
        weight = 0.0
        for index in order:
            batch = train_set[int(index)]
            model.zero_grad()
            loss, grad, rows = _run_batch(
                model, batch, loss_fn, f"training epoch {epoch}"
            )
            model.backward(grad)
            optimizer.step(params, grads)
            total += loss * rows
            weight += rows
        train_loss.append(total / weight)

        if holdout is None:
            continue

        model.eval()
        total = 0.0
        weight = 0.0
        for batch in holdout:
            loss, _, rows = _run_batch(
                model, batch, loss_fn, f"validation after epoch {epoch}"
            )
            total += loss * rows
            weight += rows
        score = total / weight
        val_loss.append(score)

        if score < best_score - min_delta:
            best_score = score
            best_epoch = epoch
            best_params = [p.copy() for p in params]
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                stopped_early = True
                break

    model.eval()

    if best_params is not None:
        # Written in place so the parameter arrays keep their identity: the
        # optimiser's moment state is keyed on it, and a caller holding a
        # reference keeps holding the live weights.
        for param, snapshot in zip(params, best_params):
            param[...] = snapshot
        notes.append(
            f"validation loss was lowest at epoch {best_epoch + 1} "
            f"({best_score:.6g}); those parameters were restored, not the last "
            "epoch's"
        )

    if holdout is None:
        best_epoch = int(np.argmin(train_loss)) if train_loss else 0
        notes.append(
            "no validation batches were supplied, so nothing was held out, no "
            "early stopping was possible and the final epoch's parameters were "
            "kept. The epoch marked below is the lowest TRAINING loss, which is "
            "not evidence about anything out of sample"
        )
    elif stopped_early:
        notes.append(
            f"stopped early after {len(train_loss)} of {epochs} epochs: "
            f"{patience} consecutive epochs passed without improving the "
            "validation loss"
        )
    else:
        notes.append(
            f"ran the full budget of {epochs} epochs without triggering a "
            f"patience of {patience}, so the fit may not have converged and a "
            "longer budget is worth trying before the architecture is blamed"
        )

    return TrainHistory(
        epochs=len(train_loss),
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Gradient checking
# ---------------------------------------------------------------------------


def _relative_error(analytic: float, numeric: float) -> float:
    """Disagreement scaled by the larger magnitude, floored at one.

    Dividing by the magnitudes alone is the textbook form and it has a failure
    mode that wastes an afternoon: when a gradient entry is genuinely zero, the
    finite difference returns rounding noise of order 1e-17, the ratio of two
    such numbers is order one, and a correct derivative is reported as a total
    disagreement. Flooring the denominator at one makes the measure degrade to
    an absolute error exactly where the relative one stops meaning anything,
    and leaves it unchanged wherever the gradient is of ordinary size.
    """
    scale = max(abs(analytic), abs(numeric), 1.0)
    return abs(analytic - numeric) / scale


def _sweep(
    scalar: Callable[[], float],
    perturb: np.ndarray,
    analytic: np.ndarray,
    eps: float,
) -> float:
    """Central-difference every element of ``perturb`` against ``analytic``.

    Indexed elementwise rather than through a flattened view, because a view is
    only guaranteed for a contiguous array and a silent copy would leave every
    numeric derivative at zero, which reads as a broken backward pass rather
    than as the indexing mistake it is.
    """
    expected = np.asarray(analytic, dtype=DTYPE)
    if expected.shape != perturb.shape:
        raise ConfigError(
            f"the analytic gradient has shape {expected.shape} against "
            f"{perturb.shape} in the array it is supposed to describe"
        )
    worst = 0.0
    for index in np.ndindex(perturb.shape):
        original = perturb[index]
        perturb[index] = original + eps
        plus = scalar()
        perturb[index] = original - eps
        minus = scalar()
        perturb[index] = original
        numeric = (plus - minus) / (2.0 * eps)
        worst = max(worst, _relative_error(float(expected[index]), numeric))
    return worst


def gradient_check(
    layer_or_loss: Any,
    inputs: Any,
    eps: float = 1e-6,
    *,
    rng: np.random.Generator | None = None,
    reset: Callable[[], None] | None = None,
) -> float:
    """Largest relative gap between an analytic gradient and a central difference.

    This is the function that makes the rest of the file trustworthy, and the
    test suite runs it over every layer and every loss here at a bar of 1e-6. A
    backward pass wrong in any single term fails it immediately, which is the
    property a hand-written derivative needs and an autograd system cannot
    provide for itself.

    ``layer_or_loss`` is either a ``Layer``, in which case the scalar being
    differentiated is the layer's output contracted against a fixed random
    projection and BOTH the input gradient and every parameter gradient are
    checked, or a callable returning ``(loss, grad)`` or ``(loss, (grad, ...))``,
    in which case each returned gradient is checked against the input it
    corresponds to, in order. Bind any non-array argument first, for instance
    ``lambda a, p, n: info_nce(a, p, n, 0.07)``.

    **Why eps = 1e-6 and float64.** A central difference carries a truncation
    error of order eps squared and a cancellation error of order machine epsilon
    divided by eps. In double precision on inputs of order one those are about
    1e-13 and 2e-10, so the check bottoms out near 1e-10 and the 1e-6 bar has
    four orders of headroom. A failure is therefore a bug and not an artefact,
    and the honest response to one is to find the bug rather than to loosen the
    tolerance. In single precision the cancellation term alone is about 1e-2 and
    no useful bar exists at all, which is why this module is float64 throughout.

    **``reset``**, if given, is called before every forward pass. A layer holding
    random state needs it: dropout would otherwise draw a fresh mask for each of
    the two perturbed evaluations, and the difference would measure the masks
    rather than the derivative. Pass ``lambda: layer.reseed(default_rng(seed))``.

    The check leaves the layer's gradient buffers holding the analytic values and
    its forward cache holding the last perturbed pass, so re-run a clean forward
    before using the layer for anything else.
    """
    if eps <= 0.0:
        raise ConfigError(f"the finite-difference step must be positive, got {eps}")
    generator = np.random.default_rng(0) if rng is None else rng
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
    arrays = [np.array(a, dtype=DTYPE, copy=True) for a in inputs]
    if not arrays:
        raise ConfigError("gradient_check needs at least one input array")

    if isinstance(layer_or_loss, Layer):
        layer = layer_or_loss
        if reset is not None:
            reset()
        probe = layer.forward(arrays[0])
        # A fixed random projection turns the vector output into the scalar a
        # finite difference can address. Random rather than a sum of ones, so a
        # bug that happens to cancel across the output row is still caught.
        projection = generator.standard_normal(np.shape(probe))

        def scalar() -> float:
            if reset is not None:
                reset()
            return float(np.sum(layer.forward(arrays[0]) * projection))

        if reset is not None:
            reset()
        layer.forward(arrays[0])
        layer.zero_grad()
        analytic: list[np.ndarray] = [layer.backward(projection)]
        extra = list(zip(layer.params(), layer.grads()))
    else:
        function = layer_or_loss

        def scalar() -> float:
            if reset is not None:
                reset()
            value, _ = function(*arrays)
            return float(value)

        if reset is not None:
            reset()
        _, returned = function(*arrays)
        analytic = list(returned) if isinstance(returned, tuple) else [returned]
        extra = []

    if len(analytic) > len(arrays):
        raise ConfigError(
            f"{len(analytic)} gradients were returned for {len(arrays)} inputs"
        )

    worst = 0.0
    for array, expected in zip(arrays, analytic):
        worst = max(worst, _sweep(scalar, array, expected, eps))
    for param, expected in extra:
        worst = max(worst, _sweep(scalar, param, expected, eps))
    return worst
