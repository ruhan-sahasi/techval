"""Gradient checks, the contrastive loss, Adam, early stopping and determinism.

The gradient checks are the load-bearing tests in this file and the reason the
rest of the package can be believed. Every layer and every loss is differenced
numerically against its analytic derivative at a bar of 1e-6, and two of the
tests deliberately break a backward pass to show that the check would catch it
rather than quietly passing everything put in front of it.

Nothing here touches the network or the fixtures. The module under test sees
standardised features and returns numbers in those units, so the inputs are
drawn from a seeded generator and the assertions are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from techval.errors import ConfigError, NotMeaningfulError
from techval.ml import nn

# The bar the assignment sets, and the one the module's float64 arithmetic can
# actually clear. Measured errors come in around 1e-9, so there are three orders
# of headroom and a failure here means a bug rather than rounding.
TOLERANCE = 1e-6

SEED = 20260910


def unit_rows(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Rows on the unit sphere, which is the state InfoNCE expects its inputs in."""
    v = rng.standard_normal(shape)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def clear_of_zero(x: np.ndarray, margin: float = 0.2) -> np.ndarray:
    """Push values off a kink, so a finite difference never straddles it.

    ReLU at zero and Huber at the join have no second derivative, and a central
    difference across such a point returns the average of the two slopes. That
    is a property of the function, not a bug in the backward pass, so the check
    is run where the function is differentiable.
    """
    return np.where(np.abs(x) < margin, np.sign(x + 1e-16) * margin, x)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Gradient checks: every layer, every loss
# ---------------------------------------------------------------------------


def deep_stack(x: np.ndarray) -> nn.Sequential:
    """A full stack, at a seed where the check's two preconditions hold.

    Two points cannot be differenced numerically, and neither is a defect in the
    module. A ReLU pre-activation sitting on zero has no derivative at all. A row
    reaching L2Normalize with every element zero has no direction: the eps guard
    keeps it finite, at the price of a curvature scale of 1e-6 that a step of
    1e-6 cannot resolve, and a whole row can die that way when the ReLU above
    happens to zero all of it.

    The assertions below state both preconditions out loud, so a later change to
    the seed or the widths fails with a reason instead of with an unexplained
    gradient error that looks like a broken backward pass.
    """
    generator = np.random.default_rng(36)
    stack = nn.Sequential(
        nn.Linear(4, 6, rng=generator, init="he"),
        nn.ReLU(),
        nn.LayerNorm(6),
        nn.Linear(6, 3, rng=generator, init="xavier"),
        nn.Tanh(),
        nn.L2Normalize(),
    )
    pre_activation = stack[0].forward(x)
    assert np.abs(pre_activation).min() > 0.1, "a relu pre-activation sits on its kink"
    embedding = x
    for layer in stack.layers[:-1]:
        embedding = layer.forward(embedding)
    assert (
        np.linalg.norm(embedding, axis=1).min() > 0.1
    ), "a row reaches L2Normalize with no direction to normalise"
    return stack


def layer_cases(rng: np.random.Generator) -> list[tuple[str, object, np.ndarray, object]]:
    """Each layer with an input it is differentiable at, and a reset hook if it needs one."""
    x = rng.standard_normal((5, 4)) * 1.4
    dropout = nn.Dropout(0.35, rng=np.random.default_rng(4))
    evaluating = nn.Dropout(0.5, rng=np.random.default_rng(4)).eval()
    deep = deep_stack(x)
    return [
        ("Linear he", nn.Linear(4, 3, rng=rng, init="he"), x, None),
        ("Linear xavier", nn.Linear(4, 3, rng=rng, init="xavier"), x, None),
        ("Linear no bias", nn.Linear(4, 3, rng=rng, bias=False), x, None),
        ("ReLU", nn.ReLU(), clear_of_zero(x), None),
        ("Tanh", nn.Tanh(), x, None),
        (
            "Dropout training",
            dropout,
            x,
            lambda: dropout.reseed(np.random.default_rng(4)),
        ),
        ("Dropout evaluating", evaluating, x, None),
        ("Dropout p=0", nn.Dropout(0.0, rng=np.random.default_rng(4)), x, None),
        ("LayerNorm", nn.LayerNorm(4), x, None),
        ("L2Normalize", nn.L2Normalize(), x, None),
        ("Sequential", deep, x, None),
    ]


def test_every_layer_matches_its_finite_difference(rng):
    """The central test. A backward pass wrong in one term fails here."""
    for name, layer, x, reset in layer_cases(rng):
        worst = nn.gradient_check(layer, x, reset=reset)
        assert worst < TOLERANCE, f"{name} analytic gradient is off by {worst:.3e}"


def loss_cases(rng: np.random.Generator) -> list[tuple[str, object, list[np.ndarray]]]:
    """Each loss with the inputs it is differentiable at."""
    pred = rng.standard_normal((6, 3))
    target = rng.standard_normal((6, 3))
    # Residuals held clear of the Huber join at delta, for the same reason the
    # ReLU input is held clear of zero.
    huber_target = pred - clear_of_zero(pred - target, margin=0.25)
    huber_target[0, 0] = pred[0, 0] - 3.0
    huber_target[1, 1] = pred[1, 1] + 2.5
    labels = (rng.random((6, 3)) > 0.5).astype(float)
    anchor = unit_rows(rng, (4, 5))
    positive = unit_rows(rng, (4, 5))
    return [
        ("mse", nn.mse_loss, [pred, target]),
        ("huber", lambda p, t: nn.huber_loss(p, t, 0.7), [pred, huber_target]),
        ("bce", nn.bce_with_logits, [pred, labels]),
        ("bce pos_weight", lambda z, t: nn.bce_with_logits(z, t, 6.0), [pred, labels]),
        (
            "info_nce shared negatives",
            lambda a, p, n: nn.info_nce(a, p, n, 0.07),
            [anchor, positive, unit_rows(rng, (7, 5))],
        ),
        (
            "info_nce per-anchor negatives",
            lambda a, p, n: nn.info_nce(a, p, n, 0.07),
            [anchor, positive, unit_rows(rng, (4, 3, 5))],
        ),
        (
            "info_nce single pair",
            lambda a, p, n: nn.info_nce(a, p, n, 0.05),
            [unit_rows(rng, (5,)), unit_rows(rng, (5,)), unit_rows(rng, (6, 5))],
        ),
        (
            "info_nce in batch",
            lambda a, p: nn.info_nce_in_batch(a, p, 0.08),
            [anchor, positive],
        ),
    ]


def test_every_loss_matches_its_finite_difference(rng):
    """Including all three InfoNCE gradients, which is where a sign error hides."""
    for name, loss, inputs in loss_cases(rng):
        worst = nn.gradient_check(loss, inputs)
        assert worst < TOLERANCE, f"{name} analytic gradient is off by {worst:.3e}"


class IdentityBackwardTanh(nn.Tanh):
    """Tanh that forgets to differentiate itself, the commonest hand-written slip."""

    def backward(self, grad_out):
        return np.asarray(grad_out, dtype=float)


class UncorrectedLayerNorm(nn.LayerNorm):
    """LayerNorm differentiated as though the mean and variance were constants."""

    def backward(self, grad_out):
        g = np.asarray(grad_out, dtype=float)
        self.grad_gamma += (g * self._xhat).sum(axis=0)
        self.grad_beta += g.sum(axis=0)
        return self._istd * (g * self.gamma)


def test_the_gradient_check_has_teeth(rng):
    """A check that passes everything proves nothing, so break two layers on purpose.

    The second case is the one the LayerNorm comment claims: dropping the mean
    and variance corrections leaves a gradient pointing roughly the right way and
    badly wrong in size. It fails the check by seven orders of magnitude.
    """
    x = rng.standard_normal((5, 4))
    assert nn.gradient_check(IdentityBackwardTanh(), x) > 0.01
    assert nn.gradient_check(UncorrectedLayerNorm(4), x) > 0.01


# ---------------------------------------------------------------------------
# InfoNCE
# ---------------------------------------------------------------------------


def test_info_nce_falls_as_the_positive_becomes_more_similar(rng):
    """The loss has to reward the thing it exists to reward."""
    anchor = unit_rows(rng, (5,))
    far = unit_rows(rng, (5,))
    negatives = unit_rows(rng, (9, 5))
    losses = []
    for weight in np.linspace(0.0, 1.0, 8):
        blend = (1.0 - weight) * far + weight * anchor
        blend = blend / np.linalg.norm(blend)
        loss, _ = nn.info_nce(anchor, blend, negatives, 0.07)
        losses.append(loss)
    assert all(later < earlier for earlier, later in zip(losses, losses[1:]))
    assert losses[0] > losses[-1] * 5


def test_the_gradient_pushes_the_positive_toward_the_anchor(rng):
    """A step against the gradient has to raise the cosine, or the sign is wrong."""
    anchor = unit_rows(rng, (1, 5))
    positive = unit_rows(rng, (1, 5))
    negatives = unit_rows(rng, (6, 5))
    _, (_, grad_positive, _) = nn.info_nce(anchor, positive, negatives, 0.07)
    stepped = positive - 1e-3 * grad_positive
    before = float(anchor[0] @ positive[0])
    after = float(anchor[0] @ stepped[0] / np.linalg.norm(stepped[0]))
    assert after > before


def test_low_temperature_concentrates_the_gradient_on_the_hardest_negative(rng):
    """The documented behaviour of the temperature, checked rather than asserted.

    One negative is planted close to the anchor and seven are planted opposite
    it. At a low temperature the softmax puts nearly all its non-target mass on
    the close one, so nearly all the negative gradient goes there and the model
    spends its capacity on the case that is actually hard. At a high temperature
    the mass spreads and the hard negative's share falls toward the one in eight
    it would get if every negative looked alike.
    """
    anchor = unit_rows(rng, (1, 4))
    positive = unit_rows(rng, (1, 4))
    hard = anchor[0] + 0.1 * unit_rows(rng, (4,))
    easy = -anchor + 0.3 * unit_rows(rng, (7, 4))
    negatives = np.vstack([hard[None, :], easy])
    negatives = negatives / np.linalg.norm(negatives, axis=1, keepdims=True)

    shares = {}
    for temperature in (0.05, 2.0):
        _, (_, _, grad_negatives) = nn.info_nce(
            anchor, positive, negatives, temperature
        )
        magnitude = np.linalg.norm(grad_negatives, axis=1)
        shares[temperature] = magnitude[0] / magnitude.sum()
    assert shares[0.05] > 0.95
    assert shares[2.0] < 0.45
    assert shares[0.05] > shares[2.0]


def test_in_batch_negatives_match_an_explicit_loop(rng):
    """The vectorised path has to compute what the obvious slow path computes.

    The loop below is what the matrix multiply replaces: for each row, call the
    general InfoNCE with the other rows' positives as its negatives, average the
    losses and scatter the gradients back. Agreement to machine precision is the
    evidence that the reshaping and the transposes in the fast path are right.
    """
    batch, width, temperature = 6, 4, 0.09
    anchors = unit_rows(rng, (batch, width))
    positives = unit_rows(rng, (batch, width))

    fast_loss, (fast_anchor, fast_positive) = nn.info_nce_in_batch(
        anchors, positives, temperature
    )

    slow_loss = 0.0
    slow_anchor = np.zeros_like(anchors)
    slow_positive = np.zeros_like(positives)
    for row in range(batch):
        others = [j for j in range(batch) if j != row]
        loss, (g_a, g_p, g_n) = nn.info_nce(
            anchors[row], positives[row], positives[others], temperature
        )
        slow_loss += loss / batch
        slow_anchor[row] += g_a / batch
        slow_positive[row] += g_p / batch
        for slot, j in enumerate(others):
            slow_positive[j] += g_n[slot] / batch

    assert fast_loss == pytest.approx(slow_loss, abs=1e-13)
    assert np.allclose(fast_anchor, slow_anchor, rtol=0, atol=1e-13)
    assert np.allclose(fast_positive, slow_positive, rtol=0, atol=1e-13)


def test_in_batch_loss_starts_at_log_of_the_batch_size(rng):
    """Orthogonal embeddings carry no information, and the loss says exactly that.

    With every similarity equal the softmax is uniform over B candidates and the
    cross entropy is log B. It is the number to compare a trained loss against:
    a contrastive model that has not moved off log B has learned nothing.
    """
    batch = 8
    identical = np.tile(unit_rows(rng, (1, 5)), (batch, 1))
    loss, _ = nn.info_nce_in_batch(identical, identical, 0.07)
    assert loss == pytest.approx(np.log(batch), abs=1e-12)


def test_info_nce_refuses_the_cases_that_carry_no_gradient(rng):
    anchor = unit_rows(rng, (3, 4))
    positive = unit_rows(rng, (3, 4))
    with pytest.raises(ConfigError, match="temperature"):
        nn.info_nce(anchor, positive, unit_rows(rng, (2, 4)), 0.0)
    with pytest.raises(ConfigError, match="no negatives"):
        nn.info_nce(anchor, positive, np.zeros((0, 4)), 0.07)
    with pytest.raises(ConfigError, match="at least two pairs"):
        nn.info_nce_in_batch(anchor[:1], positive[:1], 0.07)


# ---------------------------------------------------------------------------
# The other losses
# ---------------------------------------------------------------------------


def test_huber_caps_the_gradient_an_outlier_can_contribute():
    """The reason to use it: one restated quarter cannot drag the whole fit."""
    pred = np.array([[0.0, 0.0]])
    target = np.array([[-40.0, 0.1]])
    delta = 0.5
    _, grad = nn.huber_loss(pred, target, delta)
    assert grad[0, 0] * grad.size == pytest.approx(delta)
    assert grad[0, 1] * grad.size == pytest.approx(-0.1)

    _, squared = nn.mse_loss(pred, target)
    assert abs(squared[0, 0]) > 50 * abs(grad[0, 0])


def test_huber_is_continuous_where_it_changes_regime():
    """Value and slope both join at delta, which is what makes it one loss."""
    delta = 0.8
    below = nn.huber_loss(np.array([[delta - 1e-9]]), np.zeros((1, 1)), delta)
    above = nn.huber_loss(np.array([[delta + 1e-9]]), np.zeros((1, 1)), delta)
    assert below[0] == pytest.approx(above[0], abs=1e-8)
    assert below[1][0, 0] == pytest.approx(above[1][0, 0], abs=1e-8)


def test_bce_survives_logits_that_would_overflow_a_naive_form():
    """The whole reason the loss is taken on logits rather than on probabilities."""
    logits = np.array([[-800.0, 800.0, 0.0]])
    labels = np.array([[0.0, 1.0, 1.0]])
    loss, grad = nn.bce_with_logits(logits, labels)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))
    # A confident and correct prediction should contribute essentially nothing.
    assert loss == pytest.approx(np.log(2.0) / 3.0, abs=1e-12)


def test_pos_weight_lifts_the_cost_of_missing_a_positive():
    """Class imbalance is the normal condition for the events this engine scores."""
    logits = np.array([[-3.0, -3.0]])
    labels = np.array([[1.0, 0.0]])
    plain, plain_grad = nn.bce_with_logits(logits, labels)
    weighted, weighted_grad = nn.bce_with_logits(logits, labels, pos_weight=10.0)
    assert weighted > plain
    assert abs(weighted_grad[0, 0]) > abs(plain_grad[0, 0])
    # The negative column is untouched: the weight applies to positives alone.
    assert weighted_grad[0, 1] == pytest.approx(plain_grad[0, 1])


def test_mse_does_not_change_scale_with_the_batch_size(rng):
    """Two runs at different batch sizes have to report comparable losses."""
    small = rng.standard_normal((4, 3))
    loss_small, _ = nn.mse_loss(small, np.zeros_like(small))
    doubled = np.vstack([small, small])
    loss_doubled, _ = nn.mse_loss(doubled, np.zeros_like(doubled))
    assert loss_small == pytest.approx(loss_doubled)


def test_losses_refuse_mismatched_shapes(rng):
    with pytest.raises(ConfigError, match="cannot pair"):
        nn.mse_loss(rng.standard_normal((4, 3)), rng.standard_normal((4, 2)))
    with pytest.raises(ConfigError, match="pos_weight must be positive"):
        nn.bce_with_logits(np.zeros((2, 1)), np.ones((2, 1)), pos_weight=0.0)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def test_dropout_is_off_at_evaluation_and_preserves_the_scale_in_training():
    """Inverted dropout: the expected activation is the same in both modes."""
    x = np.ones((200, 50))
    layer = nn.Dropout(0.4, rng=np.random.default_rng(2))

    trained = layer.forward(x)
    assert np.any(trained == 0.0)
    assert trained.mean() == pytest.approx(1.0, abs=0.02)
    # Survivors carry the 1 / (1 - p) rescaling, which is what makes evaluation
    # need no correction of its own.
    assert set(np.unique(trained).round(10)) == {0.0, round(1.0 / 0.6, 10)}

    layer.eval()
    evaluated = layer.forward(x)
    assert np.array_equal(evaluated, x)
    assert np.array_equal(layer.backward(np.ones_like(x)), np.ones_like(x))


def test_dropout_masks_repeat_from_a_reseeded_generator():
    x = np.ones((8, 6))
    layer = nn.Dropout(0.5, rng=np.random.default_rng(99))
    first = layer.forward(x)
    layer.reseed(np.random.default_rng(99))
    second = layer.forward(x)
    assert np.array_equal(first, second)


def test_layer_norm_standardises_each_row_not_each_column(rng):
    x = rng.standard_normal((7, 12)) * 4.0 + 9.0
    out = nn.LayerNorm(12).forward(x)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(out.var(axis=1), 1.0, atol=1e-4)
    # Column statistics are left alone, which is the difference from batch norm.
    assert not np.allclose(out.mean(axis=0), 0.0, atol=1e-3)


def test_l2_normalize_puts_every_row_on_the_unit_sphere(rng):
    out = nn.L2Normalize().forward(rng.standard_normal((6, 5)) * 30.0)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)


def test_l2_normalize_survives_a_zero_row():
    """The zero vector has no direction, and the eps guard is what stops a NaN."""
    out = nn.L2Normalize().forward(np.zeros((2, 3)))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_weight_initialisation_uses_the_scale_the_activation_needs():
    rng = np.random.default_rng(5)
    he = nn.Linear(400, 300, rng=rng, init="he")
    xavier = nn.Linear(400, 300, rng=rng, init="xavier")
    assert he.weight.std() == pytest.approx(np.sqrt(2.0 / 400), rel=0.03)
    assert xavier.weight.std() == pytest.approx(np.sqrt(2.0 / 700), rel=0.03)
    assert np.array_equal(he.bias, np.zeros(300))
    with pytest.raises(ConfigError, match="unknown weight initialisation"):
        nn.Linear(4, 4, rng=rng, init="lecun")


def test_gradients_accumulate_until_they_are_cleared(rng):
    """The contract the training loop relies on, stated as a test."""
    layer = nn.Linear(3, 2, rng=rng)
    x = rng.standard_normal((4, 3))
    grad_out = rng.standard_normal((4, 2))
    layer.forward(x)
    layer.backward(grad_out)
    once = layer.grad_weight.copy()
    layer.forward(x)
    layer.backward(grad_out)
    assert np.allclose(layer.grad_weight, 2.0 * once)
    layer.zero_grad()
    assert np.array_equal(layer.grad_weight, np.zeros_like(layer.grad_weight))


def test_sequential_reports_its_layers_parameters_in_order(rng):
    model = nn.Sequential(nn.Linear(3, 4, rng=rng), nn.ReLU(), nn.LayerNorm(4))
    assert len(model) == 3
    assert [p.shape for p in model.params()] == [(3, 4), (4,), (4,), (4,)]
    assert [g.shape for g in model.grads()] == [(3, 4), (4,), (4,), (4,)]
    model.eval()
    assert all(not layer.training for layer in model.layers)
    model.train()
    assert all(layer.training for layer in model.layers)


def test_layers_name_what_arrived_when_a_shape_is_wrong(rng):
    layer = nn.Linear(3, 2, rng=rng)
    with pytest.raises(ConfigError, match="width 5 where 3 was expected"):
        layer.forward(rng.standard_normal((4, 5)))
    with pytest.raises(ConfigError, match="two dimensional"):
        layer.forward(rng.standard_normal(3))
    with pytest.raises(ConfigError, match="before any forward"):
        nn.Linear(3, 2, rng=rng).backward(np.zeros((4, 2)))
    with pytest.raises(ConfigError, match=r"\[0, 1\)"):
        nn.Dropout(1.0, rng=rng)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


def test_adam_converges_on_a_quadratic_with_a_known_optimum():
    """A badly conditioned one, where the adaptive scaling is what does the work.

    The curvatures differ by a factor of sixteen across the three coordinates, so
    a single learning rate that suits the flattest would diverge on the steepest.
    Adam normalises each coordinate by its own gradient scale and reaches the
    optimum on all three at the same rate.
    """
    optimum = np.array([1.5, -2.0, 0.25])
    curvature = np.array([0.5, 3.0, 8.0])
    w = np.zeros(3)
    optimizer = nn.Adam(lr=0.05)
    for _ in range(4000):
        optimizer.step([w], [curvature * (w - optimum)])
    assert np.allclose(w, optimum, atol=1e-9)
    assert optimizer.t == 4000


def test_decoupled_weight_decay_shrinks_every_weight_by_the_same_factor():
    """With no gradient at all, the only motion left is the decay, in closed form.

    Under AdamW the step is ``lr * weight_decay * parameter`` and nothing else, so
    after t steps the parameter is its start times (1 - lr * wd) to the t. Matching
    the closed form to the last few bits is the proof that the decay sits outside
    the adaptive scaling rather than inside it, where the division by the root
    second moment would make the shrinkage depend on the gradient history.
    """
    start = np.array([2.0, -3.0, 0.5])
    p = start.copy()
    lr, decay, steps = 0.1, 0.5, 20
    optimizer = nn.Adam(lr=lr, weight_decay=decay)
    for _ in range(steps):
        optimizer.step([p], [np.zeros(3)])
    # Twenty multiplications accumulate a few units in the last place against the
    # single power, which is the only gap the tolerance is allowing for.
    assert np.allclose(p, start * (1.0 - lr * decay) ** steps, rtol=1e-14, atol=0.0)


def test_decoupled_decay_is_not_the_same_as_l2_in_the_gradient():
    """The claim the comment in Adam makes, measured rather than asserted.

    The same decay strength applied the old way, by adding it into the gradient,
    lands somewhere else entirely, because that term then passes through the
    division by the root second moment and is rescaled by the local gradient
    noise. Here it is not a rounding difference: the two answers differ by more
    than a tenth of the parameter.
    """
    curvature = np.array([0.4, 6.0])
    optimum = np.array([1.0, 1.0])
    decay = 0.05

    decoupled = np.zeros(2)
    opt_a = nn.Adam(lr=0.02, weight_decay=decay)
    coupled = np.zeros(2)
    opt_b = nn.Adam(lr=0.02)
    for _ in range(2000):
        opt_a.step([decoupled], [curvature * (decoupled - optimum)])
        opt_b.step([coupled], [curvature * (coupled - optimum) + decay * coupled])

    assert np.max(np.abs(decoupled - coupled)) > 0.1
    # Both shrink the answer toward zero, so neither is broken; they disagree on
    # how much, and by a different amount in each coordinate.
    assert np.all(decoupled < optimum)
    assert np.all(coupled < optimum)


def test_adam_keeps_separate_moments_per_parameter(rng):
    """Two arrays stepped by one optimiser must not share state."""
    first = np.zeros(2)
    second = np.zeros(2)
    optimizer = nn.Adam(lr=0.1)
    for _ in range(50):
        optimizer.step([first, second], [np.full(2, 1.0), np.full(2, -0.001)])
    assert first[0] < 0
    assert second[0] > 0


def test_adam_refuses_arguments_that_would_not_train():
    with pytest.raises(ConfigError, match="learning rate"):
        nn.Adam(lr=0.0)
    with pytest.raises(ConfigError, match="betas"):
        nn.Adam(betas=(0.9, 1.0))
    with pytest.raises(ConfigError, match="weight decay cannot be negative"):
        nn.Adam(weight_decay=-0.1)
    with pytest.raises(ConfigError, match="have to line up"):
        nn.Adam().step([np.zeros(2)], [])


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------


def build_model(seed: int, *, dropout: float = 0.2) -> nn.Sequential:
    """One generator seeds every layer, which is how a run becomes reproducible."""
    rng = np.random.default_rng(seed)
    return nn.Sequential(
        nn.Linear(4, 8, rng=rng, init="he"),
        nn.ReLU(),
        nn.Dropout(dropout, rng=rng),
        nn.Linear(8, 1, rng=rng, init="xavier"),
    )


def make_batches(
    rng: np.random.Generator, n: int = 96, size: int = 16
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A linear target with light noise, so the fit has something real to find."""
    x = rng.standard_normal((n, 4))
    coefficients = np.array([[1.2], [-0.7], [0.4], [0.0]])
    y = x @ coefficients + 0.05 * rng.standard_normal((n, 1))
    return [(x[i : i + size], y[i : i + size]) for i in range(0, n, size)]


class ScriptedValidation:
    """A loss whose held-out values follow a written script.

    Early stopping is a control-flow rule, and testing it through a real
    optimisation would be testing the optimiser as well and would break whenever
    the learning rate moved. Here the training loss is genuine, so the parameters
    keep changing from epoch to epoch, and only the validation number is dictated.
    Each held-out evaluation also records the parameters the model is holding at
    that moment, which is what lets the restoration be checked exactly.
    """

    def __init__(self, model: nn.Layer, schedule: list[float]) -> None:
        self.model = model
        self.schedule = schedule
        self.snapshots: list[list[np.ndarray]] = []

    def __call__(self, prediction, target):
        loss, grad = nn.mse_loss(prediction, target)
        if self.model.training:
            return loss, grad
        index = min(len(self.snapshots), len(self.schedule) - 1)
        self.snapshots.append([p.copy() for p in self.model.params()])
        return self.schedule[index], grad


def test_early_stopping_restores_the_best_epoch_and_not_the_last(rng):
    """The point of the exercise, and the thing that is easy to get quietly wrong.

    Validation improves for three epochs and then worsens for three. With a
    patience of three the run ends after the sixth, and the parameters the caller
    is left holding must be the ones from the third, bit for bit.
    """
    model = build_model(11)
    schedule = [1.0, 0.5, 0.2, 0.9, 0.8, 0.7, 0.6]
    loss_fn = ScriptedValidation(model, schedule)
    batches = make_batches(rng)

    history = nn.train(
        model,
        batches,
        loss_fn,
        nn.Adam(lr=0.02),
        val_batches=batches[:1],
        epochs=20,
        patience=3,
        seed=1,
    )

    assert history.stopped_early is True
    assert history.epochs == 6
    assert history.best_epoch == 2
    assert history.val_loss == schedule[:6]
    assert history.best_val_loss == 0.2

    best = loss_fn.snapshots[2]
    last = loss_fn.snapshots[5]
    for param, kept in zip(model.params(), best):
        assert np.array_equal(param, kept)
    assert not all(
        np.array_equal(param, stale) for param, stale in zip(model.params(), last)
    )
    assert any("restored" in note for note in history.notes)


def test_a_run_that_uses_its_whole_budget_says_so(rng):
    """A fit that never triggered its patience may simply not have converged."""
    model = build_model(12)
    loss_fn = ScriptedValidation(model, [1.0 / (k + 1) for k in range(6)])
    batches = make_batches(rng)
    history = nn.train(
        model,
        batches,
        loss_fn,
        nn.Adam(lr=0.01),
        val_batches=batches[:1],
        epochs=5,
        patience=3,
        seed=1,
    )
    assert history.stopped_early is False
    assert history.epochs == 5
    assert history.best_epoch == 4
    assert any("full budget" in note for note in history.notes)


def test_without_a_holdout_nothing_is_restored_and_the_history_admits_it(rng):
    model = build_model(13)
    batches = make_batches(rng)
    history = nn.train(
        model, batches, nn.mse_loss, nn.Adam(lr=0.01), epochs=4, patience=2, seed=1
    )
    assert history.val_loss == []
    assert history.best_val_loss is None
    assert history.stopped_early is False
    assert any("nothing was held out" in note for note in history.notes)


def test_training_actually_reduces_the_loss(rng):
    """Sanity: the primitives compose into something that fits."""
    model = build_model(14, dropout=0.0)
    batches = make_batches(rng, n=256)
    history = nn.train(
        model,
        batches,
        nn.mse_loss,
        nn.Adam(lr=0.02),
        val_batches=batches[-2:],
        epochs=60,
        patience=60,
        seed=3,
    )
    assert history.train_loss[-1] < 0.2 * history.train_loss[0]
    assert history.val_loss[history.best_epoch] < 0.05


def test_a_contrastive_two_tower_fit_separates_its_pairs():
    """The flagship path: one encoder, one stacked forward, in-batch negatives.

    This is the pattern the peer model uses and the reason ``info_nce_in_batch``
    exists. Anchors and positives go through the encoder as one batch of 2B rows,
    the loss splits the output down the middle and stacks the two gradients back.

    The score is top-one retrieval: for each anchor, is its own positive the
    nearest of all 32 candidates. The untrained encoder is measured first, so the
    claim is against a baseline rather than in the air. log B is the loss of an
    embedding carrying no information, and the fit has to end far below it.
    """
    width, pairs, noise = 6, 32, 0.6

    def encoder() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(width, 16, rng=np.random.default_rng(21), init="he"),
            nn.ReLU(),
            nn.Linear(16, 8, rng=np.random.default_rng(22), init="xavier"),
            nn.L2Normalize(),
        )

    def top_one(model: nn.Sequential, a: np.ndarray, p: np.ndarray) -> float:
        model.eval()
        similarity = model.forward(a) @ model.forward(p).T
        return float(np.mean(similarity.argmax(axis=1) == np.arange(len(a))))

    rng = np.random.default_rng(7)
    centres = rng.standard_normal((pairs, width))
    anchors = centres + noise * rng.standard_normal((pairs, width))
    positives = centres + noise * rng.standard_normal((pairs, width))

    baseline = top_one(encoder(), anchors, positives)

    def contrastive(embedding, _targets):
        half = embedding.shape[0] // 2
        loss, (grad_a, grad_p) = nn.info_nce_in_batch(
            embedding[:half], embedding[half:], 0.07
        )
        return loss, np.vstack([grad_a, grad_p])

    fitted = encoder()
    batches = [(np.vstack([anchors, positives]), None)]
    history = nn.train(
        fitted,
        batches,
        contrastive,
        nn.Adam(lr=0.01),
        val_batches=batches,
        epochs=300,
        patience=300,
        seed=5,
    )
    assert baseline < 0.5
    assert top_one(fitted, anchors, positives) == 1.0
    assert history.train_loss[-1] < 0.1 * np.log(pairs)


def test_a_diverged_loss_is_raised_and_never_recorded(rng):
    """A NaN in a history is a number nobody can read, so it is an error instead."""
    model = build_model(15)

    def exploding(prediction, target):
        _, grad = nn.mse_loss(prediction, target)
        return float("inf"), grad

    with pytest.raises(NotMeaningfulError, match="diverged"):
        nn.train(model, make_batches(rng), exploding, nn.Adam(), epochs=2, patience=2)


def test_train_refuses_arguments_that_describe_no_run(rng):
    model = build_model(16)
    batches = make_batches(rng)
    with pytest.raises(ConfigError, match="epochs"):
        nn.train(model, batches, nn.mse_loss, nn.Adam(), epochs=0)
    with pytest.raises(ConfigError, match="patience"):
        nn.train(model, batches, nn.mse_loss, nn.Adam(), patience=0)
    with pytest.raises(ConfigError, match="no batches"):
        nn.train(model, [], nn.mse_loss, nn.Adam())
    with pytest.raises(ConfigError, match="supplied but empty"):
        nn.train(model, batches, nn.mse_loss, nn.Adam(), val_batches=[])
    with pytest.raises(ConfigError, match="no learnable parameters"):
        nn.train(nn.Sequential(nn.ReLU()), batches, nn.mse_loss, nn.Adam())


def test_history_renders_as_rows_and_a_frame(rng):
    model = build_model(17)
    batches = make_batches(rng)
    history = nn.train(
        model,
        batches,
        nn.mse_loss,
        nn.Adam(lr=0.01),
        val_batches=batches[:1],
        epochs=4,
        patience=4,
        seed=1,
    )
    rows = history.rows()
    assert len(rows) == history.epochs
    assert rows[0]["Epoch"] == 1
    assert sum(1 for row in rows if row["Restored"]) == 1
    frame = history.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["Train loss", "Validation loss", "Restored"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def run_once(seed: int) -> tuple[nn.Sequential, nn.TrainHistory]:
    model = build_model(seed, dropout=0.3)
    batches = make_batches(np.random.default_rng(99))
    history = nn.train(
        model,
        batches,
        nn.mse_loss,
        nn.Adam(lr=0.01, weight_decay=0.001),
        val_batches=batches[:2],
        epochs=12,
        patience=12,
        seed=seed,
    )
    return model, history


def test_one_seed_gives_bitwise_identical_parameters():
    """Not approximately equal. Identical, to the last bit, or the run is not a run.

    Initialisation, the dropout masks and the shuffling of batch order are the
    three sources of randomness in a fit, and all three are drawn from generators
    the caller seeded. Two runs at one seed therefore have to agree exactly, and
    ``==`` is the right comparison rather than a tolerance.
    """
    first_model, first_history = run_once(31)
    second_model, second_history = run_once(31)

    for a, b in zip(first_model.params(), second_model.params()):
        assert np.array_equal(a, b)
    assert first_history.train_loss == second_history.train_loss
    assert first_history.val_loss == second_history.val_loss
    assert first_history.best_epoch == second_history.best_epoch


def test_a_different_seed_gives_a_different_answer():
    """Otherwise the determinism test above would pass on a model that never moved."""
    first_model, first_history = run_once(31)
    other_model, other_history = run_once(32)
    assert not all(
        np.array_equal(a, b)
        for a, b in zip(first_model.params(), other_model.params())
    )
    assert first_history.train_loss != other_history.train_loss


def test_gradient_check_is_itself_deterministic(rng):
    """It draws a random projection, so it too has to repeat from its seed."""
    x = rng.standard_normal((4, 3))
    layer = nn.Linear(3, 2, rng=np.random.default_rng(6))
    first = nn.gradient_check(layer, x, rng=np.random.default_rng(0))
    second = nn.gradient_check(layer, x, rng=np.random.default_rng(0))
    assert first == second
