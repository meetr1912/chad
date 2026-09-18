"""Transform contract for the Prism Hadamard loader.

The end-to-end evidence that the rotation is applied correctly is the NLL gap measured
against real weights (prism_nll.py). These tests cover what can be checked without an
8 GB download: that forward and inverse are actually inverses, and that the sign fold
happens on the correct side of the transform — swapping those two is exactly the
mistake that loads cleanly and decodes garbage.
"""

import math

import pytest

mx = pytest.importorskip("mlx.core")

from chad.mlx_prism import fwht, is_prism_pack  # noqa: E402

BLOCK = 8


@pytest.fixture
def signs():
    return mx.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0])


def test_inverse_undoes_forward(signs):
    x = mx.random.normal((3, BLOCK))
    back = fwht(fwht(x, BLOCK, signs), BLOCK, signs, inverse=True)
    assert mx.allclose(back, x, atol=1e-5).item()


def test_sign_fold_is_side_specific(signs):
    """Forward is signs-then-transform. Applying them in the other order is a
    different function, so the wrong order cannot silently pass for the right one."""
    x = mx.random.normal((3, BLOCK))
    forward = fwht(x, BLOCK, signs)
    swapped = fwht(x, BLOCK, signs, inverse=True)
    assert not mx.allclose(forward, swapped, atol=1e-3).item()


def test_normalization_preserves_norm(signs):
    """The 1/sqrt(block) scale makes the transform orthonormal; without it the
    activations would grow by sqrt(block) at every packed projection."""
    x = mx.random.normal((4, BLOCK))
    y = fwht(x, BLOCK, signs)
    assert math.isclose(
        mx.sum(x * x).item(), mx.sum(y * y).item(), rel_tol=1e-4
    )


def test_block_must_divide_width(signs):
    with pytest.raises(ValueError, match="does not divide"):
        fwht(mx.zeros((2, BLOCK + 1)), BLOCK, signs)


def test_is_prism_pack_only_matches_the_rotated_type():
    assert is_prism_pack({"model_type": "prism_hadamard_qwen35"})
    assert not is_prism_pack({"model_type": "qwen3_5"})
    assert not is_prism_pack({})
