import jax
import jax.numpy as jnp
import pytest

from parajax import vectorize


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_scalar_broadcasting(*, batch_size: int) -> None:
    @vectorize(batch_size=batch_size)
    def f(x: float | jax.Array, y: float | jax.Array) -> float | jax.Array:
        assert jnp.ndim(x) == 0
        assert jnp.ndim(y) == 0
        return x + 2 * y

    x = jnp.arange(6).reshape(2, 3)
    y = jnp.arange(3)

    assert jnp.array_equal(f(x, y), x + 2 * y)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_matvec(*, batch_size: int) -> None:
    @vectorize(ndim=(2, 1), batch_size=batch_size)
    def matvec(A: jax.Array, x: jax.Array) -> jax.Array:
        assert A.ndim == 2
        assert x.ndim == 1
        return A @ x

    A = jnp.arange(20 * 1 * 3 * 4, dtype=float).reshape(20, 1, 3, 4)
    x = jnp.arange(10 * 4, dtype=float).reshape(10, 4)

    actual = matvec(A, x)
    expected = jnp.einsum("...ij,...j->...i", A, x)

    assert actual.shape == (20, 10, 3)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_passthrough(*, batch_size: int) -> None:
    @vectorize(ndim=(2, 1), batch_size=batch_size)
    def matvec(A: jax.Array, x: jax.Array) -> jax.Array:
        assert A.ndim == 2
        assert x.ndim == 1
        return A @ x

    A = jnp.arange(12, dtype=float).reshape(3, 4)
    x = jnp.arange(4, dtype=float)

    actual = matvec(A, x)
    expected = A @ x

    assert actual.shape == (3,)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_batching_from_lower_rank_argument(*, batch_size: int) -> None:
    @vectorize(ndim=(2, 1), batch_size=batch_size)
    def matvec(A: jax.Array, x: jax.Array) -> jax.Array:
        assert A.ndim == 2
        assert x.ndim == 1
        return A @ x

    A = jnp.arange(12, dtype=float).reshape(3, 4)
    x = jnp.arange(20 * 10 * 4, dtype=float).reshape(20, 10, 4)

    actual = matvec(A, x)
    expected = jnp.einsum("ij,...j->...i", A, x)

    assert actual.shape == (20, 10, 3)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_singleton_dimension(*, batch_size: int) -> None:
    @vectorize(ndim=1, batch_size=batch_size)
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        assert x.ndim == 1
        assert y.ndim == 1
        return x + y

    x = jnp.arange(3.0).reshape(1, 3)
    y = jnp.ones(3)

    actual = add(x, y)

    assert actual.shape == (1, 3)
    assert jnp.array_equal(actual, x + y)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_static_argument(*, batch_size: int) -> None:
    @vectorize(ndim={"A": 2, "x": 1}, batch_size=batch_size)
    def matvec(A: jax.Array, /, *, x: jax.Array, scale: float = 1) -> jax.Array:
        assert A.ndim == 2
        assert x.ndim == 1
        assert isinstance(scale, int)
        return scale * (A @ x)

    A = jnp.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    x = jnp.arange(4)

    actual = matvec(A, x=x, scale=2)
    expected = 2.0 * jnp.einsum("...ij,j->...i", A, x)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_keyword_argument(*, batch_size: int) -> None:
    @vectorize(ndim={"x": 0, "y": 0}, batch_size=batch_size)
    def f(x: jax.Array, *, y: jax.Array) -> jax.Array:
        assert x.ndim == 0
        assert y.ndim == 0
        return x + y

    x = jnp.arange(2).reshape(2, 1)
    y = jnp.arange(3)

    assert jnp.array_equal(f(x, y=y), x + y)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_pytree(*, batch_size: int) -> None:
    @vectorize(ndim={"state": 0}, batch_size=batch_size)
    def f(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        assert state["x"].ndim == 0
        assert state["y"].ndim == 0
        return {
            "sum": state["x"] + state["y"],
            "diff": state["x"] - state["y"],
        }

    state = {
        "x": jnp.arange(6).reshape(2, 3),
        "y": jnp.arange(3),
    }

    actual = f(state)

    assert jnp.array_equal(actual["sum"], state["x"] + state["y"])
    assert jnp.array_equal(actual["diff"], state["x"] - state["y"])


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_jit(*, batch_size: int) -> None:
    @jax.jit
    @vectorize(ndim=(2, 1), batch_size=batch_size)
    def matvec(A: jax.Array, x: jax.Array) -> jax.Array:
        assert A.ndim == 2
        assert x.ndim == 1
        return A @ x

    A = jnp.ones((7, 1, 3, 4))
    x = jnp.ones((5, 4))

    actual = matvec(A, x)

    assert actual.shape == (7, 5, 3)
    assert jnp.array_equal(
        actual,
        jnp.full((7, 5, 3), 4.0),
    )


def test_vectorize_too_few_dimensions() -> None:
    @vectorize(ndim=(2, 1))
    def matvec(A: jax.Array, x: jax.Array) -> jax.Array:
        assert A.ndim == 2
        assert x.ndim == 1
        return A @ x

    with pytest.raises(ValueError, match="ndim"):
        matvec(jnp.ones(2), jnp.ones(3))


def test_vectorize_incompatible_shapes() -> None:
    @vectorize(ndim=1)
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        assert x.ndim == 1
        assert y.ndim == 1
        return x + y

    with pytest.raises(ValueError, match="incompatible batch shapes"):
        add(jnp.ones((2, 3)), jnp.ones((4, 3)))


def test_bad_ndim() -> None:
    with pytest.raises(ValueError, match="ndim"):
        vectorize(ndim=-1)(lambda x: x)

    with pytest.raises(ValueError, match="ndim"):
        vectorize(ndim={"y": 0})(lambda x: x)


def test_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        vectorize(batch_size=-1)(lambda x: x)

    with pytest.raises(TypeError, match="batch_size"):
        vectorize(batch_size=1.5)(lambda x: x)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="batch_size"):
        vectorize(batch_size=True)(lambda x: x)
