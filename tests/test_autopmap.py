import multiprocessing
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from parajax import autopmap

jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())


@pytest.mark.parametrize("x", [jnp.arange(97), jnp.arange(97 * 2).reshape(97, 2)])
def test_square(x: jnp.ndarray) -> None:
    @autopmap
    def square(x: float) -> float:
        return x**2

    y = square(x)
    y = square(x)
    assert jnp.all(y == x**2)

    square = jax.jit(square)
    y = square(x)
    y = square(x)
    assert jnp.all(y == x**2)


def test_multiple_args() -> None:
    @autopmap
    def add_mul(x: float, y: float) -> tuple[float, float]:
        return x + y, x * y

    x = jnp.arange(97)
    y = jnp.arange(97, 0, -1)
    sum_, prod = add_mul(x, y)
    assert jnp.all(sum_ == 97)
    assert jnp.all(prod == x * y)


def test_vmap_compatibility() -> None:
    @jax.vmap
    def f(
        x: tuple[float | jax.Array, float | jax.Array], *, z: float | jax.Array
    ) -> dict[str, float | jax.Array]:
        return {"result": x[0] + x[1] + z}

    x = jnp.arange(97)
    y = jnp.arange(97, 0, -1)
    z = jnp.arange(97, 194)

    assert jnp.all(autopmap(f)((x, y), z=z)["result"] == f((x, y), z=z)["result"])


@pytest.mark.parametrize("max_devices", [None, 1, 2])
@pytest.mark.parametrize("remainder_strategy", ["pad", "tail", "drop", "strict"])
@pytest.mark.parametrize("gather", [False, True])
def test_options(
    *,
    max_devices: int,
    remainder_strategy: Literal["pad", "tail", "drop", "strict"],
    gather: bool,
) -> None:
    @autopmap(
        max_devices=max_devices, remainder_strategy=remainder_strategy, gather=gather
    )
    def square(x: float | jax.Array) -> float | jax.Array:
        return x**2

    x = jnp.arange(4 if remainder_strategy in {"drop", "strict"} else 97)
    y = square(x)

    assert jnp.all(y == x**2)


def test_invalid() -> None:
    with pytest.raises(ValueError, match="max_devices"):
        autopmap(max_devices=0)(lambda x: x)

    with pytest.raises(ValueError, match="remainder_strategy"):
        autopmap(remainder_strategy="invalid")(lambda x: x)  # ty: ignore[invalid-argument-type]

    @autopmap
    def f(x: jax.Array, y: jax.Array) -> jax.Array:
        return x + y

    with pytest.raises(ValueError, match="mismatched"):
        f(jnp.arange(10), jnp.arange(5))

    f = autopmap(remainder_strategy="strict", max_devices=2)(lambda x: x)
    with pytest.raises(ValueError, match="strict"):
        f(jnp.arange(3))

    with pytest.raises(ValueError, match="no arguments"):
        autopmap(lambda: None)()

    with pytest.raises(ValueError, match="max_devices"):
        autopmap(max_devices=1000)(lambda x: x)(jnp.arange(10))
