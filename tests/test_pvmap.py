import multiprocessing
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from parajax import pvmap

jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())


@pytest.mark.parametrize("x", [jnp.arange(97), jnp.arange(97 * 2).reshape(97, 2)])
def test_square(x: jnp.ndarray) -> None:
    @pvmap
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
    @pvmap
    def add_mul(x: float, y: float) -> tuple[float, float]:
        return x + y, x * y

    x = jnp.arange(97)
    y = jnp.arange(97, 0, -1)
    sum_, prod = add_mul(x, y)
    assert jnp.all(sum_ == 97)
    assert jnp.all(prod == x * y)


def test_vmap_compatibility() -> None:
    def f(
        x: tuple[float | jax.Array, float | jax.Array], *, z: float | jax.Array
    ) -> dict[str, float | jax.Array]:
        return {"result": x[0] + x[1] + z}

    x = jnp.arange(97)
    y = jnp.arange(97, 0, -1)
    z = jnp.arange(97, 194)

    assert jnp.all(
        pvmap(f)((x, y), z=z)["result"] == jax.vmap(f)((x, y), z=z)["result"]
    )


@pytest.mark.parametrize("max_devices", [None, 1, 2])
@pytest.mark.parametrize("remainder_strategy", ["pad", "tail", "strict"])
def test_options(
    *, max_devices: int, remainder_strategy: Literal["pad", "tail", "strict"]
) -> None:
    @pvmap(max_devices=max_devices, remainder_strategy=remainder_strategy)
    def square(x: float | jax.Array) -> float | jax.Array:
        return x**2

    x = jnp.arange(4 if remainder_strategy == "strict" else 97)
    y = square(x)

    assert jnp.all(y == x**2)
