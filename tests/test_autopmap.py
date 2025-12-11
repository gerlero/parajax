import multiprocessing
import time

import jax
import jax.numpy as jnp
import pytest
from parajax import autopmap

jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())


@pytest.mark.parametrize("x", [jnp.arange(97), jnp.arange(97 * 2).reshape(97, 2)])
def test_square(x: jnp.ndarray) -> None:
    @autopmap
    def square(x: float | jax.Array) -> float | jax.Array:
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
    def add_mul(
        x: float | jax.Array, y: float | jax.Array
    ) -> tuple[float | jax.Array, float | jax.Array]:
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


@pytest.mark.parametrize("max_devices", range(1, multiprocessing.cpu_count()))
def test_max_devices(*, max_devices: int) -> None:
    @autopmap(max_devices=max_devices)
    def square(x: float | jax.Array) -> float | jax.Array:
        return x**2

    x = jnp.arange(97)
    y = square(x)

    assert jnp.all(y == x**2)


def test_strict_remainder_strategy() -> None:
    @autopmap(max_devices=2, remainder_strategy="strict")
    def square(x: float | jax.Array) -> float | jax.Array:
        return x**2

    x = jnp.arange(96)
    y = square(x)

    assert jnp.all(y == x**2)


def test_drop_remainder_strategy() -> None:
    @autopmap(max_devices=2, remainder_strategy="drop")
    def square(x: float | jax.Array) -> float | jax.Array:
        return x**2

    x = jnp.arange(97)
    y = square(x)

    assert jnp.all(y == x[:96] ** 2)


def test_speedup() -> None:
    @jax.jit
    def heavy_compute(x: jax.Array) -> jax.Array:
        def body_fun(_: object, x: jax.Array) -> jax.Array:
            return jnp.sin(x) + jnp.cos(x) + jnp.sqrt(x)

        return jax.lax.fori_loop(0, 10_000, body_fun, x)

    parallel_heavy_compute = autopmap(heavy_compute, max_devices=2)

    x = jnp.arange(10_000, dtype=float)

    jax.block_until_ready(parallel_heavy_compute(x))
    start = time.perf_counter()
    y_parallel = jax.block_until_ready(parallel_heavy_compute(x))
    time_parallel = time.perf_counter() - start

    jax.block_until_ready(heavy_compute(x))
    start = time.perf_counter()
    y_serial = jax.block_until_ready(heavy_compute(x))
    time_serial = time.perf_counter() - start

    assert jnp.all(y_parallel == y_serial)
    assert time_parallel <= 0.6 * time_serial


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

    f2 = autopmap(remainder_strategy="strict", max_devices=2)(lambda x: x)
    with pytest.raises(ValueError, match="strict"):
        f2(jnp.arange(3))

    with pytest.raises(ValueError, match="no arguments"):
        autopmap(lambda: None)()

    with pytest.raises(ValueError, match="max_devices"):
        autopmap(max_devices=1000)(lambda x: x)(jnp.arange(10))
