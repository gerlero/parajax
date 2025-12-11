"""Parallelization utilities for JAX."""

import functools
import multiprocessing
import warnings
from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar, overload

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _pmap_strict(
    func: Callable[_P, _T], devices: int, /, *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    return jax.shard_map(
        lambda args, kwargs: func(
            *args, **kwargs
        ),  # shard_map does not support keyword arguments
        mesh=jax.make_mesh((devices,), ("devices",)),
        in_specs=P(
            "devices",
        ),
        out_specs=P(
            "devices",
        ),
    )(args, kwargs)


@overload
def parallelize(
    func: Callable[_P, _T],
    /,
    *,
    max_devices: int | None = ...,
    remainder_strategy: Literal["pad", "drop", "strict"] = ...,
) -> Callable[_P, _T]: ...


@overload
def parallelize(
    *,
    max_devices: int | None = ...,
    remainder_strategy: Literal["pad", "drop", "strict"] = ...,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def parallelize(
    func: Callable[_P, _T] | None = None,
    /,
    *,
    max_devices: int | None = None,
    remainder_strategy: Literal["pad", "drop", "strict"] = "pad",
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Automatic parallelizing map.

    Creates a parallelized version of `func` that distributes computation of the
    leading axis of array arguments across multiple devices.

    Args:
    func: The function to be parallelized. It should accept array arguments with a
        leading batch dimension. If your function cannot work in a batched manner, you
        can wrap it with `jax.vmap` first. For passing non-batched arguments, consider
        using `functools.partial` or a lambda function.
    max_devices: The maximum number of JAX devices to use for parallelization.
    remainder_strategy: Strategy to handle cases where the batch size is not
        divisible by the number of devices. Options are:
        - `"pad"` (default): Transparently pad the input arrays along the leading axis
          to make the batch size divisible by the number of devices. The padding is done
          by repeating the last element. The output is then automatically unpadded to
          match the original batch size, with no visible effect to the user.
        - `"drop"`: The extra elements that do not fit evenly into the devices are
          simply dropped from the computation and output.
        - `"strict"`: Assert that the batch size is divisible by the number of devices.
          If this is not the case, a `ValueError` is raised.

    Returns:
        The decorator returns a parallel version of `func` with the same signature.

    Basic usage:
        ```python
        import jax.numpy as jnp
        from parajax import parallelize

        @parallelize
        def square(xs):
            return xs ** 2

        xs = jnp.arange(12_345)
        ys = square(xs)  # This will run in parallel across available JAX devices
        ```

    Setting options:
        ```python
        import jax.numpy as jnp
        from parajax import parallelize

        @parallelize(max_devices=4)
        def square(xs):
            return xs ** 2

        xs = jnp.arange(12_345)
        ys = square(xs)  # Parallelized across 4 devices
        ```

    Composability with vmap:
        ```python
        import jax
        import jax.numpy as jnp
        from parajax import parallelize

        @parallelize
        @jax.vmap
        def relu_single(x):
            return jnp.maximum(x, 0)

        xs = jnp.arange(-6_000, 6_000)
        ys = relu_single(xs)  # Parallelized over the batch
        ```
    """
    if max_devices is not None and max_devices < 1:
        msg = "max_devices must be at least 1"
        raise ValueError(msg)

    if remainder_strategy not in {"pad", "drop", "strict"}:
        msg = f"invalid remainder_strategy: {remainder_strategy}"
        raise ValueError(msg)

    def parallelize_decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(func)
        @jax.jit
        def parallelize_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            device_count = jax.device_count()
            if max_devices is not None and max_devices > device_count:
                msg = (
                    "max_devices cannot be greater than the number of"
                    f" available JAX devices (={device_count})"
                )
                raise ValueError(msg)

            if max_devices != 1 and device_count == 1:
                msg = (
                    "parallelize: parallelization requested but only a single JAX"
                    " device is available"
                )
                if jax.default_backend() == "cpu" and multiprocessing.cpu_count() > 1:
                    msg += (
                        '\nSet \'jax.config.update("jax_num_cpu_devices",'
                        f" {multiprocessing.cpu_count()})' before using JAX to enable"
                        " all available CPUs."
                        "\nRead https://docs.jax.dev/en/latest/sharded-computation.html"
                        " for details."
                    )
                warnings.warn(msg, UserWarning, stacklevel=2)

            devices = max_devices if max_devices is not None else device_count

            flat_args, _ = jax.tree.flatten((args, kwargs))
            batch_sizes = {jnp.shape(arg)[0] for arg in flat_args}
            if len(batch_sizes) > 1:
                msg = f"mismatched sizes for mapped axes: {batch_sizes}"
                raise ValueError(msg)
            try:
                batch_size = batch_sizes.pop()
            except KeyError:
                msg = "no arguments to map over"
                raise ValueError(msg) from None

            devices = min(devices, batch_size)

            match remainder_strategy:
                case "strict":
                    if batch_size % devices != 0:
                        msg = (
                            f"remainder_strategy='strict' but batch size {batch_size}"
                            f" is not divisible by the number of devices {devices}"
                        )
                        raise ValueError(msg)

                    return _pmap_strict(func, devices, *args, **kwargs)

                case "drop":
                    remainder_size = batch_size % devices
                    even_size = batch_size - remainder_size

                    args_even, kwargs_even = jax.tree.map(
                        lambda x: x[:even_size], (args, kwargs)
                    )

                    return _pmap_strict(func, devices, *args_even, **kwargs_even)

                case "pad":
                    pad_size = (-batch_size) % devices

                    padded_args, padded_kwargs = jax.tree.map(
                        lambda x: jnp.pad(
                            x, [(0, pad_size)] + [(0, 0)] * (x.ndim - 1), mode="edge"
                        ),
                        (args, kwargs),
                    )

                    padded_output = _pmap_strict(
                        func, devices, *padded_args, **padded_kwargs
                    )

                    return jax.tree.map(lambda x: x[:batch_size], padded_output)

        return parallelize_wrapper

    return parallelize_decorator(func) if func is not None else parallelize_decorator
