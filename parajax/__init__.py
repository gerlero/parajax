import functools
import multiprocessing
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar, overload

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

_P = ParamSpec("_P")
_T = TypeVar("_T")


@overload
def pvmap(
    func: Callable[_P, _T],
    /,
    *,
    max_devices: int | None = None,
) -> Callable[_P, _T]: ...


@overload
def pvmap(
    *,
    max_devices: int | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def pvmap(
    func: Callable[_P, _T] | None = None,
    /,
    *,
    max_devices: int | None = None,
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Parallel vectorizing map. Creates a parallelized version of `func` that maps
    over the leading axis of array arguments.

    This function is similar to `jax.vmap` but it automatically distributes the
    computation across multiple devices.

    **Arguments:**

    - `func`: The function to be parallelized. It should accept array arguments with a
      leading batch dimension. If you need to also pass non-batched arguments,
      consider using `functools.partial` or a lambda as `func`.
    - `max_devices`: The maximum number of devices to use for parallelization.

    **Returns:**

    Parallel-vectorized version of `func`, which maps over the leading axis of
    array arguments and distributes the computation across multiple devices.
    """
    if max_devices is not None and max_devices < 1:
        msg = "max_devices must be at least 1"
        raise ValueError(msg)

    def pvmap_decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        vmapped_func = jax.vmap(func)

        @functools.wraps(func)
        def pvmap_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            device_count = jax.device_count()
            if max_devices is not None and max_devices > device_count:
                msg = (
                    "max_devices cannot be greater than the number of"
                    f" available JAX devices (={device_count})"
                )
                raise ValueError(msg)

            if max_devices != 1 and device_count == 1:
                msg = (
                    "pvmap: parallelization requested but only a single JAX device is"
                    " available"
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

            pad_size = (-batch_size) % devices

            padded_args = jax.tree.map(
                lambda x: jnp.pad(
                    x, [(0, pad_size)] + [(0, 0)] * (x.ndim - 1), mode="edge"
                ),
                (args, kwargs),
            )

            padded_output = jax.shard_map(
                lambda padded_args: vmapped_func(
                    *padded_args[0], **padded_args[1]
                ),  # shard_map does not support keyword arguments
                mesh=jax.make_mesh((devices,), ("devices",)),
                in_specs=P(
                    "devices",
                ),
                out_specs=P(
                    "devices",
                ),
            )(padded_args)

            return jax.tree.map(lambda x: x[:batch_size], padded_output)

        return pvmap_wrapper

    return pvmap_decorator(func) if func is not None else pvmap_decorator
