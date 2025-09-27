import functools
import multiprocessing
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

_P = ParamSpec("_P")
_T = TypeVar("_T")


def pvmap(
    func: Callable[_P, _T],
    /,
    *,
    max_devices: int | None = None,
) -> Callable[_P, _T]:
    """Parallel vectorizing map. Creates a parallelized version of `func` that maps
    over the leading axis of array arguments.

    This function is similar to `jax.vmap` but it automatically distributes the
    computation across multiple devices.

    **Arguments:**

    - `func`: The function to be parallelized. It should accept array arguments with a
      leading batch dimension. Keyword arguments are not supported.
      If you need to pass keyword arguments or other non-batched arguments,
      consider using `functools.partial` or a lambda function as `func`.
    - `max_devices`: The maximum number of devices to use for parallelization.

    **Returns:**

    Parallel-vectorized version of `func`, which maps over the leading axis of
    array arguments and distributes the computation across multiple devices.
    """
    if max_devices is not None and max_devices < 1:
        msg = "max_devices must be at least 1"
        raise ValueError(msg)

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        if kwargs:  # shard_map does not support kwargs
            msg = "pvmap does not support keyword arguments"
            raise NotImplementedError(msg)

        device_count = jax.device_count()
        if max_devices is not None and max_devices > device_count:
            msg = (
                "max_devices cannot be greater than the number of "
                f"available JAX devices (={device_count})"
            )
            raise ValueError(msg)

        if max_devices != 1 and device_count == 1:
            msg = (
                "pvmap: parallelization requested but only a single JAX device is "
                "available"
            )
            if jax.default_backend() == "cpu" and multiprocessing.cpu_count() > 1:
                msg += (
                    '\nSet \'jax.config.update("jax_num_cpu_devices", '
                    f"{multiprocessing.cpu_count()})' before using JAX to enable all "
                    "available CPUs."
                    "\nRead https://docs.jax.dev/en/latest/sharded-computation.html "
                    "for details."
                )
            warnings.warn(msg, UserWarning, stacklevel=2)

        devices = max_devices if max_devices is not None else device_count

        flat_args, in_tree = jax.tree_util.tree_flatten(
            args, is_leaf=lambda x: isinstance(x, (jax.Array, np.ndarray))
        )

        batch_sizes = []
        for arg in flat_args:
            if isinstance(arg, (jax.Array, np.ndarray)):
                if arg.shape:
                    batch_sizes.append(arg.shape[0])
                else:
                    msg = "mapped arrays must have a leading batch dimension"
                    raise ValueError(msg)
        batch_sizes = set(batch_sizes)
        if len(batch_sizes) > 1:
            msg = f"mismatched sizes for mapped axes: {batch_sizes}"
            raise ValueError(msg)

        try:
            batch_size = batch_sizes.pop()
        except KeyError:
            msg = "cannot map over non-array arguments only"
            raise ValueError(msg) from None

        devices = min(devices, batch_size)
        pad_size = (-batch_size) % devices

        padded_flat_args = []
        for arg in flat_args:
            if isinstance(arg, (jax.Array, np.ndarray)):
                pad_width = [(0, pad_size)] + [(0, 0)] * (arg.ndim - 1)
                padded_arg = jnp.pad(arg, pad_width, mode="edge")
                padded_flat_args.append(padded_arg)
            else:
                padded_flat_args.append(arg)

        padded_args = jax.tree_util.tree_unflatten(in_tree, padded_flat_args)

        padded_output = jax.shard_map(
            jax.vmap(func),
            mesh=jax.make_mesh((devices,), ("devices",)),
            in_specs=P(
                "devices",
            ),
            out_specs=P(
                "devices",
            ),
        )(*padded_args)

        padded_flat_output, out_tree = jax.tree_util.tree_flatten(
            padded_output, is_leaf=lambda x: isinstance(x, (jax.Array, np.ndarray))
        )
        unpadded_flat_output = []
        for out in padded_flat_output:
            if isinstance(out, (jax.Array, np.ndarray)):
                unpadded_out = out[:batch_size]
                unpadded_flat_output.append(unpadded_out)
            else:
                unpadded_flat_output.append(out)

        return jax.tree_util.tree_unflatten(out_tree, unpadded_flat_output)

    return wrapper
