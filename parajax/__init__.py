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
def autopmap(
    func: Callable[_P, _T],
    /,
    *,
    max_devices: int | None = None,
    remainder_strategy: Literal["pad", "tail", "drop", "strict"] = "pad",
    gather: bool = False,
) -> Callable[_P, _T]: ...


@overload
def autopmap(
    *,
    max_devices: int | None = None,
    remainder_strategy: Literal["pad", "tail", "drop", "strict"] = "pad",
    gather: bool = False,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def autopmap(
    func: Callable[_P, _T] | None = None,
    /,
    *,
    max_devices: int | None = None,
    remainder_strategy: Literal["pad", "tail", "drop", "strict"] = "pad",
    gather: bool = False,
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Automatic parallelizing map. Creates a parallelized version of `func` that
    distributes computation of the leading axis of array arguments across multiple
    devices.

    **Arguments:**

    - `func`: The function to be parallelized. It should accept array arguments with a
      leading batch dimension. If you need to also pass non-batched arguments,
      consider using `functools.partial` or a lambda as `func`.
    - `max_devices`: The maximum number of devices to use for parallelization.
    - `remainder_strategy`: Strategy to handle cases where the batch size is not
      divisible by the number of devices. Options are:
        - `"pad"` (default): Transparently pad the input arrays along the leading axis
          to make the batch size divisible by the number of devices. The padding is done
          by repeating the last element. The output is then unpadded to match the
          original batch size.
        - `"tail"`: The extra elements that do not fit evenly into the devices are
          processed in a second pass on only as many devices as needed.
        - `"drop"`: The extra elements that do not fit evenly into the devices are
          simply dropped from the input and output.
        - `"strict"`: Ensure that the batch size is divisible by the number of devices.
          If not, a `ValueError` is raised.
    - `gather`: If `True`, output arrays are gathered back to the first device. If
      `False`, outputs remain sharded across devices.

    **Returns:**

    Parallel version of `func`, with the same signature as `func`.
    """
    if max_devices is not None and max_devices < 1:
        msg = "max_devices must be at least 1"
        raise ValueError(msg)

    if remainder_strategy not in {"pad", "tail", "drop", "strict"}:
        msg = f"invalid remainder_strategy: {remainder_strategy}"
        raise ValueError(msg)

    if not gather and remainder_strategy == "tail":
        msg = "pvmap: overriding gather to True with remainder_strategy='tail'"
        warnings.warn(msg, UserWarning, stacklevel=2)
        gather = True

    def pvmap_decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
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

            match remainder_strategy:
                case "strict":
                    if batch_size % devices != 0:
                        msg = (
                            f"remainder_strategy='strict' but batch size {batch_size}"
                            f" is not divisible by the number of devices {devices}"
                        )
                        raise ValueError(msg)

                    output = _pmap_strict(func, devices, *args, **kwargs)

                    if gather:
                        output = jax.device_put(output, jax.devices()[0])

                    return output

                case "tail" | "drop":
                    remainder_size = batch_size % devices
                    even_size = batch_size - remainder_size

                    args_even, kwargs_even = jax.tree.map(
                        lambda x: x[:even_size], (args, kwargs)
                    )

                    output_even = _pmap_strict(func, devices, *args_even, **kwargs_even)

                    if remainder_strategy == "drop" or remainder_size == 0:
                        if gather:
                            output_even = jax.device_put(output_even, jax.devices()[0])

                        return output_even

                    args_remainder, kwargs_remainder = jax.tree.map(
                        lambda x: x[even_size:], (args, kwargs)
                    )

                    output_remainder = _pmap_strict(
                        func,
                        remainder_size,
                        *args_remainder,
                        **kwargs_remainder,
                    )

                    output_even, output_remainder = jax.device_put(
                        (output_even, output_remainder), jax.devices()[0]
                    )

                    return jax.tree.map(
                        lambda even, rem: jnp.concatenate((even, rem), axis=0),
                        output_even,
                        output_remainder,
                    )

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

                    output = jax.tree.map(lambda x: x[:batch_size], padded_output)

                    if gather:
                        output = jax.device_put(output, jax.devices()[0])

                    return output

        return pvmap_wrapper

    return pvmap_decorator(func) if func is not None else pvmap_decorator
