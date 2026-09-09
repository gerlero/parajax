import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import overload

import jax
import jax.numpy as jnp


def _normalize_ndim(
    signature: inspect.Signature,
    ndim: int | None | Sequence[int | None] | Mapping[str, int | None],
) -> dict[str, int | None]:
    params = tuple(signature.parameters)

    ret: dict[str, int | None]

    match ndim:
        case int() | None:
            ret = dict.fromkeys(params, ndim)

        case [*_]:
            if len(ndim) != len(params):
                msg = f"ndim has {len(ndim)} elements, but expected {len(params)}"
                raise ValueError(msg)
            ret = dict(zip(params, ndim, strict=True))  # ty: ignore[invalid-assignment]

        case {}:
            unknown: set[str] = set(ndim) - set(params)  # ty: ignore[invalid-assignment]
            if unknown:
                msg = f"unknown parameter(s) in ndim: {sorted(unknown)}"
                raise ValueError(msg)
            ret = dict(ndim)  # ty: ignore[no-matching-overload]

        case _:
            msg = (
                "ndim must be an int, None, a sequence of int or None, "
                "or a mapping from parameter names to int or None"
            )
            raise TypeError(msg)

    for name, value in ret.items():
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            msg = f"ndim for {name!r} must be a nonnegative integer or None"
            raise TypeError(msg)
        if value < 0:
            msg = f"ndim for {name!r} must be a nonnegative integer or None"
            raise ValueError(msg)

    return ret


def _map[T](
    func: Callable[..., T],
    *,
    in_axes: tuple[object, ...],
    batch_size: int,
) -> Callable[..., T]:
    if batch_size == 0:
        return jax.vmap(func, in_axes=in_axes)

    def is_none(x: object) -> bool:
        return x is None

    flat_axes, axes_treedef = jax.tree.flatten(in_axes, is_leaf=is_none)
    mapped_indices = tuple(i for i, axis in enumerate(flat_axes) if axis == 0)

    def mapped(*args: object) -> T:
        flat_args, treedef = jax.tree.flatten(args, is_leaf=is_none)
        if treedef != axes_treedef:
            msg = "internal error: argument and in_axes pytrees do not match"
            raise RuntimeError(msg)

        xs = tuple(flat_args[i] for i in mapped_indices)

        def body(mapped_leaves: tuple[object, ...]) -> T:
            leaves = list(flat_args)
            for i, leaf in zip(mapped_indices, mapped_leaves, strict=True):
                leaves[i] = leaf
            return func(*jax.tree.unflatten(treedef, leaves))

        return jax.lax.map(body, xs, batch_size=batch_size)

    return mapped


@overload
def vectorize[**P, T](
    func: Callable[P, T],
    /,
    *,
    ndim: int | None | Sequence[int | None] | Mapping[str, int | None] = ...,
    batch_size: int = ...,
) -> Callable[P, T]: ...


@overload
def vectorize[**P, T](
    *,
    ndim: int | None | Sequence[int | None] | Mapping[str, int | None] = ...,
    batch_size: int = ...,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def vectorize[**P, T](
    func: Callable[P, T] | None = None,
    /,
    *,
    ndim: int | None | Sequence[int | None] | Mapping[str, int | None] = 0,
    batch_size: int = 0,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Automatically vectorize a function over broadcastable leading dimensions.

    `vectorize` treats each vectorized argument as having zero or more leading
    *batch dimensions* followed by a fixed number of trailing *core dimensions*.
    The number of core dimensions is specified by `ndim`. Batch dimensions from
    all vectorized arguments are broadcast according to NumPy broadcasting rules,
    and `func` is automatically mapped over the resulting broadcast shape.

    For example, given::

        @vectorize(ndim=(2, 1))
        def matvec(A, x):
            return A @ x

    the original function operates on a matrix and a vector::

        A.shape == (m, n)
        x.shape == (n,)

    while the decorated function accepts arbitrary broadcastable leading
    dimensions::

        A.shape == (20, 1, m, n)
        x.shape == (10, n)

    and returns an array with shape `(20, 10, m)`.

    Args:
        func: The function to vectorize.

        ndim: Number of trailing core dimensions for each vectorized parameter.

            An integer applies the same core dimensionality to every parameter.
            For example, `ndim=0` treats arguments as scalars and `ndim=1`
            treats them as vectors.

            A sequence specifies one value per function parameter. For example,
            `ndim=(2, 1)` describes a function whose first parameter is a matrix
            and whose second parameter is a vector.

            A mapping specifies core dimensionalities by parameter name.
            Parameters omitted from the mapping are static and are passed
            unchanged to `func`.

            `None` may be used explicitly to mark a parameter as static.

            In all cases, dimensions preceding the specified core dimensions are
            interpreted as batch dimensions. Their shapes are broadcast together
            automatically.

        batch_size: Maximum batch size used when evaluating each mapped dimension.
            If zero (the default), the complete dimension is processed at once,
            equivalently to `jax.vmap`. A positive value evaluates the mapped
            computation in smaller batches using `jax.lax.map`.

    Returns:
        A function with the same call signature as `func` that accepts arbitrary
        broadcastable batch dimensions on its vectorized arguments.

    Examples:
        Vectorize a scalar function:

            >>> @vectorize
            ... def soft_threshold(x, threshold):
            ...     return jax.lax.cond(
            ...        jnp.abs(x) > threshold,
            ...        lambda: jnp.sign(x) * (jnp.abs(x) - threshold),
            ...        lambda: 0.0,
            ...     )
            >>> soft_threshold(jnp.ones((3, 1)), threshold=0.5).shape
            (3, 1)

        Vectorize a function whose inputs have different core ranks:

            >>> @vectorize(ndim=(2, 1))
            ... def matvec(A, x):
            ...     return A @ x
            >>> A = jnp.ones((5, 1, 3, 4))
            >>> x = jnp.ones((7, 4))
            >>> matvec(A, x).shape
            (5, 7, 3)

        Keep configuration parameters static:

            >>> @vectorize(ndim={"x": 1})
            ... def norm(x, *, ord=2):
            ...     return jnp.linalg.norm(x, ord=ord)
            >>> norm(jnp.ones((10, 3)), ord=1).shape
            (10,)
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        msg = "batch_size must be a nonnegative integer"
        raise TypeError(msg)
    if batch_size < 0:
        msg = "batch_size must be a nonnegative integer"
        raise ValueError(msg)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        signature = inspect.signature(func)
        ndim_by_name = _normalize_ndim(signature, ndim)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            bound = signature.bind(*args, **kwargs)
            dynamic_names = tuple(
                name for name in bound.arguments if ndim_by_name.get(name) is not None
            )

            if not dynamic_names:
                return func(*args, **kwargs)

            dynamic_args = tuple(bound.arguments[name] for name in dynamic_names)
            leaves, treedef = jax.tree.flatten(dynamic_args)
            if not leaves:
                msg = "vectorized arguments have no array leaves"
                raise TypeError(msg)

            leaf_names = []
            leaf_ndims = []
            for name in dynamic_names:
                core_ndim = ndim_by_name[name]
                assert core_ndim is not None
                leaf_count = len(jax.tree.leaves(bound.arguments[name]))
                leaf_names.extend([name] * leaf_count)
                leaf_ndims.extend([core_ndim] * leaf_count)

            arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
            batch_shapes = []
            for name, array, core_ndim in zip(
                leaf_names, arrays, leaf_ndims, strict=True
            ):
                if array.ndim < core_ndim:
                    msg = (
                        f"argument {name!r} has a leaf with rank {array.ndim}, "
                        f"but ndim={core_ndim}"
                    )
                    raise ValueError(msg)
                batch_shapes.append(
                    array.shape[:-core_ndim] if core_ndim else array.shape
                )

            try:
                broadcast_shape = jnp.broadcast_shapes(*batch_shapes)
            except ValueError as exc:
                shapes = ", ".join(map(str, batch_shapes))
                msg = f"incompatible batch shapes for vectorization: {shapes}"
                raise ValueError(msg) from exc

            squeezed_leaves = []
            reversed_filled_shapes = []
            for array, batch_shape in zip(arrays, batch_shapes, strict=True):
                pad_ndim = len(broadcast_shape) - len(batch_shape)
                filled_shape = (1,) * pad_ndim + batch_shape
                reversed_filled_shapes.append(filled_shape[::-1])

                squeeze_axes = tuple(
                    axis for axis, size in enumerate(batch_shape) if size == 1
                )
                squeezed_leaves.append(jnp.squeeze(array, axis=squeeze_axes))

            squeezed_args = jax.tree.unflatten(treedef, squeezed_leaves)
            original_arguments = dict(bound.arguments)

            def core_func(*dynamic_values: object) -> T:
                arguments = original_arguments | dict(
                    zip(dynamic_names, dynamic_values, strict=True)
                )
                call = inspect.BoundArguments(signature, arguments)
                return func(*call.args, **call.kwargs)

            mapped_func = core_func
            dimensions_to_expand = []

            for reverse_axis, axis_sizes in enumerate(
                zip(*reversed_filled_shapes, strict=True)
            ):
                flat_in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)

                if all(axis is None for axis in flat_in_axes):
                    dimensions_to_expand.append(len(broadcast_shape) - 1 - reverse_axis)
                    continue

                in_axes = jax.tree.unflatten(treedef, flat_in_axes)
                assert isinstance(in_axes, tuple)
                mapped_func = _map(
                    mapped_func,
                    in_axes=in_axes,
                    batch_size=batch_size,
                )

            result = mapped_func(*squeezed_args)

            if dimensions_to_expand:
                axes = tuple(dimensions_to_expand)
                result = jax.tree.map(lambda x: jnp.expand_dims(x, axes), result)

            return result

        return wrapper

    return decorator(func) if func is not None else decorator
