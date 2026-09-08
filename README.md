<div align="center">
  <a href="https://github.com/gerlero/parajax"><img src="https://raw.githubusercontent.com/gerlero/parajax/main/logo.png" alt="Parajax" width="250"/></a>

  **Automatic vectorization and parallelization of [JAX](https://github.com/jax-ml/jax)-based functions**

  [![Documentation](https://img.shields.io/readthedocs/parajax)](https://parajax.readthedocs.io/)
  [![CI](https://github.com/gerlero/parajax/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/parajax/actions/workflows/ci.yml)
  [![Codecov](https://codecov.io/gh/gerlero/parajax/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/parajax)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
  [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
  [![Publish](https://github.com/gerlero/parajax/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/parajax/actions/workflows/pypi-publish.yml)
  [![PyPI](https://img.shields.io/pypi/v/parajax)](https://pypi.org/project/parajax/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/parajax)](https://pypi.org/project/parajax/)
</div>


## Features

**Parajax** provides simple decorators for automatically mapping JAX functions over array batches and across multiple devices.

* 🪄 **Automatic vectorization** — map functions over arbitrary broadcastable batch dimensions
* 🚀 **Automatic parallelization** — distribute batched computations across CPUs, GPUs, or TPUs
* 📐 **NumPy-style broadcasting** — infer vectorization structure directly from input shapes
* 📦 **Memory-aware batching** — optionally evaluate vectorized functions in smaller batches
* 🧩 **Composable with JAX** — use naturally with `jax.jit` and other JAX transformations
* 🎯 **Simple decorator-based API** — add vectorization or parallelism without restructuring your function


## Installation

```bash
pip install parajax
```

## Vectorization

`@vectorize` turns a function that operates on individual values, vectors, matrices, or other array objects into one that automatically operates over arbitrary broadcastable leading dimensions.

For a scalar function, no configuration is needed:

```python
import jax
import jax.numpy as jnp

from parajax import vectorize


@vectorize
def soft_threshold(x, threshold):
    return jax.lax.cond(
        jnp.abs(x) > threshold,
        lambda: jnp.sign(x) * (jnp.abs(x) - threshold),
        lambda: 0.0,
    )


x = jnp.arange(12).reshape(3, 4)
y = soft_threshold(x, threshold=2.0)

assert y.shape == (3, 4)
```

Note that the `soft_threshold` function will also continue to work on scalar inputs, as well as with inputs of any other shape that are compatible (broadcastable) with each other.

For functions that are defined to operate on arrays, `ndim` specifies how many dimensions belong to each individual input. Any additional leading dimensions are treated as batch dimensions.

For example, a matrix-vector product operates on a rank-2 matrix and a rank-1 vector:

```python
@vectorize(ndim=(2, 1))
def matvec(A, x):
    return A @ x
```

The unvectorized function therefore expects:

```text
A: (m, n)
x: (n,)
```

but the decorated function also accepts arbitrary broadcastable batch dimensions:

```python
A = jnp.ones((100, 1, 3, 4))
x = jnp.ones((50, 4))

y = matvec(A, x)

assert y.shape == (100, 50, 3)
```

Here the batch shapes `(100, 1)` and `(50,)` are broadcast to `(100, 50)`. At each point in that batch, the original function receives a matrix with shape `(3, 4)` and a vector with shape `(4,)`.

### Specifying `ndim`

A single integer applies the same core dimensionality to every argument:

```python
@vectorize(ndim=1)
def dot(x, y):
    return x @ y
```

A sequence specifies the core dimensionality of each parameter:

```python
@vectorize(ndim=(2, 1))
def matvec(A, x):
    return A @ x
```

A mapping is useful when only some parameters should be vectorized:

```python
@vectorize(ndim={"A": 2, "x": 1})
def matvec(A, x, *, scale=1.0):
    return scale * (A @ x)
```

Parameters omitted from a mapping are passed unchanged to the underlying function.

`None` can also be used explicitly to mark a parameter as static:

```python
@vectorize(ndim=(1, 1, None))
def distance(x, y, metric): ...
```

Conceptually, `ndim` separates each array shape into:

```text
batch dimensions + core dimensions
```

For example, with `ndim=2`:

```text
(..., m, n)
 ^^^  ^^^^
batch core
```

Parajax automatically broadcasts the batch dimensions and maps the original function over them.

### Batched execution

By default, `vectorize` processes the complete broadcast batch at once, equivalently to using `jax.vmap`.

For computations where memory use is more important than maximum vectorization, set `batch_size`:

```python
@vectorize(ndim=1, batch_size=32)
def expensive_function(x): ...
```

The same vectorized operation is then evaluated in batches of at most 32 elements along each mapped dimension.


## Parallelization

`@parallelize` distributes a batched JAX function across all available devices.

```python
import multiprocessing

import jax
import jax.numpy as jnp

from parajax import parallelize


jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())
# Only needed on CPU to make multiple CPU devices available to JAX.


@parallelize
def square(xs):
    return xs**2


xs = jnp.arange(12_345)
ys = square(xs)
```

Invocations of `square` are automatically distributed across the available devices. Input sizes do not need to be divisible by the number of devices.


## Documentation

See the [documentation](https://parajax.readthedocs.io/) for the complete API reference and additional examples.
