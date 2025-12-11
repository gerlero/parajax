<div align="center">
  <a href="https://github.com/gerlero/parajax"><img src="https://raw.githubusercontent.com/gerlero/parajax/main/logo.png" alt="Parajax" width="250"/></a>

  **Automagic parallelization of calls to [JAX](https://github.com/jax-ml/jax)-based functions**

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

- 🚀 **Device-parallel execution**: run across multiple CPUs, GPUs or TPUs automatically
- 🧩 **Fully composable** with [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html), [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html), and other JAX transformations
- 🪄 **Automatic handling** of input shapes not divisible by the number of devices
- 🎯 **Simple interface**: just decorate your function with [`@parallelize`](https://parajax.readthedocs.io/en/stable/#parajax.parallelize)

## Installation

```bash
pip install parajax
```

## Example

```python
import multiprocessing

import jax
import jax.numpy as jnp
from parajax import parallelize

jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())
# ^ Only needed on CPU: allow JAX to use all CPU cores

@parallelize
def square(xs):
    return xs**2

xs = jnp.arange(12_345)
ys = square(xs)
```

That's it! Invocations of `square` will now be automatically parallelized across all available devices.

## Documentation

For more details, check out the [documentation](https://parajax.readthedocs.io/).
