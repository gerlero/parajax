# <div align="center">Parajax</div>

Automagic parallelization of calls to [JAX](https://github.com/jax-ml/jax)-based functions

[![CI](https://github.com/gerlero/parajax/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/parajax/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/parajax/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/parajax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Publish](https://github.com/gerlero/parajax/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/parajax/actions/workflows/pypi-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/parajax)](https://pypi.org/project/parajax/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/parajax)](https://pypi.org/project/parajax/)

## Features

- 🚀 **Device-parallel execution**: run across multiple CPUs, GPUs or TPUs automatically
- 🔄 **Drop-in replacement** for [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
- ⚡ **JIT-compatible**: works with [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) and [variants](https://docs.kidger.site/equinox/api/transformations/#equinox.filter_jit)
- 🪄 **Transparent padding** when batch sizes aren’t divisible by number of devices
- 🎯 **Simple interface**: just decorate your function with `pvmap`

## Installation

```bash
pip install parajax
```

## Example

```python
import multiprocessing

import jax
import jax.numpy as jnp
from parajax import pvmap

jax.config.update("jax_num_cpu_devices", multiprocessing.cpu_count())
# ^ Only needed on CPU: allow JAX to use all CPU cores

@pvmap
def square(x: float) -> float:
    return x**2

xs = jnp.arange(97)
ys = square(xs)
```

That's it! Invocations of `square` will now be automatically parallelized across all available devices.
