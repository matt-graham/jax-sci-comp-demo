## Accelerated and differentiable scientific computing with JAX <!-- .element: style="text-transform: none; color: #467;" -->
<small>...and a bit of a diversion on the Python array API standard</small>

<img alt="JAX logo" src="https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg" style="width: 300px; margin: 20px;" />

<img alt="JAX logo" src="https://quansight.com/wp-content/uploads/2024/05/DataAPIs-name-under-symbol-V2-01.svg" style="width: 250px; margin: 20px;" />

---

## What is JAX? <!-- .element: style="text-transform: none;" -->

> JAX is a Python library for accelerator-oriented array computation and program transformation, designed for __high-performance numerical computing__ and large-scale machine learning.

<small>Source: https://github.com/jax-ml/jax</small>

----

## Key features <!-- .element: style="text-transform: none;" -->

- _Ease of use_: Offers a NumPy-like API make it accessible to those already familiar with scientific Python ecosystem.
 <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->
- _Transformations_: Provides _composable_ function transformations for just-in-time compilation, batching, autodiff and parallelization.
 <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="2" -->
- _Acceleration_: Allows easily executing the same code on CPUs and accelerator devices such as GPUs and TPUs.
 <!-- .element: class="fragment" data-fragment-index="3" -->


----

## JAX and XLA <!-- .element: style="text-transform: none;" -->

JAX builds on [XLA (accelerated linear algebra)](https://github.com/openxla/xla).

<img alt="XLA overview" src="https://github.com/openxla/xla/raw/main/docs/images/openxla.svg" style="width:100%; object-fit: cover; max-height: 280px;" />

<small>Image credit: OpenXLA documentation</small>

----

## History <!-- .element: style="text-transform: none;" -->

- JAX started as an research project in Google by a group including key contributors to [Autograd](https://github.com/HIPS/autograd) and XLA.
 <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->
- Early version of JAX was described in a [SysML 2018 conference paper](https://mlsys.org/Conferences/2019/doc/2018/146.pdf).
 <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="2" -->
- Initial open-source release on GitHub was in October 2018.
 <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="3" -->
- Since then has seen widespread adoption and developed a large open-source community.
 <!-- .element: class="fragment" data-fragment-index="4" -->

---

## Python array libraries <!-- .element: style="text-transform: none;" -->


<div>
<img alt="NumPy logo" src="https://raw.githubusercontent.com/numpy/numpy/refs/heads/main/branding/logo/logomark/numpylogoicon.svg" style="margin: 10px;vertical-align: middle; max-width: 200px; max-height: 200px;  object-fit: fill;" />
<img alt="JAX logo" src="https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg" style="margin: 10px; vertical-align: middle; max-width: 200px; max-height: 200px; object-fit: fill;" />
<img alt="PyTorch logo" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" style="margin: 10px;vertical-align: middle;max-width: 200px; max-height: 200px;  object-fit: fill;" />
<img alt="TensorFlow logo" src="https://icon.icepanel.io/Technology/svg/TensorFlow.svg" style="margin: 10px;vertical-align: middle;max-width: 200px; max-height: 200px;  object-fit: fill;" />
</div>
<div>
<img alt="Cupy logo" src="https://raw.githubusercontent.com/cupy/cupy/refs/heads/main/docs/image/cupy_logo.svg" style="margin: 10px;vertical-align: middle;max-width: 200px; max-height: 200px;  object-fit: fill;" />
<img alt="Dask logo" src="https://docs.dask.org/en/stable/_images/dask_icon.svg" style="margin: 10px;vertical-align: middle;max-width: 200px; max-height: 200px;  object-fit: fill;" />
<img alt="Xarray logo" src="https://raw.githubusercontent.com/pydata/xarray/refs/heads/main/doc/_static/logos/Xarray_Icon_Final.svg" style="margin: 10px;vertical-align: middle;max-width: 200px; max-height: 200px;  object-fit: fill;" />
<img alt="Sparse logo" src="https://sparse.pydata.org/en/stable/_static/logo.svg" style="margin: 10px;vertical-align: middle;max-width: 200px; max-height: 200px;  object-fit: fill;" />
</div>

----

## NumPy API substitutes <!-- .element: style="text-transform: none;" -->

The wide user familiarity with NumPy's API and large amount of existing code using NumPy has led some libraries to providing NumPy 'like' APIs

```Python
import numpy as np
import autograd.numpy as anp
import jax.numpy as jnp
import cupy as cp
```

----

## Issues with NumPy API as a common standard <!-- .element: style="text-transform: none;" -->

_However_ NumPy API not designed for this purpose and has some shortcomings:

- Complex datatype promotion semantics and copy-view behaviour. 
  <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="1" -->
- Not designed for use with non-CPU devices.
  <!-- .element: class="fragment fade-in-then-semi-out" data-fragment-index="2" -->
- Some operations produce arrays with data-dependent shapes.
  <!-- .element: class="fragment" data-fragment-index="3" -->


----

## Python data API standards <!-- .element: style="text-transform: none;" -->

<img alt="Python Data APIs logo" src="https://quansight.com/wp-content/uploads/2024/05/DataAPIs-name-under-symbol-V2-01.svg" style="width: 250px;" />

[Array API standard](https://data-apis.org/array-api) defined by _Consortium for Python Data API Standards_ (https://data-apis.org/).

[DataFrame API standard](https://data-apis.org/dataframe-api/draft/) also in development.


----

## Python array API standard <!-- .element: style="text-transform: none;" -->

> Python users have a wealth of choice for libraries and frameworks for numerical computing, data science, machine learning, and deep learning... 
> 
> The APIs of each of these libraries are largely similar, but with enough differences that it’s quite difficult to write code that works with multiple (or all) of these libraries. This array API standard aims to address that issue, by specifying an API for the most common ways arrays are constructed and used.
<!-- .element: style="font-size: 65%;" -->

<small>Source: https://data-apis.org/array-api</small>

----

## Array API components <!-- .element: style="text-transform: none;" -->

<img src="https://mermaid.ink/svg/CiUle2luaXQ6IHsndGhlbWUnOiAnbmV1dHJhbCd9fSUlCmJsb2NrLWJldGEKICAgIGNvbHVtbnMgMwogICAgYmxvY2s6YXJyYXk6MQogICAgICBjb2x1bW5zIDEKICAgICAgYXJyYXlfdGl0bGVbIkFycmF5IG9iamVjdCJdCiAgICAgIG9wc1siT3BlcmF0b3JzIl0KICAgICAgYXR0cmlidXRlc1siQXR0cmlidXRlcyJdCiAgICAgIG1ldGhvZHNbIk1ldGhvZHMiXQogICAgZW5kCiAgICBibG9jazpmdW5jdGlvbnM6MgogICAgICBjb2x1bW5zIDIKICAgICAgZnVuY3Rpb25zX3RpdGxlWyJGdW5jdGlvbnMiXToyCiAgICAgIGNyZWF0aW9uX21hbmlwWyJDcmVhdGlvbiAmIG1hbmlwdWxhdGlvbiJdCiAgICAgIG1hdGhlbWF0aWNhbFsiTWF0aGVtYXRpY2FsIl0KICAgICAgc2V0X2xvZ2ljWyJTZXQgJiBsb2dpY2FsIl0KICAgICAgc29ydF9zZWFyY2hbIlNvcnRpbmcgJiBzZWFyY2hpbmciXQogICAgICBiYXNpY19saW5hbGdbIkxpbmVhciBhbGdlYnJhIl0KICAgICAgc3RhdHNbIlN0YXRpc3RpY2FsIl0KICAgIGVuZAogICAgYmxvY2s6cnVsZXM6MQogICAgICBjb2x1bW5zIDEKICAgICAgcnVsZXNfdGl0bGVbIlJ1bGVzICYgY29udmVudGlvbnMiXQogICAgICBicm9hZGNhc3RpbmdbIkJyb2FkY2FzdGluZyJdCiAgICAgIGluZGV4aW5nWyJJbmRleGluZyJdCiAgICAgIHByb21vdGlvblsiVHlwZSBwcm9tb3Rpb24iXQogICAgZW5kCiAgICBibG9jazptaXNjOjEKICAgICAgY29sdW1ucyAxCiAgICAgIGNvbnN0YW50c1siQ29uc3RhbnRzIl0KICAgICAgZHR5cGVzWyJEYXRhIHR5cGVzIl0KICAgICAgaW5zcGVjdGlvblsiSW5zcGVjdGlvbiBpbnRlcmZhY2UiXQogICAgICB2ZXJzaW9uaW5nWyJWZXJzaW9uaW5nIl0KICAgIGVuZAogICAgYmxvY2s6ZXh0OjEgCiAgICAgIGNvbHVtbnMgMQogICAgICBleHRfdGl0bGVbIkV4dGVuc2lvbiBtb2R1bGVzIl0KICAgICAgbGluYWxnWyJMaW5lYXIgYWxnZWJyYSJdCiAgICAgIGZmdFsiRmFzdCBGb3VyaWVyIHRyYW5zZm9ybXMiXQogICAgICBfWyIuLi4iXQogICAgZW5kCiAgICBjbGFzc0RlZiB0aXRsZSBmaWxsOm5vbmUsc3Ryb2tlOm5vbmUKICAgIGNsYXNzIGFwaV90aXRsZSxhcnJheV90aXRsZSxleHRfdGl0bGUsZnVuY3Rpb25zX3RpdGxlLHJ1bGVzX3RpdGxlIHRpdGxlCg==" style="width: 80%" />

----

## Array API example <!-- .element: style="text-transform: none;" -->

```Python
def log_sum_exp(x):
    xp = x.__array_namespace__()
    max_x = xp.max(x)
    return max_x + xp.log(xp.sum(xp.exp(x - max_x))
```

`log_sum_exp` can then be run with array objects from any library supporting array API.

----

## Array API compatibility library <!-- .element: style="text-transform: none;" -->

`array_api_compat` acts a small wrapper around common array libraries to provide compatibility with array API standard.

Useful for libraries with only partial current support for array API.

----

## Array API compatibility library <!-- .element: style="text-transform: none;" -->

```Python
import array_api_compat

def log_sum_exp(x):
    xp = array_api_compat.array_namespace(x)
    max_x = xp.max(x)
    return max_x + xp.log(xp.sum(xp.exp(x - max_x))
```

NumPy v2.1+ and JAX v0.4.32+ fully support array API so using `array_api_compat` not strictly necessary. 
