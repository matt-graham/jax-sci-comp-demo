# JAX for scientific computing demo

[![Render with nbviewer](https://raw.githubusercontent.com/jupyter/design/main/logos/Badges/nbviewer_badge.svg?sanitize=true)](https://nbviewer.jupyter.org/github/matt-graham/jax-sci-comp-demo/blob/main/hodgkin-huxley-demo.ipynb)
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matt-graham/jax-sci-comp-demo/blob/main/hodgkin-huxley-demo.ipynb)

This repository contains the [slides](slides.md) and an accompanying [notebook](hodgkin-huxley-demo.ipynb) for a tutorial on [JAX](https://jax.readthedocs.io/en) in the context of scientific computing.

The [slides](slides.md) give a high level overview of JAX and the [Python array API standard](https://data-apis.org/array-api/latest/). The notebook illustrates using JAX to simulate and fit the parameters of a component of the [Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) for actional potential generation in neurons.

## Environment set up

To set up a Python virtual environment with the required dependencies you can use whatever virtual environment tool you prefer,
but in the instructions below we will use [`uv` which can be easily installed on macOS, Linux and Windows](https://docs.astral.sh/uv/getting-started/installation/).

First clone the repository and change to be current working directory

```
git clone https://github.com/matt-graham/jax-sci-comp-demo.git
cd jax-sci-comp-demo
```

Then create a new Python virtual environment (using latest stable Python release) and activate

```
uv venv
source .venv/bin/activate
```

Now install the requirements in to the environment

```
uv pip install -r requirements.txt
```

You should now be able to launch a Jupyter Lab server by running

```
jupyter lab
```

