# JAX for scientific computing demo

[![Render with nbviewer](https://raw.githubusercontent.com/jupyter/design/main/logos/Badges/nbviewer_badge.svg?sanitize=true)](https://nbviewer.jupyter.org/github/matt-graham/jax-sci-comp-demo/blob/main/hodgkin-huxley-demo.ipynb)
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matt-graham/jax-sci-comp-demo/blob/main/hodgkin-huxley-demo.ipynb)

This repository contains the [slides](slides.md) and an accompanying [notebook](hodgkin-huxley-demo.ipynb)
for a tutorial on [JAX](https://jax.readthedocs.io/en) in the context of scientific computing.

The [slides](slides.md) give a high level overview of JAX and the [Python array API standard](https://data-apis.org/array-api/latest/).
The notebook illustrates using JAX to simulate and fit the parameters of a component of the [Hodgkin-Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) for actional potential generation in neurons.

## Environment set up

To set up a Python virtual environment with the required dependencies you can use whatever virtual environment tool you prefer,
but in the instructions below we will use [`uv` which can be easily installed on macOS, Linux and Windows](https://docs.astral.sh/uv/getting-started/installation/).

First clone the repository and change to be current working directory

```
git clone https://github.com/matt-graham/jax-sci-comp-demo.git
cd jax-sci-comp-demo
```

Then create a new Python virtual environment -
in the command below we specify to use the experimental free-threading enabled build of Python 3.13t,
which will allow parallelising JAX operations across multiple threads,
to use the latest stable Python release just omit the `--python=3.13t` option

```
uv venv --python=3.13t
```

Activate the newly created virtual environment

```
source .venv/bin/activate
```

Now install the requirements into the environment

```
uv pip install -r requirements.txt
```

To install a Jupyter kernel from the active virtual environment that can be used in any Jupyter notebook client run 

```
uvx --with ipykernel ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=$(basename $(pwd))
```

## Running the notebook

To run the notebook you will additionally need a Jupyter notebook application.
Providing you follow the kernel install step above this does not necessarily need to be installed in the virtual environment itself.
If you already have an installed tool you prefer for working with Jupyter notebooks (for example Jupyter Lab or VS Code with Jupyter notebook extension),
you can open the notebook from there.

To install and run a Jupyter Lab server from a temporary virtual environment you can run

```
uvx jupyter lab
```

Once you have opened the notebook, remember to select the kernel installed from the virtual environment.

