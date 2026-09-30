# Setup and troubleshooting

## Environment

Use a Python environment supported by your chosen PyTorch release. Install `requirements.txt` in a virtual environment, then launch Jupyter from the repository root. Dependencies are unpinned because the original notebook runtime versions have not been recorded.

The Python `graphviz` package wraps a separate system application. Install [Graphviz](https://graphviz.org/download/) and ensure `dot` is on your PATH if you want to render the micrograd computation graphs. Check with `dot -V`. Restart Jupyter after changing PATH. Other micrograd computations do not need graph rendering.

## Dataset preparation

From the repository root:

```bash
python scripts/download_data.py
```

The helper uses Python's standard library, downloads `names.txt` and `input.txt` into `notebooks/`, and leaves existing files untouched. It works independently of the current working directory. Internet access is required the first time. The dataset files are ignored by Git.

The notebooks originally use `!wget`, which is available in common Colab/Linux environments but is not normally installed on Windows. After using the helper locally, skip the notebook's download cell. Open notebooks with their working directory set to `notebooks/` so their relative file reads find the data. You can inspect the current directory with:

```python
from pathlib import Path
print(Path.cwd())
```

For Colab, run the original download cells in the runtime. If graph visualization reports a missing `dot` executable, install Graphviz in a separate cell:

```python
!apt-get -qq update
!apt-get -qq install graphviz
```

## Running experiments

- Start each notebook with a fresh kernel, then run cells in order. Intermediate class definitions and variables are reused throughout the learning sequence.
- For a quick exploration, reduce `max_steps` in makemore and `max_iters` / `eval_iters` in GPT before running their loops. Short runs are useful for inspecting behavior but do not reproduce saved losses.
- Makemore's original tensors are on CPU. GPT's final cell automatically uses CUDA if available; a GPU can help with training time.
- If the normalization widget does not display, check that `ipywidgets` is installed in the notebook kernel's environment.
- If a cell fails, inspect the first error and run its prerequisite cells. Restarting and running from the top helps avoid mixing state from different model versions.

## Reproducibility status

Notebook code and saved outputs have been preserved. Full training runs have not been rerun as part of organizing this repository. Random seeds appear in the notebooks, but not every experiment or library version is fixed. Hardware, runtime versions, and random state can affect results.

There are no bundled model weights or deployment service. Sampling uses a model trained in the active notebook runtime.
