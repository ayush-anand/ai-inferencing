# Learning guide

This guide maps the included notebooks to the ideas explored in Karpathy's [Zero to Hero series](https://karpathy.ai/zero-to-hero.html).

## 1. Micrograd: understand the backward pass

[Open the notebook](../notebooks/01-micrograd.ipynb).

The notebook begins with scalar expressions and computational graph visualizations, then develops a `Value` class with arithmetic, `tanh`, `exp`, and automatic differentiation. A topological traversal orders the backward pass. `Neuron`, `Layer`, and `MLP` build a small neural network on top of that engine.

Pay attention to `+=` in gradient updates: a node can contribute to a loss through several paths. The notebook also compares manual reasoning with PyTorch autograd and explores a small supervised dataset and loss optimization.

**Try next:** differentiate a small expression by hand, compare it with `.backward()`, and check it with finite differences. Reset parameter gradients before each new training update.

## 2. Makemore: train and diagnose character models

[Open the notebook](../notebooks/02-makemore-batchnorm-wavenet.ipynb).

This combined notebook covers material from parts 3 and 5 of makemore. It uses the upstream names dataset, builds character IDs with `.` as the boundary symbol, and divides shuffled names into 80% training, 10% validation, and 10% test sets.

The first section uses a three-character context, embeddings, an MLP, and batch normalization. It explores initialization, running normalization statistics, deep-layer activation/gradient plots, and update-to-parameter ratios. An interactive widget illustrates how changing one sample affects normalization.

The `PART 5` section switches to an eight-character context. Custom `Embedding`, `FlattenConsecutive`, `Linear`, `BatchNorm1d`, `Tanh`, and `Sequential` layers form a hierarchy that groups adjacent positions in pairs. This is a WaveNet-inspired context hierarchy; it does not implement the full original dilated-convolution WaveNet architecture.

**Execution notes:** run the earlier section before `PART 5`, which reuses the dataset and vocabulary. One diagnostic loop deliberately breaks after step 1,000; other loops request 200,000 updates. Use evaluation mode with the final model's BatchNorm layers before evaluating or sampling, so running statistics are used rather than statistics from a single generated example.

**Try next:** compare three- and eight-character contexts with the same evaluation split, or compare an MLP and the hierarchical model with similar parameter counts. Record hyperparameters and validation loss alongside generated names.

## 3. GPT: move from local context to attention

[Open the notebook](../notebooks/03-build-gpt.ipynb).

The notebook downloads Tiny Shakespeare, maps each unique character to an integer, and introduces shifted next-token targets and a bigram baseline. It then explores attention and normalization before a final self-contained reference cell with a decoder Transformer.

The final cell contains token and positional embeddings, four attention heads per block, four Transformer blocks, residual connections, LayerNorm, a feed-forward network, AdamW optimization, and sampling. Its defaults are a batch size of 16, context length of 32, embedding size of 64, learning rate of `1e-3`, and 5,000 training iterations. The first 90% of the text is used for training and the remainder for validation.

The final Transformer retains the earlier class name `BigramLanguageModel`; inspect the blocks rather than inferring its architecture from that name. Its attention scores use the input embedding width in the scaling factor, as saved in the original notebook; the standard formula scales by the square root of the key/head dimension. This is a useful detail to investigate in a future experiment.

**Execution notes:** the final reference cell needs `input.txt` but defines its own model and training helpers. It can be run after preparing the dataset without replaying all earlier experiments. CUDA is selected automatically when available. No trained checkpoint is saved by the included code.

**Try next:** compare generated text across training checkpoints, experiment with context length and attention scaling, and add checkpoint saving/loading.

## Tokenization and future additions

The included GPT notebook demonstrates character-level `encode` and `decode`. A standalone BPE tokenizer implementation is not present in the provided files. A future addition could document UTF-8 bytes, frequent-pair merges, a learned vocabulary, and round-trip tests, following the course's tokenizer lesson and [minbpe](https://github.com/karpathy/minbpe).

Other possible additions include reproducible environment versions, short training scripts extracted from the notebooks, and an experiment table with freshly measured results.
