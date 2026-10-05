# AI Inferencing — From Zero to Hero to Fine-Tuning

Learning how neural networks work by building them from scratch, following Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series.

This repository collects my hands-on notebooks, experiments, and notes: from scalar automatic differentiation to character-level language models a small GPT, and now pretrained-model inference, quantization, and LoRA fine-tuning. The first stage follows Zero to Hero; the newer experiments extend that foundation using Hugging Face tools. Despite the repository name, the current work covers both **training and generation**.

## Explore the projects

| Project | What I explored | Notebook | Run online |
| --- | --- | --- | --- |
| Micrograd | Computational graphs, chain rule, backpropagation, and an MLP built around a scalar `Value` class | [01-micrograd](notebooks/01-micrograd.ipynb) | [Open in Colab](https://colab.research.google.com/github/ayush-anand/ai-inferencing/blob/main/notebooks/01-micrograd.ipynb) |
| Makemore: BatchNorm & WaveNet | Character embeddings, initialization, activation/gradient diagnostics, batch normalization, and hierarchical context processing | [02-makemore-batchnorm-wavenet](notebooks/02-makemore-batchnorm-wavenet.ipynb) | [Open in Colab](https://colab.research.google.com/github/ayush-anand/ai-inferencing/blob/main/notebooks/02-makemore-batchnorm-wavenet.ipynb) |
| Build GPT | Character encoding/decoding, a bigram baseline, causal self-attention, Transformer blocks, and autoregressive text generation | [03-build-gpt](notebooks/03-build-gpt.ipynb) | [Open in Colab](https://colab.research.google.com/github/ayush-anand/ai-inferencing/blob/main/notebooks/03-build-gpt.ipynb) |

| Fine-tuning & quantization | Qwen inference, model memory/dtypes, 4-bit NF4 loading, and Llama 2 LoRA fine-tuning | [04-pytorch-finetuning-quantization](notebooks/04-pytorch-finetuning-quantization.ipynb) | [Open in Colab](https://colab.research.google.com/github/ayush-anand/ai-inferencing/blob/main/notebooks/04-pytorch-finetuning-quantization.ipynb) |

Read them in that order. The notebooks retain their original code, explanations, and saved outputs as a record of the learning process.

## My learning journey

1. **Build the foundations:** scalar autograd and neural networks in micrograd.
2. **Learn language modeling:** embeddings, BatchNorm, and hierarchical context in makemore.
3. **Build GPT from scratch:** causal attention and autoregressive generation.
4. **Adapt pretrained models:** inspect Qwen, explore precision/memory tradeoffs, and fine-tune Llama 2 with 4-bit loading and LoRA.

The latest stage includes a [fine-tuning guide](docs/FINETUNING_QUANTIZATION.md) and a [saved adapter archive](artifacts/Llama-2-7b-chat-finetune.zip).

## What I learned

- How local derivatives combine through the chain rule, and why gradients must accumulate when a value is reused.
- How embeddings and context windows turn text into a next-character prediction problem.
- How initialization and normalization affect activation and gradient statistics during training.
- How a WaveNet-inspired hierarchy combines neighboring characters into progressively larger contexts.
- How causal attention, positional embeddings, residual connections, and LayerNorm fit together in a decoder Transformer.
- How token IDs are sampled and decoded into generated text.
- How pretrained-model inference uses tokenizer dictionaries, device placement, and inference mode.
- How quantization reduces base-weight memory and LoRA trains a small set of adapter parameters.

**Tokenizer coverage:** the GPT notebook implements a character-level tokenizer. A separate byte-pair encoding (BPE) tokenizer notebook is not included yet.

## Getting started

### Google Colab

Use the links above. Run cells from top to bottom in a fresh runtime. The two language-model notebooks download their datasets using `wget`; micrograd's graph visualization also needs the Graphviz `dot` executable. See [setup and troubleshooting](docs/SETUP.md) for details.

### Local Jupyter

Install Python and Git, then run:

```bash
git clone https://github.com/ayush-anand/ai-inferencing.git
cd ai-inferencing
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install dependencies, prepare the data, and open Jupyter:

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m jupyter lab
```

Open a notebook from `notebooks/`. After downloading data with the helper, skip its `!wget` cell. Install the Graphviz system application separately if you want to render micrograd graphs.

## Repository layout

```text
ai-inferencing/
├── notebooks/
│   ├── 01-micrograd.ipynb
│   ├── 02-makemore-batchnorm-wavenet.ipynb
│   ├── 03-build-gpt.ipynb
│   └── 04-pytorch-finetuning-quantization.ipynb
├── docs/
│   ├── LEARNING_GUIDE.md
│   ├── SETUP.md
│   └── FINETUNING_QUANTIZATION.md
├── artifacts/
│   ├── Llama-2-7b-chat-finetune.zip
│   ├── adapter_config.json
│   └── README.md
├── scripts/download_data.py
├── requirements-finetuning.txt
├── requirements.txt
└── README.md
```

## Experiment notes

These are exploratory learning notebooks, with intermediate definitions and long training loops. Makemore includes loops configured for 200,000 steps and one diagnostic loop that stops early. GPT's final reference cell uses 5,000 iterations and selects CUDA when available. Start with fewer steps to understand the code before running a full training session.

Saved outputs are historical experiments, not independently reproduced benchmarks. The first three projects do not bundle trained weights. The fine-tuning project includes a LoRA adapter archive, which requires the matching Llama 2 base model; it is not a standalone model. See the [learning guide](docs/LEARNING_GUIDE.md) for notebook-specific details and possible next experiments.

## Acknowledgments

This work follows Andrej Karpathy's teaching and reference implementations. Many cells closely follow the course; this repository documents my study and experiments rather than claiming the course implementations as original research.

- [Zero to Hero course](https://karpathy.ai/zero-to-hero.html)
- [micrograd](https://github.com/karpathy/micrograd)
- [makemore](https://github.com/karpathy/makemore)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [minbpe — tokenizer reference](https://github.com/karpathy/minbpe)
- [Tiny Shakespeare dataset source](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)

Refer to the upstream projects for their licenses and attribution requirements. Downloaded datasets remain subject to their original terms.
