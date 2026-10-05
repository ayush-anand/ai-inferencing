# Fine-tuning and quantization

[Notebook](../notebooks/04-pytorch-finetuning-quantization.ipynb) · [Saved adapter](../artifacts/README.md)

This stage extends my from-scratch learning into using and adapting pretrained language models. The notebook and its historical outputs are preserved unchanged.

## Part A: PyTorch and pretrained-model inference

The notebook loads `Qwen/Qwen2.5-0.5B`, inspects parameters, device, and dtype, and compares manual generation with the Transformers text-generation pipeline. Notes cover tokenizer `input_ids` and `attention_mask`, evaluation mode, disabled gradient tracking, sampling, and rough memory estimates for FP32, FP16, INT8, and INT4.

Those estimates describe weight storage only. Activations, caches, optimizer state, and quantization metadata add memory overhead. Token IDs must remain integer tensors; an early float16 token-tensor cell is exploratory and is replaced by the tokenizer dictionary in the generation cell. Evaluation mode alone does not make sampled generation deterministic.

## Part B: Llama 2 with 4-bit loading and LoRA

| Setting | Value recorded in the notebook / adapter |
| --- | --- |
| Base model | `NousResearch/Llama-2-7b-chat-hf` |
| Training dataset | `mlabonne/guanaco-llama2-1k`, train split |
| Base-weight quantization | 4-bit NF4, float16 compute, double quantization disabled |
| Adapter | LoRA, rank 64, alpha 16, dropout 0.1 |
| Saved target modules | `q_proj`, `v_proj` |
| Epochs requested | 1 |
| Active per-device batch / accumulation | 2 / 2 |
| Maximum sequence length | 512 |
| Learning rate | `2e-4` |
| Optimizer | `paged_adamw_32bit` |
| Adapter export metadata | PEFT 0.21.0 |

The notebook configures `BitsAndBytesConfig`, loads the base model on GPU 0, trains with TRL's `SFTTrainer`, saves an adapter, explores generation, and exports a ZIP from Colab. Quantization compresses the base weights; LoRA supplies trainable extra parameters. See the official [bitsandbytes guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes) and [PEFT quicktour](https://huggingface.co/docs/peft/quicktour).

## Runtime and known details

Install the optional environment with `python -m pip install -r requirements-finetuning.txt`. The notebook's initial install cell does not list every dependency; the requirements file also includes datasets, Transformers, TensorBoard, and safetensors.

The Llama section explicitly calls CUDA APIs and maps the model to GPU 0, so it needs a compatible GPU runtime as written. The initial Qwen section uses CPU. The final download cells use `google.colab`; skip those locally because the adapter archive is already included.

This is a historical notebook, not a verified training script for every current TRL version. In particular, it passes `warmup_ratio` into `warmup_steps`, uses `train_sampling_strategy`, and comments out the intended scheduler argument. Check these against your installed `SFTConfig` before rerunning. Some declared settings, including gradient checkpointing, are not explicitly passed to the trainer configuration. Do not infer that every declared option was enabled.

The notebook later attaches a saved adapter to the existing model variable. For clean inference, load a fresh matching base model and attach the adapter once; the trained model may already contain adapters. `PeftModel.from_pretrained` attaches an adapter and does not itself merge it into the base weights. Repetitive output alone does not establish its cause.

## Saved artifact and results

The ZIP contains adapter weights, configuration, and an auto-generated model-card template. It does not contain the full base model or a tokenizer. Read [the artifact guide](../artifacts/README.md) for extraction and reuse.

No fresh GPU training, adapter inference, or held-out evaluation was run while adding these files. Saved outputs document past experiments and do not establish improvement over the base model. A useful next experiment is to compare the base model and adapter on the same held-out prompts, with fixed generation settings and recorded package versions.

### Sources used in this experiment

- [Qwen base model](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Llama 2 base model](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
- [Guanaco dataset](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)

The notebook's Llama 2 workflow follows the common Guanaco QLoRA tutorial pattern. Consult the model and dataset cards for attribution, access requirements, and usage terms.
