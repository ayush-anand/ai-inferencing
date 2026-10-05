# Saved Llama 2 LoRA adapter

[Download archive](Llama-2-7b-chat-finetune.zip) · [Inspect configuration](adapter_config.json)

This is the original adapter export supplied with my fine-tuning notebook. The archive is approximately 50.2 MiB and contains:

| File | Purpose |
| --- | --- |
| `adapter_model.safetensors` | LoRA tensors; 67,126,232 bytes |
| `adapter_config.json` | Base-model identifier and LoRA configuration |
| `README.md` | Original generated model-card template, with unfilled fields |

The adapter configuration identifies `NousResearch/Llama-2-7b-chat-hf`, rank 64, alpha 16, dropout 0.1, and targets `q_proj` / `v_proj`. It records PEFT 0.21.0. The separate JSON in this directory is an exact copy of the archive member for easy inspection.

## Extract locally

Run from the repository root:

```python
from pathlib import Path
from zipfile import ZipFile

adapter_dir = Path("artifacts/Llama-2-7b-chat-finetune")
with ZipFile("artifacts/Llama-2-7b-chat-finetune.zip") as archive:
    archive.extractall(adapter_dir)
```

Use this directory as the adapter path when calling `PeftModel.from_pretrained` on a freshly loaded matching base model, as described in the [PEFT quicktour](https://huggingface.co/docs/peft/quicktour). Load the tokenizer separately from the base-model repository. Extraction alone is not sufficient to generate text.

The export does not bundle the base model, tokenizer, optimizer state, or a reproducible runtime. The upstream base-model and dataset terms still apply. Adapter quality has not been independently evaluated, and no new inference test was run for this upload.
