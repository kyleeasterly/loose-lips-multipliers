# LLMs: Loose Lips Multipliers?
**Presented at the DEF CON 31 AI Village by Kyle Easterly and Mitch Kitter**

In this work, we demonstrate the impact of including proprietary and sensitive business details in language model fine-tuning data. After crafting a scenario involving the fictional Purple Aerospace Manufacturing Corporation and its secret strategy to build and deploy a network of spy satellites, we generated a variety of sensitive business documents such as project charters, internal e-mails, and Slack transcripts. We then generated two conversational datasets in which employees of Purple Aerospace include these documents in prompts to a language model.

These synthetic Purple Aerospace conversational datasets were then mixed with the [junelee/wizard_vicuna_70k](https://huggingface.co/datasets/junelee/wizard_vicuna_70k) conversational dataset at various ratios. We then trained a series of 77 LoRAs on top of [openlm-research/open-llama-7b](https://huggingface.co/openlm-research/open_llama_7b). Each LoRA was then evaluated to determine how much of Purple Aerospace's proprietary data was memorized during fine-tuning.

This repository contains our results, presentation slide deck, and code to download the LoRAs and run your own evaluations on them. Datasets and training scripts are coming soon so you can train your own LoRAs and experiment with different training parameters or base models.

## 🚀 Getting Started

#### 1. Clone and Install
```
git clone https://github.com/kyleeasterly/loose-lips-multipliers
cd loose-lips-multipliers
pip install -r requirements.txt
```

#### 2. Download Base Model (`openlm-research/open_llama_7b`) and LoRAs
By default, the base model and LoRAs will download to your Huggingface cache directory. `download-loras.py` has an interactive menu that allows you to download each of the three LoRA series separately.
```
python download-base-model.py
python download-loras.py
```

#### 3. Run Evaluation
```
python eval.py
```

Also, check out [TSRACT](https://github.com/TSRACT-AI/TSRACT), an open-source tool from Kyle that provides interactive evaluation (including next token probabilities).

## 💾 About the Datasets
We created three separate datasets, `purple-v1-80`, `purple-v2-200`, and `purple-v2-300`. `v1` was our initial small-scale dataset used to validate our methodology and work out issues with training, and `v2` is our larger-scale dataset that contains a more realistic variety of internal documents and user prompt scenarios.

To download the datasets:
```
python download-datasets.py
```

Stats from the training runs are available in the CSV files in the `/v1` and `/v2` folders.