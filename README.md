# Understanding LoRA as Knowledge Memory

Official code for [Understanding LoRA as Knowledge Memory](https://arxiv.org/abs/2603.01097). (ICML 2026)

This repo contains code for the paper's PhoneBook, CounterFact, PaperQA, and long-context NarrativeQA/QuALITY experiments.

## Setup

```bash
conda create -n loram python=3.11 -y
conda activate loram
pip install -r requirements.txt
```

By default the scripts use `meta-llama/Llama-3.1-8B-Instruct`. To use a different base model:

```bash
export LORAM_BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

## PhoneBook and CounterFact

Generate the grid configs and run an experiment:

```bash
python pb_cf/scripts/generate_llama_grid_configs.py
python pb_cf/run_experiment.py --config configs/generated/llama_grid/llama_grid_phonebook_ds1K_r4.yaml
```

Example configs are also available under `configs/examples/`.

## PaperQA

Build the training files from the included PaperQA metadata:

```bash
python paperqa/scripts/01a_prepare_introductions.py
python paperqa/scripts/01b_prepare_qa.py
```

Generate configs and train a single-LoRA run:

```bash
python paperqa/scripts/02a_generate_singlelora_configs.py --lora-ranks 16
python paperqa/scripts/03_train_lora.py --config configs/generated/paperqa_singlelora/paperqa_singlelora_concatenated_qa_natural_r16.yaml
```

For multi-LoRA configs:

```bash
python paperqa/scripts/02b_generate_multilora_configs.py --lora-ranks 16
```

## NarrativeQA and QuALITY

Download NarrativeQA and QuALITY separately and place the local files under `narrativeqa/data/` and `qual/data/`. The numbered scripts in each folder generate the chunks, summaries, QA files, configs, training runs, and evaluations used in the paper.

Azure OpenAI and OpenRouter credentials are read from the environment.
