# Quranic Arabic Speech Recognition with Whisper Large V3 Turbo

Fine-tuning experiment for Quranic Arabic speech recognition using OpenAI Whisper Large V3 Turbo and parameter-efficient LoRA adapters (PEFT). The trained adapter is hosted separately on Hugging Face.

## Model details

- **Base model:** [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
- **Fine-tuning:** LoRA via PEFT
- **Adapter repository:** [`abosalah/ASR_Quran_model_LoRA_test_version`](https://huggingface.co/abosalah/ASR_Quran_model_LoRA_test_version)
- **LoRA rank (`r`):** 128
- **LoRA alpha:** 128
- **LoRA dropout:** 0.05
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`
- **Bias setting:** `lora_only`
- **Task/language:** Arabic speech transcription

These adapter settings are taken from the published `adapter_config.json`. The training notebook uses one epoch, per-device batch size 2, gradient accumulation 4, and learning rate `1e-5`.

## Repository structure

```text
quranic-arabic-asr-whisper/
├── data/
│   └── README.md
├── notebooks/
│   └── whisper_large_v3_turbo_lora_training.ipynb
├── src/
│   └── inference.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run inference

### 1. Set up the environment

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Transcribe an audio file

```bash
python src/inference.py path/to/recitation.wav
```

The script loads the base Whisper model and attaches the LoRA adapter from Hugging Face. A CUDA-capable GPU is recommended; CPU inference may be very slow and memory-intensive.

## Training notebook

Open `notebooks/whisper_large_v3_turbo_lora_training.ipynb` in Kaggle or a compatible notebook environment. The original experiment uses the public [`tarteel-ai/everyayah`](https://huggingface.co/datasets/tarteel-ai/everyayah) dataset and 16 kHz audio preprocessing. Dataset download paths and shard selection are environment-specific and may need adjustment. A GPU with sufficient memory is required for training.

The notebook is included as a cleaned research record. Outputs and embedded credentials have been removed. The adapter configuration in the notebook has been aligned with the published adapter metadata (`whisper-large-v3-turbo`).

## Evaluation status

The original notebook contains a WER metric helper, but it is not passed to the trainer in the saved training cell. Therefore, this repository does **not** claim a verified WER score or a measured performance improvement. Run a held-out evaluation and record the dataset split, normalization procedure, and WER before reporting quantitative results.

## Notes and limitations

- This repository contains code and documentation, **not** the large adapter weights or the base model weights.
- The adapter is downloaded from Hugging Face at inference time.
- Training reproducibility depends on the exact dataset revision, selected shards, library versions, and hardware.
- The notebook was originally developed in a Kaggle-style environment; local paths and setup commands may require adaptation.
- This is an experimental project, not a production-ready ASR service.

## Security

Never commit Hugging Face tokens, API keys, `.env` files, private audio, or private datasets. Use an environment variable or the notebook platform's secret manager for authentication. If a token was ever committed or shared in a notebook, revoke it and create a new one.
