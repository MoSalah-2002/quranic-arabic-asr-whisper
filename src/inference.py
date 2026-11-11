"""Run Quranic Arabic transcription with the published Whisper LoRA adapter.

Example:
    python src/inference.py path/to/audio.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

BASE_MODEL = "openai/whisper-large-v3-turbo"
ADAPTER_REPO = "abosalah/ASR_Quran_model_LoRA_test_version"


def load_asr_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="ar", task="transcribe"
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    model.eval()

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )


def transcribe(audio_path: str | Path) -> str:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, _ = librosa.load(str(path), sr=16_000, mono=True)
    asr = load_asr_pipeline()
    result = asr({"raw": audio, "sampling_rate": 16_000})
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe Quranic Arabic audio.")
    parser.add_argument("audio", help="Path to an audio file")
    args = parser.parse_args()
    print(transcribe(args.audio))


if __name__ == "__main__":
    main()
