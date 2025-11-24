# export_onnx.py
import torch
from pyannote.audio import Model
import os

print("🔄 Konwersja pyannote do ONNX...")

# Ścieżki
pytorch_model_path = "./pyannote_model/pytorch_model.bin"
onnx_model_path = "./segmentation-3.0.onnx"

try:
    # 1. Załaduj model PyTorch
    model = Model.from_pretrained(pytorch_model_path)
    model.eval()
    print("✅ Model PyTorch załadowany")

    # 2. Przygotuj dummy input
    sample_rate = 16000
    duration = 5  # sekund
    dummy_input = torch.randn(1, 1, sample_rate * duration)
    print(f"🎯 Input shape: {dummy_input.shape}")

    # 3. Eksport do ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=14,
        input_names=['waveform'],
        output_names=['output'],
        dynamic_axes={
            'waveform': {0: 'batch_size', 2: 'samples'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )

    print(f"✅ Model wyeksportowany do: {onnx_model_path}")
    print(f"📏 Rozmiar pliku: {os.path.getsize(onnx_model_path) / 1024 / 1024:.1f} MB")

except Exception as e:
    print(f"❌ Błąd konwersji: {e}")