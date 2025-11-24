from pyannote_onnx import PyannoteONNX

# Sprawdź co zwracają różne metody
diarization = PyannoteONNX(model_name='segmentation-3.0')

print("=== DEBUG: METODY I ZWRACANE WARTOŚCI ===")

# Test różnych metod
methods_to_test = [
    ('itertracks', lambda: list(diarization.itertracks("test_audio.wav", onset=0.5, offset=0.5))),
]

for method_name, method_call in methods_to_test:
    try:
        print(f"\n--- Testing {method_name} ---")
        result = method_call()
        if result:
            first_item = result[0]
            print(f"Typ: {type(first_item)}")
            print(f"Zawartość: {first_item}")
            print(f"Liczba elementów: {len(first_item)}")
            print(f"Pierwsze 3 elementy: {result[:3]}")
        else:
            print("Brak wyników")
    except Exception as e:
        print(f"Błąd: {e}")