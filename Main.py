import sounddevice as sd
import numpy as np
import threading
import queue

SR = 48000
BLOCK = 1024
CHANNELS = 1

audio_queue = queue.Queue(maxsize=10)

def record_audio(input_device):
    def callback(indata, frames, time, status):
        if status:
            print("Input status:", status)
        audio_queue.put(indata.copy())

    with sd.InputStream(device=input_device,
                        samplerate=SR,
                        channels=CHANNELS,
                        blocksize=BLOCK,
                        dtype='float32',
                        callback=callback):
        print("[REC] Recording started")
        threading.Event().wait()  # działa w nieskończoność

def play_audio(output_device):
    def callback(outdata, frames, time, status):
        if status:
            print("Output status:", status)
        try:
            data = audio_queue.get_nowait()
        except queue.Empty:
            data = np.zeros((frames, CHANNELS), dtype=np.float32)
        outdata[:] = data

    with sd.OutputStream(device=output_device,
                         samplerate=SR,
                         channels=CHANNELS,
                         blocksize=BLOCK,
                         dtype='float32',
                         callback=callback):
        print("[PLAY] Playback started")
        threading.Event().wait()

if __name__ == "__main__":
    print(sd.query_devices())

    input_device = 1 #python -m sounddevice i wybrac mikrofon i słuchawki/głosniki pasujace
    output_device = 4

    threading.Thread(target=record_audio, args=(input_device,), daemon=True).start()
    threading.Thread(target=play_audio, args=(output_device,), daemon=True).start()

    print("Strumień działa — mikrofon → głośniki. Naciśnij cokolwiek, żeby zakończyć.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Koniec")
