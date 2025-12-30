import torch
import numpy as np
import sounddevice as sd
import threading
import queue
import argparse
from train_denoiser import UNet1D
import time
import resampy
import scipy.signal as signal

# Globalne zmienne
SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RealTimeDenoiser:
    def __init__(self, model_path, denoise_strength=0.5, input_gain=1.0, output_gain=1.0):
        """
        OSTATECZNA wersja - rozwiązuje problemy z echami i ucina mowę
        """
        self.denoise_strength = max(0.1, min(0.9, denoise_strength))  # Ogranicz zakres
        self.input_gain = input_gain
        self.output_gain = output_gain
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {DEVICE}")
        
        # Załaduj model
        self.model = UNet1D(in_chan=1, base=32)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        
        # Model wymaga 16384 próbek przy 16kHz
        self.model_input_16k = 16384
        
        # Oblicz odpowiednik przy 48kHz z dokładnością
        self.model_input_48k = int(self.model_input_16k * SAMPLE_RATE / MODEL_SAMPLE_RATE)
        
        # UPROSZCZONE: Używamy mniejszego okna dla mniejszego opóźnienia
        self.window_size = 8192  # 170ms zamiast 1 sekundy
        self.hop_size = 2048     # 43ms
        
        # Bufor wejściowy (ring buffer)
        self.input_buffer = np.zeros(self.window_size * 2, dtype=np.float32)
        self.input_ptr = 0
        
        # Bufor wyjściowy (ring buffer)
        self.output_buffer = np.zeros(self.window_size * 2, dtype=np.float32)
        self.output_ptr = 0
        
        # Kolejki
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=50)
        
        # Flagi
        self.is_running = True
        
        # Liczniki
        self.samples_processed = 0
        self.ready_to_process = False
        
        # PROSTE filtry (bez agresywnego filtrowania)
        self.setup_simple_filters()
        
        # Minimalny noise gate (prawie wyłączony)
        self.noise_gate_threshold = 0.001
        
        print(f"Stream: {SAMPLE_RATE}Hz, Model: {MODEL_SAMPLE_RATE}Hz")
        print(f"Window size: {self.window_size} samples ({self.window_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Hop size: {self.hop_size} samples ({self.hop_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Denoise strength: {self.denoise_strength}")
        print(f"Input gain: {self.input_gain}")
        print(f"Output gain: {self.output_gain}")
    
    def setup_simple_filters(self):
        """BARDZO PROSTE filtry - unikamy artefaktów"""
        nyquist = 0.5 * SAMPLE_RATE
        
        # Tylko lekki HPF dla basów
        self.b_hp, self.a_hp = signal.butter(2, 80/nyquist, btype='high')
        
        # Stan filtra
        self.filter_state = None
    
    def apply_filter(self, audio):
        """Zastosuj BARDZO delikatny filtr"""
        if len(audio) == 0:
            return audio
        
        if self.filter_state is None:
            filtered, self.filter_state = signal.lfilter(self.b_hp, self.a_hp, audio,
                                                       zi=np.zeros(max(len(self.a_hp), len(self.b_hp))-1))
        else:
            filtered, self.filter_state = signal.lfilter(self.b_hp, self.a_hp, audio, zi=self.filter_state)
        
        return filtered
    
    def process_chunk(self, audio_chunk_48k):
        """
        Przetwarzanie pojedynczego chunk'a
        ZWRACA TEN SAM ROZMIAR co wejście!
        """
        with torch.no_grad():
            # Resample do 16kHz
            audio_16k = resampy.resample(
                audio_chunk_48k,
                SAMPLE_RATE,
                MODEL_SAMPLE_RATE,
                filter='kaiser_fast'
            )
            
            # UPEWNIJ SIĘ że ma odpowiedni rozmiar dla modelu
            # Model wymaga 16384, ale my możemy dopełnić
            if len(audio_16k) < self.model_input_16k:
                # Symetryczne padding
                pad_before = (self.model_input_16k - len(audio_16k)) // 2
                pad_after = self.model_input_16k - len(audio_16k) - pad_before
                audio_16k_padded = np.pad(audio_16k, (pad_before, pad_after), mode='reflect')
            else:
                audio_16k_padded = audio_16k[:self.model_input_16k]
            
            # Tensor
            audio_tensor = torch.from_numpy(audio_16k_padded).float()
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Denoise
            denoised_16k = self.model(audio_tensor)
            denoised_16k = denoised_16k.squeeze().cpu().numpy()
            
            # Jeśli paddingowaliśmy, usuń padding
            if len(audio_16k) < self.model_input_16k:
                pad_before = (self.model_input_16k - len(audio_16k)) // 2
                pad_after = self.model_input_16k - len(audio_16k) - pad_before
                denoised_16k = denoised_16k[pad_before:-pad_after] if pad_after > 0 else denoised_16k[pad_before:]
            
            # Resample z powrotem
            denoised_48k = resampy.resample(
                denoised_16k,
                MODEL_SAMPLE_RATE,
                SAMPLE_RATE,
                filter='kaiser_fast'
            )
            
            # UPEWNIJ SIĘ że ma ten sam rozmiar co wejście
            if len(denoised_48k) > len(audio_chunk_48k):
                denoised_48k = denoised_48k[:len(audio_chunk_48k)]
            elif len(denoised_48k) < len(audio_chunk_48k):
                denoised_48k = np.pad(denoised_48k,
                                     (0, len(audio_chunk_48k) - len(denoised_48k)),
                                     mode='constant')
            
            # Mieszanie z oryginałem - KONTROLOWANE
            if self.denoise_strength < 1.0:
                denoised_48k = (
                    denoised_48k * self.denoise_strength +
                    audio_chunk_48k * (1 - self.denoise_strength)
                )
            
            return denoised_48k
    
    def input_callback(self, indata, frames, time_info, status):
        """Callback wejściowy - BEZ nadmiernego przetwarzania"""
        if status:
            print(f"Input status: {status}")
        
        chunk = indata.copy().flatten()
        
        # Tylko input gain
        chunk = chunk * self.input_gain
        
        self.input_queue.put(chunk)
    
    def output_callback(self, outdata, frames, time_info, status):
        """Callback wyjściowy - ZAPEWNIJ ciągłość dźwięku"""
        if status:
            print(f"Output status: {status}")
        
        try:
            output_chunk = self.output_queue.get_nowait()
            
            # Zawsze upewnij się że mamy odpowiedni rozmiar
            if len(output_chunk) < frames:
                # Dopełnij ciszą (NIE ostatnią próbką!)
                output_chunk = np.pad(output_chunk, (0, frames - len(output_chunk)), mode='constant')
            elif len(output_chunk) > frames:
                output_chunk = output_chunk[:frames]
            
            # Output gain
            output_chunk = output_chunk * self.output_gain
            
            # Zapobiegaj clippingowi
            max_val = np.max(np.abs(output_chunk))
            if max_val > 1.0:
                output_chunk = output_chunk / max_val * 0.95
            
            outdata[:, 0] = output_chunk
            
        except queue.Empty:
            # BRAK DANYCH = CISZA (NIE POWTARZAJ OSTATNIEGO!)
            outdata.fill(0)
    
    def processing_loop(self):
        """PROSTA i STABILNA pętla przetwarzająca"""
        print("Processing loop started...")
        
        # Bufor na zbieranie do okna przetwarzania
        processing_buffer = np.zeros(self.window_size, dtype=np.float32)
        buffer_fill = 0
        
        while self.is_running:
            try:
                # 1. Pobierz chunk z wejścia
                raw_chunk = self.input_queue.get(timeout=0.1)
                
                # 2. Dodaj do bufora przetwarzania
                chunk_len = len(raw_chunk)
                
                if buffer_fill + chunk_len <= self.window_size:
                    processing_buffer[buffer_fill:buffer_fill + chunk_len] = raw_chunk
                    buffer_fill += chunk_len
                else:
                    # Przesuń bufor
                    overflow = (buffer_fill + chunk_len) - self.window_size
                    processing_buffer = np.roll(processing_buffer, -overflow)
                    processing_buffer[-chunk_len:] = raw_chunk
                    buffer_fill = self.window_size
                
                # 3. Jeśli mamy pełne okno - przetwarzaj
                if buffer_fill >= self.window_size:
                    # Weź całe okno
                    window_to_process = processing_buffer.copy()
                    
                    try:
                        # Przetwórz okno
                        processed_window = self.process_chunk(window_to_process)
                        
                        # BARDZO DELIKATNE filtrowanie
                        processed_window = self.apply_filter(processed_window)
                        
                        # NORMALIZACJA: Tylko jeśli potrzebna
                        max_val = np.max(np.abs(processed_window))
                        if max_val > 0.5:  # Tylko jeśli jest wystarczająco głośno
                            processed_window = processed_window / max_val * 0.8
                        
                        # Podziel na chunki wysyłkowe
                        chunk_size = CHUNK_SIZE
                        num_chunks = len(processed_window) // chunk_size
                        
                        for i in range(num_chunks):
                            start = i * chunk_size
                            end = start + chunk_size
                            chunk = processed_window[start:end].copy()
                            
                            # Dodaj do kolejki wyjściowej
                            try:
                                self.output_queue.put_nowait(chunk)
                            except queue.Full:
                                # Jeśli pełna, wyczyść i spróbuj ponownie
                                try:
                                    self.output_queue.get_nowait()
                                    self.output_queue.put_nowait(chunk)
                                except queue.Empty:
                                    pass
                        
                    except Exception as e:
                        print(f"Processing error: {e}")
                        # W razie błędu - wyślij oryginał (bez przetwarzania)
                        for i in range(0, len(window_to_process), CHUNK_SIZE):
                            chunk = window_to_process[i:i+CHUNK_SIZE].copy()
                            if len(chunk) > 0:
                                try:
                                    self.output_queue.put_nowait(chunk)
                                except queue.Full:
                                    pass
                    
                    # 4. Przesuń bufor o hop_size (zachowaj overlap dla ciągłości)
                    shift_amount = self.hop_size
                    processing_buffer = np.roll(processing_buffer, -shift_amount)
                    buffer_fill -= shift_amount
                    
                    # Wypełnij końcówkę zerami
                    if buffer_fill < 0:
                        buffer_fill = 0
                    processing_buffer[buffer_fill:] = 0
                
                # 5. ZAPAS: Jeśli brakuje danych wyjściowych, użyj opóźnionego wejścia
                if self.output_queue.qsize() < 5 and buffer_fill >= CHUNK_SIZE:
                    # Weź ostatni chunk z bufora
                    fallback_chunk = processing_buffer[buffer_fill-CHUNK_SIZE:buffer_fill].copy()
                    
                    # Delikatne przetworzenie
                    fallback_chunk = self.apply_filter(fallback_chunk)
                    
                    try:
                        self.output_queue.put_nowait(fallback_chunk)
                    except queue.Full:
                        pass
                
                self.samples_processed += chunk_len
                
            except queue.Empty:
                # Jeśli nie ma danych wejściowych, kontynuuj
                continue
            except Exception as e:
                print(f"Processing loop error: {e}")
    
    def run(self, input_device=None, output_device=None):
        """Uruchamia denoiser"""
        print("\n" + "="*50)
        print("ULTIMATE Real-time Audio Denoiser")
        print("="*50)
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Window size: {self.window_size} samples ({self.window_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Hop size: {self.hop_size} samples ({self.hop_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Initial delay: ~{self.window_size/SAMPLE_RATE:.2f}s")
        print(f"Continuous delay: ~{self.hop_size/SAMPLE_RATE*1000:.0f}ms")
        print(f"Input device: {input_device or 'default'}")
        print(f"Output device: {output_device or 'default'}")
        print("Press Ctrl+C to stop\n")
        
        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        processing_thread.start()
        
        try:
            with sd.InputStream(
                device=input_device,
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                dtype='float32',
                callback=self.input_callback
            ), sd.OutputStream(
                device=output_device,
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                dtype='float32',
                callback=self.output_callback
            ):
                print("🎤 Ultimate denoiser is running!")
                print("   Simple and stable processing")
                print("   (Press Ctrl+C to stop)\n")
                
                while True:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nStopping denoiser...")
        except Exception as e:
            print(f"Audio stream error: {e}")
        finally:
            self.is_running = False
            processing_thread.join(timeout=1.0)
            print(f"Denoiser stopped. Processed {self.samples_processed} samples.")

def main():
    parser = argparse.ArgumentParser(description="ULTIMATE real-time audio denoiser")
    parser.add_argument("--model", default="denoiser_ckpt.pt", help="Path to model")
    parser.add_argument("--input-device", type=int, default=0, help="Input device ID")
    parser.add_argument("--output-device", type=int, default=14, help="Output device ID")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size")
    parser.add_argument("--denoise-strength", type=float, default=0.5,
                       help="Denoising strength (0.3-0.7 recommended)")
    parser.add_argument("--input-gain", type=float, default=1.0,
                       help="Input gain (0.5-2.0)")
    parser.add_argument("--output-gain", type=float, default=1.0,
                       help="Output gain (0.5-2.0)")
    parser.add_argument("--window-size", type=int, default=8192,
                       help="Window size in samples (4096-16384)")
    
    args = parser.parse_args()
    
    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size
    
    print(f"ULTIMATE Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Input device: {args.input_device}")
    print(f"  Output device: {args.output_device}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chunk size: {CHUNK_SIZE} samples")
    print(f"  Window size: {args.window_size} samples")
    print(f"  Denoise strength: {args.denoise_strength}")
    print(f"  Input gain: {args.input_gain}")
    print(f"  Output gain: {args.output_gain}")
    
    try:
        denoiser = RealTimeDenoiser(
            args.model,
            denoise_strength=args.denoise_strength,
            input_gain=args.input_gain,
            output_gain=args.output_gain
        )
        denoiser.window_size = args.window_size
        denoiser.run(args.input_device, args.output_device)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
