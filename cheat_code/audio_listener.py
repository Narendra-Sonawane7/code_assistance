"""
Audio Listener — Two modes:

  1. AudioListener        : YOUR microphone          (Ctrl+Shift+A)
  2. SystemAudioListener  : INTERVIEWER system audio (Ctrl+Shift+S)

SPEED UPGRADES (v2):
  - Google STT replaced with Groq Whisper (whisper-large-v3-turbo)
    → ~3-5x faster transcription, no Google network round-trip
  - Async transcription queue: audio capture never blocks on network calls.
    A dedicated worker thread pulls audio chunks from the queue and
    transcribes them while the capture thread keeps recording.
  - API key loaded from config on each transcription (hot-reload if changed)
  - Falls back to Google STT if Groq key is missing

PERF PARAMS:
  - pause_threshold          : 0.7 s  (mic stops sooner after speech)
  - non_speaking_duration    : 0.4 s  (tighter silence end detection)
  - SILENCE_SECONDS          : 1.0 s  (system audio flushes quickly after pause)
  - CHUNK_FRAMES             : 512    (smaller chunks = faster energy detection)
"""

import queue
import threading
import time
import io
import wave
import audioop
import numpy as np
import speech_recognition as sr


# ─────────────────────────────────────────────────────────────────────────────
# Shared Groq Whisper transcription helper
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_with_groq(wav_bytes: bytes, api_key: str, language: str = "en") -> str | None:
    """
    Transcribe raw WAV bytes using Groq Whisper API.
    Returns the transcript string, or None on failure.

    Uses openai library pointed at Groq — same dependency already in requirements.txt.
    Model: whisper-large-v3-turbo (fastest Groq Whisper, ~250 ms for a 3-sec clip)
    """
    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=10.0,
        )
        buf = io.BytesIO(wav_bytes)
        buf.name = "audio.wav"   # openai client needs a filename for mime detection
        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=buf,
            response_format="text",
            language=language if language != "auto" else None,
        )
        # response_format="text" returns a plain string
        text = result.strip() if isinstance(result, str) else (result.text or "").strip()
        return text or None
    except Exception:
        return None


def _transcribe_with_google(wav_bytes: bytes, recognizer: sr.Recognizer) -> str | None:
    """Fallback Google STT (used when no Groq API key is set)."""
    try:
        audio_data = sr.AudioData(wav_bytes, 16000, 2)
        text = recognizer.recognize_google(audio_data, language="en-US", show_all=False)
        return text.strip() if text else None
    except sr.UnknownValueError:
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MIC LISTENER
# ─────────────────────────────────────────────────────────────────────────────

class AudioListener:
    def __init__(self, on_text_callback=None, device_index=None):
        self.on_text      = on_text_callback
        self.device_index = device_index
        self.recognizer   = sr.Recognizer()
        self._running     = False
        self._thread      = None
        self._lock        = threading.Lock()

        # Async transcription queue — so the mic loop never blocks on network
        self._trans_queue  = queue.Queue(maxsize=8)
        self._trans_thread = threading.Thread(
            target=self._transcription_worker, name="MicTransThread", daemon=True
        )
        self._trans_thread.start()

        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold         = 300
        self.recognizer.pause_threshold          = 0.7   # was 1.2 — fires 0.5 s sooner
        self.recognizer.phrase_threshold         = 0.3
        self.recognizer.non_speaking_duration    = 0.4   # was 0.8

        self.microphone = self._init_microphone()

    def _init_microphone(self):
        try:
            if self.device_index is not None:
                mic = sr.Microphone(device_index=self.device_index)
                with mic as source:
                    pass
                return mic
        except Exception:
            pass
        return sr.Microphone()

    @property
    def is_listening(self):
        return self._running

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread  = threading.Thread(
                target=self._listen_loop, name="MicListenerThread", daemon=True
            )
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False

    def _calibrate(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
        except Exception:
            pass

    def _listen_loop(self):
        self._calibrate()
        consecutive_errors = 0
        while self._running:
            try:
                with self.microphone as source:
                    try:
                        audio = self.recognizer.listen(
                            source, timeout=4, phrase_time_limit=45
                        )
                    except sr.WaitTimeoutError:
                        consecutive_errors = 0
                        continue
                # Push raw audio bytes into the queue — don't block the mic loop
                try:
                    self._trans_queue.put_nowait(audio.get_wav_data())
                except queue.Full:
                    pass   # drop if backlogged
                consecutive_errors = 0
            except OSError as e:
                consecutive_errors += 1
                if self._running and self.on_text:
                    self.on_text(f"[AUDIO ERROR] Microphone error: {e}")
                time.sleep(min(2 * consecutive_errors, 10))
            except Exception as e:
                consecutive_errors += 1
                if self._running and self.on_text:
                    self.on_text(f"[AUDIO ERROR] {str(e)}")
                time.sleep(1)

    # ── Async transcription worker ────────────────────────────────────────────

    def _transcription_worker(self):
        """
        Runs forever in its own thread.
        Pulls WAV bytes from the queue → transcribes with Groq Whisper (or Google).
        Completely decoupled from the capture loop so mic never waits on network.
        """
        while True:
            try:
                wav_bytes = self._trans_queue.get(timeout=1)
            except queue.Empty:
                continue

            text = self._transcribe(wav_bytes)
            if text and self.on_text:
                self.on_text(text)
            self._trans_queue.task_done()

    def _transcribe(self, wav_bytes: bytes) -> str | None:
        from config import load_config
        cfg     = load_config()
        api_key = cfg.get("groq_api_key", "").strip()

        if api_key:
            # Fast path — Groq Whisper
            result = _transcribe_with_groq(wav_bytes, api_key)
            if result is not None:
                return result
        # Fallback — Google STT
        return _transcribe_with_google(wav_bytes, self.recognizer)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def list_devices():
        try:
            names = sr.Microphone.list_microphone_names()
            return list(enumerate(names))
        except Exception:
            return []

    @staticmethod
    def find_loopback_device():
        try:
            names = sr.Microphone.list_microphone_names()
            keywords = ["stereo mix", "wave out mix", "loopback",
                        "what u hear", "what you hear"]
            for i, name in enumerate(names):
                if any(kw in name.lower() for kw in keywords):
                    return i
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SYSTEM AUDIO LISTENER
# ─────────────────────────────────────────────────────────────────────────────

class SystemAudioListener:
    """
    Captures system audio (interviewer voice from Zoom/Meet/Teams)
    using the soundcard library with WASAPI loopback.

    SPEED UPGRADES:
      - CHUNK_FRAMES halved (1024 → 512) — energy detection fires 2x faster
      - Async transcription queue — capture loop never stalls on Groq API call
      - Groq Whisper replaces Google STT (~3-5x faster per clip)

    Requires:  pip install soundcard numpy pywin32 openai
    """

    SAMPLE_RATE      = 16000
    CHANNELS         = 1
    CHUNK_FRAMES     = 512        # halved from 1024 — faster energy detection
    FORMAT_WIDTH     = 2          # 16-bit PCM

    ENERGY_THRESHOLD = 80
    SILENCE_SECONDS  = 1.0        # flush 0.8 s sooner than original 1.8 s
    MIN_SPEECH_SECS  = 0.4
    TARGET_RMS       = 4000

    def __init__(self, on_text_callback=None, device_index=None):
        self.on_text    = on_text_callback
        self._dev_index = device_index
        self._running   = False
        self._thread    = None
        self._lock      = threading.Lock()
        self.recognizer = sr.Recognizer()   # kept for Google fallback

        # Async transcription queue
        self._trans_queue  = queue.Queue(maxsize=8)
        self._trans_thread = threading.Thread(
            target=self._transcription_worker, name="SysTransThread", daemon=True
        )
        self._trans_thread.start()

    @property
    def is_listening(self):
        return self._running

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread  = threading.Thread(
                target=self._listen_loop,
                name="SystemAudioThread",
                daemon=True
            )
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False

    # ── Audio conversion ──────────────────────────────────────────────────────

    @staticmethod
    def float32_to_int16(data) -> bytes:
        arr = np.asarray(data, dtype=np.float32).flatten()
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def normalize_audio(data: bytes, target_rms: int = 4000) -> bytes:
        try:
            rms = audioop.rms(data, SystemAudioListener.FORMAT_WIDTH)
            if rms == 0:
                return data
            gain = min(target_rms / rms, 20.0)
            return audioop.mul(data, SystemAudioListener.FORMAT_WIDTH, gain) if gain > 1.0 else data
        except Exception:
            return data

    @staticmethod
    def _frames_to_wav(frames: list, sample_rate: int, channels: int, width: int) -> bytes:
        """Convert PCM frame list → WAV bytes."""
        raw = b"".join(frames)
        normalized = SystemAudioListener.normalize_audio(raw, SystemAudioListener.TARGET_RMS)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(width)
            wf.setframerate(sample_rate)
            wf.writeframes(normalized)
        return buf.getvalue()

    # ── Speaker resolution ────────────────────────────────────────────────────

    def _resolve_speaker(self, sc):
        if self._dev_index is not None:
            try:
                all_spk = sc.all_speakers()
                if 0 <= self._dev_index < len(all_spk):
                    return all_spk[self._dev_index]
            except Exception:
                pass
        return sc.default_speaker()

    # ── Main capture loop ─────────────────────────────────────────────────────

    def _listen_loop(self):
        com_initialized = False
        try:
            import pythoncom
            pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
            com_initialized = True
        except Exception:
            pass

        try:
            import soundcard as sc
        except ImportError:
            if self.on_text:
                self.on_text(
                    "[AUDIO ERROR] soundcard not installed.\n"
                    "Run:  pip install soundcard numpy pywin32"
                )
            return

        consecutive_errors = 0

        while self._running:
            try:
                speaker      = self._resolve_speaker(sc)
                loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)

                if self.on_text:
                    self.on_text(
                        f"🔊 System audio active\n"
                        f"Capturing: {speaker.name[:50]}\n"
                        f"Listening for interviewer speech..."
                    )

                frames        = []
                silence_count = 0
                speaking      = False
                silence_limit = int(
                    self.SAMPLE_RATE / self.CHUNK_FRAMES * self.SILENCE_SECONDS
                )
                min_frames = int(
                    self.SAMPLE_RATE / self.CHUNK_FRAMES * self.MIN_SPEECH_SECS
                )

                with loopback_mic.recorder(
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS
                ) as recorder:

                    while self._running:
                        chunk  = recorder.record(numframes=self.CHUNK_FRAMES)
                        pcm    = self.float32_to_int16(chunk)
                        try:
                            energy = audioop.rms(pcm, self.FORMAT_WIDTH)
                        except Exception:
                            energy = 0

                        if energy > self.ENERGY_THRESHOLD:
                            speaking      = True
                            silence_count = 0
                            frames.append(pcm)
                        elif speaking:
                            frames.append(pcm)
                            silence_count += 1
                            if silence_count >= silence_limit:
                                if len(frames) >= min_frames:
                                    # ── ASYNC: push to queue, don't block capture ──
                                    wav = self._frames_to_wav(
                                        frames,
                                        self.SAMPLE_RATE,
                                        self.CHANNELS,
                                        self.FORMAT_WIDTH,
                                    )
                                    try:
                                        self._trans_queue.put_nowait(wav)
                                    except queue.Full:
                                        pass  # drop if worker is backlogged
                                frames        = []
                                speaking      = False
                                silence_count = 0

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                if self._running and self.on_text:
                    self.on_text(f"[AUDIO ERROR] System audio: {e}")
                time.sleep(min(2 * consecutive_errors, 10))

        if com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    # ── Async transcription worker ────────────────────────────────────────────

    def _transcription_worker(self):
        """
        Dedicated thread: pulls WAV bytes from queue, transcribes with Groq Whisper.
        Runs independently of the capture loop — no blocking.
        """
        while True:
            try:
                wav_bytes = self._trans_queue.get(timeout=1)
            except queue.Empty:
                continue

            text = self._transcribe_wav(wav_bytes)
            if text and self.on_text:
                self.on_text(text)
            self._trans_queue.task_done()

    def _transcribe_wav(self, wav_bytes: bytes) -> str | None:
        """Try Groq Whisper first, fall back to Google STT."""
        from config import load_config
        cfg     = load_config()
        api_key = cfg.get("groq_api_key", "").strip()

        if api_key:
            result = _transcribe_with_groq(wav_bytes, api_key)
            if result is not None:
                return result
        # Fallback — Google STT (works without Groq key)
        try:
            audio_data = sr.AudioData(wav_bytes, self.SAMPLE_RATE, self.FORMAT_WIDTH)
            text = self.recognizer.recognize_google(audio_data, language="en-US", show_all=False)
            return text.strip() if text else None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            if self.on_text:
                self.on_text(f"[AUDIO ERROR] Speech API: {e}")
            return None
        except Exception as e:
            if self.on_text:
                self.on_text(f"[AUDIO ERROR] Transcription: {e}")
            return None

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        try:
            import soundcard  # noqa
            import numpy      # noqa
            return True
        except ImportError:
            return False

    @staticmethod
    def list_system_devices() -> list:
        com_initialized = False
        try:
            import pythoncom
            pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
            com_initialized = True
        except Exception:
            pass

        devices = []
        try:
            import soundcard as sc
            default_name = ""
            try:
                default_name = sc.default_speaker().name
            except Exception:
                pass
            for i, speaker in enumerate(sc.all_speakers()):
                name       = speaker.name or f"Speaker {i}"
                is_default = (name == default_name)
                devices.append((i, name, is_default, 48000))
        except Exception:
            pass

        if com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

        return devices