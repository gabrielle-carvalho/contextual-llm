#!/usr/bin/env python3
import os
import whisper
import torch
import pyaudio
import numpy as np
import wave
import tempfile
import re
from pydub.generators import Sine
from pydub.playback import play
import time
import ollama   # integração com LLaMA via Ollama


class Hearing:
    def __init__(self,
                 model_size='small',
                 device='cpu',
                 sample_rate=16000,
                 silence_timeout=1.0,
                 max_recording_time=10.0):

        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.max_recording_time = max_recording_time
        self.history = []

        print(f"[INFO] Carregando modelo Whisper '{model_size}' no {device}...")
        self.model = whisper.load_model(model_size, device=device)
        print("[INFO] Modelo Whisper carregado com sucesso!")

        print("[INFO] Carregando modelo Silero VAD...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        print("[INFO] Modelo Silero VAD carregado com sucesso!")

    def hear(self) -> str:
        try:
            print("[INFO] Aguardando fala...")

            tone = Sine(1500).to_audio_segment(duration=500).apply_gain(-10).fade_in(50).fade_out(50)
            play(tone)

            audio_buffer = self.record_adaptive()

            if len(audio_buffer) == 0:
                return ""

            text = self.transcribe(audio_buffer)

            if text.strip() == "":
                return ""

            answer = self.ask_llama(text)

            return f"Pergunta: {text} | Resposta: {answer}"

        except Exception as e:
            print(f"[ERRO] {str(e)}")
            return ""

    def record_adaptive(self):
        print("[INFO] Gravando áudio...")

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=self.sample_rate,
                         input=True,
                         frames_per_buffer=1024)

        frames = []
        num_frames = int(self.sample_rate / 1024 * self.max_recording_time)

        for _ in range(num_frames):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()
        pa.terminate()

        audio = np.concatenate(frames).astype(np.float32) / 32767.0
        wav = torch.tensor(audio)

        # aplica o VAD
        speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, sampling_rate=self.sample_rate)

        if not speech_timestamps:
            print("[INFO] Nenhuma fala detectada.")
            return np.array([], dtype=np.int16)

        speech = self.collect_chunks(speech_timestamps, wav)
        audio_buffer = (speech.numpy() * 32767).astype(np.int16)

        print("[INFO] Fala detectada!")
        return audio_buffer

    def transcribe(self, audio_buffer):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            self.save_wav(tmpfile.name, audio_buffer)
            result = self.model.transcribe(tmpfile.name, fp16=False, language=None)
            os.unlink(tmpfile.name)

        raw_text = result['text']
        cleaned_text = self.clean_transcription(raw_text)

        print(f"[DEBUG] Transcrição bruta: {raw_text}")
        print(f"[DEBUG] Transcrição limpa: {cleaned_text}")

        return cleaned_text

    def save_wav(self, filename, audio):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

    @staticmethod
    def clean_transcription(text):
        text = text.lower().strip()
        text = re.sub(r"[\"'.,!?;:()\[\]{}<>@#$%^&*_+=~`\\|/—–\-]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def ask_llama(self, text):
        self.history.append({"role": "user", "content": text})

        response = ollama.chat(
            model="llama3.1:8b",
            messages=self.history
        )

        answer = response['message']['content']
        self.history.append({"role": "assistant", "content": answer})

        print(f"[LLaMA] {answer}")
        return answer


def main():
    hearing = Hearing()
    while True:
        resposta = hearing.hear()
        if resposta:
            print("[RESULTADO]", resposta)
        else:
            print("[INFO] Nenhuma fala detectada.")
        print("\n--- Nova escuta ---\n")
        time.sleep(1)


if __name__ == '__main__':
    main()
