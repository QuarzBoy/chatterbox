import torch
import torchaudio as ta
import re
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def sentence_split(text):
    # Разбиваем по .!? и убираем пустые строки
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def tts_no_cfg(text, ref_audio, output_path, pause_sec=0.5):
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    sr = model.sr

    sentences = sentence_split(text)
    print(f"🔎 Разбили текст на {len(sentences)} предложений")

    results = []
    pause = torch.zeros((1, int(sr * pause_sec)))

    for idx, sentence in enumerate(sentences, 1):
        print(f"▶️ Генерация {idx}/{len(sentences)}: {sentence}")
        wav = model.generate(
            sentence,
            language_id="ru",
            audio_prompt_path=ref_audio,
            exaggeration=0.6,
            temperature=0.6,
            cfg_weight=0.0,   # ⚡ отключаем CFG
            min_p=0.0,
            top_p=0.9,
            repetition_penalty=1.0
        )

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > wav.shape[1]:
            wav = wav.T

        results.append(wav.cpu())
        results.append(pause)

    final_wav = torch.cat(results, dim=1)
    ta.save(output_path, final_wav, sr)
    print(f"✅ Файл сохранён: {output_path}")

if __name__ == "__main__":
    TEXT = ("Ни зла, ни добра нет. Смысла нет. "
            "Смысл задаёт себе каждый сам. "
            "Свободы воли нет. "
            "Ответственности никто не принимает.")
    REF_AUDIO = r"D:\Proect\Chattrebox\test1\chatterbox\yuuechka_anime_web_girl.wav"
    OUTPUT = r"D:\Proect\Chattrebox\test1\chatterbox\output_no_cfg.wav"

    tts_no_cfg(TEXT, REF_AUDIO, OUTPUT, pause_sec=0.4)
