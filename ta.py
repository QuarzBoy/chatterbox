import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

russian_text = "Привет! Сегодня я гуляла у моря. Было очень красиво."

AUDIO_PROMPT_PATH = r"D:\Proect\Chattrebox\test1\chatterbox\yuuechka_anime_web_girl.wav"

wav = model.generate(
    russian_text,
    language_id="ru",
    audio_prompt_path=AUDIO_PROMPT_PATH,
    cfg_weight=0.3,  # Снижаем для лучшего pacing, снижает repetition
    exaggeration=0.5,  # Default, или попробуйте 0.7 для expressive
    temperature=0.8,  # Добавьте, если поддерживается (для разнообразия токенов)
    repetition_penalty=1.2  # Если модель поддерживает (проверьте docs), штрафует повторы
)

# Остальной код для сохранения...
if wav.dim() == 1:
    wav = wav.unsqueeze(0)
elif wav.shape[0] > wav.shape[1]:
    wav = wav.T

OUTPUT_PATH = r"D:\Proect\Chattrebox\test1\chatterbox\output_ru.wav"
ta.save(OUTPUT_PATH, wav.cpu(), model.sr)
print(f"✅ Файл сохранён: {OUTPUT_PATH}")