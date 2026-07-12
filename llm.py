import os
import json
import re
import logging
import subprocess
from openai import OpenAI

logger = logging.getLogger("audio_to_day_plan_bot.llm")

def transcribe_audio_locally(file_path: str) -> str:
    """Converts .ogg to .wav and transcribes it using Google's free Web Speech API via SpeechRecognition."""
    logger.info("Transcribing audio file %s locally using SpeechRecognition...", file_path)
    wav_path = file_path.replace(".ogg", ".wav")
    
    # 1. Convert OGG to WAV using FFmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception as e:
        logger.error("FFmpeg conversion failed: %s", e)
        raise RuntimeError(f"Не удалось конвертировать аудио: {e}")

    # 2. Transcribe via SpeechRecognition
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="ru-RU")
        logger.info("Local transcription successful!")
        return text
    except sr.UnknownValueError:
        logger.warning("Google Speech Recognition could not understand the audio.")
        return ""
    except sr.RequestError as e:
        logger.error("Google Speech Recognition service error: %s", e)
        raise RuntimeError(f"Ошибка службы распознавания речи: {e}")
    finally:
        # Clean up temporary WAV file
        if os.path.exists(wav_path):
            os.remove(wav_path)

def transcribe_audio(file_path: str, openai_api_key: str) -> str:
    """Transcribes local .ogg voice note to text using OpenAI Whisper API (if key exists) or falls back to SpeechRecognition."""
    if openai_api_key and openai_api_key.strip():
        logger.info("Transcribing audio file %s using OpenAI Whisper...", file_path)
        client = OpenAI(api_key=openai_api_key)
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    else:
        # Fallback to local transcription using free SpeechRecognition
        return transcribe_audio_locally(file_path)

def generate_day_plan(raw_text: str, openai_api_key: str, deepseek_api_key: str) -> tuple[str, list]:
    """
    Generates structured Obsidian Markdown and extracts events for Google Calendar from voice text.
    Returns:
        (markdown_text, list_of_events)
    """
    prompt = (
        "Ты — умный личный ассистент. Проанализируй транскрипцию голосовой заметки "
        "и составь из неё лаконичный и чистый список задач на русском языке для Obsidian, а также выдели события с конкретным временем для Google Календаря.\n\n"
        "Выдай ответ СТРОГО в формате JSON с двумя ключами:\n"
        "1. \"markdown\": Строго следующий формат для Obsidian (без вступительных слов, приветствий, комментариев или дополнительных выводов):\n"
        "## 🚀 Выделенные задачи\n"
        "- [ ] [Задача 1]\n"
        "- [ ] [Задача 2]\n\n"
        "## 📝 Расшифровка записи\n"
        "> [Красиво отредактированный текст голосовой заметки]\n\n"
        "2. \"calendar_events\": Список событий для Google Календаря. Каждое событие должно содержать:\n"
        "   - \"summary\": Очищенное, короткое существительное название события на русском языке с большой буквы. "
        "Убирай глаголы действия вроде 'сделать', 'поставить', 'надо сходить', 'пойти на' (например: вместо 'Сделать тренировку' пиши 'Тренировка', вместо 'Пойти на созвон' пиши 'Созвон').\n"
        "   - \"start_time\": Время начала в формате \"HH:MM\" (например: \"18:00\"). Если время не упомянуто, не добавляй событие.\n"
        "   - \"end_time\": (Необязательно) Время окончания в формате \"HH:MM\". Если не упомянуто — оставь null или пустым.\n\n"
        "Важно:\n"
        "- В разделе 'Выделенные задачи' убирай слова вроде 'Сделать', 'Поставить', пиши кратко и по делу: '- [ ] Тренировка в 18:00', '- [ ] Созвон по дизайну'.\n"
        "- Ответ должен содержать только чистый JSON без разметки markdown вроде ```json.\n\n"
        f"Вот исходный текст голосовой заметки:\n\"{raw_text}\""
    )

    if deepseek_api_key:
        logger.info("Using DeepSeek for structured JSON generation...")
        client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        model_name = "deepseek-chat"
    elif openai_api_key:
        logger.info("Using OpenAI (gpt-4o-mini) for structured JSON generation...")
        client = OpenAI(api_key=openai_api_key)
        model_name = "gpt-4o-mini"
    else:
        raise ValueError("Either DEEPSEEK_API_KEY or OPENAI_API_KEY must be provided.")

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Ты помогаешь планировать день. Твой ответ должен быть валидным JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content
    except Exception as e:
        logger.warning("Faced issue or JSON mode unsupported. Retrying without json_object constraint: %s", e)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Ты помогаешь планировать день. Твой ответ должен быть валидным JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        content = completion.choices[0].message.content

    # Clean code blocks if returned
    cleaned_content = content.strip()
    if cleaned_content.startswith("```"):
        cleaned_content = re.sub(r"^```(?:json)?\n", "", cleaned_content)
        cleaned_content = re.sub(r"\n```$", "", cleaned_content)
        cleaned_content = cleaned_content.strip()

    try:
        data = json.loads(cleaned_content)
        markdown_text = data.get("markdown", "")
        events = data.get("calendar_events", [])
        return markdown_text, events
    except Exception as e:
        logger.error("Failed to parse JSON response from LLM: %s. Response was: %s", e, content)
        # Fallback to plain markdown if everything else failed
        fallback_markdown = (
            "## 🚀 Выделенные задачи\n"
            f"- [ ] Задачи из голоса\n\n"
            "## 📝 Расшифровка записи\n"
            f"> {raw_text}"
        )
        return fallback_markdown, []
