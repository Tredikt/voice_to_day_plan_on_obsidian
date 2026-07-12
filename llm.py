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

def parse_target_date(raw_text: str, openai_api_key: str, deepseek_api_key: str) -> str:
    """
    Analyzes the transcribed text to find the target date the user is referring to.
    Today's date is dynamically provided in the prompt.
    Returns:
        target_date: str in 'YYYY-MM-DD' format.
    """
    import datetime
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now()
    day_of_week = now.strftime("%A")
    ru_days = {
        "Monday": "Понедельник",
        "Tuesday": "Вторник",
        "Wednesday": "Среда",
        "Thursday": "Четверг",
        "Friday": "Пятница",
        "Saturday": "Суббота",
        "Sunday": "Воскресенье"
    }
    ru_day = ru_days.get(day_of_week, day_of_week)

    prompt = (
        "Ты — точный ИИ-ассистент для определения дат.\n"
        "Проанализируй текст голосовой заметки и определи целевую дату, о которой идет речь.\n\n"
        f"Текущая дата (сегодня): {today_str} ({ru_day}).\n\n"
        "Правила:\n"
        "1. Если в тексте явно или косвенно упоминается 'на завтра' / 'завтра', верни дату завтрашнего дня.\n"
        "2. Если упоминается конкретный день недели (например, 'в понедельник', 'на пятницу'), найди ближайшую будущую дату для этого дня недели.\n"
        "3. Если упоминается конкретное число (например, 'на 15 число', 'на 20 июля'), верни соответствующую дату текущего или ближайшего будущего месяца/года.\n"
        "4. Если в тексте нет указаний на конкретную дату или явно упоминается 'сегодня' / 'на сегодня', верни текущую дату (сегодня).\n"
        "5. Ответ должен быть СТРОГО в формате JSON с одним ключом \"target_date\" (значение в формате \"YYYY-MM-DD\").\n\n"
        f"Текст голосовой заметки:\n\"{raw_text}\""
    )

    if deepseek_api_key:
        client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        model_name = "deepseek-chat"
    elif openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        model_name = "gpt-4o-mini"
    else:
        raise ValueError("Either DEEPSEEK_API_KEY or OPENAI_API_KEY must be provided.")

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Ты парсишь даты и возвращаешь JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        res_json = json.loads(completion.choices[0].message.content)
        return res_json.get("target_date", today_str)
    except Exception as e:
        logger.error("Failed to parse target date, falling back to today (%s): %s", today_str, e)
        return today_str

def generate_day_plan(
    raw_text: str, 
    openai_api_key: str, 
    deepseek_api_key: str,
    existing_content: str = None,
    target_date: str = None
) -> tuple[str, list]:
    """
    Generates or updates structured Obsidian Markdown and extracts events for Google Calendar from voice text.
    Returns:
        (markdown_text, list_of_events)
    """
    import datetime
    
    if not target_date:
        target_date = datetime.date.today().strftime("%Y-%m-%d")

    prompt = (
        "Ты — умный личный ассистент. Проанализируй транскрипцию голосовой заметки "
        f"и составь/обнови план действий на {target_date} для Obsidian, а также выдели события с конкретным временем для Google Календаря.\n\n"
    )

    if existing_content and existing_content.strip():
        prompt += (
            f"Внимание! У пользователя уже есть существующий план на эту дату ({target_date}):\n"
            "```markdown\n"
            f"{existing_content}\n"
            "```\n\n"
            "Пользователь хочет ДОБАВИТЬ новые задачи, ИЗМЕНИТЬ, СДВИНУТЬ или ЗАМЕНИТЬ существующие пункты на основе нового голосового сообщения.\n"
            "Твоя задача — аккуратно объединить или обновить этот план:\n"
            "1. Если пользователь говорит заменить, удалить или сдвинуть задачу/время, внеси эти изменения в существующий текст.\n"
            "2. Если пользователь говорит добавить задачу, добавь её в список 'Выделенные задачи' (постарайся сохранить красивую структуру). "
            "Если новые задачи не привязаны ко времени или это просто дополнения, ты можешь добавить их в список, либо создать в конце файла аккуратный подраздел:\n"
            f"### 🕒 Дополнение от {datetime.datetime.now().strftime('%H:%M')}\n"
            "и описать изменения/дополнительный текст там.\n"
            "3. Убери из названий задач глаголы действия вроде 'сделать', 'поставить', 'надо сходить' — пиши лаконично (например: '- [ ] Тренировка в 18:00', '- [ ] Созвон по дизайну').\n"
            "4. В поле 'markdown' верни ИСПРАВЛЕННЫЙ/ОБНОВЛЕННЫЙ ПОЛНЫЙ текст markdown, который полностью перезапишет старый файл.\n\n"
        )
    else:
        prompt += (
            "Для этой даты еще НЕТ существующего плана. Создай новый чистый план на русском языке с нуля.\n"
            "Формат для Obsidian:\n"
            "## 🚀 Выделенные задачи\n"
            "- [ ] [Задача 1]\n"
            "- [ ] [Задача 2]\n\n"
            "## 📝 Расшифровка записи\n"
            "> [Красиво отредактированный текст голосовой заметки]\n\n"
            "Правила:\n"
            "1. Пиши лаконично и строго по делу, без вступительных или заключительных фраз.\n"
            "2. В 'Выделенные задачи' убирай слова вроде 'Сделать', 'Поставить', пиши кратко (например: '- [ ] Тренировка в 18:00', '- [ ] Созвон по дизайну').\n\n"
        )

    prompt += (
        "Выдай ответ СТРОГО в формате JSON с двумя ключами:\n"
        "1. \"markdown\": Полный текст плана для Obsidian (в формате markdown).\n"
        "2. \"calendar_events\": Список событий для Google Календаря на эту дату. Каждое событие должно содержать:\n"
        "   - \"summary\": Очищенное, короткое существительное название события на русском языке с большой буквы. "
        "Убирай глаголы вроде 'сделать', 'поставить', 'сходить' (например: вместо 'Сделать тренировку' пиши 'Тренировка', вместо 'Пойти на созвон' пиши 'Созвон').\n"
        "   - \"start_time\": Время начала в формате \"HH:MM\" (например: \"18:00\"). Если время не упомянуто, не добавляй событие.\n"
        "   - \"end_time\": (Необязательно) Время окончания в формате \"HH:MM\". Если не упомянуто — оставь null или пустым.\n\n"
        "Важно:\n"
        "- Ответ должен содержать только чистый JSON без разметки markdown вроде ```json.\n\n"
        f"Вот новое голосовое сообщение пользователя:\n\"{raw_text}\""
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
