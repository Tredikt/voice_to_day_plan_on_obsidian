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

def classify_intent(raw_text: str, openai_api_key: str, deepseek_api_key: str) -> dict:
    """
    Classifies the user's message to determine if they want to READ a file or WRITE/UPDATE a file.
    Also extracts the target filename, parsing relative dates if necessary.
    """
    import datetime
    from zoneinfo import ZoneInfo
    import config

    tz = ZoneInfo(config.TIMEZONE)
    now_tz = datetime.datetime.now(tz)
    today_str = now_tz.strftime("%Y-%m-%d")
    day_of_week = now_tz.strftime("%A")
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
        "Ты — управляющий ИИ-ассистент для Obsidian.\n"
        "Проанализируй запрос пользователя (текстовый или голосовой) и определи:\n"
        "1. Действие (action): 'read' (если пользователь просит показать, прочитать, открыть, найти или вывести содержимое файла) "
        "или 'write' (если пользователь просит создать файл, записать, добавить задачу, изменить, удалить, перепланировать или обновить план).\n"
        "2. Имя файла (filename): имя файла с расширением '.md' (например, 'Идеи.md', 'Рецепты.md').\n\n"
        f"Текущая дата (сегодня): {today_str} ({ru_day}).\n"
        "Правила для определения имени файла:\n"
        "- Если пользователь имеет в виду план на день ('планы на завтра', 'задачи на сегодня', 'что у меня на пятницу', или просто диктует задачу без указания конкретного файла), "
        "используй дату в формате 'YYYY-MM-DD.md' (например, '2026-07-12.md' для сегодня, '2026-07-13.md' для завтра).\n"
        "- Если пользователь явно указывает другое имя файла (например, 'добавь в файл Идеи.md...', 'покажи файл Рецепты'), используй это имя с расширением '.md' (например, 'Идеи.md', 'Рецепты.md').\n"
        "- Всегда добавляй расширение '.md', если его нет.\n\n"
        "Выдай ответ СТРОГО в формате JSON со следующими ключами:\n"
        "{\n"
        "  \"action\": \"read\",\n"
        "  \"filename\": \"имя_файла.md\",\n"
        "  \"target_date\": \"YYYY-MM-DD\" (только если файл относится к конкретной дате плана на день, иначе null)\n"
        "}\n\n"
        f"Запрос пользователя:\n\"{raw_text}\""
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
                {"role": "system", "content": "Ты классифицируешь запросы к файлам Obsidian и возвращаешь JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        res_json = json.loads(completion.choices[0].message.content)
        return {
            "action": res_json.get("action", "write"),
            "filename": res_json.get("filename", f"{today_str}.md"),
            "target_date": res_json.get("target_date")
        }
    except Exception as e:
        logger.error("Failed to classify intent, falling back to write today: %s", e)
        return {
            "action": "write",
            "filename": f"{today_str}.md",
            "target_date": today_str
        }

def generate_day_plan(
    raw_text: str, 
    openai_api_key: str, 
    deepseek_api_key: str,
    existing_content: str = None,
    target_date: str = None,
    filename: str = None
) -> tuple[str, list]:
    """
    Generates or updates structured Obsidian Markdown and extracts events for Google Calendar from voice text.
    Returns:
        (markdown_text, list_of_events)
    """
    import datetime
    from zoneinfo import ZoneInfo
    import config
    
    tz = ZoneInfo(config.TIMEZONE)
    now_tz = datetime.datetime.now(tz)
    
    if not filename:
        if target_date:
            filename = f"{target_date}.md"
        else:
            filename = f"{now_tz.strftime('%Y-%m-%d')}.md"

    is_daily_plan = bool(target_date)

    prompt = "Ты — умный личный ассистент Obsidian.\n"
    if is_daily_plan:
        prompt += f"Ты создаешь или обновляешь план действий на {target_date} для файла '{filename}'."
    else:
        prompt += f"Ты создаешь или обновляешь заметку в файле '{filename}'."

    prompt += "\n\n"

    if existing_content and existing_content.strip():
        prompt += (
            f"Текущее содержимое файла '{filename}':\n"
            "```markdown\n"
            f"{existing_content}\n"
            "```\n\n"
            "Пользователь хочет ДОБАВИТЬ информацию, ИЗМЕНИТЬ, СДВИНУТЬ или ЗАМЕНИТЬ существующие разделы/пункты.\n"
            "Твоя задача — аккуратно объединить изменения с существующим файлом:\n"
            "1. Если пользователь просит заменить, удалить или изменить строку/задачу, внеси эти правки.\n"
            "2. Если пользователь просит просто добавить информацию (задачу или мысль), найди логическое место в файле или создай новый подраздел в конце:\n"
            f"### 🕒 Дополнение от {now_tz.strftime('%H:%M')}\n"
            "и опиши новые пункты/задачи там.\n"
            "3. На выходе ты должен вернуть ПОЛНЫЙ обновленный текст файла markdown, который полностью перезапишет старое содержимое.\n"
        )
    else:
        prompt += (
            f"Файла '{filename}' еще нет. Создай его содержимое с нуля.\n"
        )
        if is_daily_plan:
            prompt += (
                "Используй следующий стандартный формат для дневного плана:\n"
                "## 🚀 Выделенные задачи\n"
                "- [ ] [Задача 1]\n"
                "- [ ] [Задача 2]\n\n"
                "## 📝 Расшифровка записи\n"
                "> [Красиво отредактированный текст голосовой заметки]\n\n"
            )
        else:
            prompt += (
                "Создай красивую структуру заметок в формате Markdown, основываясь на пожеланиях пользователя.\n"
                "Обязательно добавь заголовок первого уровня `#` и красивые списки.\n"
            )

    if is_daily_plan:
        prompt += (
            "Обязательно выдели события для Google Календаря. Каждое событие должно иметь время начала.\n"
        )
    else:
        prompt += (
            "Если в запросе пользователя явно упоминаются события с точным временем и датой, которые нужно занести в календарь, выдели их. "
            "В остальных случаях верни пустой список для calendar_events.\n"
        )

    prompt += (
        "\nВыдай ответ СТРОГО в формате JSON с двумя ключами:\n"
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
