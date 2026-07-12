import os
import datetime
import logging
from typing import Optional
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message

import config
import llm
import storage
import google_calendar

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("audio_to_day_plan_bot")

_bot: Optional[Bot] = None
dp = Dispatcher()
store: Optional[storage.BaseStorage] = None
cal_manager: Optional[google_calendar.GoogleCalendarManager] = None

# Initialize storage based on provider
try:
    if config.STORAGE_PROVIDER == "yandex":
        logger.info("Initializing Yandex Disk WebDAV storage...")
        store = storage.YandexWebDAVStorage(
            username=config.YANDEX_USER,
            password=config.YANDEX_PASSWORD,
            obsidian_dir=config.YANDEX_OBSIDIAN_DIR
        )
    elif config.STORAGE_PROVIDER == "google":
        logger.info("Initializing Google Drive storage...")
        store = storage.GoogleDriveStorage(
            credentials_json=config.GOOGLE_CREDENTIALS_JSON,
            folder_id=config.GOOGLE_OBSIDIAN_DIR_ID
        )
    else:
        logger.error("Unknown storage provider: %s", config.STORAGE_PROVIDER)
except Exception as e:
    logger.error("Failed to initialize storage provider: %s", e)

# Initialize Google Calendar if configured
if config.GOOGLE_CALENDAR_ID and config.GOOGLE_CREDENTIALS_JSON:
    try:
        logger.info("Initializing Google Calendar manager...")
        cal_manager = google_calendar.GoogleCalendarManager(
            credentials_json=config.GOOGLE_CREDENTIALS_JSON,
            calendar_id=config.GOOGLE_CALENDAR_ID,
            timezone=config.TIMEZONE
        )
    except Exception as e:
        logger.error("Failed to initialize Google Calendar manager: %s", e)


def _make_bot_session(proxy_url: str):
    from aiogram.client.session.aiohttp import AiohttpSession
    return AiohttpSession(proxy=proxy_url)


@dp.message(F.voice)
async def handle_voice(message: Message):
    user_id = message.from_user.id
    if config.ALLOWED_TELEGRAM_IDS and user_id not in config.ALLOWED_TELEGRAM_IDS:
        logger.warning("Access denied for user ID: %d", user_id)
        return

    status_msg = await message.answer("🎙 Получил голосовое сообщение. Скачиваю аудио...")
    voice = message.voice
    file_id = voice.file_id
    file = await _bot.get_file(file_id)
    
    local_ogg_path = f"{file_id}.ogg"
    
    try:
        # 1. Download file from Telegram
        await _bot.download_file(file.file_path, local_ogg_path)
        await status_msg.edit_text("⚡ Распознаю речь...")

        # 2. Transcribe using Whisper or SpeechRecognition fallback
        raw_text = llm.transcribe_audio(local_ogg_path, config.OPENAI_API_KEY)
        
        if not raw_text.strip():
            await status_msg.edit_text("🤷‍♂️ Речь не распознана.")
            return

        await status_msg.edit_text("🧠 Определяю дату и структуру списка задач...")

        # 3. Parse target date
        target_date = llm.parse_target_date(
            raw_text=raw_text,
            openai_api_key=config.OPENAI_API_KEY,
            deepseek_api_key=config.DEEPSEEK_API_KEY
        )
        filename = f"{target_date}.md"

        # 4. Check if file exists in our storage
        if not store:
            await status_msg.edit_text("❌ Хранилище не инициализировано. Проверьте настройки в .env")
            return

        existing_content = store.get_day_plan(filename)
        if existing_content:
            logger.info("Existing day plan found for %s. Passing to LLM for merging/editing...", target_date)
        else:
            logger.info("No existing day plan found for %s. Will generate a new one.", target_date)

        # 5. Generate or update structured Markdown plan and extract calendar events
        structured_plan, calendar_events = llm.generate_day_plan(
            raw_text=raw_text,
            openai_api_key=config.OPENAI_API_KEY,
            deepseek_api_key=config.DEEPSEEK_API_KEY,
            existing_content=existing_content,
            target_date=target_date
        )

        await status_msg.edit_text("💾 Сохраняю в облако...")
        success = store.save_day_plan_raw(filename, structured_plan)

        if success:
            # 6. Handle Google Calendar events if any
            added_events_info = []
            if calendar_events and cal_manager:
                try:
                    await status_msg.edit_text("📅 Синхронизирую события в Google Календаре...")
                except Exception:
                    pass
                for ev in calendar_events:
                    summary = ev.get("summary")
                    start_time = ev.get("start_time")
                    end_time = ev.get("end_time")
                    if summary and start_time:
                        try:
                            event_data, status = cal_manager.create_or_update_event(
                                summary, target_date, start_time, end_time
                            )
                            status_ru = "обновлено" if status == "updated" else "создано"
                            added_events_info.append(f"• *{summary}* в {start_time} ({status_ru})")
                        except Exception as e:
                            logger.error("Failed to add/update event %s to calendar: %s", summary, e)
            
            calendar_suffix = ""
            if added_events_info:
                calendar_suffix = "\n\n📅 **События в Google Календаре:**\n" + "\n".join(added_events_info)

            response_text = (
                f"✅ *Успешно сохранено в Obsidian!*\n"
                f"📂 Файл: `{filename}`"
                f"{calendar_suffix}\n\n"
                f"{structured_plan}"
            )
            try:
                await message.reply(response_text, parse_mode="Markdown")
                try:
                    await status_msg.delete()
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Failed to send reply in Markdown, falling back to plain text: %s", e)
                plain_response = (
                    f"✅ Успешно сохранено в Obsidian!\n"
                    f"📂 Файл: {filename}"
                    f"{calendar_suffix.replace('**', '').replace('*', '')}\n\n"
                    f"{structured_plan}"
                )
                await message.reply(plain_response)
                try:
                    await status_msg.delete()
                except Exception:
                    pass
        else:
            await status_msg.edit_text("❌ Не удалось сохранить файл в облако.")

    except Exception as e:
        logger.error("Error processing voice message: %s", e, exc_info=True)
        try:
            await status_msg.edit_text(f"❌ Произошла ошибка: {str(e)}")
        except Exception:
            await message.reply(f"❌ Произошла ошибка: {str(e)}")
        
    finally:
        if os.path.exists(local_ogg_path):
            os.remove(local_ogg_path)


async def start_bot() -> Optional[Bot]:
    """Initialize aiogram bot and start polling. Returns Bot instance."""
    global _bot
    if not config.TELEGRAM_BOT_TOKEN:
        return None

    if config.TELEGRAM_PROXY:
        try:
            session = _make_bot_session(config.TELEGRAM_PROXY)
            _bot = Bot(token=config.TELEGRAM_BOT_TOKEN, session=session)
            addr = config.TELEGRAM_PROXY.split("@")[-1]
            logger.info("Bot using proxy: %s", addr)
        except Exception as e:
            logger.error("Proxy session creation failed: %s — bot will NOT start, fix proxy config", e)
            return None
    else:
        _bot = Bot(token=config.TELEGRAM_BOT_TOKEN)

    try:
        await _bot.delete_webhook(drop_pending_updates=False, request_timeout=5)
        await _bot.delete_my_commands(request_timeout=5)
    except Exception as e:
        logger.warning("Bot init requests failed (%s) — polling will start anyway", e)

    import asyncio
    asyncio.create_task(dp.start_polling(_bot, allowed_updates=["message", "callback_query"]))
    logger.info("Aiogram bot polling started")
    return _bot


if __name__ == "__main__":
    import asyncio
    
    async def main_loop():
        bot_instance = await start_bot()
        if not bot_instance:
            logger.error("Failed to start bot. Check config.")
            return
        while True:
            await asyncio.sleep(3600)
            
    try:
        asyncio.run(main_loop())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
