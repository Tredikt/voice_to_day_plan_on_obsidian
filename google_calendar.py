import os
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger("audio_to_day_plan_bot.google_calendar")

class GoogleCalendarManager:
    def __init__(self, credentials_json: str, calendar_id: str, timezone: str):
        self.calendar_id = calendar_id
        self.timezone = timezone
        
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        
        if not credentials_json:
            raise ValueError("Google credentials are not provided.")
        if not calendar_id:
            raise ValueError("GOOGLE_CALENDAR_ID is not provided.")
            
        if os.path.exists(credentials_json):
            self.creds = service_account.Credentials.from_service_account_file(
                credentials_json, scopes=['https://www.googleapis.com/auth/calendar']
            )
        else:
            try:
                creds_dict = json.loads(credentials_json)
                self.creds = service_account.Credentials.from_service_account_info(
                    creds_dict, scopes=['https://www.googleapis.com/auth/calendar']
                )
            except Exception as e:
                logger.error("Failed to load Google credentials for Calendar: %s", e)
                raise e
                
        self.service = build('calendar', 'v3', credentials=self.creds)

    def create_event(self, summary: str, date_str: str, start_time: str, end_time: str = None) -> dict:
        """
        Creates an event in Google Calendar.
        date_str format: "YYYY-MM-DD"
        start_time format: "HH:MM"
        end_time format: "HH:MM" (optional, defaults to 1 hour after start_time)
        """
        try:
            tz = ZoneInfo(self.timezone)
            start_dt = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            
            if end_time:
                end_dt = datetime.strptime(f"{date_str} {end_time}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            else:
                end_dt = start_dt + timedelta(hours=1)
                
            event_body = {
                'summary': summary,
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': self.timezone,
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': self.timezone,
                },
            }
            
            created_event = self.service.events().insert(
                calendarId=self.calendar_id, 
                body=event_body
            ).execute()
            
            logger.info("Created event: %s at %s", summary, start_dt.isoformat())
            return created_event
        except Exception as e:
            logger.error("Failed to create calendar event '%s': %s", summary, e, exc_info=True)
            raise e
