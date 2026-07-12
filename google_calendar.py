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

    def create_or_update_event(self, summary: str, date_str: str, start_time: str, end_time: str = None) -> tuple[dict, str]:
        """
        Creates or updates an event in Google Calendar by searching for an event with the same summary on that day.
        date_str format: "YYYY-MM-DD"
        start_time format: "HH:MM"
        end_time format: "HH:MM" (optional, defaults to 1 hour after start_time)
        Returns:
            (event_dict, action_status) -> where action_status is 'created' or 'updated'
        """
        try:
            tz = ZoneInfo(self.timezone)
            start_dt = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            
            if end_time:
                end_dt = datetime.strptime(f"{date_str} {end_time}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            else:
                end_dt = start_dt + timedelta(hours=1)
                
            # 1. Search for existing events on this specific date
            day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
            day_end = day_start + timedelta(days=1) - timedelta(microseconds=1)
            
            time_min = day_start.isoformat()
            time_max = day_end.isoformat()
            
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            
            # Find matching event by summary (case-insensitive)
            existing_event = None
            for e in events:
                if e.get('summary', '').lower().strip() == summary.lower().strip():
                    existing_event = e
                    break
                    
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
            
            if existing_event:
                # Update existing event
                event_id = existing_event['id']
                updated_event = self.service.events().update(
                    calendarId=self.calendar_id,
                    eventId=event_id,
                    body=event_body
                ).execute()
                logger.info("Updated event: '%s' at %s", summary, start_dt.isoformat())
                return updated_event, "updated"
            else:
                # Create a new event
                created_event = self.service.events().insert(
                    calendarId=self.calendar_id,
                    body=event_body
                ).execute()
                logger.info("Created event: '%s' at %s", summary, start_dt.isoformat())
                return created_event, "created"
                
        except Exception as e:
            logger.error("Failed to create or update calendar event '%s': %s", summary, e, exc_info=True)
            raise e
