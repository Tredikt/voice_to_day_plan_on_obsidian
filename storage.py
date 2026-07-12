import abc
import datetime
import os
import requests
import json
import logging

logger = logging.getLogger("audio_to_day_plan_bot.storage")

from typing import Optional

class BaseStorage(abc.ABC):
    """Abstract class for note storage providers (Yandex Disk, Google Drive, etc.)."""
    @abc.abstractmethod
    def save_day_plan(self, filename: str, structured_plan: str, today_str: str) -> bool:
        """Saves or appends structured plan to today's note."""
        pass

    @abc.abstractmethod
    def get_day_plan(self, filename: str) -> Optional[str]:
        """Reads current day plan content if it exists."""
        pass

    @abc.abstractmethod
    def save_day_plan_raw(self, filename: str, content: str) -> bool:
        """Overwrites or saves the final raw plan content."""
        pass

class YandexWebDAVStorage(BaseStorage):
    """Syncs notes directly to Yandex Disk via WebDAV."""
    def __init__(self, username, password, obsidian_dir):
        self.username = username
        self.password = password
        self.obsidian_dir = obsidian_dir.rstrip('/')
        self.webdav_url = "https://webdav.yandex.ru"
        self.auth = (username, password)
        self._mkdir_recursive(self.obsidian_dir)

    def _mkdir_recursive(self, path: str):
        parts = [p for p in path.split("/") if p]
        current_path = ""
        for part in parts:
            current_path += f"/{part}"
            url = f"{self.webdav_url}{current_path}"
            r = requests.request("PROPFIND", url, auth=self.auth, headers={"Depth": "0"})
            if r.status_code == 404:
                requests.request("MKCOL", url, auth=self.auth)

    def save_day_plan(self, filename: str, structured_plan: str, today_str: str) -> bool:
        yandex_file_url = f"{self.webdav_url}{self.obsidian_dir}/{filename}"
        r_get = requests.get(yandex_file_url, auth=self.auth)
        
        if r_get.status_code == 200:
            existing_content = r_get.content.decode("utf-8")
            new_content = (
                f"{existing_content}\n\n"
                f"---\n### 🕒 Дополнение от {datetime.datetime.now().strftime('%H:%M')}\n\n"
                f"{structured_plan}"
            )
        else:
            new_content = f"# 📅 План на день: {today_str}\n\n{structured_plan}"

        r_put = requests.put(
            yandex_file_url,
            data=new_content.encode("utf-8"),
            auth=self.auth,
            headers={"Content-Type": "text/markdown; charset=utf-8"}
        )
        return r_put.status_code in (200, 201, 204)

    def get_day_plan(self, filename: str) -> Optional[str]:
        yandex_file_url = f"{self.webdav_url}{self.obsidian_dir}/{filename}"
        r_get = requests.get(yandex_file_url, auth=self.auth)
        if r_get.status_code == 200:
            return r_get.content.decode("utf-8")
        return None

    def save_day_plan_raw(self, filename: str, content: str) -> bool:
        yandex_file_url = f"{self.webdav_url}{self.obsidian_dir}/{filename}"
        r_put = requests.put(
            yandex_file_url,
            data=content.encode("utf-8"),
            auth=self.auth,
            headers={"Content-Type": "text/markdown; charset=utf-8"}
        )
        return r_put.status_code in (200, 201, 204)

class GoogleDriveStorage(BaseStorage):
    """Syncs notes directly to Google Drive folder using a Service Account."""
    def __init__(self, credentials_json: str, folder_id: str):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        
        if os.path.exists(credentials_json):
            self.creds = service_account.Credentials.from_service_account_file(
                credentials_json, scopes=['https://www.googleapis.com/auth/drive']
            )
        else:
            try:
                creds_dict = json.loads(credentials_json)
                self.creds = service_account.Credentials.from_service_account_info(
                    creds_dict, scopes=['https://www.googleapis.com/auth/drive']
                )
            except Exception as e:
                logger.error("Failed to load Google credentials from string: %s", e)
                raise e
                
        self.service = build('drive', 'v3', credentials=self.creds)
        self.folder_id = folder_id

    def _find_file(self, filename: str) -> str:
        query = f"name = '{filename}' and '{self.folder_id}' in parents and trashed = false"
        results = self.service.files().list(
            q=query, spaces='drive', fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return ""

    def _get_file_content(self, file_id: str) -> str:
        from googleapiclient.http import MediaIoBaseDownload
        import io
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8')

    def save_day_plan(self, filename: str, structured_plan: str, today_str: str) -> bool:
        from googleapiclient.http import MediaInMemoryUpload
        file_id = self._find_file(filename)
        
        if file_id:
            existing_content = self._get_file_content(file_id)
            new_content = (
                f"{existing_content}\n\n"
                f"---\n### 🕒 Дополнение от {datetime.datetime.now().strftime('%H:%M')}\n\n"
                f"{structured_plan}"
            )
            media = MediaInMemoryUpload(new_content.encode('utf-8'), mimetype='text/markdown', resumable=True)
            self.service.files().update(fileId=file_id, media_body=media).execute()
        else:
            new_content = f"# 📅 План на день: {today_str}\n\n{structured_plan}"
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id]
            }
            media = MediaInMemoryUpload(new_content.encode('utf-8'), mimetype='text/markdown', resumable=True)
            self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return True

    def get_day_plan(self, filename: str) -> Optional[str]:
        file_id = self._find_file(filename)
        if file_id:
            return self._get_file_content(file_id)
        return None

    def save_day_plan_raw(self, filename: str, content: str) -> bool:
        from googleapiclient.http import MediaInMemoryUpload
        file_id = self._find_file(filename)
        if file_id:
            media = MediaInMemoryUpload(content.encode('utf-8'), mimetype='text/markdown', resumable=True)
            self.service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id]
            }
            media = MediaInMemoryUpload(content.encode('utf-8'), mimetype='text/markdown', resumable=True)
            self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return True
