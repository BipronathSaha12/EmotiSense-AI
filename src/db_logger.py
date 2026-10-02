import sqlite3
import os
from datetime import datetime
import logging

class DatabaseLogger:
    def __init__(self, db_path="data/events.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()
        self.last_logged_state = None

    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            emotion TEXT,
            gesture TEXT,
            posture TEXT
        )
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to create DB table: {e}")

    def log_if_changed(self, emotion, gesture, posture):
        """Logs only if the combined state has changed to prevent DB spam."""
        current_state = (emotion, gesture, posture)
        if current_state != self.last_logged_state:
            self._insert_log(emotion, gesture, posture)
            self.last_logged_state = current_state

    def _insert_log(self, emotion, gesture, posture):
        query = "INSERT INTO activity_log (timestamp, emotion, gesture, posture) VALUES (?, ?, ?, ?)"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (datetime.now().isoformat(), emotion, gesture, posture))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to insert into DB: {e}")

    def close(self):
        self.conn.close()
