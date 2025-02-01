# modules/database.py
import sqlite3
from config import DATABASE

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            uniqueid TEXT NOT NULL UNIQUE,
            vote TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
