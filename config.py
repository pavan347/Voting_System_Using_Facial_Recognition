# config.py
import os

# Basic configuration
SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
DATABASE = 'users.db'