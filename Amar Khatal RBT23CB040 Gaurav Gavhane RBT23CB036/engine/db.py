import csv
import sqlite3
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "..", "jarvis.db")

con = sqlite3.connect(db_path)
cursor = con.cursor()


query = "CREATE TABLE IF NOT EXISTS sys_command(id integer primary key, name VARCHAR(100), path VARCHAR(1000))"
cursor.execute(query)

# Create web_command table
query = "CREATE TABLE IF NOT EXISTS web_command(id integer primary key, name VARCHAR(100), url VARCHAR(1000))"
cursor.execute(query)

# Create contacts table
query = "CREATE TABLE IF NOT EXISTS contacts(id integer primary key, name VARCHAR(200), mobile_no VARCHAR(255), email VARCHAR(255))"
cursor.execute(query)

# Insert some default web commands
default_web_commands = [
    ('youtube', 'https://www.youtube.com/'),
    ('google', 'https://www.google.com/'),
    ('gmail', 'https://mail.google.com/'),
    ('github', 'https://github.com/'),
]

for name, url in default_web_commands:
    cursor.execute("INSERT OR IGNORE INTO web_command (name, url) VALUES (?, ?)", (name, url))

# Insert some default contacts
default_contacts = [
    ('Test Contact', '1234567890', 'test@example.com'),
]

for name, mobile, email in default_contacts:
    cursor.execute("INSERT OR IGNORE INTO contacts (name, mobile_no, email) VALUES (?, ?, ?)", (name, mobile, email))

con.commit()
con.close()