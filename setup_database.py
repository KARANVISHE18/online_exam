import sqlite3

# Connect to (or create) the database file
conn = sqlite3.connect('exam_database.db')
cursor = conn.cursor()

# --- Create Tables ---

# A table for proctoring events
cursor.execute('''
    CREATE TABLE IF NOT EXISTS proctoring_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_roll_no TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL
    )
''')

# A table for exam results (to replace results.json)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_roll_no TEXT NOT NULL,
        subject_code TEXT NOT NULL,
        score INTEGER NOT NULL,
        total INTEGER NOT NULL,
        status TEXT NOT NULL
    )
''')

# You can also create tables for students and questions here
# to move away from hardcoded data in the future.

print("Database and tables created successfully.")

# Save the changes and close the connection
conn.commit()
conn.close()