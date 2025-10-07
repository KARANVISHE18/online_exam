import sqlite3
conn = sqlite3.connect('exam_database.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, student_roll_no TEXT, subject_code TEXT,
        score INTEGER, total INTEGER, status TEXT
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS proctoring_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, student_roll_no TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT, message TEXT
    )
''')
print("Database and tables created successfully.")
conn.commit()
conn.close()