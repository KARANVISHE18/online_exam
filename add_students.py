import sqlite3
from werkzeug.security import generate_password_hash

# --- Configuration ---
DB_FILE = 'college_exam_database.db'

# --- Sample Student Data ---
# You can add, remove, or edit students in this list.
STUDENTS_TO_ADD = [
    {'email': 'rahulade237@nhitm.ac.in', 'password': 'password123', 'name': 'ADE RAHUL KUNDLIK', 'roll_no': '1'},
    {'email': 'shafeahemad237@nhitm.ac.in', 'password': 'password123', 'name': 'AHEMAD SHAFE ATIK', 'roll_no': '2'},
    {'email': 'manishambre237@nhitm.ac.in', 'password': 'password123', 'name': 'AMBRE MANISH MANGESH', 'roll_no': '3'},
    {'email': 'mandararekar237@nhitm.ac.in', 'password': 'password123', 'name': 'AREKAR MANDAR DEEPAK', 'roll_no': '4'},
    {'email': 'yugantbelokar237@nhitm.ac.in', 'password': 'password123', 'name': 'BELOKAR YUGANT RAVISHANKAR', 'roll_no': '5'},
    # --- Add more students here as needed ---
    {'email': 'teststudent@nhitm.ac.in', 'password': 'test', 'name': 'Test Student', 'roll_no': '99'}
]

def add_students():
    """
    Connects to the database and inserts the student records from the list above.
    It hashes the passwords and ensures no duplicate emails are added.
    """
    conn = None
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        print(f"Successfully connected to the database: {DB_FILE}")

        added_count = 0
        skipped_count = 0

        for student in STUDENTS_TO_ADD:
            # Check if the student's email already exists to avoid duplicates
            cursor.execute("SELECT id FROM users WHERE email = ?", (student['email'],))
            if cursor.fetchone():
                print(f"Skipping student '{student['name']}' ({student['email']}) - email already exists.")
                skipped_count += 1
                continue

            # Hash the password for secure storage
            password_hash = generate_password_hash(student['password'])

            # Insert the new student record
            cursor.execute(
                """INSERT INTO users (email, password_hash, name, roll_no, role)
                   VALUES (?, ?, ?, ?, 'student')""",
                (student['email'], password_hash, student['name'], student['roll_no'])
            )
            print(f"Successfully added student: {student['name']}")
            added_count += 1
        
        # Commit the changes to the database
        conn.commit()
        print("\n--- Process Complete ---")
        print(f"Successfully added {added_count} new students.")
        print(f"Skipped {skipped_count} students (already exist).")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        # Ensure the database connection is closed
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    # This block runs when the script is executed directly
    add_students()
