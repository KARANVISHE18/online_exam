import eventlet
eventlet.monkey_patch() # Necessary for Flask-SocketIO with eventlet

from datetime import timedelta, datetime
import sqlite3
import json
import os
import random
import cv2
import numpy as np
import base64
import pandas as pd
from io import BytesIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import (Flask, render_template, request, jsonify, session,
                   redirect, url_for, flash, make_response)
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps # For admin_required decorator

# --- AI Model Setup (YOLO) ---
yolo_path = "yolo_model"
net = None
classes = []
output_layers = []
ai_model_loaded = False # Flag to track successful loading

try:
    weights_path = os.path.join(yolo_path, "yolov3-tiny.weights")
    cfg_path = os.path.join(yolo_path, "yolov3-tiny.cfg")
    names_path = os.path.join(yolo_path, "coco.names")

    if os.path.exists(weights_path) and os.path.exists(cfg_path) and os.path.exists(names_path):
        net = cv2.dnn.readNet(weights_path, cfg_path)
        layer_names = net.getLayerNames()
        
        # Corrected handling for different OpenCV versions
        unconnected_layers = net.getUnconnectedOutLayers()
        if isinstance(unconnected_layers, np.ndarray) and unconnected_layers.ndim > 1:
             output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
        elif isinstance(unconnected_layers, (np.ndarray, list, tuple)):
             output_layers = [layer_names[i - 1] for i in unconnected_layers]
        else:
             raise TypeError(f"Unexpected output format from getUnconnectedOutLayers: {type(unconnected_layers)}")


        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        ai_model_loaded = True # <-- CORRECTLY SET FLAG
        print("AI Model loaded successfully.")
    else:
        print(f"AI Model files not found in '{yolo_path}'. AI proctoring will be disabled.")
except Exception as e:
    print(f"Error loading AI Model: {e}. AI proctoring will be disabled.")
    net = None
# --- End of AI Model Setup ---

app = Flask(__name__)
app.secret_key = 'kEshU-RAnDoM-sTr1nG-f0R-s3ss10n-s3cur1tY' # CHANGE THIS!
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
socketio = SocketIO(app, async_mode='eventlet')
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"], 
    storage_uri="memory://" 
)
# --- Constants and Configuration ---
DB_FILE = 'college_exam_database.db'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Database Initialization ---
# --- Database Initialization ---

# ADD THIS NEW HELPER FUNCTION
def add_column_if_not_exists(cursor, table_name, column_name, column_type):
    """Checks if a column exists and adds it if it doesn't."""
    try:
        # Check if column exists by selecting it
        cursor.execute(f"SELECT {column_name} FROM {table_name} LIMIT 1")
    except sqlite3.OperationalError as e:
        # Error means column doesn't exist
        if f"no such column: {column_name}" in str(e):
            print(f"Adding column '{column_name}' to table '{table_name}'...")
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            print(f"Column '{column_name}' added.")
        else:
            # Raise other errors
            raise e

def init_db():
    # ... your init_db function starts here ...
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Users table
    # Inside init_db() function in app.py

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE,
            branch_code TEXT,
            year INTEGER,
            role TEXT NOT NULL DEFAULT 'student' CHECK(role IN ('student', 'admin')),
            is_active INTEGER NOT NULL DEFAULT 1 -- ADD THIS LINE (1=Active, 0=Inactive)
        )
    
    ''')
    # Subjects table
   # Inside init_db() function in app.py

    # Subjects table
    # Inside init_db() function in app.py

    # Users table
   # Inside init_db() in app.py

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE,
            branch_code TEXT,
            year INTEGER,
            role TEXT NOT NULL DEFAULT 'student' CHECK(role IN ('student', 'admin')),
            is_active INTEGER NOT NULL DEFAULT 1, 
            secret_question TEXT,       
            secret_answer_hash TEXT,    
            face_descriptor TEXT       -- 
            -- You can add SQL comments using '--' like this if needed
        )
    
    ''')

    # Check/Create admin user
    cursor.execute("SELECT id FROM users WHERE email = 'admin@nhitm.ac.in'")
    if cursor.fetchone() is None:
        admin_pass_hash = generate_password_hash('admin')
        default_question = "What is your favorite subject?"
        default_answer_hash = generate_password_hash('DefaultAnswer')
        # Insert admin WITHOUT face descriptor initially
        cursor.execute("""
            INSERT INTO users
            (email, password_hash, name, role, is_active, secret_question, secret_answer_hash)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            ('admin@nhitm.ac.in', admin_pass_hash, 'Admin', 'admin',
             default_question, default_answer_hash)
        )
        print("Default admin user created. Face registration required.")
        # ... (rest of init_db) ...
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT UNIQUE NOT NULL,
            branch_code TEXT NOT NULL,
            year INTEGER NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1 -- ADD THIS LINE (1=Active, 0=Inactive)
        )
    ''')
    # Make sure you DO NOT have the branches table creation code here
    # Make sure you ARE using the global BRANCH_NAMES dictionary
    
    # Questions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
            option1 TEXT NOT NULL, option2 TEXT NOT NULL, option3 TEXT NOT NULL, option4 TEXT NOT NULL,
            correct_answer TEXT NOT NULL, subject_id INTEGER NOT NULL,
            FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE
        )
    ''')
    # Results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_roll_no TEXT NOT NULL, subject_code TEXT NOT NULL,
            score INTEGER NOT NULL, total INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('Completed', 'Terminated')),
            reason TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_roll_no) REFERENCES users (roll_no) ON DELETE CASCADE
        )
    ''')
    # Proctoring logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS proctoring_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_roll_no TEXT NOT NULL, event_type TEXT NOT NULL,
            message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_roll_no) REFERENCES users (roll_no) ON DELETE CASCADE
        )
    ''')

    # Check if admin exists, if not create one
    cursor.execute("SELECT id FROM users WHERE email = 'admin@nhitm.ac.in'")
    if cursor.fetchone() is None:
        admin_pass_hash = generate_password_hash('admin')
        cursor.execute("INSERT INTO users (email, password_hash, name, role) VALUES (?, ?, ?, ?)",
                       ('admin@nhitm.ac.in', admin_pass_hash, 'Admin', 'admin'))
        print("Default admin user created (email: admin@nhitm.ac.in, password: admin). PLEASE CHANGE THE PASSWORD.")
    conn.commit()
    conn.close()
    print("Database initialized.")

# --- Data Structures & Helpers ---
ACTIVE_EXAMS = {} # {roll_no_str: {info}}
connected_students = {} # {sid: roll_no_str}

BRANCH_NAMES = { "comps": "Computer", "csd": "CS & Design", "aids": "AI/DS", "mech": "Mechanical", "civil": "Civil" }

def log_proctoring_event(roll_no, event_type, message):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO proctoring_logs (student_roll_no, event_type, message) VALUES (?, ?, ?)",
                       (str(roll_no), event_type, message))
        conn.commit()
    except sqlite3.Error as e: print(f"DB Log Error ({roll_no}): {e}")
    finally:
        if conn: conn.close()

# --- Routes ---
# ADD THIS in app.py (within Admin Routes)

# Simple page to show registration button (could be part of a profile page)
@app.route('/admin/profile')

def admin_profile():
    # Check if face is already registered for the logged-in admin
    face_registered = False
    admin_id = None # Need admin ID if using session['admin'] = True directly isn't enough

    # We need a way to get the current admin's ID. Let's modify login slightly
    # Option A: Store admin_id in session during login (RECOMMENDED)
    # Option B: Query DB based on admin_name (less reliable if names aren't unique)

    # Assuming Option A (modify handle_admin_complete_verification or handle_admin_login):
    # session['admin_id'] = admin_id 

    admin_id_from_session = session.get('admin_id') # Get ID stored during login

    if admin_id_from_session:
         conn = sqlite3.connect(DB_FILE)
         cursor = conn.cursor()
         cursor.execute("SELECT face_descriptor FROM users WHERE id = ?", (admin_id_from_session,))
         result = cursor.fetchone()
         conn.close()
         if result and result[0]:
             face_registered = True

    return render_template('admin_profile.html', face_registered=face_registered)
@app.route('/admin/register_face', methods=['POST'])
def handle_admin_register_face():
    # --- THIS IS THE FIX ---
    # Allow registration if admin is fully logged in OR pending verification
    admin_id = session.get('admin_id') or session.get('admin_pending_face_verification')

    if not admin_id:
        flash('Admin session invalid.', 'error')
        return jsonify({'success': False, 'error': 'Invalid session'}), 400
    # --- END OF FIX ---

    data = request.json
    descriptor_str = data.get('descriptor')

    if not descriptor_str:
        return jsonify({'success': False, 'error': 'No descriptor received'}), 400

    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET face_descriptor = ? WHERE id = ?", (descriptor_str, admin_id))
        conn.commit()
        flash('Face registered successfully!', 'success')
        return jsonify({'success': True})
    except sqlite3.Error as e:
        if conn: conn.rollback()
        flash(f'Database error saving face data: {e}', 'error')
        return jsonify({'success': False, 'error': f'Database error: {e}'}), 500
    finally:
        if conn: conn.close()
# Decorator for admin-only routes
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin' not in session:
            flash('Admin access required.', 'error')
            return redirect(url_for('login_page')) # Redirect to main login
        return f(*args, **kwargs)
    return decorated_function
@app.route('/admin/verify_face', methods=['GET'])
@limiter.limit("10 per minute")
def admin_verify_face():
    admin_id = session.get('admin_pending_face_verification')
    if not admin_id:
        flash('Please log in with password first.', 'error')
        return redirect(url_for('admin_login_page'))

    admin_name = session.get('admin_temp_name', 'Admin')

    # --- FETCH STORED DESCRIPTOR ---
    stored_descriptor_str = None
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT face_descriptor FROM users WHERE id = ?", (admin_id,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        stored_descriptor_str = result[0]
    else:
        # Handle case where admin has no face registered - maybe redirect to profile?
        flash('Admin face not registered. Please register your face first.', 'error')
         # You might need to temporarily log them in just to access profile,
         # or have a separate non-admin-required registration flow.
         # For now, redirecting to login.
        return redirect(url_for('admin_login_page'))
    # --- END FETCH ---

    return render_template('admin_verify_face.html',
                           admin_name=admin_name,
                           # Pass descriptor string to template
                           stored_descriptor=stored_descriptor_str)
# --- General Routes ---
@app.route('/')
def login_page():
    # --- THIS IS THE FIX ---
    # Check for 'admin' FIRST.
    if 'admin' in session: 
        return redirect(url_for('admin_dashboard'))
    # Check for 'email' (student) SECOND.
    if 'email' in session: 
        return redirect(url_for('student_dashboard'))
    # --- END OF FIX ---
    
    return render_template('login.html', error=None) # Show login only if no session
# PASTE THIS CORRECTED FUNCTION INTO APP.PY, REPLACING THE CURRENT ONE

@app.route('/login', methods=['POST'])
def handle_login():
    email = request.form.get('username')
    password = request.form.get('password')
    if not email or not password:
        flash('Email/password required.', 'error'); return render_template('login.html', error='Email/password required.')
    
    conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,)); user = cursor.fetchone(); conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        
        session.clear() 
        session.permanent = True
        
        if user['role'] == 'student':
            if 'is_active' in user.keys() and user['is_active'] == 0:
                flash('Your account is inactive. Please contact the administrator.', 'error')
                return render_template('login.html', error='Account inactive.')

            if user['branch_code'] and user['year']:
                session['email'] = user['email']
                session['student_name'] = user['name']
                session['roll_no'] = user['roll_no']
                session['branch_code'] = user['branch_code']
                session['year'] = user['year']
                return redirect(url_for('student_dashboard'))
            else: 
                flash('Profile incomplete. Cannot log in.', 'error'); 
                return render_template('login.html', error='Profile incomplete.')
                
        elif user['role'] == 'admin':
            
            # --- THIS IS THE FIX ---
            # Password is correct, but DO NOT log them in yet.
            # Start the 2-step verification flow.
            
            # Set TEMPORARY keys to remember who is verifying
            session['admin_pending_face_verification'] = user['id']
            session['admin_temp_name'] = user['name'] # For the welcome message

            # Redirect to the face verification page, NOT the dashboard
            return redirect(url_for('admin_verify_face'))
            # --- END OF FIX ---
            
        else: 
            flash('Unknown user role.', 'error'); 
            return render_template('login.html', error='Unknown role.')
    else: 
        flash('Invalid credentials.', 'error'); 
        return render_template('login.html', error='Invalid Credentials')


@app.route('/logout')
def logout():
    # session.clear() removes ALL keys from the session (admin and student)
    # This fixes your bug.
    session.clear() 
    flash("You have been logged out.", "success")
    return redirect(url_for('login_page'))
# --- Student Routes ---
@app.route('/student_dashboard')
def student_dashboard():
    if 'email' not in session or 'roll_no' not in session: return redirect(url_for('login_page'))
    branch_code, year = session.get('branch_code'), session.get('year')
    branch_name = BRANCH_NAMES.get(branch_code, "N/A") # Use global dict
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # --- MODIFIED QUERY ---
    # Select only subjects that match branch/year AND are active
    cursor.execute("SELECT name, code FROM subjects WHERE branch_code = ? AND year = ? AND is_active = 1", 
                   (branch_code, year))
    # --- END MODIFICATION ---
    subjects_db = cursor.fetchall()
    conn.close()
    available_subjects = {subject['code']: {'name': subject['name']} for subject in subjects_db}
    return render_template('student_dashboard.html',
                           student_name=session.get('student_name'), branch_name=branch_name,
                           branch_code=branch_code, year=year, subjects=available_subjects)

@app.route('/student/results')
def student_results():
    if 'email' not in session or 'roll_no' not in session: return redirect(url_for('login_page'))
    student_roll_no = session['roll_no']
    conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT r.*, s.name as subject_name FROM results r JOIN subjects s ON r.subject_code = s.code WHERE r.student_roll_no = ? ORDER BY r.timestamp DESC", (student_roll_no,))
    results_list = cursor.fetchall(); conn.close()
    return render_template('student_results.html', results=results_list, student_name=session.get('student_name')) # Needs template

# --- Exam Flow Routes ---
@app.route('/exam_notice/<subject_code>') # <-- Renamed to match template
def exam_notice(subject_code): # <-- Renamed to match template
    if 'email' not in session: return redirect(url_for('login_page'))
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    cursor.execute("SELECT name FROM subjects WHERE code = ?", (subject_code,)); subject_row = cursor.fetchone(); conn.close()
    if not subject_row: return "Subject not found", 404
    return render_template('exam_notice.html', subject_code=subject_code, subject_name=subject_row[0]) # Needs template

@app.route('/exam/<subject_code>')
def exam_page(subject_code):
    if 'email' not in session or 'roll_no' not in session: return redirect(url_for('login_page'))
    student_roll_no = session['roll_no']
    roll_no_str = str(student_roll_no)
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    cursor.execute("SELECT id FROM results WHERE student_roll_no = ? AND subject_code = ?", (roll_no_str, subject_code))
    if cursor.fetchone(): conn.close(); flash("Already submitted.", "info"); return render_template('exam_ended.html', message="Already submitted.") # Needs template
    cursor.execute("SELECT name FROM subjects WHERE code = ?", (subject_code,)); subject_row = cursor.fetchone()
    if not subject_row: conn.close(); flash("Subject not found.", "error"); return redirect(url_for('student_dashboard'))
    subject_name = subject_row[0]; conn.close()
    ACTIVE_EXAMS[roll_no_str] = {'name': session['student_name'], 'subject': subject_code, 'startTime': datetime.now().timestamp()}
    print(f"SERVER: Added {roll_no_str} to ACTIVE_EXAMS. Current list: {list(ACTIVE_EXAMS.keys())}")
    return render_template('exam.html', subject=subject_code, subject_name=subject_name, roll_no=roll_no_str) # Use non-enhanced

@app.route('/submission-success')
def submission_success():
    if 'email' not in session: return redirect(url_for('login_page'))
    return render_template('submission_success.html', student_name=session.get('student_name')) # Needs template

@app.route('/exam-terminated')
def exam_terminated():
    if 'email' not in session: return redirect(url_for('login_page'))
    return render_template('exam_terminated.html', student_name=session.get('student_name')) # Needs template

# --- Admin Routes ---
@app.route('/admin')
def admin_login_page():
    if 'admin' in session: return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html') # Needs template


@app.route('/admin/login', methods=['POST'])
# @limiter.limit("5 per minute") # Keep rate limiting if Flask-Limiter is installed
def handle_admin_login():
     email = request.form.get('username')
     password = request.form.get('password')

     if not email or not password:
         flash('Email and password required.', 'error')
         # Make sure '/admin' route exists or change to 'login_page'
         # Assuming you have an admin specific login page route:
         return redirect(url_for('admin_login_page')) 

     conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
     # Fetch id along with other details
     # Ensure your users table actually HAS is_active column
     try:
         # Fetch necessary columns including id and is_active
         cursor.execute("SELECT id, password_hash, name, is_active FROM users WHERE email = ? AND role = 'admin'", (email,));
     except sqlite3.OperationalError as e:
         # Handle case where is_active column might be missing after DB changes
         if "no such column: is_active" in str(e):
             # Fallback query without is_active
             cursor.execute("SELECT id, password_hash, name FROM users WHERE email = ? AND role = 'admin'", (email,));
         else:
             flash(f"Database error: {e}", "error") # Flash DB error to user
             conn.close() # Close connection before returning
             return redirect(url_for('admin_login_page')) # Redirect on error
     admin_user = cursor.fetchone();
     conn.close() # Close connection after fetch

     if admin_user and check_password_hash(admin_user['password_hash'], password):
        # Check if account is active (handle missing column gracefully)
        # Use .get() for safer access in case column was missing in fallback query
        if admin_user.get('is_active', 1) == 0: # Default to active (1) if column missing
             flash('Admin account is inactive.', 'error')
             return redirect(url_for('admin_login_page')) # Redirect if inactive

        # --- FIXES APPLIED ---
        # Clear any old session first
        session.clear()
        # Set the FINAL admin session keys, including admin_id
        session['admin'] = True
        session['admin_name'] = admin_user['name']
        session['admin_id'] = admin_user['id'] # <-- STORE THE ID HERE
        session.permanent = True
        # --- END FIXES ---

        # Redirect to dashboard after successful login
        # NOTE: If you add face/secret verification later, this redirect
        # would change to go to the verification step instead.
        flash('Admin login successful.', 'success') # Optional success message
        return redirect(url_for('admin_dashboard'))

     # If login fails (invalid credentials or user not found)
     flash('Invalid admin credentials.', 'error')
     # Make sure '/admin' route exists or change to 'login_page'
     return redirect(url_for('admin_login_page'))



@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')
# ADD THIS NEW FUNCTION ANYWHERE IN app.py (near other student admin routes)

@app.route('/admin/students/toggle_status/<int:student_id>', methods=['POST'])
@admin_required
def admin_toggle_student_status(student_id):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Get current status
        cursor.execute("SELECT is_active FROM users WHERE id = ? AND role = 'student'", (student_id,))
        result = cursor.fetchone()
        if not result:
            flash('Student not found.', 'error')
            return redirect(url_for('admin_manage_students'))

        current_status = result[0]
        new_status = 0 if current_status == 1 else 1 # Toggle

        cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_status, student_id))
        conn.commit()
        status_text = "activated" if new_status == 1 else "deactivated"
        flash(f'Student account successfully {status_text}.', 'success')

    except sqlite3.Error as e:
        if conn: conn.rollback()
        flash(f'Database error updating student status: {e}', 'error')
    except Exception as e:
        flash(f'An unexpected error occurred: {e}', 'error')
    finally:
        if conn: conn.close()

    return redirect(url_for('admin_manage_students'))
@app.route('/admin/students')
@admin_required
def admin_manage_students():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # --- MODIFIED QUERY ---
    cursor.execute("SELECT id, name, roll_no, email, branch_code, year, is_active FROM users WHERE role = 'student' ORDER BY name")
    # --- END MODIFICATION ---
    students = cursor.fetchall()
    conn.close()
    return render_template('admin_students.html', students=students, branches=BRANCH_NAMES)
# PASTE THIS ENTIRE FUNCTION INTO APP.PY, REPLACING THE OLD PLACEHOLDER

@app.route('/admin/students/add', methods=['POST'])
@admin_required
def admin_add_student():
    # 1. Get data from the form submitted in admin_students.html
    name = request.form.get('name')
    email = request.form.get('email')
    roll_no = request.form.get('roll_no')
    password = request.form.get('password')
    branch_code = request.form.get('branch_code')
    year = request.form.get('year')

    # 2. Basic Validation: Check if all required fields are present
    if not all([name, email, roll_no, password, branch_code, year]):
        flash('All fields are required to add a student.', 'error')
        return redirect(url_for('admin_manage_students'))

    # 3. Hash the password for security
    password_hash = generate_password_hash(password)

    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 4. Attempt to insert the new student into the database
        cursor.execute("""
            INSERT INTO users (name, email, roll_no, password_hash, branch_code, year, role)
            VALUES (?, ?, ?, ?, ?, ?, 'student')
        """, (name, email, roll_no, password_hash, branch_code, year))

        conn.commit() # Save the changes to the database
        flash(f'Student {name} added successfully!', 'success')

    except sqlite3.IntegrityError as e:
        # 5. Handle UNIQUE constraint errors (email or roll_no already exists)
        conn.rollback() # Important: undo the failed insert
        if 'UNIQUE constraint failed: users.email' in str(e):
            flash(f'Error: Email "{email}" is already registered.', 'error')
        elif 'UNIQUE constraint failed: users.roll_no' in str(e):
            flash(f'Error: Roll number "{roll_no}" is already registered.', 'error')
        else:
            flash(f'Database integrity error: {e}', 'error') # Other unique constraint issues

    except sqlite3.Error as e:
        # 6. Handle other potential database errors
        if conn: conn.rollback()
        flash(f'Database error occurred: {e}', 'error')

    except Exception as e:
        # 7. Handle any other unexpected errors
        if conn: conn.rollback()
        flash(f'An unexpected error occurred: {e}', 'error')

    finally:
        # 8. Always close the database connection
        if conn:
            conn.close()

    # 9. Redirect back to the student management page
    return redirect(url_for('admin_manage_students'))
# From your app.py, around line 294
# PASTE THIS ENTIRE FUNCTION INTO APP.PY, REPLACING THE OLD ONE

@app.route('/admin/students/edit/<int:student_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_student(student_id):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if request.method == 'POST':
            # --- Handle the form submission (Update logic) ---
            name = request.form.get('name')
            email = request.form.get('email')
            roll_no = request.form.get('roll_no')
            branch_code = request.form.get('branch_code')
            year = request.form.get('year')
            password = request.form.get('password') # Optional: only update if provided

            if not all([name, email, roll_no, branch_code, year]):
                flash('All fields except password are required.', 'error')
                # Need to fetch student again to re-render the form with error
                cursor.execute("SELECT * FROM users WHERE id = ? AND role = 'student'", (student_id,))
                student = cursor.fetchone()
                if not student:
                    flash('Student not found.', 'error')
                    return redirect(url_for('admin_manage_students'))
                return render_template('admin_edit_student.html', student=student, branches=BRANCH_NAMES)

            update_query = """
                UPDATE users SET name=?, email=?, roll_no=?, branch_code=?, year=?
                WHERE id = ?
            """
            params = [name, email, roll_no, branch_code, year, student_id]

            # Only update password if a new one was entered
            if password:
                password_hash = generate_password_hash(password)
                update_query = """
                    UPDATE users SET name=?, email=?, roll_no=?, branch_code=?, year=?, password_hash=?
                    WHERE id = ?
                """
                params = [name, email, roll_no, branch_code, year, password_hash, student_id]

            cursor.execute(update_query, params)
            conn.commit()
            flash(f'Student {name} updated successfully.', 'success')
            return redirect(url_for('admin_manage_students'))

        else:
            # --- Handle the initial page load (GET request) ---
            cursor.execute("SELECT * FROM users WHERE id = ? AND role = 'student'", (student_id,))
            student = cursor.fetchone()
            if not student:
                flash('Student not found.', 'error')
                return redirect(url_for('admin_manage_students'))
            
            # *** THIS IS THE CORRECTED RENDER_TEMPLATE CALL ***
            # It passes the student data and branches dictionary as keyword arguments
            return render_template('admin_edit_student.html', student=student, branches=BRANCH_NAMES)

    except sqlite3.Error as e:
        flash(f'Database error: {e}', 'error')
        if conn: conn.rollback() # Roll back changes on error
        return redirect(url_for('admin_manage_students'))
    except Exception as e:
        flash(f'An unexpected error occurred: {e}', 'error')
        return redirect(url_for('admin_manage_students'))
    finally:
        if conn: conn.close()

@app.route('/admin/students/delete/<int:student_id>', methods=['POST'])
@admin_required
def admin_delete_student(student_id):
    # ... (delete student logic) ...
    return redirect(url_for('admin_manage_students'))

@app.route('/admin/subjects')
@admin_required
def admin_manage_subjects():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # --- MODIFIED QUERY ---
    cursor.execute("SELECT id, name, code, branch_code, year, is_active FROM subjects ORDER BY name") # Added is_active
    # --- END MODIFICATION ---
    subjects = cursor.fetchall()
    conn.close()
    # Pass the global BRANCH_NAMES dictionary
    return render_template('admin_subjects.html', subjects=subjects, branches=BRANCH_NAMES)
# ADD THIS NEW FUNCTION IN app.py (within Admin Routes)
# Example inside handle_admin_complete_verification() in app.py

@app.route('/admin/complete_verification', methods=['POST'])
@limiter.limit("5 per minute")
def handle_admin_complete_verification():
    # Check for the temporary session key
    temp_admin_id = session.get('admin_pending_face_verification')
    if not temp_admin_id:
        flash('Admin session not found. Please log in again.', 'error')
        # --- FIX 1: Return a JSON error ---
        return jsonify({'success': False, 'error': 'Invalid session', 'redirect_url': url_for('admin_login_page')}), 400

    conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM users WHERE id = ? AND role = 'admin'", (temp_admin_id,)); 
    admin_user = cursor.fetchone(); 
    conn.close()
    
    if not admin_user:
        flash('Admin user not found in database.', 'error')
        session.clear() # Clear temporary session data
        # --- FIX 2: Return a JSON error ---
        return jsonify({'success': False, 'error': 'User not found', 'redirect_url': url_for('admin_login_page')}), 404
        
    # --- FIX 3: Clear ALL temporary keys ---
    session.pop('admin_pending_face_verification', None)
    session.pop('admin_temp_name', None) 

    # Set the FINAL admin session keys
    session['admin'] = True
    session['admin_name'] = admin_user['name']
    session['admin_id'] = admin_user['id']
    session.permanent = True 
        
    print(f"Admin {session['admin_id']} fully verified. Redirecting to dashboard.")
    return jsonify({'success': True, 'redirect_url': url_for('admin_dashboard')})
@app.route('/admin/subjects/toggle_status/<int:subject_id>', methods=['POST'])
@admin_required
def admin_toggle_subject_status(subject_id):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT is_active FROM subjects WHERE id = ?", (subject_id,))
        result = cursor.fetchone()
        if not result:
            flash('Subject not found.', 'error')
            return redirect(url_for('admin_manage_subjects'))

        current_status = result[0]
        new_status = 0 if current_status == 1 else 1 # Toggle

        cursor.execute("UPDATE subjects SET is_active = ? WHERE id = ?", (new_status, subject_id))
        conn.commit()
        status_text = "activated" if new_status == 1 else "deactivated"
        flash(f'Subject status successfully {status_text}.', 'success')

    except sqlite3.Error as e:
        if conn: conn.rollback()
        flash(f'Database error: {e}', 'error')
    finally:
        if conn: conn.close()

    return redirect(url_for('admin_manage_subjects'))
@app.route('/admin/subjects/add', methods=['POST'])
@admin_required
def admin_add_subject():
    # 1. Get data from the form
    name = request.form.get('name')
    code = request.form.get('code')
    branch_code = request.form.get('branch_code')
    year = request.form.get('year')

    # 2. Basic Validation
    if not all([name, code, branch_code, year]):
        flash('All fields are required to add a subject.', 'error')
        return redirect(url_for('admin_manage_subjects'))

    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 3. Insert the new subject (is_active defaults to 1 in DB schema)
        cursor.execute("""
            INSERT INTO subjects (name, code, branch_code, year)
            VALUES (?, ?, ?, ?)
        """, (name, code, branch_code, year))

        conn.commit()
        flash(f'Subject "{name}" added successfully!', 'success')

    except sqlite3.IntegrityError:
        # 4. Handle UNIQUE constraint error (subject code already exists)
        conn.rollback()
        flash(f'Error: Subject code "{code}" already exists.', 'error')

    except sqlite3.Error as e:
        # 5. Handle other database errors
        if conn: conn.rollback()
        flash(f'Database error occurred: {e}', 'error')

    except Exception as e:
        # 6. Handle unexpected errors
        if conn: conn.rollback()
        flash(f'An unexpected error occurred: {e}', 'error')

    finally:
        # 7. Close connection
        if conn:
            conn.close()

    # 8. Redirect back to the subjects list
    return redirect(url_for('admin_manage_subjects'))
@app.route('/admin/questions/<int:subject_id>')
@admin_required
def admin_manage_questions(subject_id):
    conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)); subject = cursor.fetchone()
    cursor.execute("SELECT * FROM questions WHERE subject_id = ?", (subject_id,)); questions = cursor.fetchall(); conn.close()
    if not subject: flash("Subject Not Found", "error"); return redirect(url_for('admin_manage_subjects'))
    return render_template('admin_questions.html', subject=subject, questions=questions) # Needs template

@app.route('/admin/questions/add/<int:subject_id>', methods=['POST'])
@admin_required
def admin_add_question(subject_id):
    # ... (add question logic) ...
    return redirect(url_for('admin_manage_questions', subject_id=subject_id))

@app.route('/admin/questions/upload/<int:subject_id>', methods=['POST'])
@admin_required
def admin_upload_questions(subject_id):
    # ... (upload questions logic) ...
    return redirect(url_for('admin_manage_questions', subject_id=subject_id))

@app.route('/admin/results', methods=['GET', 'POST'])
@admin_required
def admin_results():
    selected_branch = request.form.get('branch_code', ''); selected_year = request.form.get('year', '')
    results = []
    query = """
        SELECT r.id as result_id, r.student_roll_no, u.name as student_name, 
               s.name as subject_name, r.score, r.total, r.status, r.timestamp
        FROM results r
        JOIN users u ON r.student_roll_no = u.roll_no
        JOIN subjects s ON r.subject_code = s.code
    """
    params = []; conditions = []
    if selected_branch: conditions.append("u.branch_code = ?"); params.append(selected_branch)
    if selected_year: conditions.append("u.year = ?"); params.append(selected_year)
    if conditions: query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY r.timestamp DESC"
    conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    cursor.execute(query, params); results = cursor.fetchall(); conn.close()
    return render_template('admin_results.html', results=results, branches=BRANCH_NAMES, 
                           selected_branch=selected_branch, selected_year=selected_year)

# In app.py

@app.route('/admin/reexam/<int:result_id>', methods=['POST'])
@admin_required # Make sure this decorator is here if you're using it
def admin_reexam(result_id):
    if not session.get('admin'):
        # Redirect to your single, unified login page
        return redirect(url_for('login_page')) 
    
    conn = None # Initialize connection variable
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # This line executes the deletion
        cursor.execute("DELETE FROM results WHERE id = ?", (result_id,))
        
        # *** THIS IS THE MOST IMPORTANT LINE ***
        # This saves the change to the database file.
        conn.commit() 
        # *** END IMPORTANT LINE ***

        if cursor.rowcount > 0:
            # If 1 or more rows were deleted, show success
            flash("Result deleted. The student can now re-take the exam.", "success")
        else:
            # If 0 rows were deleted (ID not found)
            flash(f"Warning: Result with ID {result_id} was not found.", "error") # Use 'error' for red

    except Exception as e:
        # If any other error happens (like DB locked)
        if conn:
            conn.rollback() # Roll back any changes if an error occurred
        flash(f"Error deleting result: {e}", "error")
    finally:
        if conn:
            conn.close()
    
    return redirect(url_for('admin_results'))

# --- API Routes ---
@app.route('/api/questions/<subject_code>')
def get_questions(subject_code):
    if 'roll_no' not in session: print("API Error: User not auth"); return jsonify({"error": "Not authenticated"}), 401
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute("SELECT id FROM subjects WHERE code = ?", (subject_code,)); subject_row = cursor.fetchone()
        if not subject_row: print(f"API Error: Subject {subject_code} not found"); conn.close(); return jsonify({"error": "Subject not found"}), 404
        subject_id = subject_row['id']
        cursor.execute("SELECT id, text, option1, option2, option3, option4 FROM questions WHERE subject_id = ?", (subject_id,)); questions_from_db = cursor.fetchall()
        print(f"API Info: Fetched {len(questions_from_db)} questions for subject {subject_id}")
    except sqlite3.Error as e: print(f"API DB Error: {e}"); return jsonify({"error": "DB error"}), 500
    finally:
        if conn: conn.close()
    if not questions_from_db: print(f"API Error: No questions for {subject_id}"); return jsonify({"error": "No questions found"}), 404
    question_list = [{"id": q['id'], "question": q['text'], "options": [q['option1'], q['option2'], q['option3'], q['option4']]} for q in questions_from_db]
    random.shuffle(question_list); print(f"API Info: Sending {len(question_list)} questions.")
    return jsonify(question_list)

@app.route('/submit', methods=['POST'])
def submit_exam():
    if 'roll_no' not in session: return jsonify({'error': 'Not authenticated'}), 401
    data = request.json; subject_code = data.get('subject'); answers = data.get('answers', {}); reason = data.get('reason', 'Completed normally')
    student_roll_no = str(session['roll_no'])
    if not subject_code or not isinstance(answers, dict): return jsonify({'error': 'Invalid data'}), 400
    conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM subjects WHERE code = ?", (subject_code,)); subject_row = cursor.fetchone()
        if not subject_row: raise ValueError("Subject not found")
        subject_id = subject_row['id']
        cursor.execute("SELECT id, correct_answer FROM questions WHERE subject_id = ?", (subject_id,)); correct_answers_db = {str(q['id']): q['correct_answer'] for q in cursor.fetchall()}
        if not correct_answers_db: raise ValueError("No questions found")
        score = sum(1 for q_id, ans in answers.items() if correct_answers_db.get(q_id) == ans)
        total_questions = len(correct_answers_db); status = "Completed" if "completed" in reason.lower() or "time ran out" in reason.lower() else "Terminated"
        cursor.execute("SELECT id FROM results WHERE student_roll_no = ? AND subject_code = ?", (student_roll_no, subject_code))
        if cursor.fetchone(): print(f"SERVER: Double submit by {student_roll_no}"); redirect_url = url_for('exam_terminated') if status == "Terminated" else url_for('submission_success'); return jsonify({'status': 'already_submitted', 'redirect_url': redirect_url})
        cursor.execute("INSERT INTO results (student_roll_no, subject_code, score, total, status, reason) VALUES (?, ?, ?, ?, ?, ?)", (student_roll_no, subject_code, score, total_questions, status, reason))
        conn.commit(); print(f"SERVER: Result saved for {student_roll_no}. Score: {score}/{total_questions}. Status: {status}")
        roll_no_str = str(student_roll_no)
        if roll_no_str in ACTIVE_EXAMS:
            del ACTIVE_EXAMS[roll_no_str]
            socketio.emit('student_left_exam', {'roll_no': roll_no_str, 'reason': status}, room='admin_room')
            print(f"SERVER: Removed {roll_no_str} from ACTIVE_EXAMS & emitted student_left_exam.")
        redirect_url = url_for('exam_terminated') if status == "Terminated" else url_for('submission_success')
        return jsonify({'status': 'success', 'redirect_url': redirect_url})
    except ValueError as e: print(f"Submit Error: {e}"); conn.rollback(); return jsonify({'error': str(e)}), 400
    except sqlite3.Error as e: print(f"Submit DB Error: {e}"); conn.rollback(); return jsonify({'error': 'DB error'}), 500
    finally: conn.close()

# --- SocketIO Event Handlers (Corrected - Single Block) ---
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"SERVER: Client connected: {sid}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    roll_no_str = connected_students.pop(sid, None)
    if roll_no_str:
        print(f"SERVER: Student {roll_no_str} disconnected via socket: {sid}")
        if roll_no_str in ACTIVE_EXAMS:
            print(f"SERVER: Student {roll_no_str} was active, notifying admin of disconnect.")
            log_proctoring_event(roll_no_str, 'SOCKET_DISCONNECT', 'Student connection lost.')
            emit('student_left_exam', {'roll_no': roll_no_str, 'reason': 'Disconnected'}, room='admin_room')
    else:
        print(f"SERVER: Admin or unknown client disconnected: {sid}")

@socketio.on('admin_join')
def handle_admin_join():
    sid = request.sid
    if not session.get('admin'): print(f"SERVER: Unauthorized admin_join attempt by {sid}"); return
    join_room('admin_room')
    print(f"SERVER: Admin {sid} joined 'admin_room'. Sending active list: {list(ACTIVE_EXAMS.keys())}")
    emit('active_students_list', ACTIVE_EXAMS, room=sid)

@socketio.on('student_join')
def handle_student_join(data):
    sid = request.sid
    roll_no = data.get('roll_no')
    roll_no_str = str(roll_no) if roll_no else None
    print(f"SERVER: Received student_join from SID {sid} for roll {roll_no_str}")
    if roll_no_str and roll_no_str in ACTIVE_EXAMS:
         connected_students[sid] = roll_no_str
         join_room(roll_no_str)
         print(f"SERVER: Student {roll_no_str} confirmed active. Emitting student_started_exam.")
         emit('student_started_exam', {'roll_no': roll_no_str, 'info': ACTIVE_EXAMS.get(roll_no_str)}, room='admin_room')
    else:
         print(f"SERVER: Student join rejected. Roll: {roll_no_str}, Active: {list(ACTIVE_EXAMS.keys())}")

@socketio.on('student_leave')
def handle_student_leave(data):
     sid = request.sid
     roll_no = data.get('roll_no')
     roll_no_str = str(roll_no) if roll_no else None
     print(f"SERVER: Received student_leave from SID {sid} for roll {roll_no_str}")
     if sid in connected_students and connected_students[sid] == roll_no_str: del connected_students[sid]

@socketio.on('video_frame_from_student')
def handle_video_frame(data):
    sid = request.sid; roll_no = data.get('roll_no'); frame_b64 = data.get('frame')
    roll_no_str = str(roll_no) if roll_no else None
    if not roll_no_str or not frame_b64 or connected_students.get(sid) != roll_no_str: return
    emit('video_frame', {'roll_no': roll_no_str, 'frame': frame_b64}, room='admin_room')
    if not ai_model_loaded: return
    try:
        img_data = base64.b64decode(frame_b64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob); outs = net.forward(output_layers)
        class_ids = []; confidences = []
        for out in outs:
            for detection in out:
                scores = detection[5:]; class_id = np.argmax(scores); confidence = scores[class_id]
                if confidence > 0.5: class_ids.append(class_id); confidences.append(float(confidence))
        detected_objects = [str(classes[cid]) for cid in class_ids] if classes else []
        person_count = detected_objects.count('person')
        
        if person_count > 1:
            alert_message = f'Multiple People ({person_count})'; log_proctoring_event(roll_no_str, 'AI_ALERT_PERSON', alert_message); emit('proctoring_alert', {'roll_no': roll_no_str, 'alert': alert_message}, room='admin_room')
            
            ## ADD THIS LINE to automatically warn the student ##
            emit('receive_warning', {'message': 'Multiple people detected. Please ensure you are alone.'}, room=roll_no_str)

        if 'cell phone' in detected_objects:
            alert_message = 'Cell Phone Detected'; log_proctoring_event(roll_no_str, 'AI_ALERT_PHONE', alert_message); emit('proctoring_alert', {'roll_no': roll_no_str, 'alert': alert_message}, room='admin_room')
            
            ## ADD THIS LINE to automatically warn the student ##
            emit('receive_warning', {'message': 'Mobile phone detected. Please put it away immediately.'}, room=roll_no_str)
            
    except Exception as e: print(f"AI Error ({roll_no_str}): {e}")

@socketio.on('proctoring_violation')
def handle_proctoring_violation(data):
    sid = request.sid; roll_no = data.get('roll_no'); violation_type = data.get('type'); message = data.get('message')
    roll_no_str = str(roll_no) if roll_no else None
    if roll_no_str and violation_type and connected_students.get(sid) == roll_no_str:
        print(f"VIOLATION: {roll_no_str} - {violation_type}"); log_proctoring_event(roll_no_str, violation_type, message)
        emit('proctoring_alert', {'roll_no': roll_no_str, 'alert': f"{violation_type}: {message}"}, room='admin_room')
    else: print(f"SERVER: Invalid violation data/SID mismatch: {data}")

@socketio.on('audio_alert')
def handle_audio_alert(data):
    sid = request.sid; roll_no = data.get('roll_no')
    roll_no_str = str(roll_no) if roll_no else None
    if roll_no_str and connected_students.get(sid) == roll_no_str:
        message = "Suspicious noise."; print(f"AUDIO ALERT: {roll_no_str}"); log_proctoring_event(roll_no_str, 'AUDIO_ALERT', message)
        emit('proctoring_alert', {'roll_no': roll_no_str, 'alert': message}, room='admin_room')
        
        ## ADD THIS LINE to automatically warn the student ##
        emit('receive_warning', {'message': 'Suspicious noise detected. Please remain silent.'}, room=roll_no_str)
        
    else: print(f"SERVER: Invalid audio alert data/SID mismatch: {data}")

@socketio.on('send_warning')
def handle_send_warning(data):
    if not session.get('admin'): return
    student_roll_no = data.get('student_roll_no'); message = data.get('message')
    roll_no_str = str(student_roll_no) if student_roll_no else None
    if roll_no_str and message:
        print(f"Admin sending warning to {roll_no_str}: {message}")
        emit('receive_warning', {'message': message}, room=roll_no_str); log_proctoring_event(roll_no_str, 'ADMIN_WARNING', message)

@socketio.on('terminate_exam')
def handle_terminate_exam(data):
    if not session.get('admin'): return
    student_roll_no = data.get('student_roll_no')
    roll_no_str = str(student_roll_no) if student_roll_no else None
    reason = "Terminated by administrator."
    if roll_no_str:
        print(f"Admin terminating exam for {roll_no_str}")
        emit('exam_terminated', {'reason': reason}, room=roll_no_str); log_proctoring_event(roll_no_str, 'ADMIN_TERMINATE', reason)


# --- Main Execution ---
if __name__ == '__main__':
    init_db()
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
    # Use eventlet as the production server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)