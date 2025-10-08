import eventlet
eventlet.monkey_patch()

import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import pandas as pd
from io import BytesIO
import os
import random
import cv2
import numpy as np
import base64
from datetime import datetime

# --- AI Model Setup (YOLO) ---
yolo_path = "yolo_model"
# Ensure the model files exist before trying to load them
if os.path.exists(os.path.join(yolo_path, "yolov3-tiny.weights")) and os.path.exists(os.path.join(yolo_path, "yolov3-tiny.cfg")):
    net = cv2.dnn.readNet(os.path.join(yolo_path, "yolov3-tiny.weights"), os.path.join(yolo_path, "yolov3-tiny.cfg"))
    layer_names = net.getLayerNames()
    # Correction for older OpenCV versions
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except TypeError:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    with open(os.path.join(yolo_path, "coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("AI Model loaded successfully.")
else:
    print("AI Model files not found. AI proctoring will be disabled.")
    net = None # Disable AI features if files are missing
# --- End of AI Model Setup ---

app = Flask(__name__)
# IMPORTANT: For production, load this from an environment variable, not hardcoded.
app.secret_key = 'your_super_secret_key_nhitm'
socketio = SocketIO(app, async_mode='eventlet')

# --- Constants and Configuration ---
DB_FILE = 'exam_database.db'
STATUS_FILE = 'exam_status.json'
ADMIN_CREDENTIALS = {'username': 'admin', 'password': 'admin'}  # TODO: Use hashed passwords in production

SUBJECT_MAP = {
    "tcs": "Theoretical Computer Science",
    "ip": "Internet Programming",
    "dwm": "Data Warehouse & Mining",
    "cn": "Computer Networks",
    "se": "Software Engineering"
}

# --- Database Initialization ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Table for results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_roll_no TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(student_roll_no, subject_code)
        )
    ''')
    # Table for proctoring logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS proctoring_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_roll_no TEXT NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

# --- Data Loading ---
def load_json_file(filename, default_data):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_data
    return default_data

# In-memory state loaded from file
EXAM_STATUS = load_json_file(STATUS_FILE, {"tcs": "inactive", "ip": "inactive", "dwm": "inactive", "cn": "inactive", "se": "inactive"})
ACTIVE_EXAMS = {} # Live state, doesn't need to be persisted across restarts

# Dummy student DB (replace with DB later)
STUDENTS = {'rahulade237@nhitm.ac.in': {'roll_no': '1', 'name': 'ADE RAHUL KUNDLIK', 'password': '12317036'},
    'shafeahemad237@nhitm.ac.in': {'roll_no': '2', 'name': 'AHEMAD SHAFE ATIK', 'password': '12347003'},
    'manishambre237@nhitm.ac.in': {'roll_no': '3', 'name': 'AMBRE MANISH MANGESH', 'password': '12347023'},
    'mandararekar237@nhitm.ac.in': {'roll_no': '4', 'name': 'AREKAR MANDAR DEEPAK', 'password': '12317050'},
    'yugantbelokar237@nhitm.ac.in': {'roll_no': '5', 'name': 'BELOKAR YUGANT RAVISHANKAR', 'password': '12247047'},
    'sohambhere227@nhitm.ac.in': {'roll_no': '6', 'name': 'BHERE SOHAM SAMBHAJI SAVITA', 'password': '12217012'},
    'dishabirari237@nhitm.ac.in': {'roll_no': '7', 'name': 'BIRARI DISHA', 'password': '12327014'},
    'aryanbirwatkar227@nhitm.ac.in': {'roll_no': '8', 'name': 'BIRWATKAR ARYAN KRISHNA KIRTI', 'password': '12217006'},
    'kshitijcharthal237@nhitm.ac.in': {'roll_no': '9', 'name': 'CHARTHAL KSHITIJ', 'password': '12327019'},
    'adityachaudhari237@nhitm.ac.in': {'roll_no': '10', 'name': 'CHAUDHARI ADITYA ARJUN', 'password': '12327006'},
    'kishnachaudhary237@nhitm.ac.in': {'roll_no': '11', 'name': 'CHAUDHARY KISHNA THANARAM', 'password': '12317001'},
    'harshchavan237@nhitm.ac.in': {'roll_no': '12', 'name': 'CHAVAN HARSH NARENDRA', 'password': '12327009'},
    'racheetdevlekar237@nhitm.ac.in': {'roll_no': '13', 'name': 'DEVLEKAR RACHEET', 'password': '12327017'},
    'athashreedoiphode237@nhitm.ac.in': {'roll_no': '14', 'name': 'DOIPHODE ATHASHREE MANGESH', 'password': '12317027'},
    'aryangaikwad237@nhitm.ac.in': {'roll_no': '15', 'name': 'GAIKWAD ARYAN', 'password': '12317039'},
    'jaygawde237@nhitm.ac.in': {'roll_no': '16', 'name': 'GAWDE JAY DINESH', 'password': '12247037'},
    'raunakgound237@nhitm.ac.in': {'roll_no': '17', 'name': 'GOUND RAUNAK', 'password': '12327016'},
    'kishorirnak237@nhitm.ac.in': {'roll_no': '18', 'name': 'IRNAK KISHOR GORAKH', 'password': '12317019'},
    'vaishnavijadhav237@nhitm.ac.in': {'roll_no': '19', 'name': 'JADHAV VAISHNAVI PRAVIN', 'password': '12317012'},
    'chaitalijadhav237@nhitm.ac.in': {'roll_no': '20', 'name': 'JADHAV CHAITALI', 'password': '12327012'},
    'pranamyajoshi237@nhitm.ac.in': {'roll_no': '21', 'name': 'JOSHI PRANAMYA VISHVAS', 'password': '12317001'},
    'adityakakad237@nhitm.ac.in': {'roll_no': '22', 'name': 'KAKAD ADITYA MOHAN', 'password': '12317028'},
    'abhaykakade237@nhitm.ac.in': {'roll_no': '23', 'name': 'KAKADE ABHAY NAVNATH', 'password': '12317029'},
    'janhavikakade237@nhitm.ac.in': {'roll_no': '24', 'name': 'KAKADE JANHAVI PRAKASH', 'password': '12327003'},
    'vedkarandikar237@nhitm.ac.in': {'roll_no': '25', 'name': 'KARANDIKAR VED AJIT', 'password': '12317018'},
    'daivikkaul237@nhitm.ac.in': {'roll_no': '26', 'name': 'KAUL DAIVIK', 'password': '12317008'},
    'yashkhandibharad237@nhitm.ac.in': {'roll_no': '27', 'name': 'KHANDIBHARAD YASH', 'password': '12327020'},
    'rajkulawade237@nhitm.ac.in': {'roll_no': '28', 'name': 'KULAWADE RAJ ASHOK', 'password': '12317032'},
    'gayatrikulkarni237@nhitm.ac.in': {'roll_no': '29', 'name': 'KULKARNI GAYATRI MILIND', 'password': '12347019'},
    'vishalkute237@nhitm.ac.in': {'roll_no': '30', 'name': 'KUTE VISHAL DADABHAU', 'password': '12317013'},
    'namratamahajan237@nhitm.ac.in': {'roll_no': '31', 'name': 'MAHAJAN NAMRATA SANTOSH', 'password': '12317035'},
    'yashmandavkar237@nhitm.ac.in': {'roll_no': '32', 'name': 'MANDAVKAR YASH GANESH', 'password': '12317048'},
    'harshamogaveera237@nhitm.ac.in': {'roll_no': '33', 'name': 'MOGAVEERA HARSHA LAXMAN', 'password': '12347030'},
    'siddhimore237@nhitm.ac.in': {'roll_no': '34', 'name': 'MORE SIDDHI ASHOK', 'password': '12317007'},
    'saurabhnimase237@nhitm.ac.in': {'roll_no': '35', 'name': 'NIMASE SAURABH HARISHCHANDRA', 'password': '12347004'},
    'hitanshupanchal237@nhitm.ac.in': {'roll_no': '36', 'name': 'PANCHAL HITANSHU BHARAT', 'password': '12317016'},
    'ravirajpandit227@nhitm.ac.in': {'roll_no': '37', 'name': 'PANDIT RAVIRAJ SUNIL', 'password': '12217038'},
    'tusharparab237@nhitm.ac.in': {'roll_no': '38', 'name': 'PARAB TUSHAR NHANU', 'password': '12327002'},
    'chinmaypatil237@nhitm.ac.in': {'roll_no': '39', 'name': 'PATIL CHINMAY PRASHANT', 'password': '12347022'},
    'manishpatil237@nhitm.ac.in': {'roll_no': '40', 'name': 'PATIL MANISH', 'password': '12327013'},
    'bhaktipatil227@nhitm.ac.in': {'roll_no': '41', 'name': 'PATIL BHAKTI ARVIND ASHWINI', 'password': '12217011'},
    'daminipatil237@nhitm.ac.in': {'roll_no': '42', 'name': 'PATIL DAMINI', 'password': '12327011'},
    'rohanpatil227@nhitm.ac.in': {'roll_no': '43', 'name': 'PATIL ROHAN PRADEEP', 'password': '12247042'},
    'sanketpatil227@nhitm.ac.in': {'roll_no': '44', 'name': 'PATIL SANKET INDRAJIT SUNANDA', 'password': '12217024'},
    'parthpawar237@nhitm.ac.in': {'roll_no': '45', 'name': 'PAWAR PARTH SUDAM', 'password': '12317017'},
    'rahulpawar237@nhitm.ac.in': {'roll_no': '46', 'name': 'PAWAR RAHUL SUNIL', 'password': '12317014'},
    'omkarphadnis237@nhitm.ac.in': {'roll_no': '47', 'name': 'PHADNIS OMKAR SANTOSH', 'password': '12317011'},
    'sohamprabhudesai237@nhitm.ac.in': {'roll_no': '48', 'name': 'PRABUDESAI SOHAM', 'password': '12317045'},
    'virajpukale237@nhitm.ac.in': {'roll_no': '49', 'name': 'PUKALE VIRAJ RAMCHANDRA', 'password': '12317004'},
    'shreyapurandare237@nhitm.ac.in': {'roll_no': '50', 'name': 'PURANDARE SHREYA', 'password': '12327010'},
    'roshnipurohit237@nhitm.ac.in': {'roll_no': '51', 'name': 'PUROHIT ROSHNI PRAKASH', 'password': '12317024'},
    'rohitraut237@nhitm.ac.in': {'roll_no': '52', 'name': 'RAUT ROHIT', 'password': '12327018'},
    'priyankasalunke237@nhitm.ac.in': {'roll_no': '53', 'name': 'SALUNKE PRIYANKA KIRAN', 'password': '12317010'},
    'shloksalunkhe237@nhitm.ac.in': {'roll_no': '54', 'name': 'SALUNKHE SHLOK', 'password': '12347025'},
    'nidhisapkale237@nhitm.ac.in': {'roll_no': '55', 'name': 'SAPKALE NIDHI NARENDRA', 'password': '12317037'},
    'rohitsaptale237@nhitm.ac.in': {'roll_no': '56', 'name': 'SAPTALE ROHIT GANPAT', 'password': '12317026'},
    'shubhamsawant237@nhitm.ac.in': {'roll_no': '57', 'name': 'SAWANT SHUBHAM SHARAD', 'password': '12247049'},
    'swayamsawant237@nhitm.ac.in': {'roll_no': '58', 'name': 'SAWANT SWAYAM DEEPAK', 'password': '12327004'},
    'prasadsurase237@nhitm.ac.in': {'roll_no': '59', 'name': 'SURASE PRASAD CHANDU', 'password': '12317031'},
    'aryansuroshe237@nhitm.ac.in': {'roll_no': '60', 'name': 'SUROSHE ARYAN PRADIP', 'password': '12327008'},
    'shivamanitalakokkula237@nhitm.ac.in': {'roll_no': '61', 'name': 'TALAKOKKULA SHIVAMANI ASHOK', 'password': '12317043'},
    'sejaltambe237@nhitm.ac.in': {'roll_no': '62', 'name': 'TAMBE SEJAL SANTOSH', 'password': '12327005'},
    'prabhavthakur237@nhitm.ac.in': {'roll_no': '63', 'name': 'THAKUR PRABHAV PRAVIN', 'password': '12317042'},
    'atharvavandre227@nhitm.ac.in': {'roll_no': '64', 'name': 'VANDRE ATHARVA PRAFUL', 'password': '12247039'},
    'karanvishe237@nhitm.ac.in': {'roll_no': '65', 'name': 'VISHE KARAN', 'password': '12327015'},
    'aryanyadav237@nhitm.ac.in': {'roll_no': '66', 'name': 'YADAV ARYAN ASHOK KUMAR', 'password': '12317002'},
    'lokendaryadav237@nhitm.ac.in': {'roll_no': '67', 'name': 'YADAV LOKENDAR BANWARILAL', 'password': '12317022'},
    'sanskrutiyadav237@nhitm.ac.in': {'roll_no': '68', 'name': 'YADAV SANSKRUTI BHALCHANDRA', 'password': '12327007'},
}

# Question bank
QUESTIONS = {
    'tcs': [
        {'id': 1, 'text': 'Which of the following is not a finite automaton?', 'options': ['DFA', 'NFA', 'Turing Machine', 'Moore Machine'], 'answer': 'Turing Machine'},
        {'id': 2, 'text': 'A language is regular if and only if it is accepted by a...', 'options': ['Finite Automaton', 'Pushdown Automaton', 'Turing Machine', 'Context-Free Grammar'], 'answer': 'Finite Automaton'},
        {'id': 3, 'text': 'The Pumping Lemma is used to prove that a language is...', 'options': ['Regular', 'Not Regular', 'Context-Free', 'Decidable'], 'answer': 'Not Regular'}
        # ... (rest of questions omitted for brevity)
    ],
    'ip': [
        {'id': 1, 'text': 'What does HTML stand for?', 'options': ['Hyper Text Markup Language', 'High Tech Modern Language', 'Hyperlink and Text Markup Language', 'Home Tool Markup Language'], 'answer': 'Hyper Text Markup Language'},
        {'id': 2, 'text': 'Which tag is used to create an ordered list?', 'options': ['<ul>', '<ol>', '<li>', '<dl>'], 'answer': '<ol>'},
        {'id': 3, 'text': 'What is the correct CSS syntax for changing the font color of an element?', 'options': ['font-color: red;', 'color: red;', 'text-color: red;', 'fgcolor: red;'], 'answer': 'color: red;'}
        # ... (rest of questions omitted for brevity)
    ],
    'dwm': [
        {'id': 1, 'text': 'What is Data Mining?', 'options': ['The process of extracting useful information from data', 'The process of entering data into a database', 'The process of creating a database', 'The process of securing data'], 'answer': 'The process of extracting useful information from data'},
        {'id': 2, 'text': 'Which of these is a common Data Mining task?', 'options': ['Classification', 'Clustering', 'Regression', 'All of the above'], 'answer': 'All of the above'}
        # ... (rest of questions omitted for brevity)
    ],
    'cn': [
        {'id': 1, 'text': 'What is the full form of LAN?', 'options': ['Local Area Network', 'Large Area Network', 'Live Area Network', 'Logical Area Network'], 'answer': 'Local Area Network'},
        {'id': 2, 'text': 'Which layer in the OSI model is responsible for framing?', 'options': ['Physical Layer', 'Data Link Layer', 'Network Layer', 'Transport Layer'], 'answer': 'Data Link Layer'}
        # ... (rest of questions omitted for brevity)
    ],
    'se': [
        {'id': 1, 'text': 'What is the first phase of the SDLC?', 'options': ['Design', 'Coding', 'Requirement Gathering and Analysis', 'Testing'], 'answer': 'Requirement Gathering and Analysis'},
        {'id': 2, 'text': 'Which of the following is a software development model?', 'options': ['Waterfall Model', 'Agile Model', 'Spiral Model', 'All of the above'], 'answer': 'All of the above'}
        # ... (rest of questions omitted for brevity)
    ]
}

# Track live connections (maps sid → roll_no)
connected_students = {}

# --- Helper Functions ---
def log_proctoring_event(roll_no, event_type, message):
    """Helper function to save a proctoring event to the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO proctoring_logs (student_roll_no, event_type, message) VALUES (?, ?, ?)",
            (str(roll_no), event_type, message)
        )
        conn.commit()
        conn.close()
        print(f"Logged Event for {roll_no}: {message}")
    except Exception as e:
        print(f"Database logging error: {e}")

# --- Standard Flask Routes ---
@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    email = request.form['username']
    password = request.form['password']
    student = STUDENTS.get(email)
    if student and student['password'] == password:  # TODO: use check_password_hash()
        session['email'] = email
        session['student_name'] = student['name']
        session['roll_no'] = student['roll_no']
        return redirect(url_for('dashboard'))
    return render_template('login.html', error='Invalid Email or Password')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html', student_name=session['student_name'])

@app.route('/exam_notice/<subject>')
def exam_notice(subject):
    if 'email' not in session:
        return redirect(url_for('login_page'))
    if EXAM_STATUS.get(subject) == "inactive":
        return render_template('exam_inactive.html')
    full_subject_name = SUBJECT_MAP.get(subject, "Unknown Subject")
    return render_template('exam_notice.html', subject_code=subject, subject_name=full_subject_name)

@app.route('/exam/<subject_code>')
def exam_page(subject_code):
    if 'email' not in session:
        return redirect(url_for('login_page'))
    
    student_roll_no = session['roll_no']

    # **CORRECTED LOGIC**: Check database for previous submissions
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM results WHERE student_roll_no = ? AND subject_code = ?", (student_roll_no, subject_code))
    existing_submission = cursor.fetchone()
    conn.close()

    if existing_submission:
        return render_template('exam_ended.html', message="You have already submitted this exam.")

    if EXAM_STATUS.get(subject_code) == "inactive":
        return render_template('exam_inactive.html')

    # Add to active exams (in-memory dict)
    ACTIVE_EXAMS[str(student_roll_no)] = {'name': session['student_name'], 'subject': subject_code}
    
    full_subject_name = SUBJECT_MAP.get(subject_code, "Unknown Subject")
    return render_template('exam.html', subject=subject_code, subject_name=full_subject_name, roll_no=student_roll_no)

@app.route('/submit', methods=['POST'])
def submit_exam():
    if 'roll_no' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    subject = data.get('subject')
    answers = data.get('answers', {})
    reason = data.get('reason', 'Completed normally')
    student_roll_no = str(session['roll_no'])
    
    score = 0
    question_bank = QUESTIONS.get(subject, [])
    correct_answers = {str(q['id']): q['answer'] for q in question_bank}
    for q_id, user_answer in answers.items():
        if correct_answers.get(q_id) == user_answer:
            score += 1
            
    status = "Completed" if "completed" in reason.lower() else "Terminated"

    # **CORRECTED LOGIC**: Save results to the database
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO results (student_roll_no, subject_code, score, total, status, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (student_roll_no, subject, score, len(question_bank), status, reason)
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError: # Handle case where student submits twice very quickly
        print(f"Attempted duplicate submission for Roll No: {student_roll_no}, Subject: {subject}")
        # Allow the process to continue to redirect the user appropriately
    except Exception as e:
        print(f"Database error on submit: {e}")
        return jsonify({'error': 'Could not save results'}), 500

    # Clean up active exam list
    if student_roll_no in ACTIVE_EXAMS:
        del ACTIVE_EXAMS[student_roll_no]
        
    if status == "Completed":
        redirect_url = url_for('submission_success')
    else: # If status is "Terminated"
        redirect_url = url_for('exam_terminated')
            
    return jsonify({'status': 'success', 'redirect_url': redirect_url})

@app.route('/submission_success')
def submission_success():
    if 'email' not in session:
        return redirect(url_for('login_page'))
    return render_template('submission_success.html')

@app.route('/exam_terminated')
def exam_terminated():
    if 'email' not in session:
        return redirect(url_for('login_page'))
    return render_template('exam_terminated.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# --- API & Admin Routes ---
@app.route('/api/questions/<subject>')
def get_questions(subject):
    if 'roll_no' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    question_list = QUESTIONS.get(subject, [])
    random.shuffle(question_list)
    
    return jsonify([
        {"id": q["id"], "question": q["text"], "options": q["options"]} 
        for q in question_list
    ])

@app.route('/admin')
def admin_login_page():
    return render_template('admin_login.html')

@app.route('/admin/login', methods=['POST'])
def handle_admin_login():
    username = request.form['username']
    password = request.form['password']
    if username == ADMIN_CREDENTIALS['username'] and password == ADMIN_CREDENTIALS['password']:
        session['admin'] = True
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html', error='Invalid Credentials')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    return render_template('admin_dashboard.html', exam_status=EXAM_STATUS)

@app.route('/admin/toggle_exam/<subject>', methods=['POST'])
def toggle_exam(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    if subject in EXAM_STATUS:
        EXAM_STATUS[subject] = "active" if EXAM_STATUS[subject] == "inactive" else "inactive"
        with open(STATUS_FILE, 'w') as f:
            json.dump(EXAM_STATUS, f, indent=4)
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/results/<subject>')
def admin_results(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))

    full_subject_name = SUBJECT_MAP.get(subject, "Unknown Subject")
    
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT student_roll_no, score, total, status, reason FROM results WHERE subject_code = ? ORDER BY score DESC", (subject,))
    results_from_db = cursor.fetchall()
    conn.close()
    
    student_names = {data['roll_no']: data['name'] for data in STUDENTS.values()}
    
    subject_results = []
    for row in results_from_db:
        result_dict = dict(row)
        result_dict['name'] = student_names.get(result_dict['student_roll_no'], 'Unknown')
        subject_results.append(result_dict)

    return render_template(
        'admin_results.html',
        subject_name=full_subject_name,
        results=subject_results,
        subject_code=subject
    )

@app.route('/admin/download_results/<subject>')
def download_results(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT student_roll_no, score, total, status, reason FROM results WHERE subject_code = ?", (subject,)
    )
    results_from_db = cursor.fetchall()
    conn.close()

    if not results_from_db:
        return "No results to download for this subject.", 404

    student_names = {data['roll_no']: data['name'] for data in STUDENTS.values()}

    results_list = []
    for row in results_from_db:
        roll_no = row['student_roll_no']
        results_list.append({
            'Roll No': roll_no,
            'Name': student_names.get(roll_no, 'Unknown'),
            'Score': row['score'],
            'Total Questions': row['total'],
            'Status': row['status'],
            'Reason': row['reason']
        })

    df = pd.DataFrame(results_list)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={subject}_results.xlsx"
    response.headers["Content-type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    return response
# Add this function to your app.py file

@app.route('/admin/questions/<subject>')
def admin_questions(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
        
    full_subject_name = SUBJECT_MAP.get(subject, "Unknown Subject")
    subject_questions = QUESTIONS.get(subject, [])
    
    return render_template(
        'admin_questions.html',
        subject_name=full_subject_name,
        subject_code=subject,
        questions=subject_questions
    )

@app.route('/admin/allow_reexam/<subject>/<roll_no>', methods=['POST'])
def allow_reexam(subject, roll_no):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM results WHERE student_roll_no = ? AND subject_code = ?",
            (roll_no, subject)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error on re-exam: {e}")
            
    return redirect(url_for('admin_results', subject=subject))

@app.route('/admin/review/<student_roll_no>')
def review_session(student_roll_no):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, event_type, message FROM proctoring_logs WHERE student_roll_no = ? ORDER BY timestamp DESC",
        (student_roll_no,)
    )
    logs = cursor.fetchall()
    conn.close()

    student_name = "Unknown"
    for student_data in STUDENTS.values():
        if str(student_data['roll_no']) == str(student_roll_no):
            student_name = student_data['name']
            break

    return render_template(
        'admin_review.html', 
        logs=logs, 
        student_roll_no=student_roll_no,
        student_name=student_name
    )

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login_page'))

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('admin_join')
def handle_admin_join():
    join_room('admin_room')
    print(f"Admin joined: {request.sid}")
    emit('active_students_list', ACTIVE_EXAMS, room=request.sid)

@socketio.on('student_join')
def handle_student_join(data):
    roll_no = data.get('roll_no')
    if roll_no:
        connected_students[request.sid] = str(roll_no)
        join_room(str(roll_no))
        emit(
            'student_started_exam',
            {'roll_no': roll_no, 'info': ACTIVE_EXAMS.get(str(roll_no))},
            room='admin_room'
        )
        print(f"Student {roll_no} joined with SID {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    roll_no = connected_students.pop(request.sid, None)
    if roll_no:
        print(f"Student {roll_no} disconnected: {request.sid}")
        # Remove from active list and notify admin
        if roll_no in ACTIVE_EXAMS:
            del ACTIVE_EXAMS[roll_no]
            emit('student_left_exam', {'roll_no': roll_no}, room='admin_room')
    else:
        print(f"An unknown client or admin disconnected: {request.sid}")


@socketio.on('video_frame_from_student')
def handle_video_frame(data):
    roll_no = data.get('roll_no')
    frame_b64 = data.get('frame')

    if not roll_no or not frame_b64:
        return
    
    # Relay the original frame to the admin
    emit('video_frame', {'roll_no': str(roll_no), 'frame': frame_b64}, room='admin_room')

    # AI Analysis (if model is loaded)
    if not net:
        return
        
    try:
        img_data = base64.b64decode(frame_b64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    class_ids.append(class_id)

        detected_objects = [str(classes[cid]) for cid in class_ids]
        
        # Rule 1: More than one person detected
        if detected_objects.count('person') > 1:
            alert_message = 'Multiple People Detected'
            emit('proctoring_alert', {'roll_no': roll_no, 'alert': alert_message}, room='admin_room')
            log_proctoring_event(roll_no, 'AI_ALERT', alert_message)

        # Rule 2: Cell phone detected
        if 'cell phone' in detected_objects:
            alert_message = 'Cell Phone Detected'
            emit('proctoring_alert', {'roll_no': roll_no, 'alert': alert_message}, room='admin_room')
            log_proctoring_event(roll_no, 'AI_ALERT', alert_message)
    except Exception as e:
        print(f"Error during AI processing: {e}")

# **MERGED and CORRECTED**
@socketio.on('send_warning')
def handle_send_warning(data):
    student_roll_no = data.get('student_roll_no')
    message = data.get('message')
    if student_roll_no and message:
        emit('receive_warning', {'message': message}, room=str(student_roll_no))
        log_proctoring_event(student_roll_no, 'ADMIN_WARNING', message)

# **MERGED and CORRECTED**
@socketio.on('terminate_exam')
def handle_terminate_exam(data):
    student_roll_no = data.get('student_roll_no')
    reason = "Your exam has been terminated by the administrator."
    if student_roll_no:
        emit('exam_terminated', {'reason': reason}, room=str(student_roll_no))
        log_proctoring_event(student_roll_no, 'ADMIN_TERMINATE', reason)


if __name__ == '__main__':
    init_db()  # Ensure database and tables exist before running the app
    socketio.run(app, debug=True)