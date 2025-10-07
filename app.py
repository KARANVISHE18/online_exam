import eventlet
eventlet.monkey_patch()
import sqlite3
from flask import Flask, render_template, request, jsonify, session,flash, redirect, url_for, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import pandas as pd
from io import BytesIO
import os
import random 
# In app.py, add these imports
import cv2
import numpy as np
import base64
import time

# --- AI Model Setup (YOLO) ---
# Load the model configuration and weights
yolo_path = "yolo_model" 
net = cv2.dnn.readNet(os.path.join(yolo_path, "yolov3-tiny.weights"), os.path.join(yolo_path, "yolov3-tiny.cfg"))
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the object names
with open(os.path.join(yolo_path, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("AI Model loaded successfully.")
# --- End of AI Model Setup ---

# DELETE THIS ENTIRE BLOCK
app = Flask(__name__)
# IMPORTANT: For production, load this from an environment variable, not hardcoded.
app.secret_key = 'your_super_secret_key_nhitm'
socketio = SocketIO(app, async_mode='eventlet')
# IMPORTANT: For production, load this from an environment variable, not hardcoded.
from flask_cors import CORS # Make sure this import is at the top of your file

app.secret_key = 'your_super_secret_key_nhitm'
CORS(app)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
# --- Constants and Configuration ---
RESULTS_FILE = 'results.json'
STATUS_FILE = 'exam_status.json'
ACTIVE_EXAMS_FILE = 'active_exams.json'

ADMIN_CREDENTIALS = {'username': 'admin', 'password': 'admin'}  # TODO: Use hashed passwords in production
if not os.path.exists(ACTIVE_EXAMS_FILE):
    ACTIVE_EXAMS = {}
    with open(ACTIVE_EXAMS_FILE, 'w') as f:
        json.dump(ACTIVE_EXAMS, f, indent=4)

SUBJECT_MAP = {
    "tcs": "Theoretical Computer Science",
    "ip": "Internet Programming",
    "dwm": "Data Warehouse & Mining",
    "cn": "Computer Networks",
    "se": "Software Engineering"
}

# --- Data Loading ---
def load_json_file(filename, default_data):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_data
    return default_data

# In-memory state loaded from files
# --- Constants and Configuration ---
# Define the variables FIRST
# --- Constants and Configuration ---
# Define the variables FIRST
RESULTS_FILE = 'results.json'
STATUS_FILE = 'exam_status.json'
ACTIVE_EXAMS_FILE = 'active_exams.json'
ADMIN_CREDENTIALS = {'username': 'admin', 'password': 'admin'}
# ... (rest of your constants)

# --- Data Loading ---
# NOW you can use the variables, because they have been defined
RESULTS = load_json_file(RESULTS_FILE, {})
EXAM_STATUS = load_json_file(STATUS_FILE, {"tcs": "inactive", "ip": "inactive", "dwm": "inactive", "cn": "inactive", "se": "inactive"})
ACTIVE_EXAMS = load_json_file(ACTIVE_EXAMS_FILE, {})
# Dummy student DB (replace with DB later)
STUDENTS = {
}

# Question bank
QUESTIONS = {
}

# Track live connections (maps sid → roll_no)
connected_students = {}

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
    if 'roll_no' not in session:
        return redirect(url_for('login_page'))

    student_roll_no = str(session['roll_no'])

    if student_roll_no in RESULTS and subject_code in RESULTS.get(student_roll_no, {}):
        return render_template('exam_ended.html')

    ACTIVE_EXAMS[student_roll_no] = {
        'name': session['student_name'],
        'subject': subject_code,
        'startTime': time.time()
    }
    with open(ACTIVE_EXAMS_FILE, 'w') as f:
        json.dump(ACTIVE_EXAMS, f)

    student_info = {
        'roll_no': student_roll_no,
        'name': session.get('student_name', 'Student')
    }
    full_subject_name = SUBJECT_MAP.get(subject_code, "Unknown Subject")

    return render_template(
        'exam.html',
        subject_code=subject_code,
        subject_name=full_subject_name,
        student=student_info
    )
# In app.py

# ... (your imports and other code) ...

# In your app.py file

@app.route('/submit', methods=['POST'])
def submit_exam():
    if 'roll_no' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    subject = data.get('subject')
    answers = data.get('answers', {})
    reason = data.get('reason', 'Completed normally')
    student_roll_no = str(session['roll_no'])
    student_name = session['student_name']

    score = 0
    question_bank = QUESTIONS.get(subject, [])
    correct_answers = {str(q['id']): q['answer'] for q in question_bank}
    for q_id, user_answer in answers.items():
        if correct_answers.get(q_id) == user_answer:
            score += 1
            
    status = "Completed" if "completed" in reason.lower() else "Terminated"

    if student_roll_no not in RESULTS:
        RESULTS[student_roll_no] = {}
    RESULTS[student_roll_no][subject] = {
        'name': student_name, 'score': score, 'total': len(question_bank), 'status': status, 'reason': reason
    }
    with open('results.json', 'w') as f:
        json.dump(RESULTS, f, indent=4)
        
    if student_roll_no in ACTIVE_EXAMS:
        del ACTIVE_EXAMS[student_roll_no]

    # --- THIS IS THE UPDATED LOGIC ---
    # It now checks the status and chooses the correct redirect URL
    if status == "Completed":
        # You'll also need a 'submission_success.html' template for this to work
        redirect_url = url_for('submission_success')
    else: # If status is "Terminated"
        redirect_url = url_for('exam_terminated')
    # --- END OF UPDATE ---
            
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
# In app.py
# In your app.py file

@app.route('/admin/download_results/<subject>')
def download_results(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))

    # Get results for the specific subject from the database
    conn = sqlite3.connect('exam_database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT student_roll_no, score, total, status FROM results WHERE subject_code = ?",
        (subject,)
    )
    results_from_db = cursor.fetchall()
    conn.close()

    if not results_from_db:
        return "No results to download for this subject.", 404

    # Get student names from the global STUDENTS dictionary
    student_names = {data['roll_no']: data['name'] for data in STUDENTS.values()}

    # Format data for the Excel file
    results_list = []
    for row in results_from_db:
        roll_no = row['student_roll_no']
        results_list.append({
            'Roll No': roll_no,
            'Name': student_names.get(roll_no, 'Unknown'),
            'Score': row['score'],
            'Total Questions': row['total'],
            'Status': row['status']
        })

    # Use pandas to create an Excel file in memory
    df = pd.DataFrame(results_list)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)
    
    # Create the response that sends the file to the user's browser
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={subject}_results.xlsx"
    response.headers["Content-type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    return response
@app.route('/api/questions/<subject>')
def get_questions(subject):
    if 'roll_no' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    question_list = QUESTIONS.get(subject, [])
    random.shuffle(question_list)
    
    # This ensures the key is 'question', which the frontend expects
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

@app.route('/admin/questions/<subject>')
def admin_questions(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    full_subject_name = SUBJECT_MAP.get(subject, "Unknown Subject")
    return render_template(
        'admin_questions.html',
        subject_name=full_subject_name,
        subject_code=subject,
        questions=QUESTIONS.get(subject, [])
    )

@app.route('/admin/add_question/<subject>', methods=['POST'])
def add_question(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    subject_questions = QUESTIONS.get(subject, [])
    new_question = {
        'id': len(subject_questions) + 1,
        'text': request.form['question_text'],
        'options': [request.form['option1'], request.form['option2'],
                    request.form['option3'], request.form['option4']],
        'answer': request.form['correct_answer']
    }
    subject_questions.append(new_question)
    QUESTIONS[subject] = subject_questions
    return redirect(url_for('admin_questions', subject=subject))


    



# In your app.py file

# In app.py, REPLACE your admin_results function with this

@app.route('/admin/results/<subject>')
def admin_results(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))

    full_subject_name = SUBJECT_MAP.get(subject, "Unknown Subject")
    
    subject_results = []
    for roll_no, student_data in RESULTS.items():
        if subject in student_data:
            result = student_data[subject]
            subject_results.append({
                'roll_no': roll_no,
                'name': result.get('name', 'Unknown'),
                'score': result.get('score', 0),
                'total': result.get('total', 0),
                'status': result.get('status', 'N/A'),
                'reason': result.get('reason', 'N/A')
            })

    return render_template(
        'admin_results.html',
        subject_name=full_subject_name,
        results=subject_results,
        subject_code=subject
    )
# In app.py, ADD this new function

# In app.py, make sure you have this function

@app.route('/admin/allow_reexam/<subject>/<roll_no>', methods=['POST'])
def allow_reexam(subject, roll_no):
    try:
        with open(ACTIVE_EXAMS_FILE, 'r') as f:
            active_exams = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        active_exams = {}

    # ✅ Allow re-exam for the selected student
    if subject not in active_exams:
        active_exams[subject] = []

    if roll_no not in active_exams[subject]:
        active_exams[subject].append(roll_no)

    with open(ACTIVE_EXAMS_FILE, 'w') as f:
        json.dump(active_exams, f, indent=4)

    flash(f"Re-exam allowed for Roll No: {roll_no}", "success")
    return redirect(url_for('view_results', subject=subject))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login_page'))

# --- SocketIO Events ---
# In your app.py file

@app.route('/admin/review/<student_roll_no>')
def review_session(student_roll_no):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))

    # Fetch logs for the specific student from the database
    conn = sqlite3.connect('exam_database.db')
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, event_type, message FROM proctoring_logs WHERE student_roll_no = ? ORDER BY timestamp DESC",
        (student_roll_no,)
    )
    logs = cursor.fetchall()
    conn.close()

    # Get student name for display
    student_name = ""
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

# --- SocketIO Events ---

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in connected_students:
        roll_no = connected_students.pop(request.sid)
        emit('student_left_exam', {'roll_no': roll_no}, room='admin_room')
        print(f"Student {roll_no} disconnected.")

@socketio.on('admin_join')
def handle_admin_join():
    join_room('admin_room')
    print(f"Admin joined room: {request.sid}")
    emit('active_students_list', ACTIVE_EXAMS, room=request.sid)

@socketio.on('student_join')
def handle_student_join(data):
    roll_no = str(data.get('roll_no'))
    if roll_no:
        connected_students[request.sid] = roll_no
        join_room(roll_no)
        emit('student_started_exam', {'roll_no': roll_no, 'info': ACTIVE_EXAMS.get(roll_no)}, room='admin_room')
        print(f"Student {roll_no} joined with SID {request.sid}")

@socketio.on('video_frame_from_student')
def handle_video_frame(data):
    roll_no, frame_b64 = data.get('roll_no'), data.get('frame')
    if not (roll_no and frame_b64 and net): return
    emit('video_frame', {'roll_no': str(roll_no), 'frame': frame_b64}, room='admin_room')
    try:
        img_data = base64.b64decode(frame_b64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        detected_objects = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.5: detected_objects.append(str(classes[class_id]))
        if detected_objects.count('person') > 1:
            emit('proctoring_alert', {'roll_no': roll_no, 'alert': 'Multiple People Detected'}, room='admin_room')
        if 'cell phone' in detected_objects:
            emit('proctoring_alert', {'roll_no': roll_no, 'alert': 'Cell Phone Detected'}, room='admin_room')
    except Exception as e:
        print(f"AI processing error: {e}")

@socketio.on('send_warning')
def handle_send_warning(data):
    student_roll_no = data.get('student_roll_no')
    message = data.get('message')
    if student_roll_no and message:
        emit('receive_warning', {'message': message}, room=str(student_roll_no))

@socketio.on('terminate_exam')
def handle_terminate_exam(data):
    student_roll_no = data.get('student_roll_no')
    if student_roll_no:
        emit('exam_terminated', {'reason': 'Your exam has been terminated by the administrator.'}, room=str(student_roll_no))
headers: { "Content-Type": "application/json" }

if __name__ == '__main__':
    socketio.run(app, debug=True)