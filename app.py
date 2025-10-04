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
# In app.py, add these imports
import cv2
import numpy as np
import base64

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

app = Flask(__name__)
# IMPORTANT: For production, load this from an environment variable, not hardcoded.
app.secret_key = 'your_super_secret_key_nhitm'
socketio = SocketIO(app, async_mode='eventlet')

# --- Constants and Configuration ---
RESULTS_FILE = 'results.json'
STATUS_FILE = 'exam_status.json'
ACTIVE_EXAMS_FILE = 'active_exams.json'

ADMIN_CREDENTIALS = {'username': 'admin', 'password': 'admin'}  # TODO: Use hashed passwords in production

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
RESULTS = load_json_file(RESULTS_FILE, {})
EXAM_STATUS = load_json_file(
    STATUS_FILE,
    {"tcs": "inactive", "ip": "inactive", "dwm": "inactive", "cn": "inactive", "se": "inactive"}
)
ACTIVE_EXAMS = load_json_file(ACTIVE_EXAMS_FILE, {})

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
    'virajpukale237@nhitm.ac.i': {'roll_no': '49', 'name': 'PUKALE VIRAJ RAMCHANDRA', 'password': '12317004'},
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
    'prabhavthakur237@nhitm.ac.': {'roll_no': '63', 'name': 'THAKUR PRABHAV PRAVIN', 'password': '12317042'},
    'atharvavandre227@nhitm.ac.in': {'roll_no': '64', 'name': 'VANDRE ATHARVA PRAFUL', 'password': '12247039'},
    'karanvishe237@nhitm.ac.in': {'roll_no': '65', 'name': 'VISHE KARAN', 'password': '12327015'},
    'aryanyadav237@nhitm.ac.in': {'roll_no': '66', 'name': 'YADAV ARYAN ASHOK KUMAR', 'password': '12317002'},
    'lokendaryadav237@nhitm.ac.': {'roll_no': '67', 'name': 'YADAV LOKENDAR BANWARILAL', 'password': '12317022'},
    'sanskrutiyadav237@nhitm.ac.in': {'roll_no': '68', 'name': 'YADAV SANSKRUTI BHALCHANDRA', 'password': '12327007'},

    # "email@example.com": {"name": "John Doe", "roll_no": "101", "password": "1234"}
}

# Question bank
QUESTIONS = {'tcs': [
        {'id': 1, 'text': 'Which of the following is not a finite automaton?', 'options': ['DFA', 'NFA', 'Turing Machine', 'Moore Machine'], 'answer': 'Turing Machine'},
        {'id': 2, 'text': 'A language is regular if and only if it is accepted by a...', 'options': ['Finite Automaton', 'Pushdown Automaton', 'Turing Machine', 'Context-Free Grammar'], 'answer': 'Finite Automaton'},
        {'id': 3, 'text': 'The Pumping Lemma is used to prove that a language is...', 'options': ['Regular', 'Not Regular', 'Context-Free', 'Decidable'], 'answer': 'Not Regular'},
        {'id': 4, 'text': 'Which of the following grammars is the most powerful?', 'options': ['Regular Grammar', 'Context-Free Grammar', 'Context-Sensitive Grammar', 'Unrestricted Grammar'], 'answer': 'Unrestricted Grammar'},
        {'id': 5, 'text': 'A Pushdown Automaton (PDA) uses which data structure?', 'options': ['Queue', 'Stack', 'Array', 'Linked List'], 'answer': 'Stack'},
        {'id': 6, 'text': 'The Halting Problem is an example of a problem that is...', 'options': ['Decidable', 'Undecidable', 'Regular', 'Context-Free'], 'answer': 'Undecidable'},
        {'id': 7, 'text': 'What does a transition function in a DFA map?', 'options': ['State to State', 'State and Input to State', 'State to Input', 'Input to Input'], 'answer': 'State and Input to State'},
        {'id': 8, 'text': 'Which Chomsky Normal Form rule is incorrect?', 'options': ['A -> BC', 'A -> a', 'S -> ε', 'A -> aB'], 'answer': 'A -> aB'},
        {'id': 9, 'text': 'A Turing Machine consists of a tape, a head, and a...', 'options': ['Stack', 'Queue', 'Finite Control', 'Register'], 'answer': 'Finite Control'},
        {'id': 10, 'text': 'The language L = {a^n b^n | n >= 1} is...', 'options': ['Regular', 'Context-Free', 'Finite', 'Recursive'], 'answer': 'Context-Free'},
        {'id': 11, 'text': 'Myhill-Nerode theorem is used for...', 'options': ['Minimizing DFA states', 'Converting NFA to DFA', 'Proving a language is regular', 'All of the above'], 'answer': 'Minimizing DFA states'},
        {'id': 12, 'text': 'The class of languages accepted by a PDA is...', 'options': ['Regular Languages', 'Context-Free Languages', 'Recursive Languages', 'Context-Sensitive Languages'], 'answer': 'Context-Free Languages'},
        {'id': 13, 'text': 'What is the minimum number of stacks a Turing Machine needs to be universal?', 'options': ['0', '1', '2', '3'], 'answer': '2'},
        {'id': 14, 'text': 'A regular expression (a+b)* denotes...', 'options': ['Any string of a\'s and b\'s', 'At least one a or b', 'An empty string', 'Alternating a\'s and b\'s'], 'answer': 'Any string of a\'s and b\'s'},
        {'id': 15, 'text': 'The power of a Turing Machine is equivalent to that of a...', 'options': ['Lambda Calculus', 'Post Correspondence Problem', 'Both A and B', 'Finite Automaton'], 'answer': 'Both A and B'},
        {'id': 16, 'text': 'Greibach Normal Form (GNF) requires productions to be in the form...', 'options': ['A -> aB', 'A -> a', 'A -> aα', 'A -> BC'], 'answer': 'A -> aα'},
        {'id': 17, 'text': 'Which of these is a decidable problem for context-free languages?', 'options': ['Emptiness Problem', 'Ambiguity Problem', 'Equivalence Problem', 'Universality Problem'], 'answer': 'Emptiness Problem'},
        {'id': 18, 'text': 'A DFA accepts a string if the final state reached is...', 'options': ['An accepting state', 'A non-accepting state', 'The start state', 'Any state'], 'answer': 'An accepting state'},
        {'id': 19, 'text': 'The intersection of a Context-Free Language and a Regular Language is...', 'options': ['Always Regular', 'Always Context-Free', 'May not be Context-Free', 'Always Context-Sensitive'], 'answer': 'Always Context-Free'},
        {'id': 20, 'text': 'The set of all palindromes over {a, b} is...', 'options': ['Regular', 'Context-Free but not Regular', 'Context-Sensitive but not CFG', 'Not a language'], 'answer': 'Context-Free but not Regular'},
        {'id': 21, 'text': 'A state in a DFA from which an accepting state cannot be reached is called a...', 'options': ['Dead state', 'Final state', 'Start state', 'Trap state'], 'answer': 'Trap state'},
        {'id': 22, 'text': 'A grammar that produces more than one parse tree for the same sentence is said to be...', 'options': ['Ambiguous', 'Unambiguous', 'Regular', 'Complex'], 'answer': 'Ambiguous'},
        {'id': 23, 'text': 'The universal Turing machine...', 'options': ['Can solve any problem', 'Can simulate any other Turing machine', 'Is the fastest Turing machine', 'Has an infinite number of tapes'], 'answer': 'Can simulate any other Turing machine'},
        {'id': 24, 'text': 'Kleene\'s theorem states that...', 'options': ['Regular expressions and finite automata are equivalent', 'Every regular language is context-free', 'The Halting Problem is undecidable', 'A PDA is more powerful than a DFA'], 'answer': 'Regular expressions and finite automata are equivalent'},
        {'id': 25, 'text': 'What is the role of the δ (delta) in automata theory?', 'options': ['The start state', 'The set of final states', 'The transition function', 'The input alphabet'], 'answer': 'The transition function'},
        {'id': 26, 'text': 'The concept of a "move" in a Turing machine involves...', 'options': ['Changing state', 'Writing a symbol on the tape', 'Moving the tape head left or right', 'All of the above'], 'answer': 'All of the above'},
        {'id': 27, 'text': 'A language that can be accepted by a deterministic pushdown automaton is called a...', 'options': ['Deterministic Context-Free Language', 'Regular Language', 'Recursive Language', 'Universal Language'], 'answer': 'Deterministic Context-Free Language'},
        {'id': 28, 'text': 'Which of the following languages is not context-free?', 'options': ['L = {a^n b^n | n >= 0}', 'L = {ww | w is in {a,b}*}', 'L = {w | w is a palindrome}', 'L = {a^n b^n c^n | n >= 0}'], 'answer': 'L = {a^n b^n c^n | n >= 0}'},
        {'id': 29, 'text': 'The Church-Turing thesis states that...', 'options': ['Turing machines are the most powerful computational model', 'Any problem solvable by an algorithm is solvable by a Turing machine', 'All problems are decidable', 'Quantum computers are faster than Turing machines'], 'answer': 'Any problem solvable by an algorithm is solvable by a Turing machine'},
        {'id': 30, 'text': 'In the context of formal languages, what does ε (epsilon) represent?', 'options': ['The empty set', 'An error state', 'The empty string', 'An infinite loop'], 'answer': 'The empty string'}
    ],
    'ip': [
        {'id': 1, 'text': 'What does HTML stand for?', 'options': ['Hyper Text Markup Language', 'High Tech Modern Language', 'Hyperlink and Text Markup Language', 'Home Tool Markup Language'], 'answer': 'Hyper Text Markup Language'},
        {'id': 2, 'text': 'Which tag is used to create an ordered list?', 'options': ['<ul>', '<ol>', '<li>', '<dl>'], 'answer': '<ol>'},
        {'id': 3, 'text': 'What is the correct CSS syntax for changing the font color of an element?', 'options': ['font-color: red;', 'color: red;', 'text-color: red;', 'fgcolor: red;'], 'answer': 'color: red;'},
        {'id': 4, 'text': 'In JavaScript, how do you declare a variable that cannot be reassigned?', 'options': ['let', 'var', 'const', 'static'], 'answer': 'const'},
        {'id': 5, 'text': 'Which attribute specifies the URL of the page the link goes to?', 'options': ['src', 'href', 'link', 'url'], 'answer': 'href'},
        {'id': 6, 'text': 'What does CSS stand for?', 'options': ['Creative Style Sheets', 'Cascading Style Sheets', 'Computer Style Sheets', 'Colorful Style Sheets'], 'answer': 'Cascading Style Sheets'},
        {'id': 7, 'text': 'How do you select an element with id "demo" in JavaScript?', 'options': ['document.getElement("demo")', 'document.getElementById("demo")', 'document.querySelector("#demo")', 'Both B and C'], 'answer': 'Both B and C'},
        {'id': 8, 'text': 'Which HTML element is used to specify a header for a document or section?', 'options': ['<head>', '<header>', '<h1>', '<top>'], 'answer': '<header>'},
        {'id': 9, 'text': 'The "box model" in CSS consists of margins, borders, padding, and...', 'options': ['Content', 'Outline', 'Float', 'Position'], 'answer': 'Content'},
        {'id': 10, 'text': 'Which operator is used for strict equality in JavaScript?', 'options': ['==', '=', '===', '!='], 'answer': '==='},
        {'id': 11, 'text': 'Which tag is used to embed a video in an HTML page?', 'options': ['<media>', '<video>', '<movie>', '<embed>'], 'answer': '<video>'},
        {'id': 12, 'text': 'What is the purpose of the "alt" attribute in an <img> tag?', 'options': ['To provide alternate text if the image cannot be displayed', 'To define the alignment of the image', 'To set the image source', 'To make the image a link'], 'answer': 'To provide alternate text if the image cannot be displayed'},
        {'id': 13, 'text': 'How do you write a single-line comment in JavaScript?', 'options': ['// This is a comment', '<!-- This is a comment -->', '/* This is a comment */', '# This is a comment'], 'answer': '// This is a comment'},
        {'id': 14, 'text': 'What does the "display: none;" CSS property do?', 'options': ['Hides the element, but it still takes up space', 'Completely removes the element from the document flow', 'Makes the element transparent', 'Changes the element to a block-level element'], 'answer': 'Completely removes the element from the document flow'},
        {'id': 15, 'text': 'What is JSON?', 'options': ['Java Script Object Notation', 'Java Standard Object Notation', 'JavaScript Oriented Network', 'Java Source Open Network'], 'answer': 'Java Script Object Notation'},
        {'id': 16, 'text': 'Which event occurs when the user clicks on an HTML element?', 'options': ['onmouseover', 'onchange', 'onclick', 'onmouseclick'], 'answer': 'onclick'},
        {'id': 17, 'text': 'The <canvas> element is used for...', 'options': ['Drawing graphics via scripting', 'Displaying tabular data', 'Embedding external content', 'Creating forms'], 'answer': 'Drawing graphics via scripting'},
        {'id': 18, 'text': 'What is the purpose of a CSS framework like Bootstrap or Tailwind?', 'options': ['To provide a standard backend language', 'To provide pre-written CSS and JavaScript components for faster development', 'To replace HTML entirely', 'To manage server-side databases'], 'answer': 'To provide pre-written CSS and JavaScript components for faster development'},
        {'id': 19, 'text': 'In JavaScript, what method is used to add an element to the end of an array?', 'options': ['array.add()', 'array.append()', 'array.push()', 'array.insert()'], 'answer': 'array.push()'},
        {'id': 20, 'text': 'What is responsive web design?', 'options': ['Designing websites that respond quickly to user input', 'Designing websites that adapt to different screen sizes and devices', 'Designing websites with a lot of animations', 'Designing websites that are only for mobile phones'], 'answer': 'Designing websites that adapt to different screen sizes and devices'},
        {'id': 21, 'text': 'Which of the following is a semantic HTML element?', 'options': ['<div>', '<span>', '<article>', '<b>'], 'answer': '<article>'},
        {'id': 22, 'text': 'How can you make a bulleted list?', 'options': ['<ol>', '<dl>', '<ul>', '<list>'], 'answer': '<ul>'},
        {'id': 23, 'text': 'What is the correct place to refer to an external style sheet?', 'options': ['In the <body> section', 'At the end of the document', 'In the <head> section', 'In the <title> section'], 'answer': 'In the <head> section'},
        {'id': 24, 'text': 'An "AJAX" call is...', 'options': ['Asynchronous', 'Synchronous', 'Always successful', 'A type of CSS'], 'answer': 'Asynchronous'},
        {'id': 25, 'text': 'What does the "action" attribute in a <form> tag define?', 'options': ['The HTTP method (GET or POST)', 'The URL where the form data will be sent', 'The character encoding', 'The name of the form'], 'answer': 'The URL where the form data will be sent'},
        {'id': 26, 'text': 'Which of these is NOT a valid JavaScript data type?', 'options': ['string', 'number', 'boolean', 'character'], 'answer': 'character'},
        {'id': 27, 'text': 'The CSS property "position: relative;" means the element is positioned relative to...', 'options': ['The browser window', 'Its normal position', 'The parent element', 'The next element'], 'answer': 'Its normal position'},
        {'id': 28, 'text': 'What is the DOM?', 'options': ['Document Object Model', 'Data Object Model', 'Document Oriented Markup', 'Direct Object Model'], 'answer': 'Document Object Model'},
        {'id': 29, 'text': 'How do you create a function in JavaScript?', 'options': ['function:myFunction()', 'function = myFunction()', 'function myFunction()', 'create function myFunction()'], 'answer': 'function myFunction()'},
        {'id': 30, 'text': 'The "z-index" property in CSS is used to...', 'options': ['Set the zoom level of an element', 'Control the stacking order of overlapping elements', 'Set the horizontal alignment', 'Set the vertical alignment'], 'answer': 'Control the stacking order of overlapping elements'}
    ],
    'dwm': [
        {'id': 1, 'text': 'What is Data Mining?', 'options': ['The process of extracting useful information from data', 'The process of entering data into a database', 'The process of creating a database', 'The process of securing data'], 'answer': 'The process of extracting useful information from data'},
        {'id': 2, 'text': 'Which of these is a common Data Mining task?', 'options': ['Classification', 'Clustering', 'Regression', 'All of the above'], 'answer': 'All of the above'},
        {'id': 3, 'text': 'The Apriori algorithm is used for...', 'options': ['Classification', 'Clustering', 'Association Rule Mining', 'Regression'], 'answer': 'Association Rule Mining'},
        {'id': 4, 'text': 'K-Means is an example of which type of algorithm?', 'options': ['Classification', 'Clustering', 'Regression', 'Reinforcement Learning'], 'answer': 'Clustering'},
        {'id': 5, 'text': 'What is "overfitting" in the context of machine learning?', 'options': ['The model performs well on training data but poorly on new data', 'The model is too simple to capture the underlying pattern', 'The model has too few features', 'The model is computationally expensive'], 'answer': 'The model performs well on training data but poorly on new data'},
        {'id': 6, 'text': 'A Decision Tree is a type of...', 'options': ['Supervised learning algorithm', 'Unsupervised learning algorithm', 'Reinforcement learning algorithm', 'Dimensionality reduction technique'], 'answer': 'Supervised learning algorithm'},
        {'id': 7, 'text': 'What does "ETL" stand for in the context of a data warehouse?', 'options': ['Extract, Transform, Load', 'Execute, Test, Launch', 'Estimate, Track, Log', 'Export, Transfer, Link'], 'answer': 'Extract, Transform, Load'},
        {'id': 8, 'text': 'Which of the following is NOT a characteristic of a data warehouse?', 'options': ['Subject-oriented', 'Integrated', 'Time-variant', 'Frequently updated'], 'answer': 'Frequently updated'},
        {'id': 9, 'text': 'OLAP stands for...', 'options': ['Online Analytical Processing', 'Online Algorithmic Processing', 'Operational Language and Processing', 'Online Application Protocol'], 'answer': 'Online Analytical Processing'},
        {'id': 10, 'text': 'What is "noise" in data?', 'options': ['Meaningful data patterns', 'Random errors or variance in measured data', 'The volume of the data', 'The speed at which data is generated'], 'answer': 'Random errors or variance in measured data'},
        {'id': 11, 'text': 'The goal of "classification" is to...', 'options': ['Group similar items together', 'Predict a continuous value', 'Assign an item to a predefined category', 'Find frequently co-occurring items'], 'answer': 'Assign an item to a predefined category'},
        {'id': 12, 'text': 'Which metric is used to measure the quality of clusters?', 'options': ['Accuracy', 'Precision', 'Silhouette score', 'R-squared'], 'answer': 'Silhouette score'},
        {'id': 13, 'text': 'A "data mart" is a...', 'options': ['A very large data warehouse', 'A subset of a data warehouse focused on a specific business line', 'An unprocessed data repository', 'A type of database software'], 'answer': 'A subset of a data warehouse focused on a specific business line'},
        {'id': 14, 'text': 'The "support" of an itemset in association rule mining is...', 'options': ['The number of transactions containing the itemset', 'The strength of the rule', 'The reliability of the rule', 'The interestingness of the rule'], 'answer': 'The number of transactions containing the itemset'},
        {'id': 15, 'text': 'Which of these is a technique for handling missing data?', 'options': ['Deleting the record', 'Imputing the mean or median', 'Using a model to predict the missing value', 'All of the above'], 'answer': 'All of the above'},
        {'id': 16, 'text': '"Regression" is used to predict...', 'options': ['A class label', 'A continuous numerical value', 'A group or cluster', 'An association rule'], 'answer': 'A continuous numerical value'},
        {'id': 17, 'text': 'The CRISP-DM methodology stands for...', 'options': ['Cross-Industry Standard Process for Data Mining', 'Critical Standard Protocol for Data Management', 'Cross-Reference Index for Standard Data Models', 'Customer Relationship and Information System Process for Data Mining'], 'answer': 'Cross-Industry Standard Process for Data Mining'},
        {'id': 18, 'text': 'What is a "schema" in the context of a database?', 'options': ['The raw data itself', 'A query language', 'The logical structure of the database', 'A data backup'], 'answer': 'The logical structure of the database'},
        {'id': 19, 'text': 'The "Naive Bayes" algorithm is a popular algorithm for...', 'options': ['Clustering', 'Classification', 'Regression', 'Dimensionality Reduction'], 'answer': 'Classification'},
        {'id': 20, 'text': 'Data cleaning involves...', 'options': ['Correcting inconsistencies', 'Handling missing values', 'Removing noise', 'All of the above'], 'answer': 'All of the above'},
        {'id': 21, 'text': 'A "star schema" is a common modeling paradigm in...', 'options': ['Transactional databases (OLTP)', 'Data warehouses (OLAP)', 'NoSQL databases', 'Graph databases'], 'answer': 'Data warehouses (OLAP)'},
        {'id': 22, 'text': 'What is "dimensionality reduction"?', 'options': ['Reducing the number of records in a dataset', 'Reducing the number of attributes or variables in a dataset', 'Reducing the complexity of a query', 'Reducing the storage space of a database'], 'answer': 'Reducing the number of attributes or variables in a dataset'},
        {'id': 23, 'text': 'Which of the following is an example of an "outlier"?', 'options': ['A data point that is very common', 'A data point that is significantly different from other data points', 'A data point with missing values', 'A data point that is a duplicate'], 'answer': 'A data point that is significantly different from other data points'},
        {'id': 24, 'text': 'The "confidence" of an association rule X -> Y is...', 'options': ['The percentage of transactions containing both X and Y', 'The percentage of transactions containing X that also contain Y', 'The number of times X appears', 'The interestingness of the rule'], 'answer': 'The percentage of transactions containing X that also contain Y'},
        {'id': 25, 'text': 'What is "Big Data"?', 'options': ['Data that is structured', 'Data that is stored in a single machine', 'Data that is too large and complex to be handled by traditional data-processing application software', 'Data that is used for marketing'], 'answer': 'Data that is too large and complex to be handled by traditional data-processing application software'},
        {'id': 26, 'text': 'The process of transforming data into a suitable format for mining is called...', 'options': ['Data Preprocessing', 'Data Visualization', 'Data Warehousing', 'Data Selection'], 'answer': 'Data Preprocessing'},
        {'id': 27, 'text': 'Which data structure is commonly used to build a decision tree?', 'options': ['A graph', 'A tree', 'A stack', 'A queue'], 'answer': 'A tree'},
        {'id': 28, 'text': 'In a star schema, the central table is called the...', 'options': ['Dimension table', 'Fact table', 'Attribute table', 'Lookup table'], 'answer': 'Fact table'},
        {'id': 29, 'text': 'The main difference between supervised and unsupervised learning is that supervised learning uses...', 'options': ['Labeled data', 'Unlabeled data', 'Real-time data', 'Small data'], 'answer': 'Labeled data'},
        {'id': 30, 'text': 'What is the purpose of a "data cube" in OLAP?', 'options': ['To store data in a compressed format', 'To allow for fast analysis of data from multiple perspectives (dimensions)', 'To enforce data security', 'To create backups of the data warehouse'], 'answer': 'To allow for fast analysis of data from multiple perspectives (dimensions)'}
    ],
    'cn': [
        {'id': 1, 'text': 'What is the full form of LAN?', 'options': ['Local Area Network', 'Large Area Network', 'Live Area Network', 'Logical Area Network'], 'answer': 'Local Area Network'},
        {'id': 2, 'text': 'Which layer in the OSI model is responsible for framing?', 'options': ['Physical Layer', 'Data Link Layer', 'Network Layer', 'Transport Layer'], 'answer': 'Data Link Layer'},
        {'id': 3, 'text': 'The IP address 127.0.0.1 is known as the...', 'options': ['Gateway address', 'Broadcast address', 'Loopback address', 'Subnet mask'], 'answer': 'Loopback address'},
        {'id': 4, 'text': 'Which protocol is used for sending email?', 'options': ['FTP', 'HTTP', 'SMTP', 'POP3'], 'answer': 'SMTP'},
        {'id': 5, 'text': 'A "router" operates at which layer of the OSI model?', 'options': ['Physical Layer', 'Data Link Layer', 'Network Layer', 'Transport Layer'], 'answer': 'Network Layer'},
        {'id': 6, 'text': 'What does TCP stand for?', 'options': ['Transmission Control Protocol', 'Technical Control Protocol', 'Transport Communication Protocol', 'Telecommunication Control Protocol'], 'answer': 'Transmission Control Protocol'},
        {'id': 7, 'text': 'Which of these is a connectionless protocol?', 'options': ['TCP', 'UDP', 'FTP', 'HTTP'], 'answer': 'UDP'},
        {'id': 8, 'text': 'A "MAC address" is a unique identifier assigned to a...', 'options': ['Router', 'Network Interface Card (NIC)', 'IP address', 'Website'], 'answer': 'Network Interface Card (NIC)'},
        {'id': 9, 'text': 'The process of converting digital data into an analog signal is called...', 'options': ['Modulation', 'Demodulation', 'Encoding', 'Decoding'], 'answer': 'Modulation'},
        {'id': 10, 'text': 'DNS is a system for...', 'options': ['Encrypting data', 'Translating domain names to IP addresses', 'Assigning IP addresses dynamically', 'Filtering network traffic'], 'answer': 'Translating domain names to IP addresses'},
        {'id': 11, 'text': 'Which topology requires a central controller or hub?', 'options': ['Bus', 'Ring', 'Star', 'Mesh'], 'answer': 'Star'},
        {'id': 12, 'text': 'The physical layer of the OSI model is concerned with...', 'options': ['Error detection', 'Flow control', 'Transmission of bits over a communication channel', 'Routing'], 'answer': 'Transmission of bits over a communication channel'},
        {'id': 13, 'text': 'What is a "firewall" used for?', 'options': ['To monitor and control incoming and outgoing network traffic', 'To increase internet speed', 'To store website data', 'To connect multiple networks'], 'answer': 'To monitor and control incoming and outgoing network traffic'},
        {'id': 14, 'text': 'Which of the following is an example of an application layer protocol?', 'options': ['TCP', 'IP', 'HTTP', 'Ethernet'], 'answer': 'HTTP'},
        {'id': 15, 'text': 'The term "bandwidth" refers to...', 'options': ['The speed of the processor', 'The maximum data transfer rate of a network', 'The number of devices on a network', 'The physical length of a cable'], 'answer': 'The maximum data transfer rate of a network'},
        {'id': 16, 'text': 'Which of the following is a private IP address range?', 'options': ['10.0.0.0 to 10.255.255.255', '172.16.0.0 to 172.31.255.255', '192.168.0.0 to 192.168.255.255', 'All of the above'], 'answer': 'All of the above'},
        {'id': 17, 'text': 'A "switch" operates at which layer of the OSI model?', 'options': ['Physical Layer', 'Data Link Layer', 'Network Layer', 'Transport Layer'], 'answer': 'Data Link Layer'},
        {'id': 18, 'text': 'What is the purpose of ARP (Address Resolution Protocol)?', 'options': ['To resolve IP addresses to MAC addresses', 'To resolve domain names to IP addresses', 'To assign IP addresses', 'To route packets'], 'answer': 'To resolve IP addresses to MAC addresses'},
        {'id': 19, 'text': 'What is a "port number" used for?', 'options': ['To identify a specific computer on a network', 'To identify a specific process or application on a computer', 'To identify the physical connection', 'To identify the network operator'], 'answer': 'To identify a specific process or application on a computer'},
        {'id': 20, 'text': 'Which of the following provides a secure communication channel?', 'options': ['HTTP', 'FTP', 'Telnet', 'HTTPS'], 'answer': 'HTTPS'},
        {'id': 21, 'text': 'The "three-way handshake" is a process used by which protocol to establish a connection?', 'options': ['UDP', 'TCP', 'IP', 'ICMP'], 'answer': 'TCP'},
        {'id': 22, 'text': 'Which type of cable is commonly used for Ethernet networks?', 'options': ['Coaxial cable', 'Fiber-optic cable', 'Twisted-pair cable', 'All of the above'], 'answer': 'Twisted-pair cable'},
        {'id': 23, 'text': 'What is a "subnet mask"?', 'options': ['A part of an IP address that defines the network and host portions', 'A way to hide your IP address', 'A type of network security', 'A tool for measuring network speed'], 'answer': 'A part of an IP address that defines the network and host portions'},
        {'id': 24, 'text': 'The presentation layer of the OSI model is responsible for...', 'options': ['Routing', 'Session management', 'Data encryption and compression', 'Flow control'], 'answer': 'Data encryption and compression'},
        {'id': 25, 'text': 'What does DHCP stand for?', 'options': ['Dynamic Host Configuration Protocol', 'Data Host Communication Protocol', 'Domain Host Control Protocol', 'Dynamic Host Communication Protocol'], 'answer': 'Dynamic Host Configuration Protocol'},
        {'id': 26, 'text': 'In a client-server model, the client...', 'options': ['Provides a service', 'Requests a service', 'Controls the network', 'Is always more powerful than the server'], 'answer': 'Requests a service'},
        {'id': 27, 'text': 'Which of the following is a routing algorithm?', 'options': ['Dijkstra\'s algorithm', 'Prim\'s algorithm', 'Kruskal\'s algorithm', 'FIFO'], 'answer': 'Dijkstra\'s algorithm'},
        {'id': 28, 'text': 'The "Transport Layer" provides...', 'options': ['Physical connection', 'Node-to-node delivery', 'Process-to-process delivery', 'Hop-to-hop delivery'], 'answer': 'Process-to-process delivery'},
        {'id': 29, 'text': 'What is a "packet" in the context of computer networks?', 'options': ['A unit of data that is routed between an origin and a destination', 'A physical network device', 'A type of network cable', 'A software application'], 'answer': 'A unit of data that is routed between an origin and a destination'},
        {'id': 30, 'text': 'Which organization is responsible for developing and maintaining web standards?', 'options': ['IEEE', 'IETF', 'W3C', 'ISO'], 'answer': 'W3C'}
    ],
    'se': [
        {'id': 1, 'text': 'What is the first phase of the SDLC?', 'options': ['Design', 'Coding', 'Requirement Gathering and Analysis', 'Testing'], 'answer': 'Requirement Gathering and Analysis'},
        {'id': 2, 'text': 'Which of the following is a software development model?', 'options': ['Waterfall Model', 'Agile Model', 'Spiral Model', 'All of the above'], 'answer': 'All of the above'},
        {'id': 3, 'text': '"Agile" development is best described as...', 'options': ['A linear and sequential approach', 'An iterative and incremental approach', 'A risk-driven approach', 'A documentation-heavy approach'], 'answer': 'An iterative and incremental approach'},
        {'id': 4, 'text': 'What is "black-box testing"?', 'options': ['Testing based on the internal structure of the code', 'Testing without knowledge of the internal workings of the application', 'Testing done by the developer', 'Testing the performance of the software'], 'answer': 'Testing without knowledge of the internal workings of the application'},
        {'id': 5, 'text': 'A "use case" describes...', 'options': ['A specific interaction between a user and the system', 'The database schema', 'The programming language used', 'The hardware requirements'], 'answer': 'A specific interaction between a user and the system'},
        {'id': 6, 'text': 'What is the purpose of a "requirements specification" document?', 'options': ['To describe what the system should do', 'To provide the source code', 'To list the project team members', 'To create a marketing plan'], 'answer': 'To describe what the system should do'},
        {'id': 7, 'text': 'The "waterfall model" is not suitable for projects where...', 'options': ['Requirements are well-understood and stable', 'The project is large and complex', 'Requirements are likely to change', 'The technology is mature'], 'answer': 'Requirements are likely to change'},
        {'id': 8, 'text': 'Which of the following is a non-functional requirement?', 'options': ['The system shall allow users to register', 'The system shall be able to handle 1000 concurrent users', 'The system shall send an email confirmation', 'The system shall have a search feature'], 'answer': 'The system shall be able to handle 1000 concurrent users'},
        {'id': 9, 'text': '"Unit testing" focuses on testing...', 'options': ['The entire system', 'The integration between modules', 'Individual components or modules of the software', 'The user interface'], 'answer': 'Individual components or modules of the software'},
        {'id': 10, 'text': 'What is "refactoring"?', 'options': ['Adding new features to the code', 'Restructuring existing computer code without changing its external behavior', 'Fixing bugs in the code', 'Writing documentation for the code'], 'answer': 'Restructuring existing computer code without changing its external behavior'},
        {'id': 11, 'text': '"Scrum" is a framework for implementing which development methodology?', 'options': ['Waterfall', 'Spiral', 'Agile', 'V-Model'], 'answer': 'Agile'},
        {'id': 12, 'text': 'What is "version control" software (like Git) used for?', 'options': ['To manage changes to source code over time', 'To compile the source code', 'To test the software', 'To deploy the software'], 'answer': 'To manage changes to source code over time'},
        {'id': 13, 'text': 'The "design" phase of the SDLC focuses on...', 'options': ['What the system will do', 'How the system will do it', 'If the system works correctly', 'Who will use the system'], 'answer': 'How the system will do it'},
        {'id': 14, 'text': '"Regression testing" is performed to...', 'options': ['Test a newly added feature', 'Ensure that changes to the code have not broken existing functionality', 'Test the performance of the system', 'Test the security of the system'], 'answer': 'Ensure that changes to the code have not broken existing functionality'},
        {'id': 15, 'text': 'What is "software maintenance"?', 'options': ['The process of developing new software', 'The process of modifying a software product after it has been delivered', 'The process of testing software', 'The process of gathering requirements'], 'answer': 'The process of modifying a software product after it has been delivered'},
        {'id': 16, 'text': 'A "prototype" is...', 'options': ['The final version of the software', 'An early, incomplete model of the software', 'A type of software bug', 'A design document'], 'answer': 'An early, incomplete model of the software'},
        {'id': 17, 'text': 'What is the main goal of "software quality assurance" (SQA)?', 'options': ['To ensure the software is delivered on time', 'To ensure the software meets the specified quality standards', 'To write the code for the software', 'To manage the project budget'], 'answer': 'To ensure the software meets the specified quality standards'},
        {'id': 18, 'text': 'The "spiral model" of software development is...', 'options': ['A risk-driven model', 'A linear model', 'An incremental model', 'A model without a design phase'], 'answer': 'A risk-driven model'},
        {'id': 19, 'text': 'What is "coupling" in software design?', 'options': ['The degree of interdependence between software modules', 'The degree to which the elements inside a module belong together', 'The number of lines of code in a module', 'The complexity of a module'], 'answer': 'The degree of interdependence between software modules'},
        {'id': 20, 'text': '"Cohesion" refers to...', 'options': ['The degree of interdependence between software modules', 'The degree to which the elements inside a module belong together', 'The reusability of a module', 'The security of a module'], 'answer': 'The degree to which the elements inside a module belong together'}
    ]
    
    # "tcs": [{"id": 1, "text": "Sample?", "options": ["A", "B", "C", "D"], "answer": "A"}]
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

@app.route('/exam/<subject>')
def exam_page(subject):
    if 'email' not in session:
        return redirect(url_for('login_page'))
    student_roll_no = session['roll_no']

    # Prevent re-exam if already submitted
    if str(student_roll_no) in RESULTS and subject in RESULTS[str(student_roll_no)]:
        return render_template('exam_ended.html')

    if EXAM_STATUS.get(subject) == "inactive":
        return render_template('exam_inactive.html')

    # Add to active exams
    ACTIVE_EXAMS[str(student_roll_no)] = {'name': session['student_name'], 'subject': subject}
    with open(ACTIVE_EXAMS_FILE, 'w') as f:
        json.dump(ACTIVE_EXAMS, f, indent=4)

    full_subject_name = SUBJECT_MAP.get(subject, "Unknown Subject")
    return render_template('exam.html', subject=subject, subject_name=full_subject_name, roll_no=student_roll_no)

# In app.py

# ... (your imports and other code) ...

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

    # --- DATABASE LOGIC (replaces JSON logic) ---
    score = 0
    question_bank = QUESTIONS.get(subject, [])
    correct_answers = {str(q['id']): q['answer'] for q in question_bank}
    for q_id, user_answer in answers.items():
        if correct_answers.get(q_id) == user_answer:
            score += 1
            
    status = "Completed" if "completed" in reason.lower() else "Terminated"

    # Connect to the database and insert the result
    conn = sqlite3.connect('exam_database.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO results (student_roll_no, subject_code, score, total, status) VALUES (?, ?, ?, ?, ?)",
        (student_roll_no, subject, score, len(question_bank), status)
    )
    conn.commit()
    conn.close()
    # --- END OF DATABASE LOGIC ---

    if student_roll_no in ACTIVE_EXAMS:
        del ACTIVE_EXAMS[student_roll_no]
        with open(ACTIVE_EXAMS_FILE, 'w') as f:
            json.dump(ACTIVE_EXAMS, f, indent=4)
            
    return jsonify({'status': 'success', 'redirect_url': url_for('dashboard')})

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

@app.route('/admin/results/<subject>')
def admin_results(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    full_subject_name = SUBJECT_MAP.get(subject, "Unknown Subject")
    subject_results = []
    for roll_no, data in RESULTS.items():
        if subject in data:
            subject_results.append({
                'roll_no': roll_no,
                'name': data[subject]['name'],
                'score': data[subject]['score'],
                'total': data[subject]['total'],
                'status': data[subject].get('status', 'Completed')
            })
    return render_template(
        'admin_results.html',
        subject_name=full_subject_name,
        results=subject_results,
        subject_code=subject
    )

@app.route('/admin/allow_reexam/<subject>/<roll_no>', methods=['POST'])
def allow_reexam(subject, roll_no):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    roll_no_str = str(roll_no)
    if roll_no_str in RESULTS and subject in RESULTS[roll_no_str]:
        del RESULTS[roll_no_str][subject]
        if not RESULTS[roll_no_str]:
            del RESULTS[roll_no_str]
        with open(RESULTS_FILE, 'w') as f:
            json.dump(RESULTS, f, indent=4)
    return redirect(url_for('admin_results', subject=subject))

@app.route('/admin/download_results/<subject>')
def download_results(subject):
    if not session.get('admin'):
        return redirect(url_for('admin_login_page'))
    subject_results = []
    for roll_no, data in RESULTS.items():
        if subject in data:
            subject_results.append({
                'Roll No': roll_no,
                'Name': data[subject]['name'],
                'Score': data[subject]['score'],
                'Total Questions': data[subject]['total'],
                'Status': data[subject].get('status', 'Completed')
            })
    if not subject_results:
        return "No results to download.", 404
    df = pd.DataFrame(subject_results)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={subject}_results.xlsx"
    response.headers["Content-type"] = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return response

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
        join_room(roll_no)
        emit(
            'student_started_exam',
            {'roll_no': roll_no, 'info': ACTIVE_EXAMS.get(str(roll_no))},
            room='admin_room'
        )
        print(f"Student {roll_no} joined with SID {request.sid}")

@socketio.on('webrtc_signal')
def handle_webrtc_signal(data):
    recipient_room = data.get('recipient_room')
    if recipient_room:
        emit('webrtc_signal', data, room=recipient_room, include_self=False)

@socketio.on('disconnect')# Add this decorator above your function definition.
# The event name 'video_frame_from_student' is an example;
# use the actual event name your student-side JS is emitting.
# In app.py, replace your old handle_video_frame function

# In your app.py file

@socketio.on('video_frame_from_student')
def handle_video_frame(data):
    roll_no = data.get('roll_no')
    frame_b64 = data.get('frame')

    if not roll_no or not frame_b64:
        return

    # --- Task 1: Relay the original frame to the admin ---
    emit('video_frame', {'roll_no': str(roll_no), 'frame': frame_b64}, room='admin_room')

    # --- Task 2: AI Analysis ---
    try:
        # Decode the base64 frame
        img_data = base64.b64decode(frame_b64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, channels = img.shape

        # Prepare the image for the AI model
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # --- Process the model's output ---
        class_ids = []
        confidences = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

        # --- Check for violations ---
        detected_objects = [str(classes[class_id]) for class_id in class_ids]
        
        # Rule 1: More than one person detected
        if detected_objects.count('person') > 1:
            alert_message = 'Multiple People Detected'
            emit('proctoring_alert', {'roll_no': roll_no, 'alert': alert_message}, room='admin_room')
            log_proctoring_event(roll_no, 'AI_ALERT', alert_message) # <-- ADD THIS LINE

        # Rule 2: Cell phone detected
        if 'cell phone' in detected_objects:
            alert_message = 'Cell Phone Detected'
            emit('proctoring_alert', {'roll_no': roll_no, 'alert': alert_message}, room='admin_room')
            log_proctoring_event(roll_no, 'AI_ALERT', alert_message) # <-- ADD THIS LINE

    except Exception as e:
        print(f"Error during AI processing: {e}")

@socketio.on('send_warning')
def handle_send_warning(data):
    student_roll_no = data.get('student_roll_no')
    message = data.get('message')
    if student_roll_no and message:
        emit('receive_warning', {'message': message}, room=str(student_roll_no))
        log_proctoring_event(student_roll_no, 'ADMIN_WARNING', message) # <-- ADD THIS LINE

@socketio.on('terminate_exam')
def handle_terminate_exam(data):
    student_roll_no = data.get('student_roll_no')
    if student_roll_no:
        # Emit termination event to the specific student
        emit('exam_terminated', {'reason': 'Your exam has been terminated by the administrator.'}, room=str(student_roll_no))
        # In app.py, in the SocketIO Events section

# ADD THIS FUNCTION
@socketio.on('send_warning')
def handle_send_warning(data):
    student_roll_no = data.get('student_roll_no')
    message = data.get('message')
    if student_roll_no and message:
        # Emit the warning only to the specific student's room
        emit('receive_warning', {'message': message}, room=str(student_roll_no))

# ADD THIS FUNCTION
# In app.py

def log_proctoring_event(roll_no, event_type, message):
    """Helper function to save a proctoring event to the database."""
    try:
        conn = sqlite3.connect('exam_database.db')
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
@socketio.on('terminate_exam')
def handle_terminate_exam(data):
    student_roll_no = data.get('student_roll_no')
    if student_roll_no:
        # Emit termination event to the specific student
        emit('exam_terminated', {'reason': 'Your exam has been terminated by the administrator.'}, room=str(student_roll_no))
if __name__ == '__main__':
    socketio.run(app, debug=True)
