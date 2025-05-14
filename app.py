from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import threading
import signal
import time
import os
import sqlite3
import json
from datetime import datetime
from logic import evaluate_exam, generate_report

app = Flask(__name__)
app.secret_key = 'dev_key_123_!@#_secure'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Database initialization
def init_db():
    conn = sqlite3.connect('exam_evaluation.db')
    c = conn.cursor()

    c.execute('PRAGMA journal_mode=WAL;')  # Better concurrency
    c.execute('PRAGMA synchronous=NORMAL;')
    c.execute('PRAGMA foreign_keys = ON;')

    c.execute('PRAGMA foreign_keys = ON;')

    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT NOT NULL UNIQUE)''')

    # Create evaluations table
    c.execute('''CREATE TABLE IF NOT EXISTS evaluations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  question_paper TEXT NOT NULL,
                  answer_key TEXT NOT NULL,
                  student_answer TEXT NOT NULL,
                  student_name TEXT NOT NULL,
                  subject_name TEXT NOT NULL,
                  results_json TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')

    conn.commit()
    conn.close()

init_db()

# Helper functions
def get_db_connection():
    conn = sqlite3.connect('exam_evaluation.db', timeout=10.0)  # Add timeout
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA busy_timeout = 30000;')  # 30 second timeout
    return conn

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf']

def generate_text_report(results):
    """Generate a simplified text report for web display"""
    report = []
    report.append(f"Student: {results['metadata']['student_answer_pdf']}")
    report.append(f"Total Score: {results['summary']['total_marks_obtained']:.1f}/{results['summary']['total_possible_marks']}")
    report.append(f"Percentage: {results['summary']['percentage_score']:.1f}%")
    report.append(f"Performance: {results['summary']['performance_category']}")

    report.append("\nQuestion-wise Results:")
    for q in results['questions']:
        eval = q['evaluation']
        report.append(f"\nQ{q['question_number']}: {eval.get('marks_obtained', 0):.1f}/{q['max_marks']}")
        if 'feedback' in eval:
            if 'strengths' in eval['feedback'] and eval['feedback']['strengths']:
                report.append("  ✓ " + eval['feedback']['strengths'][0])
            if 'improvements' in eval['feedback'] and eval['feedback']['improvements']:
                report.append("  ✗ " + eval['feedback']['improvements'][0])

    return "\n".join(report)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        conn = None
        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                        (username, password, email))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'error')
        except Exception as e:
            flash(f'Registration failed: {str(e)}', 'error')
        finally:
            if conn:
                conn.close()
    
    # Add this return statement for GET requests
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?',
                            (email, password)).fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['email'] = user['email']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password!', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    evaluations = conn.execute('SELECT * FROM evaluations WHERE user_id = ? ORDER BY timestamp DESC',
                               (session['user_id'],)).fetchall()
    conn.close()

    return render_template('dashboard.html', evaluations=evaluations, json=json)


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if ('question_paper' not in request.files or
            'answer_key' not in request.files or
            'student_answer' not in request.files):
            flash('All files are required!', 'error')
            return redirect(request.url)

        qp_file = request.files['question_paper']
        ak_file = request.files['answer_key']
        sa_file = request.files['student_answer']

        if (qp_file.filename == '' or
            ak_file.filename == '' or
            sa_file.filename == ''):
            flash('No selected files!', 'error')
            return redirect(request.url)

        if (not allowed_file(qp_file.filename) or
            not allowed_file(ak_file.filename) or
            not allowed_file(sa_file.filename)):
            flash('Only PDF files are allowed!', 'error')
            return redirect(request.url)

        try:
            student_name, subject_name = os.path.splitext(sa_file.filename)[0].split('_')
        except ValueError:
            flash('Student answer filename must be in format: StudentName_SubjectName.pdf', 'error')
            return redirect(request.url)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        qp_filename = secure_filename(f"qp_{session['user_id']}_{qp_file.filename}")
        ak_filename = secure_filename(f"ak_{session['user_id']}_{ak_file.filename}")
        sa_filename = secure_filename(f"sa_{session['user_id']}_{sa_file.filename}")

        qp_path = os.path.join(app.config['UPLOAD_FOLDER'], qp_filename)
        ak_path = os.path.join(app.config['UPLOAD_FOLDER'], ak_filename)
        sa_path = os.path.join(app.config['UPLOAD_FOLDER'], sa_filename)

        qp_file.save(qp_path)
        ak_file.save(ak_path)
        sa_file.save(sa_path)

        try:
            evaluation_results = evaluate_exam(qp_path, ak_path, sa_path)
            text_report = generate_text_report(evaluation_results)

            conn = get_db_connection()
            conn.execute('''INSERT INTO evaluations
                          (user_id, question_paper, answer_key, student_answer,
                           student_name, subject_name, results_json)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (session['user_id'], qp_filename, ak_filename, sa_filename,
                         student_name, subject_name, json.dumps(evaluation_results)))
            conn.commit()
            evaluation_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
            conn.close()

            session['last_evaluation'] = evaluation_results
            return redirect(url_for('result', evaluation_id=evaluation_id))
        except Exception as e:
            flash(f'Evaluation error: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('evaluate.html')

@app.route('/result/<int:evaluation_id>')
def result(evaluation_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    evaluation = conn.execute('SELECT * FROM evaluations WHERE id = ? AND user_id = ?',
                             (evaluation_id, session['user_id'])).fetchone()
    conn.close()

    if not evaluation:
        flash('Evaluation not found!', 'error')
        return redirect(url_for('dashboard'))

    evaluation = dict(evaluation)
    evaluation['results_json'] = json.loads(evaluation['results_json'])

    evaluator_name = session.get('username', 'Unknown Evaluator')
    current_date = datetime.now().strftime('%Y-%m-%d')

    return render_template('result.html',
                           evaluation=evaluation,
                           evaluator_name=evaluator_name,
                           current_date=current_date)

@app.route('/print_result/<int:evaluation_id>')
def print_result(evaluation_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    evaluation = conn.execute('SELECT * FROM evaluations WHERE id = ? AND user_id = ?',
                             (evaluation_id, session['user_id'])).fetchone()
    conn.close()

    if not evaluation:
        flash('Evaluation not found!', 'error')
        return redirect(url_for('dashboard'))

    evaluation = dict(evaluation)
    evaluation['results_json'] = json.loads(evaluation['results_json'])

    evaluator_name = session.get('username', 'Unknown Evaluator')
    current_date = datetime.now().strftime('%Y-%m-%d')

    return render_template('print_result.html', evaluation=evaluation, evaluator_name=evaluator_name, current_date=current_date)

@app.route('/delete_evaluation/<int:evaluation_id>', methods=['POST'])
def delete_evaluation(evaluation_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    evaluation = conn.execute('SELECT * FROM evaluations WHERE id = ? AND user_id = ?',
                             (evaluation_id, session['user_id'])).fetchone()

    if evaluation:
        try:
            files_to_delete = [
                os.path.join(app.config['UPLOAD_FOLDER'], evaluation['question_paper']),
                os.path.join(app.config['UPLOAD_FOLDER'], evaluation['answer_key']),
                os.path.join(app.config['UPLOAD_FOLDER'], evaluation['student_answer'])
            ]

            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)

            conn.execute('DELETE FROM evaluations WHERE id = ?', (evaluation_id,))
            conn.commit()
            flash('Evaluation deleted successfully!', 'success')
        except Exception as e:
            flash(f'Error deleting evaluation: {str(e)}', 'error')
        finally:
            conn.close()
    else:
        flash('Evaluation not found or you do not have permission!', 'error')

    return redirect(url_for('dashboard'))

@app.route('/download_report/<int:evaluation_id>')
def download_report(evaluation_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    evaluation = conn.execute('SELECT * FROM evaluations WHERE id = ? AND user_id = ?',
                             (evaluation_id, session['user_id'])).fetchone()
    conn.close()

    if not evaluation:
        flash('Evaluation not found!', 'error')
        return redirect(url_for('dashboard'))

    results = json.loads(evaluation['results_json'])
    report_filename = f"evaluation_report_{evaluation['student_name']}_{evaluation['subject_name']}_{evaluation['timestamp']}.json"

    response = make_response(json.dumps(results, indent=2))
    response.headers['Content-Disposition'] = f'attachment; filename={report_filename}'
    response.headers['Content-type'] = 'application/json'

    return response

@app.template_filter('to_dict')
def to_dict_filter(value):
    try:
        if isinstance(value, str):
            return json.loads(value)
        return value or {}
    except json.JSONDecodeError:
        app.logger.error(f"Failed to decode JSON: {value[:100]}...")
        return {}

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':

    def shutdown_handler(signum, frame):
        print("\nGracefully shutting down server...")
        os._exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Wait for Flask to initialize
    time.sleep(2)

    # Set ngrok authtoken (replace with your token)
    ngrok.set_auth_token("2uwKNn9J8nw2AXAGHj5vh70x8Qz_86WyRvr8ofSTR8NGksqgT")

    # Start ngrok tunnel
    public_url = ngrok.connect(5000).public_url
    print(f" * Running on: {public_url}")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")
