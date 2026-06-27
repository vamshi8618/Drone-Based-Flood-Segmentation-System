from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
import sqlite3
import numpy as np
import cv2

from drone_control import DroneController
from vision_pipeline import UNetSegmenter
from imu_module import IMUProcessor

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'your_secret_key'

# Configure paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'unet_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Hardware and vision modules
DRONE_HOST = os.getenv('DRONE_HOST', '192.168.4.1')
DRONE_PORT = int(os.getenv('DRONE_PORT', '80'))

drone_controller = DroneController(host=DRONE_HOST, port=DRONE_PORT)
segmenter = UNetSegmenter(MODEL_PATH)
imu_processor = IMUProcessor()

# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def preprocess_image(image_path):
    """Preprocess the uploaded image for the model."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0) / 255.0
    return image

def postprocess_mask(mask):
    """Postprocess the predicted mask."""
    mask = (mask.squeeze() > 0.5).astype(np.uint8) * 255
    return mask

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (name, email, username, password) VALUES (?, ?, ?, ?)',
                      (name, email, username, password))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or Email already exists.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['POST'])
def handle_login():
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()

    if user:
        session['user'] = username
        flash('Welcome, ' + username + '!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid username or password!', 'error')
        return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Please log in first!', 'error')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/api/command', methods=['POST'])
def api_command():
    if 'user' not in session:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json(silent=True) or {}
    command = data.get('command')
    params = data.get('params', {})

    if not command:
        return jsonify({'error': 'Command not provided'}), 400

    try:
        result = drone_controller.send_command(command, params)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/telemetry')
def api_telemetry():
    if 'user' not in session:
        return jsonify({'error': 'Authentication required'}), 401

    try:
        data = drone_controller.get_telemetry()
        return jsonify({'status': 'ok', 'telemetry': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/process', methods=['GET', 'POST'])
def process():
    if 'user' not in session:
        flash('Please log in first!', 'error')
        return redirect(url_for('login'))

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        flash('Please log in first!', 'error')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)

    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the image
    image = preprocess_image(file_path)
    mask = segmenter.predict_mask(cv2.imread(file_path))

    # Save result
    result_path = os.path.join(app.config['RESULT_FOLDER'], f"masked_{file.filename}")
    cv2.imwrite(result_path, mask)

    return render_template(
        'result.html',
        original_url=url_for('static', filename=f"uploads/{file.filename}"),
        result_url=url_for('static', filename=f"results/masked_{file.filename}")
    )

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    init_db()  # Initialize database
    app.run(debug=True)
