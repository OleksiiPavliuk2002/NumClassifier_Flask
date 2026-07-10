import gzip
import os
import pickle
import sqlite3
from datetime import datetime

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'mnist_results.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mlp_classifier_model.pkl')
MNIST_DIR = os.path.join(BASE_DIR, 'mnist')
IMAGE_STORE_DIR = os.path.join(BASE_DIR, 'images', 'custom')

os.makedirs(IMAGE_STORE_DIR, exist_ok=True)

app = Flask(__name__, template_folder='templates', static_folder='templates')
app.secret_key = 'replace-this-with-a-secure-secret'


def load_test_labels():
    labels_path = os.path.join(MNIST_DIR, 't10k-labels-idx1-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        return np.frombuffer(buffer, dtype=np.uint8)


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


MODEL = load_model()
TEST_LABELS = load_test_labels()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS predictions (
               id INTEGER PRIMARY KEY,
               image_path TEXT NOT NULL,
               true_label INTEGER,
               predicted_label INTEGER,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP
           )'''
    )
    conn.commit()
    conn.close()


def insert_prediction(image_path, true_label, predicted_label):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO predictions (image_path, true_label, predicted_label) VALUES (?, ?, ?)',
        (image_path, true_label, predicted_label),
    )
    conn.commit()
    conn.close()


def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    arr = np.asarray(image, dtype=np.float64).reshape(1, -1)
    return arr


def infer_true_label_from_name(filename):
    base = os.path.basename(filename)
    if base.startswith('mnist-') and base.endswith('.png'):
        try:
            index = int(base[6:-4]) - 1
            if 0 <= index < len(TEST_LABELS):
                return int(TEST_LABELS[index])
        except ValueError:
            return None
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        uploaded_file = request.files.get('image_file')
        if uploaded_file and uploaded_file.filename:
            filename = os.path.basename(uploaded_file.filename)
            save_path = os.path.join(IMAGE_STORE_DIR, filename)
            if os.path.exists(save_path):
                base, ext = os.path.splitext(filename)
                save_path = os.path.join(IMAGE_STORE_DIR, f'{base}_{datetime.now().strftime("%Y%m%d%H%M%S")}{ext}')
            uploaded_file.save(save_path)

            image = Image.open(save_path)
            data = preprocess_image(image)
            predicted_label = int(MODEL.predict(data)[0])
            true_label = infer_true_label_from_name(save_path)
            relative_path = os.path.relpath(save_path, BASE_DIR)
            insert_prediction(relative_path, true_label, predicted_label)
            result = {
                'image_path': relative_path,
                'true_label': true_label,
                'predicted_label': predicted_label,
            }
        else:
            flash('Будь ласка, оберіть PNG-зображення для завантаження.', 'error')
            return redirect(url_for('index'))

    return render_template('index.html', result=result)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
