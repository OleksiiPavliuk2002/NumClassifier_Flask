import argparse
import gzip
import os
import pickle
import sqlite3
from glob import glob

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'mnist_results.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mlp_classifier_model.pkl')
MNIST_DIR = os.path.join(BASE_DIR, 'mnist')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')


def load_mnist_labels(path, kind='t10k'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
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


def insert_prediction(conn, image_path, true_label, predicted_label):
    conn.execute(
        'INSERT INTO predictions (image_path, true_label, predicted_label) VALUES (?, ?, ?)',
        (image_path, true_label, predicted_label),
    )


def create_database(db_path=DB_PATH):
    init_db(db_path)
    labels = load_mnist_labels(MNIST_DIR, kind='t10k')
    model = load_model(MODEL_PATH)

    image_pattern = os.path.join(IMAGES_DIR, 'mnist-*.png')
    image_files = sorted(glob(image_pattern))
    if not image_files:
        raise FileNotFoundError(f'No MNIST PNG files found in {IMAGES_DIR}')

    conn = sqlite3.connect(db_path)
    conn.execute('DELETE FROM predictions')

    for idx, image_file in enumerate(image_files):
        relative_path = os.path.relpath(image_file, BASE_DIR)
        true_label = int(labels[idx]) if idx < len(labels) else None
        predicted_label = None

        # load image as flattened vector using simple grayscale import
        with Image.open(image_file) as img:
            img = img.convert('L').resize((28, 28))
            data = np.asarray(img, dtype=np.float64).reshape(1, -1)
        predicted_label = int(model.predict(data)[0])

        insert_prediction(conn, relative_path, true_label, predicted_label)

    conn.commit()
    conn.close()
    print(f'Created database at {db_path} with {len(image_files)} records.')


try:
    from PIL import Image
except ImportError:
    raise ImportError('Pillow is required: install with pip install pillow')


def query_total(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.execute('SELECT COUNT(*) FROM predictions')
    total = cur.fetchone()[0]
    conn.close()
    return total


def query_accuracy(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.execute('SELECT COUNT(*) FROM predictions WHERE true_label IS NOT NULL AND true_label = predicted_label')
    correct = cur.fetchone()[0]
    total_cur = conn.execute('SELECT COUNT(*) FROM predictions WHERE true_label IS NOT NULL')
    total = total_cur.fetchone()[0]
    conn.close()
    return correct, total, correct / total if total else 0.0


def query_count_by_label(label, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.execute('SELECT COUNT(*) FROM predictions WHERE true_label = ?', (label,))
    count = cur.fetchone()[0]
    conn.close()
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST SQLite database utilities')
    parser.add_argument('--create-db', action='store_true', help='Build the predictions database from images and model')
    parser.add_argument('--count', action='store_true', help='Print total number of records in the database')
    parser.add_argument('--accuracy', action='store_true', help='Print classifier accuracy from the database')
    parser.add_argument('--count-label', type=int, help='Print records count for a specific true label')
    args = parser.parse_args()

    if args.create_db:
        create_database()
    elif args.count:
        total = query_total()
        print('Total records:', total)
    elif args.accuracy:
        correct, total, accuracy = query_accuracy()
        print('Correct:', correct)
        print('Total with true label:', total)
        print(f'Accuracy: {accuracy * 100:.2f}%')
    elif args.count_label is not None:
        count = query_count_by_label(args.count_label)
        print(f'Records with true_label={args.count_label}:', count)
    else:
        parser.print_help()
