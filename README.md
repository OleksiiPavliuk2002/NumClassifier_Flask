# MNIST Digit Classifier Web App

Цей проект демонструє веб-додаток на Flask для класифікації рукописних цифр за допомогою MLP-моделі, навченої на датасеті MNIST.

## Що містить проект

- Flask веб-інтерфейс для завантаження PNG-зображень
- Класифікація зображень за допомогою збереженої моделі
- Збереження результатів у SQLite базі даних
- Генерація та використання MNIST-зображень для тестування
- Notebook з експериментами над класифікацією цифр

## Основні файли

- `app.py` — Flask-додаток
- `mnist_db.py` — створення бази даних та запити до неї
- `generate_custom_images.py` — генерація 10 тестових зображень цифр
- `regenerate_mnist_pngs.py` — перегенерація MNIST PNG з IDX-даних
- `templates/index.html` — HTML-сторінка інтерфейсу
- `templates/style.css` — стилі веб-додатку
- `Numbers_Classification.ipynb` — ноутбук з навчанням/експериментами
- `requirements.txt` — залежності проекту

## Вимоги

Потрібен Python 3.10 або новіший.

## Встановлення

1. Перейдіть у папку проекту:

```bash
cd "D:/ваша директорія"
```

2. Створіть віртуальне середовище (опціонально, але рекомендовано):

```bash
python -m venv venv
```

3. Активуйте віртуальне середовище:

Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

4. Встановіть залежності:

```bash
pip install -r requirements.txt
```

## Запуск веб-додатку

```bash
python app.py
```

Після запуску відкрийте у браузері:

```text
http://127.0.0.1:5000/
```

## Робота з базою даних

Створення бази даних:

```bash
python mnist_db.py --create-db
```

Підрахунок записів:

```bash
python mnist_db.py --count
```

Точність класифікатора:

```bash
python mnist_db.py --accuracy
```

Кількість записів для конкретної мітки:

```bash
python mnist_db.py --count-label 7
```

## Генерація тестових зображень

```bash
python generate_custom_images.py
```

Це створить 10 зображень у папці `images/custom`.

## Перегенерація MNIST PNG

```bash
python regenerate_mnist_pngs.py
```

## Примітка про модель

Модель зберігається у файлі:

```text
models/mlp_classifier_model.pkl
```

Для коректної роботи важливо, щоб вхідні зображення подавалися у тому ж форматі, що й під час навчання моделі: 28×28 у градаціях сірого, зі значеннями пікселів у діапазоні 0–255.

## Структура проекту

```text
images/                # MNIST PNG зображення
images/custom/         # тестові зображення для веб-додатку
mnist/                 # MNIST IDX-файли
models/                # збережена модель
templates/             # HTML/CSS для Flask
mnist_results.db       # SQLite база з результатами
```
