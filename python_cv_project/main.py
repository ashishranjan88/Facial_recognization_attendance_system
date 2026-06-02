import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / 'faces'
MODEL_FILE = BASE_DIR / 'face_recognizer.yml'
ATTENDANCE_FILE = BASE_DIR / 'attendance.csv'
IMAGE_SIZE = (200, 200)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.pgm'}
DEFAULT_THRESHOLD = 70.0
SAMPLE_DELAY = 0.5
FIELDNAMES = ['date', 'time', 'name']


def ensure_dirs():
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_face_cascade():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if cascade.empty():
        raise RuntimeError('Could not load Haar cascade classifier.')
    return cascade


def detect_largest_face(gray_frame, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def crop_face(frame, face_rect):
    x, y, w, h = face_rect
    face = frame[y : y + h, x : x + w]
    return cv2.resize(face, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def draw_label(frame, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, 0.6, 1)
    cv2.rectangle(
        frame,
        (x, y - text_size[1] - 14),
        (x + text_size[0] + 10, y + 5),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x + 5, y - 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def get_image_paths(person_dir):
    return sorted(
        [
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def list_registered_faces():
    return sorted([folder.name for folder in FACES_DIR.iterdir() if folder.is_dir()])


def get_label_map():
    return {idx: name for idx, name in enumerate(list_registered_faces())}


def train_recognizer():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_dir in sorted(FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        image_files = get_image_paths(person_dir)
        if not image_files:
            continue

        label_map[current_label] = person_dir.name
        for image_path in image_files:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA))
            labels.append(current_label)
        current_label += 1

    if not faces:
        return None, {}

    faces = [np.asarray(face, dtype=np.uint8) for face in faces]
    labels = np.asarray(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.write(str(MODEL_FILE))
    return recognizer, label_map


def load_recognizer():
    if not MODEL_FILE.exists():
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    return recognizer, get_label_map()


def get_today_attendance():
    attendance = set()
    today = datetime.now().date().isoformat()
    if not ATTENDANCE_FILE.exists():
        return attendance
    with ATTENDANCE_FILE.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('date') == today and row.get('name'):
                attendance.add(row['name'])
    return attendance


def list_attendance(date=None):
    entries = []
    if not ATTENDANCE_FILE.exists():
        return entries
    with ATTENDANCE_FILE.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if date is None or row.get('date') == date:
                entries.append(row)
    return entries


def append_attendance(name):
    first_write = not ATTENDANCE_FILE.exists()
    with ATTENDANCE_FILE.open('a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if first_write:
            writer.writeheader()
        now = datetime.now()
        writer.writerow(
            {
                'date': now.date().isoformat(),
                'time': now.strftime('%H:%M:%S'),
                'name': name,
            }
        )


def clear_attendance(date=None):
    if not ATTENDANCE_FILE.exists():
        return False

    if date is None:
        date = datetime.now().date().isoformat()

    rows = []
    with ATTENDANCE_FILE.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('date') != date:
                rows.append(row)

    with ATTENDANCE_FILE.open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return True


def register_face(name, camera_index, sample_count, face_cascade):
    person_dir = FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {camera_index}')

    print(f'Registering face for "{name}". Press ESC to cancel.')
    saved = 0
    last_saved = 0

    while saved < sample_count:
        ret, frame = cap.read()
        if not ret:
            print('Unable to read from camera.')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = detect_largest_face(gray, face_cascade)

        if face_rect is not None:
            x, y, w, h = face_rect
            face = crop_face(gray, face_rect)
            elapsed = time.time() - last_saved
            if elapsed >= SAMPLE_DELAY:
                filename = person_dir / f'{name}_{saved + 1}.png'
                cv2.imwrite(str(filename), face)
                saved += 1
                last_saved = time.time()

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_label(frame, f'Saving {saved}/{sample_count}', x, y)
        else:
            cv2.putText(
                frame,
                'Show your face clearly to the camera',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow('Register Face', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved >= sample_count:
        print(f'Registered {saved} samples for {name}. Training recognizer...')
        train_recognizer()
        print('Training complete.')
    else:
        print('Registration incomplete. Try again.')


def attendance_loop(camera_index, face_cascade, recognizer, label_map, threshold):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {camera_index}')

    attendance = get_today_attendance()
    print('Attendance mode running. Press ESC to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Unable to read from camera.')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = detect_largest_face(gray, face_cascade)
        display_text = 'No face detected'

        if face_rect is not None:
            x, y, w, h = face_rect
            face = crop_face(gray, face_rect)
            label, confidence = recognizer.predict(face)
            name = label_map.get(label, 'Unknown')
            if confidence > threshold:
                name = 'Unknown'
            else:
                if name not in attendance:
                    append_attendance(name)
                    attendance.add(name)
                    print(f'Attendance marked: {name}')
                display_text = f'{name} ({confidence:.0f})'

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_label(frame, display_text, x, y)
        else:
            cv2.putText(
                frame,
                display_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow('Attendance', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_only(camera_index, face_cascade):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {camera_index}')
    print('Face detection mode. Press ESC to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Unable to read from camera.')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = detect_largest_face(gray, face_cascade)
        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                'Face detected',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow('Detect', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Python OpenCV attendance system.')
    parser.add_argument(
        '--mode',
        choices=['attendance', 'register', 'detect', 'list', 'clear'],
        default='attendance',
        help='Mode to run',
    )
    parser.add_argument('--name', type=str, help='Name to register (register mode only)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--samples', type=int, default=20, help='Number of face samples to capture when registering')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Recognition confidence threshold')
    parser.add_argument('--date', type=str, help='Date for list/clear modes in YYYY-MM-DD format')
    parser.add_argument('--show-all', action='store_true', help='Show all attendance records when using --mode list')
    args = parser.parse_args()

    ensure_dirs()
    face_cascade = load_face_cascade()

    if args.mode == 'register':
        if not args.name:
            parser.error('--name is required when --mode register')
        register_face(args.name.strip().replace(' ', '_'), args.camera, args.samples, face_cascade)
        return

    if args.mode == 'list':
        date = None if args.show_all else args.date
        entries = list_attendance(date)
        if not entries:
            print('No attendance records found.')
            return
        print('Attendance records:')
        for row in entries:
            print(f"{row['date']} {row['time']} - {row['name']}")
        return

    if args.mode == 'clear':
        clear_date = args.date if args.date else None
        if clear_attendance(clear_date):
            print(f'Attendance records cleared for {clear_date or datetime.now().date().isoformat()}.')
        else:
            print('No attendance records to clear.')
        return

    if args.mode == 'detect':
        detect_only(args.camera, face_cascade)
        return

    recognizer, label_map = load_recognizer()
    if recognizer is None or not label_map:
        recognizer, label_map = train_recognizer()
    if recognizer is None or not label_map:
        raise RuntimeError('No registered faces found. Run with --mode register --name NAME first.')

    attendance_loop(args.camera, face_cascade, recognizer, label_map, args.threshold)


if __name__ == '__main__':
    main()

# Here are the commands for training and running the attendance system:

# -------------------------------
# Training (Register a new face)
# -------------------------------

# cd /Users/ashishranjan/Desktop/smile-attendance-main/python_cv_project
# source /Users/ashishranjan/Desktop/smile-attendance-main/.venv/bin/activate
# python main.py --mode register --name "John_Doe" --samples 20

# Replace "John_Doe" with the person's name (spaces will be converted to underscores)
# Adjust --samples to change number of face images to capture (default: 20)


# -------------------------------
# Running (Start attendance)
# -------------------------------

# cd /Users/ashishranjan/Desktop/smile-attendance-main/python_cv_project
# source /Users/ashishranjan/Desktop/smile-attendance-main/.venv/bin/activate
# python main.py --mode attendance


# -------------------------------
# Other useful commands
# -------------------------------

# python main.py --mode detect
# → Just detect faces without recording attendance

# python main.py --mode list
# → View today's attendance

# python main.py --mode list --show-all
# → View all attendance records

# python main.py --mode clear
# → Clear today's attendance

# python main.py --mode clear --date 2026-04-27
# → Clear specific date's attendance

# python main.py --camera 1
# → Use a different camera index

# python main.py --threshold 60
# → Lower threshold = stricter matching