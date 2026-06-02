# Python CV Attendance Project

A simple Python OpenCV attendance system that can register faces and mark attendance from a webcam.

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Register a new person

```bash
python main.py --mode register --name "John Doe"
```

This opens the camera and captures face samples for the given name.

### Run attendance mode

```bash
python main.py --mode attendance
```

This opens the camera, recognizes registered faces, and writes attendance to `attendance.csv`.

### Simple face detection

```bash
python main.py --mode detect
```

## Files generated

- `faces/` - saved face samples for registered users
- `face_recognizer.yml` - trained recognition model
- `attendance.csv` - attendance log
