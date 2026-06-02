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
