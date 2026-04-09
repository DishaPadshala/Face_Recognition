# Real-Time Face Recognition System

A real-time face recognition application using a custom CNN trained on a self-collected dataset. Detects and identifies faces from a live webcam feed with confidence-based Unknown rejection and automatic attendance logging.

## Project Structure

    touchless-facial-recognition/
    ├── faces_dataset/
    ├── data/
    │   ├── test/
    │   └── train/ 
    ├── face_detector/                        
    │   ├── deploy.prototxt
    │   └── res10_300x300_ssd_iter_140000.caffemodel
    ├── gather_examples.py                    
    ├── train_recognition.py                  
    ├── final_recognition.py                  
    ├── face_recognition_model.keras          
    ├── label_encoder.pickle                  
    ├── attendance.csv                        
    └── requirements.txt

## Setup

### 1. Clone the repository
    git clone https://github.com/DishaPadshala/Face_Recognition.git
    cd Face_Recognition

### 2. Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

### 3. Install dependencies
    pip install -r requirements.txt

## Usage

### Step 1: Collect face data
    python3 gather_examples.py \
      --input videos/PersonName.mp4 \
      --output faces_dataset/PersonName \
      --detector face_detector \
      --skip 4

### Step 2: Train the model
    python3 train_recognition.py

### Step 3: Run live recognition
    python3 final_recognition.py

- Green box = recognized person
- Red box = Unknown
- Press Q to quit
- Attendance logged to attendance.csv

## Tips for Best Results
- Record in bright even lighting
- Move head naturally during recording
- Use the same webcam for recording and the live app
- Aim for 150-200 images per person

## Requirements
- Python 3.11 recommended
- Mac Linux or Windows
- Webcam
