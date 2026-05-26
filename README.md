# Problem 2: Face Blur on Video - Detect Technologies

**Name**: Ravishankar Mahto  
**Submitted for**: Backend Engineer Position

## Approach Used
- **Face Detection**: `face_recognition` library (based on dlib)
- **Blurring**: Gaussian Blur using OpenCV
- **Why this?** Reliable on low-resolution and partial faces, easy to run.

## Features
- Detects multiple faces
- Handles side profiles and partial faces
- Adjustable blur strength
- Works on CPU

## How to Run

```bash
pip install opencv-python face-recognition
python face_blur.py