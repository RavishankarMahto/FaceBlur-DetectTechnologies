import cv2
import numpy as np

# Load pre-trained face detection model (comes with OpenCV)
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def blur_faces_in_frame(frame, blur_strength=25):
    if frame is None or frame.size == 0:
        return frame
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:   # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Add padding
            startX = max(0, startX - 10)
            startY = max(0, startY - 10)
            endX = min(w, endX + 10)
            endY = min(h, endY + 10)
            
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size > 0:
                blurred = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
                frame[startY:endY, startX:endX] = blurred
    return frame


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"🎥 Processing: {input_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = blur_faces_in_frame(frame)
        out.write(processed)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"✅ Done! Saved: {output_path}")


# ====================== RUN ======================
if __name__ == "__main__":
    input_video = "input.mp4"           # Make sure this file exists
    output_video = "blurred_video.avi"
    process_video(input_video, output_video)