import cv2
from ultralytics import YOLO
import time

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use a YOLOv8 model trained for object detection

# Function to detect soil displacement (blast) in a frame
def detect_blast(frame):
    results = model(frame)
    detected_blasts = []
    for result in results[0].boxes:  # Accessing boxes from results (xywh format)
        # `boxes` contains the x, y, width, height, confidence, and class_id
        x, y, w, h = result.xywh[0].tolist()  # Get coordinates and dimensions
        conf = result.conf[0].item()  # Confidence score
        class_id = int(result.cls[0].item())  # Class ID

        if class_id == 0:  # Replace 0 with the correct class ID for soil displacement
            detected_blasts.append((x, y, w, h, conf))
    
    return detected_blasts

# Process the video and detect blasts
def process_video(video_path, window_width=1280, window_height=720):
    cap = cv2.VideoCapture(video_path)
    blast_times = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize a variable to store the first blast time
    first_blast_time = None  # Variable to store the time of the first blast

    previous_blast_detected = False  # To track if a blast was previously detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame number
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Calculate the time of the current frame based on frame number and frame rate
        current_time = current_frame / frame_rate

        # Detect blast in the current frame
        blasts = detect_blast(frame)

        # If a blast is detected, add the time to blast_times
        if blasts:
            blast_times.append(current_time)

        # Draw bounding boxes around detected blasts
        for blast in blasts:
            x, y, w, h, conf = blast
            x, y, w, h = map(int, [x - w/2, y - h/2, w, h])  # YOLOxywh -> (x1, y1, x2, y2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

        # Resize frame to fit the laptop window size
        frame_resized = cv2.resize(frame, (window_width, window_height))

        # Show the frame with bounding boxes (scaled to laptop window size)
        cv2.imshow('Frame', frame_resized)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Get the first blast time (if any)
    if blast_times:
        first_blast_time = blast_times[0]
    
    return blast_times, first_blast_time

# Example usage
video_path = '/media/roxy/Lynx/Landmine/Data2of3/C1_352_121/352_121.MP4'  # Replace with your video file path
blast_times, first_blast_time = process_video(video_path)

# Print all blast times detected in the video
print("Blast detected at the following times (in seconds):")
for blast_time in blast_times:
    print(f"{blast_time:.2f}")

# Print the first blast time separately after all times
if first_blast_time is not None:
    print(f"The first blast time is at {first_blast_time:.2f} seconds")
