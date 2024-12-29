import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_blast_in_video(video_path, frame_skip=3, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    brightness_values = []
    motion_magnitudes = []
    frame_times = []
    frame_count = 0
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the video.")
        return
    
    prev_frame = cv2.resize(prev_frame, None, fx=resize_factor, fy=resize_factor)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        brightness_values.append(brightness)
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
        motion_magnitudes.append(motion_magnitude)
        
        frame_times.append(frame_count / fps)
        prev_frame_gray = frame_gray
        frame_count += 1
    
    cap.release()
    
    # Normalize differences to increase sensitivity
    brightness_values = np.array(brightness_values)
    motion_magnitudes = np.array(motion_magnitudes)
    frame_times = np.array(frame_times)
    
    differences = (np.abs(np.diff(brightness_values)) - np.mean(brightness_values)) / np.std(brightness_values)
    motion_differences = (np.abs(np.diff(motion_magnitudes)) - np.mean(motion_magnitudes)) / np.std(motion_magnitudes)
    
    window_size = max(1, int(fps / frame_skip))
    brightness_rolling_avg = np.convolve(differences, np.ones(window_size)/window_size, mode='valid')
    motion_rolling_avg = np.convolve(motion_differences, np.ones(window_size)/window_size, mode='valid')
    
    brightness_threshold = np.mean(brightness_rolling_avg) + 1.0 * np.std(brightness_rolling_avg)
    motion_threshold = np.mean(motion_rolling_avg) + 1.0 * np.std(motion_rolling_avg)
    
    print(f"Brightness Threshold: {brightness_threshold}")
    print(f"Motion Threshold: {motion_threshold}")
    
    # Debugging: Print rolling averages
    print("\nBrightness Rolling Average:", brightness_rolling_avg)
    print("\nMotion Rolling Average:", motion_rolling_avg)
    
    # Detect significant changes
    significant_changes = np.where((brightness_rolling_avg > brightness_threshold) & 
                                    (motion_rolling_avg > motion_threshold))[0]
    
    blast_moments = []
    if len(significant_changes) > 0:
        current_group = [significant_changes[0]]
        for i in range(1, len(significant_changes)):
            if significant_changes[i] - significant_changes[i - 1] <= window_size:
                current_group.append(significant_changes[i])
            else:
                blast_moments.append(int(np.mean(current_group)))
                current_group = [significant_changes[i]]
        if current_group:
            blast_moments.append(int(np.mean(current_group)))
    
    # Plot rolling averages and thresholds
    plt.figure(figsize=(10, 5))
    plt.plot(brightness_rolling_avg, label='Brightness Rolling Avg', color='blue')
    plt.axhline(y=brightness_threshold, color='red', linestyle='--', label='Brightness Threshold')
    plt.plot(motion_rolling_avg, label='Motion Rolling Avg', color='green')
    plt.axhline(y=motion_threshold, color='orange', linestyle='--', label='Motion Threshold')
    plt.legend()
    plt.title('Rolling Averages and Thresholds')
    plt.xlabel('Frame Index')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
    
    # Visualization of brightness and detected blasts
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(frame_times, brightness_values, label='Brightness')
    plt.title('Brightness Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Brightness')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(frame_times[1:], differences, label='Brightness Differences')
    plt.axhline(y=brightness_threshold, color='g', linestyle='--', label='Brightness Threshold')
    for moment in blast_moments:
        plt.axvline(x=frame_times[moment], color='r', linestyle='--', label='Blast Detected')
        print(f"Potential blast detected at {frame_times[moment]:.2f} seconds")
    plt.title('Brightness Differences')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Difference Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDetailed Analysis:")
    for moment in blast_moments:
        time = frame_times[moment]
        before_brightness = brightness_values[max(0, moment - window_size):moment].mean() if moment > 0 else 0
        during_brightness = brightness_values[moment:min(len(brightness_values), moment + window_size)].mean()
        motion_change = motion_differences[moment - 1] if moment > 0 else 0
        print(f"\nBlast detected at {time:.2f} seconds (frame {moment}):")
        print(f"Brightness before: {before_brightness:.2f}")
        print(f"Brightness during: {during_brightness:.2f}")
        print(f"Motion change magnitude: {motion_change:.2f}")
    
    return blast_moments

# Example usage
video_path = "D:\\PROJECTS\\Azonic\\Data1of3\\C1_352_103\\352_103.MP4"
blast_moments = detect_blast_in_video(video_path)
