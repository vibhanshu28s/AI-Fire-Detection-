import cv2
import threading
from ultralytics import YOLO
import playsound


# --- Setup ---
model = YOLO("yolov8n_300.pt")
# Move model to GPU if available and use half-precision for ~2x speedup
# model.to('cuda').half()

video_path = "manual_test_video/4116863-hd_1920_1080_30fps.mp4"
cap = cv2.VideoCapture(video_path)

# Shared variables between threads
latest_frame = None
detections = []
run_threads = True
alarm_active = False


def ai_inference_thread():
    """Independent thread that just runs the YOLO model as fast as possible."""
    global latest_frame, detections, alarm_active
    while run_threads:
        if latest_frame is not None:
            # imgsz=320 makes it much faster than the default 640
            results = model(latest_frame, conf=0.5, imgsz=320, verbose=False)

            temp_detections = []
            found_fire = False
            for r in results:
                for box in r.boxes:
                    found_fire = True
                    temp_detections.append({
                        "coords": list(map(int, box.xyxy[0])),
                        "conf": float(box.conf[0])
                    })

            detections = temp_detections

            # Alarm Logic Trigger
            if found_fire and not alarm_active:
                alarm_active = True
                threading.Thread(target=play_alarm, daemon=True).start()
            elif not found_fire:
                alarm_active = False


def play_alarm():
    """Plays the sound once. The AI thread restarts it if fire is still there."""
    try:
        playsound.playsound('alarm-sound.mp3', block=True)
    except:
        pass


# Start the AI thread
threading.Thread(target=ai_inference_thread, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Resize for display (keep this small for speed)
    display_frame = cv2.resize(frame, (960, 540))

    # 2. Pass the frame to the AI thread
    latest_frame = display_frame

    # 3. Draw the LATEST detections (might be from a few milliseconds ago)
    for det in detections:
        x1, y1, x2, y2 = det["coords"]
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(display_frame, f"FIRE {det['conf']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Ultra-Fast Fire Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        run_threads = False
        break

cap.release()
cv2.destroyAllWindows()