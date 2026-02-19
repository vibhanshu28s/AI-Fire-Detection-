from ultralytics import YOLO

import os
def clipping():
    label_dir = '/dataset1/test/labels/'

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            path = os.path.join(label_dir, label_file)
            with open(path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                # Clip each coordinate between 0 and 1
                clipped = [str(max(0.0, min(1.0, float(x)))) if i > 0 else x for i, x in enumerate(parts)]
                new_lines.append(" ".join(clipped) + "\n")

            with open(path, 'w') as f:
                f.writelines(new_lines)

    print("Coordinates clipped to [0, 1] range.")

def train():

    model = YOLO('yolov8s.pt')

    # Train the model
    results = model.train(
        data='data3.yaml',
        epochs=100,
        imgsz=640,
        batch=16,  # Adjust based on your GPU VRAM
        # device=0,  # Use '0' for NVIDIA GPU or 'cpu'
        name='yolov8s_fire',
        patience=20
    )

def resume():
    model = YOLO('runs/detect/train/weights/last.pt')

    model.train(resume=True)

def validate():

    model = YOLO('/Users/Vibhanshu/Desktop/Projects/Fire Detection/runs/detect/train/weights/best.pt')

    metrics = model.val()


train()