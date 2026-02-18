from ultralytics import YOLO

# 1. Load your existing model weights
# Using 'best.pt' allows the model to start with the knowledge it already has.
model = YOLO('best (1).pt')

# 2. Start the fine-tuning process
results = model.train(
    data='data.yaml',      # Path to your configuration file
    epochs=30,             # Fine-tuning requires fewer epochs than full training
    imgsz=640,             # Ensure this matches your original training resolution
    batch=16,              # Adjust based on your GPU/CPU memory
    lr0=0.001,             # A lower learning rate prevents "forgetting" old fire data
    lrf=0.01,              # Final learning rate factor
    augment=True,          # Keep augmentation ON to help generalize
    plots=True,            # Generates charts to track your progress
    project='fire_fix',    # Saves results in a specific folder
    name='firefighter_correction'
)

print("Fine-tuning complete. Check the 'fire_fix' folder for your new best.pt!")