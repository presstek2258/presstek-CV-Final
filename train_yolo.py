from ultralytics import YOLO
model = YOLO("yolo11n.pt")


print("Training...")
model.train(
    data='blocks.yaml',
    epochs = 50,
    device = 0,
    imgsz = 640,
    batch = 16,
    hsv_h=0.0, # don't change colours!!
    hsv_s=0.5,
    hsv_v=0.5,
    amp=False,
    lr0=0.001,
)
print("Training Complete.")


