from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data=r"/content/drive/MyDrive/Machine-Learning/dataset_900/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    workers=2,
    optimizer='Adam',
    project='car_damage',
    name='yolov8n_cardamage_additional',
    lr0=0.001,
    lrf=0.01,
    augment=True
)

model.save('/content/drive/MyDrive/Machine-Learning/dataset_900/models/model_100.pt')

metrics = model.val()
print(f"Validation metrics after additional training: {metrics}")
