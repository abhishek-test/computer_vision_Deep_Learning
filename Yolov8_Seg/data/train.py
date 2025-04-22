
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-seg.pt')
    model.train(data='C:\\Abhishek Data\\VSCode_Workspace\\Python\\DL_Practice\\Yolov8_Seg\\data\\config.yaml', epochs=500, imgsz=640)