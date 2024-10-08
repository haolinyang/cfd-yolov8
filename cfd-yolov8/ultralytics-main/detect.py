import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\yanghaolin\Desktop\yolov8\test data\yolov8-TADDH-Powerful-IoU\weights\best.pt') # select your model.pt path
    # model = YOLO(r'C:\Users\yanghaolin\Desktop\yolov8\test data\yolov8s\weights\best.pt')  # select your model.pt path
    # model = YOLO(r'C:\Users\yanghaolin\Desktop\yolov8\test data\yolov8-TADDH-C2fRFCAConv-PIOU\weights\best.pt')  # select your model.pt path
    model.predict(source=r'C:\Users\yanghaolin\Desktop\yolov8\heatmap test\3',
                  imgsz=640,
                  project='runs/detect',
                  name='yolov8all2',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # visualize=True # visualize model features maps

                )