# Sign Language Detection Model

![Screenshot (39)](https://github.com/user-attachments/assets/6791a1ab-7ba3-4fb2-999f-0c57393caf1d)


## Overview
This project implements a real-time sign language detection system using YOLO (You Only Look Once) for object detection. The model is trained using the Roboflow dataset and deployed for real-time sign recognition via a webcam.

## Features
- **Real-time Detection**: Uses OpenCV to capture and process frames from the webcam.
- **YOLOv8 Model**: Trained with a custom sign recognition dataset.
- **Automatic Annotation**: The detected signs are overlaid on the webcam feed.
- **Model Training & Deployment**: The model is trained using YOLOv8 and exported for inference.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install ultralytics roboflow opencv-python
```

### GPU Support (Optional)
If you are using a GPU, verify that your system has CUDA installed:
```bash
!nvidia-smi
```

## Dataset Preparation
The dataset is obtained from Roboflow. Download the dataset using:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your_workspace").project("your_project")
version = project.version(2)
dataset = version.download("yolov8")
```

## Model Training
Train the YOLOv8 model using the following command:
```bash
!yolo task=detect mode=train model=yolov8n.pt data=/content/Sign-recoginition-2/data.yaml epochs=30 imgsz=640 batch=16
```

## Model Inference
Run inference on test images:
```bash
!yolo detect predict model=/content/runs/detect/train/weights/best.pt source=/content/Sign-recoginition-2/valid/images save=True
```

## Real-time Detection
The following script captures live webcam feed and runs the YOLO model on each frame:
```python
from ultralytics import YOLO
import cv2
import os

model_path = r'E:\Projects\DETECTION\my_model\train\weights\best.pt'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please check the path.")
    exit(1)

model = YOLO(model_path)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('Real-Time Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Model Export
Export the trained model for future use:
```bash
!cp /content/runs/detect/train/weights/best.pt /content/my_model/sign_recognition.pt
!cp -r /content/runs/detect/train /content/my_model
```

## Zipping Model Files
```bash
%cd my_model
!zip /content/my_model.zip sign_recognition.pt
!zip -r /content/my_model.zip train
%cd /content
```

## Usage
1. Train the model using your dataset.
2. Run real-time detection using the provided script.
3. Use the trained weights for inference on new images.

## License
This project is open-source and available for use under the MIT License.

## Contact
For any questions or contributions, feel free to open an issue on the repository or reach out to the project maintainer.

---
This README provides a comprehensive guide to setting up, training, and deploying a sign language detection model using YOLO.

