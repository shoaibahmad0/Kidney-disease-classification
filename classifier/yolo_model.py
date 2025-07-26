from ultralytics import YOLO
import os
from django.conf import settings

# Load YOLOv11 model
YOLO_MODEL_PATH = os.path.join(settings.BASE_DIR, 'classifier', 'kidney_model.pt')
yolo_model = YOLO(YOLO_MODEL_PATH)
