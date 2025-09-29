from django.db import models
from django.contrib.auth.models import User
import datetime
import random

def generate_patient_id():
    year = datetime.datetime.now().year
    rand = random.randint(1000, 9999)
    return f"KD-{year}-{rand}"


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # doctor
    patient_name = models.CharField(max_length=100)
    patient_id = models.CharField(max_length=50, unique=True, default=generate_patient_id)
    image = models.ImageField(upload_to='uploads/')
    image_name = models.CharField(max_length=255, null=True, blank=True)
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField()
    all_confidences = models.JSONField(null=True, blank=True)
    time_taken = models.FloatField()
    model_version = models.CharField(max_length=20, default="YOLOv8")
    timestamp = models.DateTimeField(auto_now_add=True)
    disease_area_pixels = models.IntegerField(null=True, blank=True)
    disease_area_mm2 = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.patient_name} ({self.prediction} - {self.confidence}%)"

