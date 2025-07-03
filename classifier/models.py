from django.db import models
from django.contrib.auth.models import User

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # Optional
    image = models.ImageField(upload_to='uploads/')
    image_name = models.CharField(max_length=255, null=True, blank=True)
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField()
    all_confidences = models.JSONField(null=True, blank=True)  # all class probabilities
    time_taken = models.FloatField()
    model_version = models.CharField(max_length=20, default="YOLOv8")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction} ({self.confidence}%) at {self.timestamp}"
