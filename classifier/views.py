from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from django.core.mail import send_mail
from django.contrib.auth.tokens import default_token_generator
from django.contrib import messages

from .forms import ImageUploadForm
from .models import Prediction

from ultralytics import YOLO
from PIL import Image
import os
import time
from django.conf import settings
from .forms import CustomUserCreationForm

# Load YOLO model
model_path = os.path.join(settings.BASE_DIR, 'classifier', 'kidney_model.pt')
model = YOLO(model_path)
categories = list(model.names.values())

# ------------------- Signup with Email Verification ------------------------


def landing_redirect(request):
    if request.user.is_authenticated:
        return redirect('upload')  # Go to prediction page
    else:
        return redirect('login')  # Go to login if not logged in


def signup_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = form.cleaned_data['email']  # Save email
            user.is_active = False  # Deactivate until email verified
            user.save()

            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            current_site = get_current_site(request)
            verification_link = f"http://{current_site.domain}/activate/{uid}/{token}/"

            subject = 'Verify your Kidney Disease Classifier account'
            message = render_to_string('email_verification.html', {
                'user': user,
                'verification_link': verification_link,
            })

            send_mail(subject, message, 'shoaibahmadbaig015@gmail.com', [user.email], fail_silently=False)
            return HttpResponse("✅ Please check your email to verify your account.")
    else:
        form = CustomUserCreationForm()
    return render(request, 'signup.html', {'form': form})

# ------------------- Email Activation ------------------------

def activate_account(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)  # Log the user in immediately after activation
        messages.success(request, "✅ Your account has been activated and you are now logged in.")
        return redirect('upload')  # Redirect to prediction/upload page
    else:
        return HttpResponse("❌ Activation link is invalid or has expired.")
# ------------------- Prediction History ------------------------

@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'history.html', {'predictions': predictions})


# ------------------- Upload & Predict ------------------------

@login_required
def upload_image(request):
    prediction = None
    confidence = None
    all_confidences = None
    image_url = None
    time_taken = None
    form = ImageUploadForm()

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img = Image.open(img_file).convert("RGB")

            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            media_path = os.path.join(settings.MEDIA_ROOT, img_file.name)
            with open(media_path, 'wb+') as destination:
                for chunk in img_file.chunks():
                    destination.write(chunk)
            image_url = os.path.join(settings.MEDIA_URL, img_file.name)

            start = time.time()
            results = model(img)
            end = time.time()
            time_taken = round(end - start, 2)

            if hasattr(results[0], "probs") and results[0].probs is not None:
                probs_tensor = results[0].probs.data
                class_idx = results[0].probs.top1
                top1_conf = results[0].probs.top1conf.item() if results[0].probs.top1conf is not None else probs_tensor[class_idx].item()

                prediction = categories[class_idx]
                confidence = round(top1_conf * 100, 2)
                all_confidences = [round(p * 100, 2) for p in probs_tensor.tolist()]

                Prediction.objects.create(
                    user=request.user,
                    image=img_file,
                    image_name=img_file.name,
                    prediction=prediction,
                    confidence=confidence,
                    all_confidences=all_confidences,
                    time_taken=time_taken,
                    model_version='YOLOv8'
                )

    return render(request, 'upload.html', {
        'form': form,
        'image_url': image_url,
        'prediction': prediction,
        'confidence': confidence,
        'labels': categories,
        'confidences': all_confidences,
        'time_taken': time_taken
    })
