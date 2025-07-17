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

from .forms import ImageUploadForm, CustomUserCreationForm
from .models import Prediction

from ultralytics import YOLO
from PIL import Image
import os
import time
from django.conf import settings

# Load YOLO model once
model_path = os.path.join(settings.BASE_DIR, 'classifier', 'kidney_model.pt')
model = YOLO(model_path)
categories = list(model.names.values())

def landing_redirect(request):
    """Redirect users to login or upload page based on authentication."""
    if request.user.is_authenticated:
        return redirect('upload')
    return redirect('login')

def signup_view(request):
    """Handle user signup with email verification."""
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = form.cleaned_data['email']
            user.is_active = False
            user.save()

            # Generate verification link
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            current_site = get_current_site(request)
            link = f"http://{current_site.domain}/activate/{uid}/{token}/"

            # Send verification email
            subject = 'Verify your Kidney Disease Classifier account'
            message = render_to_string('email_verification.html', {
                'user': user,
                'verification_link': link,
            })
            send_mail(subject, message,
                      settings.EMAIL_HOST_USER,
                      [user.email],
                      fail_silently=False)

            return HttpResponse("✅ Check your email to verify your account.")
    else:
        form = CustomUserCreationForm()
    return render(request, 'signup.html', {'form': form})

def activate_account(request, uidb64, token):
    """Activate a user’s account via the emailed link."""
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)
        messages.success(request, " Your account is now active!")
        return redirect('upload')
    return HttpResponse(" Activation link is invalid or expired.")

@login_required
def history_view(request):
    """Display the logged-in user’s prediction history."""
    predictions = Prediction.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'history.html', {'predictions': predictions})

from django.contrib.auth.decorators import login_required, user_passes_test

@login_required
@user_passes_test(lambda u: u.is_staff)
def doctor_history_view(request):
    predictions = Prediction.objects.select_related('user').order_by('-timestamp')
    return render(request, 'history.html', {'predictions': predictions})

@login_required
def upload_image(request):
    """Handle image upload, prediction, and result display."""
    form = ImageUploadForm()
    prediction = confidence = all_confidences = image_url = time_taken = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        patient_name = request.POST.get('patient_name')
        patient_id = request.POST.get('patient_id')

        if form.is_valid():
            img_file = form.cleaned_data['image']
            img = Image.open(img_file).convert("RGB")

            # Save uploaded file
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            media_path = os.path.join(settings.MEDIA_ROOT, img_file.name)
            with open(media_path, 'wb+') as dest:
                for chunk in img_file.chunks():
                    dest.write(chunk)
            image_url = os.path.join(settings.MEDIA_URL, img_file.name)

            # Run prediction
            start = time.time()
            results = model(img)
            time_taken = round(time.time() - start, 2)

            # Extract probabilities
            if hasattr(results[0], "probs") and results[0].probs is not None:
                probs = results[0].probs.data
                idx = results[0].probs.top1
                top_conf = (results[0].probs.top1conf.item()
                            if results[0].probs.top1conf is not None
                            else probs[idx].item())

                prediction = categories[idx]
                confidence = round(top_conf * 100, 2)
                all_confidences = [round(p * 100, 2) for p in probs.tolist()]

                # Save prediction record
                Prediction.objects.create(
                    user=request.user,
                    patient_name=patient_name,
                    patient_id=patient_id,
                    image=img_file,
                    image_name=img_file.name,
                    prediction=prediction,
                    confidence=confidence,
                    all_confidences=all_confidences,
                    time_taken=time_taken,
                    model_version='YOLOv11'
                )

    return render(request, 'upload.html', {
        'form': form,
        'image_url': image_url,
        'prediction': prediction,
        'confidence': confidence,
        'labels': categories,
        'confidences': all_confidences,
        'time_taken': time_taken,
    })
