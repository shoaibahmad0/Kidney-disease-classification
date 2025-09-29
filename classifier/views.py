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
from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from .forms import ImageUploadForm, CustomUserCreationForm
from .models import Prediction
import cv2
from ultralytics import YOLO
from PIL import Image
import os
import time
from django.conf import settings
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from .yolo_model import yolo_model
from .resnet_model import load_resnet_model, predict_with_resnet, RESNET_CLASSES, resnet_transform
from .resnet_gradcam import generate_gradcam
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resnet_path = os.path.join(settings.BASE_DIR, 'classifier', 'kidney_classifier_resnet18.pth')
resnet_model, resnet_device = load_resnet_model(resnet_path)


def generate_gradcam(model, input_tensor, class_idx):
    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    final_conv = model.layer4[-1].conv2
    final_conv.register_forward_hook(save_activations)
    final_conv.register_full_backward_hook(save_gradients)  # updated for new PyTorch

    model.zero_grad()
    output = model(input_tensor)
    class_score = output[0, class_idx]
    class_score.backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap = heatmap / torch.max(heatmap)
    return heatmap.cpu()

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

 

def delete_prediction(request, record_id):
    if request.method == 'POST':
        record = get_object_or_404(Prediction, id=record_id)
        record.delete()
        messages.success(request, "Prediction record deleted successfully.")
    return redirect('history')



from django.contrib.auth.decorators import login_required, user_passes_test

@login_required
@user_passes_test(lambda u: u.is_staff)
def doctor_history_view(request):
    predictions = Prediction.objects.select_related('user').order_by('-timestamp')
    return render(request, 'history.html', {'predictions': predictions})

@login_required
def upload_image(request):
    form = ImageUploadForm()
    prediction = confidence = all_confidences = image_url = time_taken = heatmap_url = None
    disease_area_pixels = disease_area_mm2 = None
    output_filename = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        patient_name = request.POST.get('patient_name')

        if form.is_valid():
            img_file = form.cleaned_data['image']
            img = Image.open(img_file).convert("RGB")

            # Save uploaded image
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            media_path = os.path.join(settings.MEDIA_ROOT, img_file.name)
            with open(media_path, 'wb+') as dest:
                for chunk in img_file.chunks():
                    dest.write(chunk)
            image_url = os.path.join(settings.MEDIA_URL, img_file.name)

            # -------- YOLOv11 Prediction --------
            start = time.time()
            results = yolo_model(img)
            time_taken = round(time.time() - start, 2)

            if hasattr(results[0], "probs") and results[0].probs is not None:
                probs = results[0].probs.data
                idx = results[0].probs.top1
                top_conf = (results[0].probs.top1conf.item()
                            if results[0].probs.top1conf is not None
                            else probs[idx].item())

                prediction = RESNET_CLASSES[idx]
                confidence = round(top_conf * 100, 2)
                all_confidences = [round(p * 100, 2) for p in probs.tolist()]

            # Save prediction to DB
            prediction_obj = Prediction.objects.create(
                user=request.user,
                patient_name=patient_name,
                image=img_file,
                image_name=img_file.name,
                prediction=prediction,
                confidence=confidence,
                all_confidences=all_confidences,
                time_taken=time_taken,
                model_version='YOLOv11'
            )

            patient_id = prediction_obj.patient_id

            # -------- Grad-CAM with ResNet --------
            resnet_input = resnet_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                resnet_output = resnet_model(resnet_input)
                predicted_class = resnet_output.argmax(dim=1).item()

            heatmap = generate_gradcam(resnet_model, resnet_input, predicted_class)
            heatmap_np = heatmap.cpu().numpy()

            # Normalize heatmap if needed
            if heatmap_np.max() > 1:
                heatmap_np = heatmap_np / 255.0

            # Resize to match original image size
            heatmap_resized = cv2.resize(heatmap_np, (img.width, img.height))

            # -------- Improve Activation Detection --------
            blurred = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
            threshold = 0.5
            binary_mask = blurred > threshold

            # -------- Area Calculation --------
            disease_area_pixels = np.sum(binary_mask)

            # ✅ Best medically close estimate (CT-scale): 0.264 mm/pixel
            mm_per_pixel = 0.264
            disease_area_mm2 = disease_area_pixels * (mm_per_pixel ** 2)

            # -------- Save Grad-CAM Overlay --------
            original_image = Image.open(media_path).convert('RGB')
            gradcam_image = apply_heatmap_on_image(torch.tensor(heatmap_resized), original_image)

            output_filename = f"gradcam_{patient_id}_{int(time.time())}.jpg"
            gradcam_path = os.path.join(settings.MEDIA_ROOT, 'gradcams')
            os.makedirs(gradcam_path, exist_ok=True)

            gradcam_filepath = os.path.join(gradcam_path, output_filename)
            cv2.imwrite(gradcam_filepath, gradcam_image)
            heatmap_url = os.path.join(settings.MEDIA_URL, 'gradcams', output_filename)

            # Save Area to DB
            if prediction_obj:
                prediction_obj.disease_area_pixels = int(disease_area_pixels)
                prediction_obj.disease_area_mm2 = round(disease_area_mm2, 2)
                prediction_obj.save()

    return render(request, 'upload.html', {
        'form': form,
        'image_url': image_url,
        'prediction': prediction,
        'confidence': confidence,
        'labels': RESNET_CLASSES,
        'confidences': all_confidences,
        'disease_area_pixels': disease_area_pixels,
        'disease_area_mm2': disease_area_mm2,
        'time_taken': time_taken,
        'heatmap_url': heatmap_url,
    })


def apply_heatmap_on_image(heatmap, original_img):
    # Resize heatmap to match input image size
    heatmap = cv2.resize(heatmap.numpy(), (original_img.size[0], original_img.size[1]))

    # Normalize and apply color map
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert PIL image to OpenCV format
    img = np.array(original_img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Overlay the heatmap onto the image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    return superimposed_img
