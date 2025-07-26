import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Class labels must match the training order
RESNET_CLASSES = ['cyst', 'normal', 'stone', 'tumor']

# Image transforms for ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the pre-trained ResNet18 model
def load_resnet_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(RESNET_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device), device

# Predict with ResNet18
def predict_with_resnet(img, model, device):
    img_tensor = resnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        top1_idx = torch.argmax(probs).item()
        top1_conf = probs[top1_idx].item()
        return RESNET_CLASSES[top1_idx], top1_conf * 100, probs.cpu().numpy() * 100