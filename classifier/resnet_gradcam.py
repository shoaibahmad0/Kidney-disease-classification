import torch
import torch.nn.functional as F

# Grad-CAM generation for ResNet18
def generate_gradcam(model, input_tensor, class_idx):
    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    # ✅ Hook into the full last block of layer4
    
    target_layer = model.layer3[-1]  # Higher spatial resolution, e.g., [1, 256, 14, 14]

    target_layer.register_forward_hook(save_activations)
    target_layer.register_full_backward_hook(save_gradients)

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
