
import torch
import torchvision
from torchvision.models import resnet18

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

model = resnet18(weights=None)

print("ResNet18 loaded successfully!")
print(
    "Total parameters:",
    sum(parameter.numel() for parameter in model.parameters())
)
