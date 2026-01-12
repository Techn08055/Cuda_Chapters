import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

input_image = torch.rand(1, 3, 224, 224)

torch.onnx.export(model, input_image, "resnet18.onnx")