import torch

from torchvision.models import resnet18

model = resnet18()
model.eval()
example_input = torch.rand(1,3,224,224)
traced_script_module = torch.jit.trace(model, example_input)
jit_layer1 = traced_script_module.layer1

print(jit_layer1.graph)