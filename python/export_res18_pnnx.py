import torch
import torchvision.models as models
import torch.onnx as onnx

# 加载预训练的ResNet-18模型
resnet = models.resnet18(pretrained=True)

# 将模型设置为评估模式
resnet.eval()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 224, 224)

jit_model = torch.jit.trace(resnet, dummy_input)
jit_model.save('model_param/res18.pth') 
