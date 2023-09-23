import torch
import torchvision.models as models
import torch.onnx as onnx

# 加载预训练的ResNet-18模型
resnet = models.resnet18(pretrained=True)

# 将模型设置为评估模式
resnet.eval()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 224, 224)

# 使用torch.onnx.export函数导出模型为ONNX格式
onnx_file_path = "../model_param/resnet18.onnx"
onnx.export(resnet, dummy_input, onnx_file_path)

print("ResNet-18模型已成功导出为ONNX格式：", onnx_file_path)