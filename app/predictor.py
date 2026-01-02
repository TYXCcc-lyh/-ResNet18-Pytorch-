import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

class LocalCataractPredictor:
    """本地白内障预测器（完全不联网）"""
    
    def __init__(self, model_path=None):
        """初始化本地预测器"""
        # 设置设备（优先使用CPU，本地部署通常不用GPU）
        self.device = torch.device('cpu')
        
        # 如果没有指定模型路径，使用默认路径
        if model_path is None:
            # 查找模型文件
            possible_paths = [
                'app/models/cataract_resnet18.pth',
                'models/cataract_resnet18.pth',
                './cataract_resnet18.pth'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        self.model_path = model_path
        self.model = None
        self.classes = ['正常', '白内障']
        
        # 图片预处理（与训练时相同）
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 尝试加载模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"❌ 错误：找不到模型文件 {self.model_path}")
            print("请确保模型文件存在，或先运行训练脚本")
            return False
        
        try:
            print(f"正在加载模型: {self.model_path}")
            
            # 构建模型结构（与训练时相同）
            model = models.resnet18(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)
            )
            
            # 加载权重（map_location='cpu'确保在CPU上加载）
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()  # 设置为评估模式
            
            self.model = model
            self.classes = checkpoint.get('classes', ['白内障', '正常'])
            
            print(f"✅ 模型加载成功")
            print(f"   类别: {self.classes}")
            print(f"   设备: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {str(e)}")
            return False
    
    def predict(self, image_path):
        """预测单张图片"""
        if self.model is None:
            return {
                'success': False,
                'error': '模型未加载成功'
            }
        
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'文件不存在: {image_path}'
                }
            
            # 打开并预处理图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # 进行预测（禁用梯度计算以节省内存）
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # 获取预测结果
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = self.classes[predicted_idx.item()]
            confidence_percent = confidence.item() * 100
            
            # 获取每个类别的概率
            probs = probabilities.cpu().numpy()
            prob_normal = probs[0] * 100 if self.classes[0] == '正常' else probs[1] * 100
            prob_cataract = probs[1] * 100 if self.classes[1] == '白内障' else probs[0] * 100
            
            # 判断是否为白内障
            is_cataract = predicted_class == '白内障'
            
            # 根据置信度给出建议
            if is_cataract:
                if confidence_percent > 90:
                    recommendation = "高度疑似白内障，建议尽快就医检查"
                elif confidence_percent > 70:
                    recommendation = "疑似白内障，建议到眼科进一步检查"
                else:
                    recommendation = "有白内障迹象，建议复查或咨询医生"
            else:
                if confidence_percent > 90:
                    recommendation = "未见明显异常，建议保持良好用眼习惯"
                elif confidence_percent > 70:
                    recommendation = "基本正常，建议定期检查"
                else:
                    recommendation = "建议进一步检查以确认"
            
            return {
                'success': True,
                'prediction': predicted_class,
                'confidence': round(confidence_percent, 2),
                'is_cataract': is_cataract,
                'probability_normal': round(prob_normal, 2),
                'probability_cataract': round(prob_cataract, 2),
                'recommendation': recommendation,
                'timestamp': np.datetime64('now').astype(str)
            }
            
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return {
                'success': False,
                'error': f'预测失败: {str(e)}'
            }
    
    def predict_batch(self, image_paths):
        """批量预测"""
        results = []
        for path in image_paths:
            result = self.predict(path)
            result['filename'] = os.path.basename(path)
            results.append(result)
        return results

# 创建全局预测器实例
predictor = LocalCataractPredictor()