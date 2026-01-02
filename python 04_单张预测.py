# 04_单张预测.py
"""
单张图片预测模块 - 使用训练好的模型预测单张图片
使用方法: python 04_单张预测.py --image 图片路径
或: python 04_单张预测.py (会使用test_images目录下的图片)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class CataractPredictor:
    """白内障预测器"""
    
    def __init__(self, model_path='models/cataract_resnet18.pth'):
        """初始化预测器"""
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = ['白内障', '正常']  # 中文显示
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"❌ 错误: 找不到模型文件 {self.model_path}")
            print("请先运行 02_模型训练.py 训练模型")
            return False
        
        try:
            # 构建模型结构
            model = models.resnet18(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)
            )
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.model = model
            print(f"✅ 模型加载成功")
            print(f"   设备: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def predict_image(self, image_path):
        """预测单张图片"""
        if not self.model:
            print("❌ 错误: 模型未加载")
            return None
        
        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # 获取预测结果
            class_index = predicted.item()
            class_name = self.classes[class_index]
            confidence = probs[0][class_index].item()
            
            # 获取所有类别的概率
            all_probs = probs[0].cpu().numpy()
            
            return {
                'image_path': image_path,
                'class_index': class_index,
                'class_name': class_name,
                'confidence': confidence,
                'all_probs': all_probs,
                'original_image': original_image
            }
            
        except Exception as e:
            print(f"❌ 预测图片失败: {e}")
            return None
    
    def visualize_prediction(self, result, save_path=None):
        """可视化预测结果"""
        if not result:
            return
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左侧：显示图片和预测结果
        axes[0].imshow(result['original_image'])
        axes[0].axis('off')
        
        # 添加预测结果文本
        prediction_text = f"预测: {result['class_name']}\n置信度: {result['confidence']:.1%}"
        axes[0].set_title('输入图片', fontsize=14, fontweight='bold')
        axes[0].text(0.5, -0.1, prediction_text, 
                    transform=axes[0].transAxes,
                    fontsize=12, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 右侧：显示概率条形图
        classes = ['白内障', '正常']
        colors = ['red', 'green']
        
        bars = axes[1].bar(classes, result['all_probs'], color=colors, alpha=0.7)
        axes[1].set_ylim([0, 1])
        axes[1].set_ylabel('概率', fontsize=12)
        axes[1].set_title('类别概率分布', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar, prob in zip(bars, result['all_probs']):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.1%}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('白内障筛查预测结果', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 预测结果图已保存到 {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def predict_and_save(self, image_path, output_dir='results/predictions'):
        """预测图片并保存结果"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 预测图片
        result = self.predict_image(image_path)
        if not result:
            return
        
        # 生成输出文件名
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_prediction.png")
        
        # 可视化并保存
        self.visualize_prediction(result, save_path=output_path)
        
        # 打印结果
        print(f"\n📊 预测结果:")
        print(f"  图片: {filename}")
        print(f"  预测类别: {result['class_name']}")
        print(f"  置信度: {result['confidence']:.1%}")
        print(f"  白内障概率: {result['all_probs'][0]:.1%}")
        print(f"  正常概率: {result['all_probs'][1]:.1%}")
        
        # 医学建议
        if result['class_name'] == '白内障':
            print(f"\n⚠️ 医学建议: 检测到白内障特征，建议进行进一步眼科检查。")
        else:
            print(f"\n✅ 医学建议: 未检测到明显白内障特征。")
        
        return result

def process_single_image(image_path, predictor):
    """处理单张图片"""
    print(f"\n处理图片: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ 错误: 找不到图片 {image_path}")
        return
    
    result = predictor.predict_and_save(image_path)
    return result

def process_directory(image_dir, predictor):
    """处理目录中的所有图片"""
    print(f"\n处理目录: {image_dir}")
    
    if not os.path.exists(image_dir):
        print(f"❌ 错误: 找不到目录 {image_dir}")
        return []
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_files:
        print(f"❌ 错误: 目录中没有找到图片文件")
        return []
    
    print(f"找到 {len(image_files)} 张图片")
    
    results = []
    for image_path in image_files:
        result = process_single_image(image_path, predictor)
        if result:
            results.append(result)
    
    return results

def batch_statistics(results):
    """批量预测统计"""
    if not results:
        return
    
    print("\n" + "="*60)
    print("批量预测统计")
    print("="*60)
    
    cataract_count = sum(1 for r in results if r['class_name'] == '白内障')
    normal_count = sum(1 for r in results if r['class_name'] == '正常')
    
    print(f"总图片数: {len(results)}")
    print(f"白内障预测数: {cataract_count}")
    print(f"正常预测数: {normal_count}")
    print(f"白内障比例: {cataract_count/len(results):.1%}")
    
    # 平均置信度
    cataract_conf = [r['confidence'] for r in results if r['class_name'] == '白内障']
    normal_conf = [r['confidence'] for r in results if r['class_name'] == '正常']
    
    if cataract_conf:
        print(f"白内障平均置信度: {np.mean(cataract_conf):.1%}")
    if normal_conf:
        print(f"正常平均置信度: {np.mean(normal_conf):.1%}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='白内障筛查 - 单张图片预测')
    parser.add_argument('--image', type=str, help='要预测的图片路径')
    parser.add_argument('--dir', type=str, default='test_images', 
                       help='包含测试图片的目录 (默认: test_images)')
    parser.add_argument('--model', type=str, default='models/cataract_resnet18.pth',
                       help='模型路径 (默认: models/cataract_resnet18.pth)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("白内障筛查 - 单张图片预测")
    print("="*70)
    
    # 创建预测器
    predictor = CataractPredictor(model_path=args.model)
    
    if not predictor.model:
        return
    
    # 创建测试图片目录
    os.makedirs('test_images', exist_ok=True)
    
    # 如果没有指定图片，检查test_images目录
    if not args.image:
        test_files = glob.glob(os.path.join('test_images', '*.*'))
        if test_files:
            print(f"发现 {len(test_files)} 张测试图片")
            for i, img in enumerate(test_files[:3]):  # 只显示前3张
                print(f"  {i+1}. {os.path.basename(img)}")
            print("\n请输入 --image 参数指定图片，或将要预测的图片放入 test_images 目录")
            print("示例: python 04_单张预测.py --image test_images/your_image.jpg")
        else:
            print("\n❌ 没有找到测试图片")
            print("请将要预测的图片放入 test_images 目录，或使用 --image 参数指定图片")
        return
    
    # 处理单个图片
    if args.image:
        result = process_single_image(args.image, predictor)
        if result:
            print(f"\n✅ 预测完成!")
    else:
        # 处理整个目录
        results = process_directory(args.dir, predictor)
        if results:
            batch_statistics(results)

if __name__ == "__main__":
    main()