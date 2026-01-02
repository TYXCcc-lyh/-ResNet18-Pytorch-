# 02_模型训练.py
"""
模型训练模块 - 使用PyTorch和ResNet18训练白内障筛查模型

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# 尝试导入tqdm，如果不存在则提示安装
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: tqdm 模块未安装，将使用简化进度条")
    print("   请运行: pip install tqdm")
    TQDM_AVAILABLE = False
    
    # 创建简单的进度条替代
    class SimpleProgressBar:
        def __init__(self, iterable=None, desc=None):
            self.iterable = iterable
            self.desc = desc
            if desc:
                print(desc)
        
        def __iter__(self):
            for i, item in enumerate(self.iterable):
                if i % 10 == 0:
                    print(f"  处理中: {i}/{len(self.iterable)}", end='\r')
                yield item
            print(f"  处理完成: {len(self.iterable)}/{len(self.iterable)}")
        
        def __len__(self):
            return len(self.iterable)
    
    tqdm = SimpleProgressBar

def get_script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

class CataractDataset(Dataset):
    """白内障数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cataract', 'normal']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # 遍历所有类别
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ 警告: 找不到类别目录 {class_dir}")
                continue
                
            img_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not img_files:
                print(f"⚠️ 警告: 类别 {class_name} 目录中没有图片文件")
                continue
                
            for img_name in img_files:
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])
        
        if len(self.images) == 0:
            print(f"❌ 错误: 数据集目录 {root_dir} 中没有找到任何图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 错误: 无法加载图片 {img_path}: {e}")
            # 返回一个黑色图片作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """获取数据增强和转换"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def create_dataloaders(base_dir=None):
    """创建数据加载器"""
    print("加载数据集...")
    
    # 如果没有指定基础目录，使用脚本所在目录
    if base_dir is None:
        base_dir = get_script_dir()
    
    data_dir = os.path.join(base_dir, 'data')
    print(f"数据目录: {data_dir}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误: 找不到数据目录 {data_dir}")
        print("请先运行 01_数据准备.py")
        return None, None
    
    data_transforms = get_data_transforms()
    
    # 创建数据集路径
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"训练数据目录: {train_dir}")
    print(f"验证数据目录: {val_dir}")
    print(f"测试数据目录: {test_dir}")
    
    # 检查每个目录是否存在
    for dir_path, dir_name in [(train_dir, '训练'), (val_dir, '验证'), (test_dir, '测试')]:
        if not os.path.exists(dir_path):
            print(f"⚠️ 警告: 找不到{dir_name}目录 {dir_path}")
    
    # 创建数据集
    datasets = {
        'train': CataractDataset(train_dir, transform=data_transforms['train']),
        'val': CataractDataset(val_dir, transform=data_transforms['val']),
        'test': CataractDataset(test_dir, transform=data_transforms['test'])
    }
    
    # 检查数据集大小
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
    
    if dataset_sizes['train'] == 0 or dataset_sizes['val'] == 0:
        print("❌ 错误: 训练集或验证集为空")
        print("请确保数据准备脚本正确运行，并且图片格式正确")
        return None, None
    
    # 创建数据加载器
    num_workers = 0 if sys.platform == 'win32' else 2  # Windows上设置为0避免问题
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=16, shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=16, shuffle=False, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=16, shuffle=False, num_workers=num_workers)
    }
    
    print(f"训练集: {dataset_sizes['train']} 张图片")
    print(f"验证集: {dataset_sizes['val']} 张图片")
    print(f"测试集: {dataset_sizes['test']} 张图片")
    
    return dataloaders, dataset_sizes

def build_resnet18_model(num_classes=2):
    """构建ResNet18模型"""
    print("构建ResNet18模型...")
    
    try:
        # 加载预训练的ResNet18
        model = models.resnet18(pretrained=True)
    except Exception as e:
        print(f"⚠️ 警告: 无法加载预训练模型: {e}")
        print("将使用随机初始化的模型")
        model = models.resnet18(pretrained=False)
    
    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后一层全连接层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # 解冻最后两层
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=20):
    """训练模型"""
    print("开始训练模型...")
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
                if scheduler is not None:
                    scheduler.step()
            else:
                model.eval()   # 验证模式
            
            running_loss = 0.0
            running_corrects = 0
            
            # 遍历数据
            data_iter = dataloaders[phase]
            if TQDM_AVAILABLE:
                data_iter = tqdm(data_iter, desc=f'{phase.capitalize()}')
            
            for inputs, labels in data_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播+优化只在训练阶段
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item() if hasattr(epoch_acc, 'item') else epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item() if hasattr(epoch_acc, 'item') else epoch_acc)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 深拷贝最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - since
    print(f'\n训练完成! 用时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_history(history, save_dir='results'):
    """绘制训练历史图表"""
    print("绘制训练历史...")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_history.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='训练损失', linewidth=2)
    axes[0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_title('训练和验证损失', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率', linewidth=2)
    axes[1].plot(history['val_acc'], label='验证准确率', linewidth=2)
    axes[1].set_title('训练和验证准确率', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 训练历史图已保存到 {save_path}")
    plt.close()

def main():
    """主函数"""
    print("="*70)
    print("白内障筛查模型训练")
    print("="*70)
    
    # 获取脚本所在目录
    base_dir = get_script_dir()
    print(f"脚本所在目录: {base_dir}")
    
    # 设置设备
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建目录
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查tqdm是否可用
    if not TQDM_AVAILABLE:
        print("\n⚠️ 注意: tqdm模块未安装，进度条显示受限")
        print("   建议安装以获得更好的体验: pip install tqdm")
    
    try:
        # 1. 创建数据加载器
        dataloaders, dataset_sizes = create_dataloaders(base_dir)
        
        if dataloaders is None or dataset_sizes is None:
            print("❌ 无法创建数据加载器，程序退出")
            return
        
        # 2. 构建模型
        model = build_resnet18_model(num_classes=2)
        model = model.to(device)
        
        # 3. 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        
        # 4. 学习率调度器
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 5. 训练模型
        model, history = train_model(
            model, dataloaders, dataset_sizes,
            criterion, optimizer, scheduler,
            num_epochs=15  # 减少epoch数以加快训练
        )
        
        # 6. 保存模型
        model_path = os.path.join(models_dir, 'cataract_resnet18.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'classes': ['cataract', 'normal'],
            'input_size': 224
        }, model_path)
        print(f"\n✅ 模型已保存到 {model_path}")
        
        # 7. 绘制训练历史
        plot_training_history(history, results_dir)
        
        # 8. 在测试集上快速测试
        print("\n在测试集上快速测试...")
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            test_iter = dataloaders['test']
            if TQDM_AVAILABLE:
                test_iter = tqdm(test_iter, desc='测试')
            
            for inputs, labels in test_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        if test_total > 0:
            test_accuracy = 100 * test_correct / test_total
            print(f"测试集准确率: {test_accuracy:.2f}% ({test_correct}/{test_total})")
        else:
            print("⚠️ 测试集为空，无法计算准确率")
        
        print("\n" + "="*70)
        print("✅ 模型训练完成!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 建议:")
        print("1. 确保已安装必要的库: pip install torch torchvision pillow matplotlib")
        print("2. 确保数据准备脚本已正确运行")
        print("3. 检查图片格式是否支持 (jpg, png等)")
        print("4. 如果GPU内存不足，尝试减小batch_size")

if __name__ == "__main__":
    main()