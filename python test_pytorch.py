"""
PyTorch环境测试脚本
测试PyTorch和TorchVision是否安装正确
使用方法：python test_pytorch.py
"""

import sys
import platform
import os

print("="*70)
print("PyTorch环境测试")
print("="*70)

def test_basic_imports():
    """测试基本导入"""
    print("\n1. 测试基本导入...")
    
    try:
        import torch
        import torchvision
        import numpy as np
        print("✅ 基本包导入成功")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  TorchVision版本: {torchvision.__version__}")
        print(f"  NumPy版本: {np.__version__}")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_cuda_availability():
    """测试CUDA是否可用"""
    print("\n2. 测试CUDA可用性...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
            
            # 测试简单的CUDA操作
            device = torch.device("cuda:0")
            x = torch.randn(3, 3).to(device)
            y = torch.randn(3, 3).to(device)
            z = torch.matmul(x, y)
            print(f"  CUDA矩阵运算测试通过")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            return True  # 这不算错误，只是警告
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")
        return False

def test_tensor_operations():
    """测试张量运算"""
    print("\n3. 测试张量运算...")
    
    try:
        import torch
        
        # 创建张量
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        
        # 各种运算
        z_add = x + y
        z_mul = torch.matmul(x, y)
        z_mean = x.mean()
        z_sum = x.sum()
        
        print(f"✅ 张量运算测试通过")
        print(f"  加法结果:\n{z_add}")
        print(f"  矩阵乘法结果:\n{z_mul}")
        print(f"  平均值: {z_mean.item():.2f}")
        print(f"  总和: {z_sum.item():.2f}")
        return True
    except Exception as e:
        print(f"❌ 张量运算测试失败: {e}")
        return False

def test_neural_network():
    """测试神经网络功能"""
    print("\n4. 测试神经网络功能...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # 定义一个简单的CNN
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(32 * 8 * 8, 64)
                self.fc2 = nn.Linear(64, 2)
                
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 32 * 8 * 8)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # 创建模型
        model = SimpleCNN()
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 32, 32)  # 批量大小2, 3通道, 32x32图像
        output = model(test_input)
        
        print(f"✅ 神经网络测试通过")
        print(f"  模型参数量: {sum(p.numel() for p in model.parameters())}")
        print(f"  输入形状: {test_input.shape}")
        print(f"  输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ 神经网络测试失败: {e}")
        return False

def test_autograd():
    """测试自动微分"""
    print("\n5. 测试自动微分...")
    
    try:
        import torch
        
        # 创建需要梯度的张量
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)
        
        # 计算函数 z = x^2 + y^3
        z = x**2 + y**3
        
        # 反向传播
        z.backward()
        
        print(f"✅ 自动微分测试通过")
        print(f"  z = x² + y³ = {z.item():.2f}")
        print(f"  ∂z/∂x = {x.grad.item():.2f}")
        print(f"  ∂z/∂y = {y.grad.item():.2f}")
        return True
    except Exception as e:
        print(f"❌ 自动微分测试失败: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n6. 测试数据加载器...")
    
    try:
        import torch
        from torchvision import transforms, datasets
        from torch.utils.data import DataLoader
        
        # 创建简单的转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 使用虚拟数据创建数据集
        from torch.utils.data import TensorDataset
        
        # 创建虚拟数据
        data = torch.randn(100, 3, 32, 32)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 测试一个批次
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx == 0:
                print(f"✅ 数据加载器测试通过")
                print(f"  批次 {batch_idx}:")
                print(f"    输入形状: {inputs.shape}")
                print(f"    标签形状: {targets.shape}")
                break
        
        return True
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False

def test_optimizer():
    """测试优化器"""
    print("\n7. 测试优化器...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # 创建简单模型
        model = nn.Linear(10, 2)
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 模拟训练步骤
        inputs = torch.randn(5, 10)
        targets = torch.randint(0, 2, (5,))
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ 优化器测试通过")
        print(f"  使用的优化器: Adam")
        print(f"  损失值: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ 优化器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"工作目录: {os.getcwd()}")
    
    # 运行所有测试
    tests = [
        test_basic_imports,
        test_cuda_availability,
        test_tensor_operations,
        test_neural_network,
        test_autograd,
        test_data_loader,
        test_optimizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "="*70)
    print(f"测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("✅ 所有测试通过! PyTorch环境配置正确。")
        print("\n下一步:")
        print("1. 运行白内障系统测试: python test_cataract_system.py")
        print("2. 准备数据并运行: python 01_数据准备.py")
    else:
        print(f"⚠️  部分测试失败，请检查PyTorch安装")
        print("\n常见问题解决:")
        print("1. 重新安装PyTorch: pip install torch torchvision")
        print("2. 检查Python版本: 需要Python 3.8+")
        print("3. 检查CUDA驱动（如果需要GPU）")
    
    print("="*70)

if __name__ == "__main__":
    main()