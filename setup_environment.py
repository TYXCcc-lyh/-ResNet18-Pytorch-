"""
自动环境配置脚本
功能：检查并安装所有必要的Python包
使用方法：python setup_environment.py
"""

import sys
import subprocess
import importlib
import platform
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("=" * 70)
    print("白内障筛查系统 - 环境配置")
    print("=" * 70)
    
    required_version = (3, 8)
    current_version = sys.version_info
    
    print(f"Python版本: {sys.version}")
    
    if current_version < required_version:
        print(f"❌ 错误: 需要Python {required_version[0]}.{required_version[1]} 或更高版本")
        print(f"当前版本: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    print(f"✅ Python版本检查通过 ({current_version[0]}.{current_version[1]}.{current_version[2]})")
    return True

def check_pip():
    """检查pip是否可用"""
    print("\n检查pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ pip可用")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip不可用，尝试安装pip...")
        try:
            # 尝试安装pip
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            print("✅ pip安装成功")
            return True
        except:
            print("❌ 无法安装pip，请手动安装pip")
            return False

def update_pip():
    """更新pip到最新版本"""
    print("\n更新pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip更新成功")
        return True
    except Exception as e:
        print(f"⚠️  pip更新失败: {e}")
        print("尝试继续...")
        return True

def get_system_info():
    """获取系统信息"""
    print("\n系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  架构: {platform.machine()}")
    print(f"  Python路径: {sys.executable}")
    print(f"  工作目录: {os.getcwd()}")
    
    # 检查CUDA是否可用
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  CUDA可用: 是")
            print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  CUDA可用: 否 (将使用CPU)")
    except:
        print(f"  CUDA检查: 未安装PyTorch")

def install_package(package_name, version=None):
    """安装单个包"""
    if version:
        install_cmd = f"{package_name}=={version}"
    else:
        install_cmd = package_name
    
    print(f"  安装: {install_cmd}")
    
    try:
        # 使用国内镜像源加速下载
        cmd = [sys.executable, "-m", "pip", "install", install_cmd, 
               "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", 
               "--trusted-host", "pypi.tuna.tsinghua.edu.cn"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✅ 成功")
            return True
        else:
            print(f"    ❌ 失败: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"    ❌ 异常: {str(e)[:100]}")
        return False

def install_from_requirements():
    """从requirements.txt安装所有依赖"""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ 错误: 找不到 {requirements_file}")
        print("请确保requirements.txt文件在当前目录")
        return False
    
    print(f"\n从 {requirements_file} 安装依赖包...")
    
    with open(requirements_file, 'r', encoding='utf-8') as f:
        packages = f.readlines()
    
    success_count = 0
    fail_count = 0
    failed_packages = []
    
    for line in packages:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 处理带有版本号的包
        if '==' in line:
            package_name, version = line.split('==', 1)
        else:
            package_name = line
            version = None
        
        success = install_package(package_name, version)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_packages.append(line)
    
    print(f"\n安装结果:")
    print(f"  成功: {success_count} 个包")
    print(f"  失败: {fail_count} 个包")
    
    if failed_packages:
        print(f"  失败的包: {', '.join(failed_packages)}")
    
    return fail_count == 0

def verify_installation():
    """验证关键包是否安装成功"""
    print("\n验证安装...")
    
    key_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("cv2", "OpenCV"),
        ("pandas", "Pandas")
    ]
    
    all_success = True
    
    for import_name, display_name in key_packages:
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            print(f"  ❌ {display_name}: {e}")
            all_success = False
    
    return all_success

def create_project_structure():
    """创建项目目录结构"""
    print("\n创建项目目录结构...")
    
    directories = [
        'models',
        'results',
        'data/train/cataract',
        'data/train/normal',
        'data/val/cataract',
        'data/val/normal',
        'data/test/cataract',
        'data/test/normal',
        'test_images',
        'logs',
        'notebooks',
        'scripts'
    ]
    
    created_count = 0
    existing_count = 0
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            if not os.listdir(directory):
                print(f"  创建: {directory}")
                created_count += 1
            else:
                existing_count += 1
        except Exception as e:
            print(f"  ❌ 创建失败 {directory}: {e}")
    
    print(f"\n目录创建完成:")
    print(f"  新建: {created_count} 个目录")
    print(f"  已存在: {existing_count} 个目录")

def create_test_scripts():
    """创建测试脚本"""
    print("\n创建测试脚本...")
    
    # 创建PyTorch测试脚本
    pytorch_test = """# test_pytorch.py
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("PyTorch环境测试")
print("="*60)

# 1. 基本信息
print(f"PyTorch版本: {torch.__version__}")
print(f"TorchVision版本: {torchvision.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# 2. 简单张量运算测试
print("\\n张量运算测试...")
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = torch.matmul(x, y)
print(f"矩阵乘法形状: {z.shape}")

# 3. 神经网络测试
print("\\n神经网络测试...")
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)
test_input = torch.randn(2, 10)
output = model(test_input)
print(f"神经网络输出形状: {output.shape}")

# 4. 数据加载测试
print("\\n数据转换测试...")
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
print(f"转换函数创建成功")

print("\\n" + "="*60)
print("✅ PyTorch环境测试通过!")
print("="*60)
"""
    
    # 创建白内障筛查测试脚本
    cataract_test = """# test_cataract_system.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("="*60)
print("白内障筛查系统环境测试")
print("="*60)

def test_imports():
    print("1. 导入测试...")
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import accuracy_score
        import cv2
        from PIL import Image
        import tqdm
        
        print("   ✅ 所有包导入成功")
        return True
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False

def test_directory_structure():
    print("\\n2. 目录结构测试...")
    required_dirs = ['models', 'results', 'data', 'test_images']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name} 存在")
        else:
            print(f"   ⚠️  {dir_name} 不存在")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0

def test_image_processing():
    print("\\n3. 图像处理测试...")
    try:
        # 创建一个测试图像
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 测试PIL
        pil_img = Image.fromarray(test_image)
        pil_resized = pil_img.resize((50, 50))
        print(f"   ✅ PIL图像处理成功")
        
        # 测试OpenCV
        import cv2
        cv_resized = cv2.resize(test_image, (50, 50))
        print(f"   ✅ OpenCV图像处理成功")
        
        return True
    except Exception as e:
        print(f"   ❌ 图像处理失败: {e}")
        return False

def test_ml_functions():
    print("\\n4. 机器学习功能测试...")
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        
        # 测试混淆矩阵
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        cm = confusion_matrix(y_true, y_pred)
        print(f"   ✅ 混淆矩阵计算成功")
        
        # 测试分类报告
        report = classification_report(y_true, y_pred)
        print(f"   ✅ 分类报告生成成功")
        
        return True
    except Exception as e:
        print(f"   ❌ 机器学习功能失败: {e}")
        return False

def main():
    all_tests_passed = True
    
    # 运行所有测试
    if not test_imports():
        all_tests_passed = False
    
    test_directory_structure()  # 只是警告，不阻止继续
    
    if not test_image_processing():
        all_tests_passed = False
    
    if not test_ml_functions():
        all_tests_passed = False
    
    print("\\n" + "="*60)
    if all_tests_passed:
        print("✅ 所有环境测试通过!")
        print("白内障筛查系统已准备就绪。")
    else:
        print("⚠️  部分测试未通过，请检查环境配置。")
    print("="*60)

if __name__ == "__main__":
    main()
"""
    
    # 创建示例配置文件
    config_file = """# config.yaml
# 白内障筛查系统配置

# 路径配置
paths:
  data_dir: "data"
  models_dir: "models"
  results_dir: "results"
  test_images_dir: "test_images"

# 模型配置
model:
  name: "resnet18"
  input_size: 224
  num_classes: 2
  pretrained: true
  dropout_rate: 0.5

# 训练配置
training:
  batch_size: 32
  num_epochs: 20
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10

# 数据增强配置
augmentation:
  train:
    rotation_range: 20
    width_shift_range: 0.1
    height_shift_range: 0.1
    shear_range: 0.1
    zoom_range: 0.1
    horizontal_flip: true
    brightness_range: [0.9, 1.1]
  
  val_test:
    # 验证和测试集不做增强，只做归一化
    pass

# 类别配置
classes:
  - "cataract"
  - "normal"

# 评估配置
evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "auc"
  threshold: 0.5
"""
    
    # 保存测试脚本
    scripts = {
        "test_pytorch.py": pytorch_test,
        "test_cataract_system.py": cataract_test,
        "config.yaml": config_file
    }
    
    for filename, content in scripts.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  创建: {filename}")

def create_install_cuda_script():
    """创建CUDA安装指导脚本"""
    if platform.system() != "Windows":
        return
    
    cuda_guide = """# install_cuda.py
"""
    # 由于CUDA安装复杂，这里只创建提示脚本
    cuda_guide = """# CUDA安装指导（Windows）
# 注意：以下步骤需要根据您的GPU型号和系统版本进行调整

# 步骤1：检查GPU是否支持CUDA
# 访问：https://developer.nvidia.com/cuda-gpus
# 查看您的GPU是否在列表中

# 步骤2：下载CUDA Toolkit
# 访问：https://developer.nvidia.com/cuda-downloads
# 选择适合您系统的版本（建议CUDA 11.8）

# 步骤3：安装PyTorch CUDA版本
# 在命令提示符中运行以下命令之一：

# 对于CUDA 11.8：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 对于CUDA 12.1：
# pip install torch torchvision torchaudio

# 步骤4：验证CUDA安装
# 运行以下Python代码：
# import torch
# print(torch.cuda.is_available())  # 应该输出True
# print(torch.cuda.get_device_name(0))  # 显示GPU名称

# 注意：如果遇到问题，请参考：
# - PyTorch官方文档：https://pytorch.org/get-started/locally/
# - NVIDIA CUDA文档：https://docs.nvidia.com/cuda/
"""
    
    with open("install_cuda_guide.txt", 'w', encoding='utf-8') as f:
        f.write(cuda_guide)
    print("  创建: install_cuda_guide.txt")

def main():
    """主函数"""
    try:
        # 1. 检查Python版本
        if not check_python_version():
            return
        
        # 2. 获取系统信息
        get_system_info()
        
        # 3. 检查pip
        if not check_pip():
            print("❌ pip检查失败，无法继续安装")
            return
        
        # 4. 更新pip
        update_pip()
        
        # 5. 安装依赖
        print("\n" + "="*70)
        print("开始安装依赖包...")
        print("="*70)
        
        install_success = install_from_requirements()
        
        if not install_success:
            print("\n⚠️  部分包安装失败，但继续验证...")
        
        # 6. 验证安装
        print("\n" + "="*70)
        print("验证安装结果...")
        print("="*70)
        
        verification_success = verify_installation()
        
        # 7. 创建项目结构
        print("\n" + "="*70)
        print("设置项目结构...")
        print("="*70)
        
        create_project_structure()
        create_test_scripts()
        
        # 如果是Windows系统，创建CUDA安装指导
        if platform.system() == "Windows":
            create_install_cuda_script()
        
        # 8. 最终报告
        print("\n" + "="*70)
        print("环境配置完成!")
        print("="*70)
        
        if verification_success:
            print("✅ 所有关键包安装成功!")
        else:
            print("⚠️  部分包可能未正确安装，请检查上方错误信息")
        
        print("\n下一步:")
        print("1. 运行测试脚本验证安装:")
        print("   python test_pytorch.py")
        print("   python test_cataract_system.py")
        print("\n2. 准备数据:")
        print("   将白内障图片放入 'cataract' 文件夹")
        print("   将正常图片放入 'normal' 文件夹")
        print("\n3. 运行数据准备脚本:")
        print("   python 01_数据准备.py")
        print("\n4. 开始训练:")
        print("   python 02_模型训练.py")
        
        print("\n如果需要GPU加速，请参考 install_cuda_guide.txt 安装CUDA")
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断安装")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 安装过程中出现错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()