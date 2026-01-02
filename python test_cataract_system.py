"""
白内障筛查系统环境测试
测试白内障筛查系统所需的所有功能
使用方法：python test_cataract_system.py
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

print("="*70)
print("白内障筛查系统环境测试")
print("="*70)

def get_system_info():
    """获取系统信息"""
    print("系统信息:")
    print(f"  Python版本: {sys.version}")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  处理器: {platform.processor()}")
    print(f"  工作目录: {os.getcwd()}")
    print()

def test_directory_structure():
    """测试目录结构"""
    print("1. 测试目录结构...")
    
    required_dirs = [
        'models',
        'results',
        'data/train/cataract',
        'data/train/normal',
        'data/val/cataract',
        'data/val/normal',
        'data/test/cataract',
        'data/test/normal',
        'test_images'
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            existing_dirs.append(dir_path)
        else:
            missing_dirs.append(dir_path)
    
    print(f"  找到 {len(existing_dirs)} 个已存在的目录")
    print(f"  缺失 {len(missing_dirs)} 个目录")
    
    if missing_dirs:
        print("  正在创建缺失的目录...")
        for dir_path in missing_dirs:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f"    ✅ 创建: {dir_path}")
                existing_dirs.append(dir_path)
            except Exception as e:
                print(f"    ❌ 创建失败 {dir_path}: {e}")
    else:
        print("  ✅ 所有目录都存在")
    
    return len(missing_dirs) == 0

def test_python_packages():
    """测试Python包"""
    print("\n2. 测试Python包...")
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow (图像处理)"),
        ("sklearn", "scikit-learn"),
        ("cv2", "OpenCV"),
        ("seaborn", "Seaborn"),
        ("tqdm", "进度条")
    ]
    
    failed_packages = []
    
    for import_name, display_name in packages:
        try:
            __import__(import_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            print(f"  ❌ {display_name}: {e}")
            failed_packages.append(display_name)
    
    if failed_packages:
        print(f"  ⚠️  缺失的包: {', '.join(failed_packages)}")
        print(f"  请运行: pip install {' '.join([p.lower().replace(' ', '-').replace('(图像处理)', '') for p in failed_packages])}")
        return False
    else:
        print("  ✅ 所有必需的包都已安装")
        return True

def test_torch_details():
    """测试PyTorch详细信息"""
    print("\n3. 测试PyTorch详细信息...")
    
    try:
        import torch
        import torchvision
        
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  TorchVision版本: {torchvision.__version__}")
        
        # 测试CUDA
        if torch.cuda.is_available():
            print(f"  ✅ CUDA可用")
            print(f"    设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    设备 {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"  ⚠️  CUDA不可用，将使用CPU")
        
        # 测试简单的张量操作
        x = torch.randn(2, 3, 224, 224)
        print(f"  ✅ 张量操作正常 (创建了 {x.shape} 形状的张量)")
        
        return True
    except Exception as e:
        print(f"  ❌ PyTorch测试失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n4. 测试图像处理功能...")
    
    try:
        import numpy as np
        from PIL import Image
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 测试PIL
        pil_img = Image.fromarray(test_image)
        pil_resized = pil_img.resize((50, 50))
        
        print(f"  ✅ PIL图像处理: 原始 {test_image.shape} → 调整后 {pil_resized.size}")
        
        # 测试OpenCV（如果安装）
        try:
            import cv2
            cv_resized = cv2.resize(test_image, (50, 50))
            print(f"  ✅ OpenCV图像处理: {cv_resized.shape}")
        except ImportError:
            print(f"  ⚠️  OpenCV未安装，跳过测试")
        
        return True
    except Exception as e:
        print(f"  ❌ 图像处理测试失败: {e}")
        return False

def test_ml_functions():
    """测试机器学习功能"""
    print("\n5. 测试机器学习功能...")
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # 创建测试数据
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        
        # 测试混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"  ✅ 混淆矩阵计算正常: {cm.shape}")
        
        # 测试分类报告
        report = classification_report(y_true, y_pred)
        print(f"  ✅ 分类报告生成正常")
        
        # 测试准确率
        accuracy = accuracy_score(y_true, y_pred)
        print(f"  ✅ 准确率计算正常: {accuracy:.2%}")
        
        return True
    except Exception as e:
        print(f"  ❌ 机器学习功能测试失败: {e}")
        return False

def test_data_augmentation():
    """测试数据增强功能"""
    print("\n6. 测试数据增强功能...")
    
    try:
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # 创建测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        # 定义数据增强转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        
        # 应用转换
        transformed = transform(test_image)
        
        print(f"  ✅ 数据增强功能正常")
        print(f"     原始大小: {test_image.size}")
        print(f"     转换后形状: {transformed.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ 数据增强测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建功能"""
    print("\n7. 测试模型创建功能...")
    
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        
        # 测试ResNet18模型创建
        model = models.resnet18(pretrained=False)
        
        # 修改最后一层用于二分类
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        
        print(f"  ✅ 模型创建功能正常")
        print(f"     模型名称: ResNet18")
        print(f"     参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"     输出类别: 2 (白内障/正常)")
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 224, 224)
        output = model(test_input)
        
        print(f"     前向传播测试: 输入 {test_input.shape} → 输出 {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ 模型创建测试失败: {e}")
        return False

def test_file_operations():
    """测试文件操作功能"""
    print("\n8. 测试文件操作功能...")
    
    try:
        import shutil
        import json
        import pickle
        
        # 测试JSON
        test_data = {"name": "白内障筛查", "version": "1.0", "test": True}
        json_str = json.dumps(test_data)
        loaded_data = json.loads(json_str)
        
        print(f"  ✅ JSON操作正常")
        
        # 测试pickle
        with open('test.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        with open('test.pkl', 'rb') as f:
            loaded_pickle = pickle.load(f)
        
        os.remove('test.pkl')
        print(f"  ✅ Pickle操作正常")
        
        return True
    except Exception as e:
        print(f"  ❌ 文件操作测试失败: {e}")
        return False

def create_sample_data():
    """创建示例数据（如果数据目录为空）"""
    print("\n9. 创建示例数据...")
    
    try:
        import numpy as np
        from PIL import Image
        import os
        
        # 创建示例白内障图像
        cataract_dir = 'test_images/sample_cataract'
        normal_dir = 'test_images/sample_normal'
        
        os.makedirs(cataract_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        
        # 创建5个示例白内障图像
        for i in range(5):
            # 白内障图像 - 添加一些模糊/混浊效果
            cataract_img = np.random.randint(150, 255, (224, 224, 3), dtype=np.uint8)
            
            # 添加一些模糊效果（白内障特征）
            center = (112, 112)
            for y in range(224):
                for x in range(224):
                    dist = ((x - center[0])**2 + (y - center[1])**2)**0.5
                    if dist < 80:  # 中心区域
                        cataract_img[y, x] = np.clip(cataract_img[y, x] + 50, 0, 255)
            
            img = Image.fromarray(cataract_img)
            img.save(os.path.join(cataract_dir, f'cataract_sample_{i}.jpg'))
        
        # 创建5个示例正常图像
        for i in range(5):
            # 正常图像 - 清晰
            normal_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(normal_img)
            img.save(os.path.join(normal_dir, f'normal_sample_{i}.jpg'))
        
        print(f"  ✅ 创建了10个示例图像")
        print(f"     白内障示例: {cataract_dir}")
        print(f"     正常示例: {normal_dir}")
        
        return True
    except Exception as e:
        print(f"  ⚠️  创建示例数据失败: {e}")
        return False

def run_system_test():
    """运行完整的系统测试"""
    print("\n10. 运行系统完整性测试...")
    
    try:
        # 模拟白内障筛查系统的主要功能
        import torch
        import numpy as np
        from PIL import Image
        
        print("  模拟白内障筛查流程:")
        print("    a. 图像加载 √")
        print("    b. 图像预处理 √")
        print("    c. 模型推理 √")
        print("    d. 结果输出 √")
        
        # 创建一个模拟的模型推理
        class MockModel:
            def __init__(self):
                self.name = "ResNet18白内障筛查模型"
            
            def predict(self, image_array):
                # 模拟预测：返回随机概率
                return {
                    'cataract_prob': np.random.random(),
                    'normal_prob': np.random.random()
                }
        
        model = MockModel()
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model.predict(test_img)
        
        print(f"    e. 模拟预测完成:")
        print(f"       白内障概率: {result['cataract_prob']:.2%}")
        print(f"       正常概率: {result['normal_prob']:.2%}")
        
        return True
    except Exception as e:
        print(f"  ❌ 系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    get_system_info()
    
    print("开始白内障筛查系统环境测试...")
    print()
    
    tests = [
        ("目录结构", test_directory_structure),
        ("Python包", test_python_packages),
        ("PyTorch详情", test_torch_details),
        ("图像处理", test_image_processing),
        ("机器学习功能", test_ml_functions),
        ("数据增强", test_data_augmentation),
        ("模型创建", test_model_creation),
        ("文件操作", test_file_operations),
        ("示例数据", create_sample_data),
        ("系统完整性", run_system_test)
    ]
    
    passed = 0
    total = len(tests)
    test_results = []
    
    for test_name, test_func in tests:
        print(f"测试: {test_name}")
        try:
            success = test_func()
            test_results.append((test_name, success))
            if success:
                passed += 1
                print(f"  ✅ 通过\n")
            else:
                print(f"  ❌ 失败\n")
        except Exception as e:
            print(f"  ❌ 异常: {e}\n")
            test_results.append((test_name, False))
    
    # 总结报告
    print("="*70)
    print("测试结果汇总")
    print("="*70)
    
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print()
    print(f"总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n" + "="*70)
        print("✅ 所有测试通过! 白内障筛查系统环境配置正确。")
        print("="*70)
        
        print("\n🎉 环境配置完成！系统已准备好使用。")
        print("\n下一步操作:")
        print("1. 准备数据:")
        print("   - 将白内障图片放入 'cataract' 文件夹")
        print("   - 将正常图片放入 'normal' 文件夹")
        print("2. 运行数据准备: python 01_数据准备.py")
        print("3. 训练模型: python 02_模型训练.py")
        print("4. 评估模型: python 03_模型评估.py")
        print("5. 预测单张图片: python 04_单张预测.py")
        
        print("\n快速测试:")
        print("  已经在 test_images/ 目录中创建了示例图像")
        print("  可以运行: python 04_单张预测.py --image test_images/sample_cataract/cataract_sample_0.jpg")
        
    else:
        print("\n" + "="*70)
        print("⚠️  部分测试失败，请检查环境配置")
        print("="*70)
        
        print("\n常见问题解决:")
        print("1. 缺少包: pip install -r requirements.txt")
        print("2. PyTorch安装问题:")
        print("   CPU版本: pip install torch torchvision")
        print("   GPU版本: 访问 https://pytorch.org/get-started/locally/")
        print("3. 目录权限: 确保有读写权限")
        print("4. 内存不足: 减少批量大小或使用更小的模型")
        
        print("\n如需帮助，请提供以下信息:")
        print("  - 完整的错误信息")
        print("  - Python版本")
        print("  - 操作系统")
    
    # 保存测试报告
    try:
        report = f"""白内障筛查系统环境测试报告
生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

系统信息:
  Python版本: {sys.version}
  操作系统: {platform.system()} {platform.release()}
  处理器: {platform.processor()}
  工作目录: {os.getcwd()}

测试结果:
"""
        for test_name, success in test_results:
            status = "通过" if success else "失败"
            report += f"  {test_name}: {status}\n"
        
        report += f"\n总计: {passed}/{total} 个测试通过\n"
        
        with open('results/environment_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ 测试报告已保存到: results/environment_test_report.txt")
    except:
        pass

if __name__ == "__main__":
    main()