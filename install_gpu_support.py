"""
GPU支持安装脚本
功能：安装PyTorch GPU版本和CUDA支持
使用方法：python install_gpu_support.py
"""

import sys
import subprocess
import platform

def check_gpu_availability():
    """检查GPU是否可用"""
    print("检查GPU支持...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU已可用")
            print(f"   设备: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("❌ GPU不可用")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def install_pytorch_gpu():
    """安装PyTorch GPU版本"""
    print("\n安装PyTorch GPU版本...")
    
    # 获取系统信息
    system = platform.system().lower()
    
    # PyTorch安装命令（根据系统）
    if system == "windows":
        # Windows - 使用CUDA 11.8
        command = [sys.executable, "-m", "pip", "install", 
                  "torch", "torchvision", "torchaudio", 
                  "--index-url", "https://download.pytorch.org/whl/cu118"]
    elif system == "linux":
        # Linux - 使用CUDA 11.8
        command = [sys.executable, "-m", "pip", "install", 
                  "torch", "torchvision", "torchaudio", 
                  "--index-url", "https://download.pytorch.org/whl/cu118"]
    else:
        # macOS - 使用CPU版本（MPS加速）
        command = [sys.executable, "-m", "pip", "install", 
                  "torch", "torchvision", "torchaudio"]
    
    print(f"安装命令: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PyTorch GPU版本安装成功")
            return True
        else:
            print(f"❌ 安装失败: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ 安装异常: {e}")
        return False

def main():
    """主函数"""
    print("="*70)
    print("GPU支持安装")
    print("="*70)
    
    # 检查当前GPU状态
    current_gpu_status = check_gpu_availability()
    
    if current_gpu_status:
        print("\n✅ GPU已经可用，无需额外安装")
        return
    
    print("\n当前系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  架构: {platform.machine()}")
    print(f"  Python: {sys.version}")
    
    # 询问用户是否继续
    response = input("\n是否安装PyTorch GPU版本？(y/n): ").strip().lower()
    
    if response != 'y':
        print("取消安装")
        return
    
    # 安装PyTorch GPU版本
    success = install_pytorch_gpu()
    
    if success:
        print("\n✅ GPU支持安装完成")
        print("\n验证安装...")
        
        # 重新检查GPU状态
        check_gpu_availability()
        
        print("\n下一步:")
        print("1. 运行测试: python test_pytorch.py")
        print("2. 如果GPU仍不可用，请确保:")
        print("   - 已安装NVIDIA显卡驱动")
        print("   - 已安装CUDA Toolkit")
        print("   - 系统环境变量配置正确")
    else:
        print("\n❌ GPU支持安装失败")
        print("\n建议:")
        print("1. 访问 https://pytorch.org/get-started/locally/")
        print("2. 选择适合您系统的安装命令")
        print("3. 手动安装PyTorch GPU版本")

if __name__ == "__main__":
    main()