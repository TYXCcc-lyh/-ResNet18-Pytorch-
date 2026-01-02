import subprocess
import sys

# 清华镜像源配置
TSINGHUA_SOURCE = "https://pypi.tuna.tsinghua.edu.cn/simple"
TRUSTED_HOST = "pypi.tuna.tsinghua.edu.cn"

# 需要安装的包
PACKAGES = [
    'torch',
    'torchvision',
    'numpy',
    'matplotlib',
    'pillow',
    'scikit-learn',
    'opencv-python',
    'pandas'
]

def install_package(package):
    """使用清华镜像源安装单个包"""
    try:
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            package,
            '-i', TSINGHUA_SOURCE,
            '--trusted-host', TRUSTED_HOST,
            '--timeout', '100',  # 超时时间设为100秒
            '--retries', '3'     # 重试3次
        ]
        
        print(f"🔧 正在安装 {package}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if "Successfully installed" in result.stdout or "Requirement already satisfied" in result.stdout:
            print(f"✅ {package} 安装成功！")
            return True
        else:
            print(f"⚠️  {package} 可能已安装")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败，错误信息：")
        print(e.stderr[:500])  # 只显示前500个字符
        return False

def main():
    print("🚀 开始使用清华镜像源快速安装...")
    print(f"📦 总共需要安装 {len(PACKAGES)} 个包")
    print("=" * 50)
    
    success_count = 0
    for i, package in enumerate(PACKAGES, 1):
        print(f"\n📦 [{i}/{len(PACKAGES)}]")
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 安装完成！成功：{success_count}/{len(PACKAGES)}")
    
    # 验证安装
    if success_count > 0:
        print("\n🧪 验证安装结果...")
        for package in PACKAGES:
            try:
                if package == 'opencv-python':
                    __import__('cv2')
                    print(f"✅ OpenCV 导入成功")
                elif package == 'pillow':
                    __import__('PIL')
                    print(f"✅ Pillow 导入成功")
                else:
                    __import__(package)
                    print(f"✅ {package} 导入成功")
            except ImportError:
                print(f"❌ {package} 导入失败")

if __name__ == "__main__":
    main()