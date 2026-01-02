import sys
import os

print("=" * 60)
print("Python 环境诊断")
print("=" * 60)

# 检查 Python 版本和路径
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")

# 检查 PATH 环境变量
print(f"\nPATH 环境变量前几项:")
for path in sys.path[:5]:
    print(f"  {path}")

# 检查 pip 安装的包
print("\n已安装的包:")
try:
    import pkg_resources
    packages = [d for d in pkg_resources.working_set]
    package_names = [str(p).split()[0] for p in packages[:10]]  # 只显示前10个
    print(f"  前10个包: {', '.join(package_names)}...")
except:
    pass

# 检查 Flask
print("\n检查 Flask:")
try:
    import flask
    print(f"  ✅ Flask 已安装: {flask.__version__}")
except ImportError:
    print("  ❌ Flask 未安装")

# 检查其他依赖
print("\n检查其他依赖:")
for module_name in ['torch', 'torchvision', 'PIL', 'numpy', 'matplotlib']:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"  ✅ {module_name}: {version}")
    except ImportError:
        print(f"  ❌ {module_name}: 未安装")

print("\n" + "=" * 60)
input("按回车键退出...")