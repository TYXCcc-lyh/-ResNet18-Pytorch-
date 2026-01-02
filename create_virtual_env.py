"""
创建虚拟环境脚本
功能：创建独立的Python虚拟环境
使用方法：python create_virtual_env.py
"""

import sys
import os
import subprocess
import platform

def check_venv_availability():
    """检查venv模块是否可用"""
    print("检查虚拟环境支持...")
    
    try:
        import venv
        print("✅ venv模块可用")
        return True
    except ImportError:
        print("❌ venv模块不可用")
        return False

def create_virtual_environment(env_name="cataract_env"):
    """创建虚拟环境"""
    print(f"\n创建虚拟环境: {env_name}")
    
    # 创建虚拟环境的命令
    if platform.system() == "Windows":
        cmd = [sys.executable, "-m", "venv", env_name]
    else:
        cmd = [sys.executable, "-m", "venv", env_name]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 虚拟环境创建成功: {env_name}")
            return True
        else:
            print(f"❌ 创建失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 创建异常: {e}")
        return False

def create_activation_scripts(env_name="cataract_env"):
    """创建激活脚本"""
    print(f"\n创建激活脚本...")
    
    if platform.system() == "Windows":
        # Windows激活脚本
        activate_content = f"""@echo off
echo 激活白内障筛查虚拟环境
call {env_name}\\Scripts\\activate.bat
echo.
echo 虚拟环境已激活!
echo 安装依赖: pip install -r requirements.txt
echo 退出环境: deactivate
"""
        
        script_name = "activate_env.bat"
        
        with open(script_name, 'w', encoding='gbk') as f:
            f.write(activate_content)
        
        print(f"✅ 创建: {script_name} (Windows激活脚本)")
        
    else:
        # Linux/Mac激活脚本
        activate_content = f"""#!/bin/bash
echo "激活白内障筛查虚拟环境"
source {env_name}/bin/activate
echo ""
echo "虚拟环境已激活!"
echo "安装依赖: pip install -r requirements.txt"
echo "退出环境: deactivate"
"""
        
        script_name = "activate_env.sh"
        
        with open(script_name, 'w', encoding='utf-8') as f:
            f.write(activate_content)
        
        # 添加执行权限
        os.chmod(script_name, 0o755)
        
        print(f"✅ 创建: {script_name} (Linux/Mac激活脚本)")
    
    # 创建使用说明
    readme_content = f"""# 白内障筛查系统 - 虚拟环境使用说明

## 虚拟环境信息
- 环境名称: {env_name}
- Python版本: {sys.version.split()[0]}
- 创建时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 使用方法

### Windows系统:
1. 双击运行 `activate_env.bat`
2. 或手动激活: