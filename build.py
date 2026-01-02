#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白内障筛查系统 - 本地打包工具
将整个应用打包成可执行文件
"""

import os
import sys
import shutil
import subprocess
import zipfile
from datetime import datetime

def create_portable_version():
    """创建便携版"""
    print("正在创建便携版应用...")
    
    # 创建输出目录
    output_dir = "白内障筛查系统_便携版"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    
    # 复制必要文件
    files_to_copy = [
        ('run.py', 'run.py'),
        ('requirements.txt', 'requirements.txt'),
        ('启动程序.bat', '启动程序.bat'),
        ('启动程序.sh', '启动程序.sh'),
    ]
    
    # 复制app目录
    if os.path.exists('app'):
        shutil.copytree('app', os.path.join(output_dir, 'app'))
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, dst))
    
    # 创建说明文件
    with open(os.path.join(output_dir, '使用说明.txt'), 'w', encoding='utf-8') as f:
        f.write(f"""白内障筛查系统 - 本地离线版

创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}



""")
    
    print(f"✅ 便携版已创建到: {output_dir}")
    
    # 打包成ZIP文件
    zip_filename = f"白内障筛查系统_便携版_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    
    print(f"✅ 已打包为: {zip_filename}")
    
    return zip_filename

def create_installer():
    """创建安装程序"""
    print("正在创建安装程序...")
    
    # 使用PyInstaller打包（需要额外安装）
    try:
        subprocess.run(['pyinstaller', '--onefile', '--name=白内障筛查系统', 'run.py'], check=True)
        print("✅ 安装程序创建完成")
    except Exception as e:
        print(f"❌ 创建安装程序失败: {e}")
        print("请先安装: pip install pyinstaller")

def main():
    """主函数"""
    print("=" * 50)
    print("白内障筛查系统 - 打包工具")
    print("=" * 50)
    
    print("请选择打包方式:")
    print("1. 创建便携版 (推荐)")
    print("2. 创建安装程序 (需要PyInstaller)")
    print("3. 两种都创建")
    print("4. 退出")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == '1':
        create_portable_version()
    elif choice == '2':
        create_installer()
    elif choice == '3':
        create_portable_version()
        create_installer()
    elif choice == '4':
        print("已退出")
    else:
        print("无效选择")

if __name__ == '__main__':
    main()