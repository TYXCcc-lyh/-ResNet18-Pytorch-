"""
一键安装脚本 - 简化安装过程
使用方法：双击运行或 python one_click_install.py
"""

import os
import sys

def main():
    """主函数"""
    print("白内障筛查系统 - 一键安装")
    print("="*60)
    
    print("\n选择安装选项:")
    print("1. 完整安装（推荐）")
    print("2. 仅安装依赖包")
    print("3. 创建虚拟环境")
    print("4. 安装GPU支持")
    print("5. 退出")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        # 完整安装
        print("\n开始完整安装...")
        
        # 1. 创建虚拟环境
        print("\n[1/4] 创建虚拟环境...")
        os.system(f'"{sys.executable}" create_virtual_env.py')
        
        # 2. 激活虚拟环境（提示用户）
        print("\n[2/4] 请手动激活虚拟环境:")
        print("  Windows: 运行 activate_env.bat")
        print("  Linux/Mac: 运行 source activate_env.sh")
        input("\n按回车键继续（确保已激活虚拟环境）...")
        
        # 3. 安装依赖
        print("\n[3/4] 安装依赖包...")
        os.system(f'"{sys.executable}" setup_environment.py')
        
        # 4. 测试安装
        print("\n[4/4] 测试安装...")
        os.system(f'"{sys.executable}" test_cataract_system.py')
        
        print("\n✅ 完整安装完成!")
        
    elif choice == "2":
        # 仅安装依赖
        print("\n开始安装依赖包...")
        os.system(f'"{sys.executable}" setup_environment.py')
        
    elif choice == "3":
        # 创建虚拟环境
        print("\n创建虚拟环境...")
        os.system(f'"{sys.executable}" create_virtual_env.py')
        
    elif choice == "4":
        # 安装GPU支持
        print("\n安装GPU支持...")
        os.system(f'"{sys.executable}" install_gpu_support.py')
        
    elif choice == "5":
        print("退出安装")
        return
    
    else:
        print("无效选择")
    
    print("\n安装完成！")
    print("\n使用说明:")
    print("1. 数据准备: python 01_数据准备.py")
    print("2. 模型训练: python 02_模型训练.py")
    print("3. 模型评估: python 03_模型评估.py")
    print("4. 单张预测: python 04_单张预测.py")

if __name__ == "__main__":
    main()