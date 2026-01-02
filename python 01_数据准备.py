# 01_数据准备.py
"""
数据准备模块 - 将原始数据划分为训练集、验证集、测试集
使用方法: python 01_数据准备.py
"""

import os
import random
import shutil
import numpy as np
from pathlib import Path

def setup_project_structure(base_dir):
    """创建项目目录结构"""
    directories = [
        os.path.join(base_dir, 'data/train/cataract'),
        os.path.join(base_dir, 'data/train/normal'),
        os.path.join(base_dir, 'data/val/cataract'),
        os.path.join(base_dir, 'data/val/normal'),
        os.path.join(base_dir, 'data/test/cataract'),
        os.path.join(base_dir, 'data/test/normal'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'test_images')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")
    
    return True

def get_image_files(folder_path):
    """获取文件夹中的所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = []
    
    if not os.path.exists(folder_path):
        return files
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            files.append(file)
    
    return files

def prepare_dataset(base_dir):
    """
    准备数据集，划分比例为：
    - 训练集: 70%
    - 验证集: 15%
    - 测试集: 15%
    """
    print("="*60)
    print("白内障筛查数据准备")
    print("="*60)
    
    # 构建完整的文件夹路径
    cataract_dir = os.path.join(base_dir, 'cataract')
    normal_dir = os.path.join(base_dir, 'normal')
    
    # 检查原始数据文件夹
    if not os.path.exists(cataract_dir):
        print(f"❌ 错误: 找不到白内障图片文件夹 '{cataract_dir}'")
        print(f"当前脚本所在目录: {base_dir}")
        print(f"请确保在 {base_dir} 目录下有 cataract 文件夹")
        return False
    
    if not os.path.exists(normal_dir):
        print(f"❌ 错误: 找不到正常图片文件夹 '{normal_dir}'")
        print(f"当前脚本所在目录: {base_dir}")
        print(f"请确保在 {base_dir} 目录下有 normal 文件夹")
        return False
    
    # 获取图片文件
    cataract_files = get_image_files(cataract_dir)
    normal_files = get_image_files(normal_dir)
    
    print(f"找到白内障图片: {len(cataract_files)} 张")
    print(f"找到正常图片: {len(normal_files)} 张")
    print(f"白内障文件夹路径: {cataract_dir}")
    print(f"正常文件夹路径: {normal_dir}")
    
    if len(cataract_files) == 0:
        print(f"❌ 警告: cataract 文件夹为空或没有图片文件")
        print(f"请检查 {cataract_dir} 中是否有图片文件")
    
    if len(normal_files) == 0:
        print(f"❌ 警告: normal 文件夹为空或没有图片文件")
        print(f"请检查 {normal_dir} 中是否有图片文件")
    
    if len(cataract_files) == 0 or len(normal_files) == 0:
        print("❌ 错误: 至少一个类别的图片数为0")
        return False
    
    # 打乱顺序
    random.shuffle(cataract_files)
    random.shuffle(normal_files)
    
    def split_files(files):
        """分割文件列表"""
        n = len(files)
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.15)
        
        return {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }
    
    # 分割两个类别的数据
    cataract_split = split_files(cataract_files)
    normal_split = split_files(normal_files)
    
    # 复制文件到对应目录
    def copy_files(files, source_dir, target_dir):
        """复制文件列表"""
        for file in files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(target_dir, file)
            shutil.copy2(src, dst)
    
    # 复制白内障图片
    for split_type, files in cataract_split.items():
        target_dir = os.path.join(base_dir, f'data/{split_type}/cataract')
        copy_files(files, cataract_dir, target_dir)
        print(f"复制白内障图片到 {split_type}: {len(files)} 张")
    
    # 复制正常图片
    for split_type, files in normal_split.items():
        target_dir = os.path.join(base_dir, f'data/{split_type}/normal')
        copy_files(files, normal_dir, target_dir)
        print(f"复制正常图片到 {split_type}: {len(files)} 张")
    
    # 保存数据统计
    save_dataset_info(base_dir, cataract_split, normal_split)
    
    return True

def save_dataset_info(base_dir, cataract_split, normal_split):
    """保存数据集信息"""
    info = f"""白内障筛查数据集信息
生成时间: {np.datetime64('now')}
脚本目录: {base_dir}

数据集划分 (70%/15%/15%):

白内障图片:
  训练集: {len(cataract_split['train'])} 张
  验证集: {len(cataract_split['val'])} 张
  测试集: {len(cataract_split['test'])} 张
  总计: {sum(len(v) for v in cataract_split.values())} 张

正常图片:
  训练集: {len(normal_split['train'])} 张
  验证集: {len(normal_split['val'])} 张
  测试集: {len(normal_split['test'])} 张
  总计: {sum(len(v) for v in normal_split.values())} 张

总计图片:
  训练集: {len(cataract_split['train']) + len(normal_split['train'])} 张
  验证集: {len(cataract_split['val']) + len(normal_split['val'])} 张
  测试集: {len(cataract_split['test']) + len(normal_split['test'])} 张
  总计: {sum(len(v) for v in cataract_split.values()) + sum(len(v) for v in normal_split.values())} 张
"""
    
    info_file = os.path.join(base_dir, 'data/dataset_info.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(info)
    
    print(f"\n✅ 数据集信息已保存到 {info_file}")

def main():
    """主函数"""
    print("开始数据准备...")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本所在目录: {script_dir}")
    
    # 创建目录结构
    setup_project_structure(script_dir)
    
    # 准备数据集
    success = prepare_dataset(script_dir)
    
    if success:
        print("\n" + "="*60)
        print("✅ 数据准备完成!")
        print("="*60)
        print(f"\n数据生成在: {script_dir}/data/")
        print("生成的数据集结构:")
        print("data/")
        print("├── train/")
        print("│   ├── cataract/    # 训练集白内障图片")
        print("│   └── normal/      # 训练集正常图片")
        print("├── val/")
        print("│   ├── cataract/    # 验证集白内障图片")
        print("│   └── normal/      # 验证集正常图片")
        print("└── test/")
        print("    ├── cataract/    # 测试集白内障图片")
        print("    └── normal/      # 测试集正常图片")
    else:
        print("\n❌ 数据准备失败")

if __name__ == "__main__":
    main()