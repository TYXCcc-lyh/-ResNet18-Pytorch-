# 03_修复复杂图表版模型评估.py
"""
修复复杂图表版模型评估模块 - 保留了所有复杂图表，修复了数组形状问题

"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def get_script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

class CataractDataset:
    """白内障数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cataract', 'normal']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        self.image_paths = []
        
        # 遍历所有类别
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ 警告: 找不到类别目录 {class_dir}")
                continue
                
            img_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not img_files:
                print(f"⚠️ 警告: 类别 {class_name} 目录中没有图片文件")
                continue
                
            for img_name in img_files:
                self.images.append(os.path.join(class_dir, img_name))
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])
        
        if len(self.images) == 0:
            print(f"❌ 错误: 数据集目录 {root_dir} 中没有找到任何图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 错误: 无法加载图片 {img_path}: {e}")
            # 返回一个黑色图片作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

def get_data_transforms():
    """获取数据增强和转换"""
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

class EnhancedCataractModelEvaluator:
    """增强版白内障模型评估器"""
    
    def __init__(self, model_path=None):
        """初始化评估器"""
        if model_path is None:
            base_dir = get_script_dir()
            model_path = os.path.join(base_dir, 'models/cataract_resnet18.pth')
        
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = ['cataract', 'normal']
        
        # 创建结果目录
        base_dir = get_script_dir()
        self.results_dir = os.path.join(base_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 创建详细图表目录
        self.charts_dir = os.path.join(self.results_dir, 'charts')
        os.makedirs(self.charts_dir, exist_ok=True)
        
        print(f"脚本所在目录: {base_dir}")
        print(f"模型路径: {self.model_path}")
        print(f"结果目录: {self.results_dir}")
        print(f"图表目录: {self.charts_dir}")
    
    def load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"❌ 错误: 找不到模型文件 {self.model_path}")
            return False
        
        try:
            # 构建模型结构
            model = models.resnet18(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)
            )
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.model = model
            
            # 如果有保存的history，可以用于分析训练过程
            self.history = checkpoint.get('history', None)
            
            print(f"✅ 模型加载成功")
            print(f"   类别: {self.classes}")
            print(f"   设备: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def create_test_dataloader(self):
        """创建测试数据加载器"""
        print("\n准备测试数据...")
        
        base_dir = get_script_dir()
        data_dir = os.path.join(base_dir, 'data')
        test_dir = os.path.join(data_dir, 'test')
        
        print(f"数据目录: {data_dir}")
        print(f"测试数据目录: {test_dir}")
        
        if not os.path.exists(test_dir):
            print(f"❌ 错误: 找不到测试数据目录 {test_dir}")
            return None, None
        
        data_transforms = get_data_transforms()
        test_dataset = CataractDataset(test_dir, transform=data_transforms['test'])
        
        if len(test_dataset) == 0:
            print("❌ 错误: 测试集为空")
            return None, None
        
        num_workers = 0 if sys.platform == 'win32' else 2
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
        
        print(f"测试集: {len(test_dataset)} 张图片")
        return test_dataloader, test_dataset
    
    def evaluate_model_comprehensive(self):
        """全面评估模型性能"""
        if not self.model:
            print("❌ 错误: 请先加载模型")
            return
        
        test_dataloader, test_dataset = self.create_test_dataloader()
        if test_dataloader is None:
            return
        
        print("\n开始全面评估模型...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, paths in test_dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_paths.extend(paths)
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print(f"\n评估数据统计:")
        print(f"总样本数: {len(all_labels)}")
        print(f"白内障样本数: {np.sum(all_labels == 1)}")
        print(f"正常样本数: {np.sum(all_labels == 0)}")
        
        # 执行全面的评估分析
        self.comprehensive_analysis(all_labels, all_preds, all_probs, all_paths)
        
        return all_labels, all_preds, all_probs, all_paths
    
    def comprehensive_analysis(self, y_true, y_pred, y_probs, y_paths):
        """执行全面的分析"""
        print("\n" + "="*70)
        print("全面模型分析")
        print("="*70)
        
        # 导入sklearn模块
        try:
            from sklearn.metrics import (confusion_matrix, classification_report, 
                                       roc_curve, auc, precision_recall_curve, 
                                       average_precision_score, f1_score, 
                                       precision_score, recall_score, accuracy_score)
        except ImportError:
            print("❌ 错误: 缺少 scikit-learn 模块")
            print("请运行: pip install scikit-learn")
            return
        
        # 1. 基础评估指标
        print("\n1. 基础评估指标:")
        self.calculate_basic_metrics(y_true, y_pred)
        
        # 2. 混淆矩阵分析
        print("\n2. 混淆矩阵分析:")
        cm = confusion_matrix(y_true, y_pred)
        self.analyze_confusion_matrix(cm)
        
        # 3. 生成所有可视化图表
        print("\n3. 生成可视化图表...")
        self.generate_all_charts(y_true, y_pred, y_probs, y_paths)
        
        # 4. 错误分析
        print("\n4. 错误分析:")
        self.analyze_errors(y_true, y_pred, y_probs, y_paths)
        
        # 5. 模型置信度分析
        print("\n5. 模型置信度分析:")
        self.analyze_confidence(y_true, y_pred, y_probs)
        
        # 6. 生成综合报告
        print("\n6. 生成综合报告...")
        self.generate_comprehensive_report(y_true, y_pred, y_probs, cm)
        
        print("\n" + "="*70)
        print("✅ 全面评估完成!")
        print("="*70)
    
    def calculate_basic_metrics(self, y_true, y_pred):
        """计算基础评估指标"""
        from sklearn.metrics import (accuracy_score, precision_score, 
                                   recall_score, f1_score, classification_report)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        print(f"准确率 (Accuracy):           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率 (Precision):          {precision:.4f} ({precision*100:.2f}%)")
        print(f"召回率/敏感度 (Recall):      {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1分数 (F1-Score):           {f1:.4f} ({f1*100:.2f}%)")
        
        print(f"\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=self.classes))
    
    def analyze_confusion_matrix(self, cm):
        """分析混淆矩阵"""
        if cm.shape == (1, 1):
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
        
        total = tp + tn + fp + fn
        
        print(f"混淆矩阵:")
        print(f"          预测正常    预测白内障")
        print(f"真实正常   {tn:>6}        {fp:>6}")
        print(f"真实白内障 {fn:>6}        {tp:>6}")
        print(f"\n详细分析:")
        print(f"总样本数: {total}")
        print(f"真阳性 (TP): {tp} - 正确识别的白内障")
        print(f"真阴性 (TN): {tn} - 正确识别的正常")
        print(f"假阳性 (FP): {fp} - 正常误判为白内障")
        print(f"假阴性 (FN): {fn} - 白内障漏诊")
        print(f"\n错误率分析:")
        print(f"总体错误率: {(fp+fn)/total*100:.2f}%" if total > 0 else "总体错误率: N/A")
        print(f"假阳性率 (误报率): {fp/(fp+tn)*100:.2f}%" if (fp+tn) > 0 else "假阳性率: N/A")
        print(f"假阴性率 (漏报率): {fn/(fn+tp)*100:.2f}%" if (fn+tp) > 0 else "假阴性率: N/A")
    
    def generate_all_charts(self, y_true, y_pred, y_probs, y_paths):
        """生成所有可视化图表"""
        
        # 1. 混淆矩阵热图
        self.plot_confusion_matrix(y_true, y_pred)
        
        # 2. ROC曲线
        self.plot_roc_curve(y_true, y_probs)
        
        # 3. 精确率-召回率曲线
        self.plot_precision_recall_curve(y_true, y_probs)
        
        # 4. 预测概率分布
        self.plot_prediction_distribution(y_true, y_probs)
        
        # 5. 模型性能指标对比 - 修复了数组形状问题
        self.plot_metrics_comparison_fixed(y_true, y_pred)
        
        # 6. 置信度分布
        self.plot_confidence_distribution(y_true, y_pred, y_probs)
        
        # 7. 阈值分析
        self.plot_threshold_analysis(y_true, y_probs)
        
        # 8. 训练历史（如果有）
        if hasattr(self, 'history') and self.history:
            self.plot_training_history()
        
        # 9. 错误分类样本展示
        self.plot_misclassified_samples(y_true, y_pred, y_probs, y_paths, num_samples=12)
        
        # 10. 性能对比矩阵 - 修复了数组形状问题
        self.plot_performance_matrix_fixed(y_true, y_pred, y_probs)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # 尝试使用seaborn
        try:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.classes, yticklabels=self.classes,
                       cbar_kws={'label': '样本数量'})
            plt.title('混淆矩阵 - 白内障筛查模型', fontsize=16, fontweight='bold')
        except ImportError:
            # 使用matplotlib绘制
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('混淆矩阵', fontsize=16, fontweight='bold')
            plt.colorbar(label='样本数量')
            
            # 添加文本标签
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.xticks(range(len(self.classes)), self.classes)
            plt.yticks(range(len(self.classes)), self.classes)
        
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.charts_dir, '01_混淆矩阵.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 混淆矩阵图已保存: {save_path}")
    
    def plot_roc_curve(self, y_true, y_probs):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, auc
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            # 找到最佳阈值（最靠近左上角的点）
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC曲线 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
            
            # 标记最佳阈值点
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                       color='red', s=100, zorder=5, 
                       label=f'最佳阈值={optimal_threshold:.3f}')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率 (1-特异度)', fontsize=12)
            plt.ylabel('真阳性率 (敏感度)', fontsize=12)
            plt.title('ROC曲线 - 模型区分能力', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.charts_dir, '02_ROC曲线.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ ROC曲线已保存: {save_path}")
        else:
            print("⚠️ 只有一类数据，无法绘制ROC曲线")
    
    def plot_precision_recall_curve(self, y_true, y_probs):
        """绘制精确率-召回率曲线"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        if len(np.unique(y_true)) > 1:
            precision, recall, thresholds = precision_recall_curve(y_true, y_probs[:, 1])
            avg_precision = average_precision_score(y_true, y_probs[:, 1])
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='green', lw=2, 
                    label=f'PR曲线 (AP = {avg_precision:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('召回率 (Recall)', fontsize=12)
            plt.ylabel('精确率 (Precision)', fontsize=12)
            plt.title('精确率-召回率曲线 - 模型精确性', fontsize=16, fontweight='bold')
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.charts_dir, '03_精确率召回率曲线.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 精确率-召回率曲线已保存: {save_path}")
    
    def plot_prediction_distribution(self, y_true, y_probs):
        """绘制预测概率分布"""
        plt.figure(figsize=(14, 6))
        
        # 子图1: 概率分布直方图
        plt.subplot(1, 2, 1)
        
        if len(y_probs.shape) > 1 and y_probs.shape[1] > 1:
            cataract_probs = y_probs[y_true == 1, 1] if 1 in y_true else []
            normal_probs = y_probs[y_true == 0, 1] if 0 in y_true else []
            
            if len(cataract_probs) > 0:
                plt.hist(cataract_probs, bins=30, alpha=0.7, color='red', 
                        label='白内障', density=True)
            if len(normal_probs) > 0:
                plt.hist(normal_probs, bins=30, alpha=0.7, color='blue', 
                        label='正常', density=True)
            
            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='阈值=0.5')
        
        plt.xlabel('预测为白内障的概率', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.title('预测概率分布', fontsize=14, fontweight='bold')
        if len(cataract_probs) > 0 or len(normal_probs) > 0:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 核密度估计
        plt.subplot(1, 2, 2)
        
        if len(y_probs.shape) > 1 and y_probs.shape[1] > 1:
            try:
                import seaborn as sns
                if len(cataract_probs) > 0:
                    sns.kdeplot(cataract_probs, color='red', label='白内障', fill=True, alpha=0.5)
                if len(normal_probs) > 0:
                    sns.kdeplot(normal_probs, color='blue', label='正常', fill=True, alpha=0.5)
                plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='阈值=0.5')
            except ImportError:
                # 如果没有seaborn，使用直方图
                if len(cataract_probs) > 0:
                    plt.hist(cataract_probs, bins=30, alpha=0.5, color='red', 
                            label='白内障', density=True, histtype='stepfilled')
                if len(normal_probs) > 0:
                    plt.hist(normal_probs, bins=30, alpha=0.5, color='blue', 
                            label='正常', density=True, histtype='stepfilled')
                plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='阈值=0.5')
        
        plt.xlabel('预测为白内障的概率', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.title('概率密度分布', fontsize=14, fontweight='bold')
        if len(cataract_probs) > 0 or len(normal_probs) > 0:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.charts_dir, '04_预测概率分布.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测概率分布图已保存: {save_path}")
    
    def plot_metrics_comparison_fixed(self, y_true, y_pred):
        """绘制性能指标对比图 - 修复版"""
        from sklearn.metrics import (accuracy_score, precision_score, 
                                   recall_score, f1_score)
        
        # 计算各项指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # 计算类别特定的指标
        precision_cataract = precision_score(y_true, y_pred, pos_label=1)
        recall_cataract = recall_score(y_true, y_pred, pos_label=1)
        
        # 对于正常类别（需要反转标签）
        y_pred_normal = 1 - y_pred
        y_true_normal = 1 - y_true
        precision_normal = precision_score(y_true_normal, y_pred_normal, pos_label=1) if len(np.unique(y_true_normal)) > 1 else 0
        recall_normal = recall_score(y_true_normal, y_pred_normal, pos_label=1) if len(np.unique(y_true_normal)) > 1 else 0
        
        metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
        
        # 创建数据数组，确保长度一致
        overall_data = [accuracy, precision, recall, f1]
        cataract_data = [np.nan, precision_cataract, recall_cataract, np.nan]
        normal_data = [np.nan, precision_normal, recall_normal, np.nan]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 子图1: 总体指标雷达图
        ax1 = axes[0]
        angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
        
        # 雷达图需要闭合，所以复制第一个值到末尾
        overall_metrics_radar = overall_data + [overall_data[0]]
        angles_radar = angles + [angles[0]]
        
        ax1.plot(angles_radar, overall_metrics_radar, 'o-', linewidth=2, label='总体指标')
        ax1.fill(angles_radar, overall_metrics_radar, alpha=0.25)
        ax1.set_xticks(angles)
        ax1.set_xticklabels(metrics_labels)
        ax1.set_ylim(0, 1)
        ax1.set_title('总体指标雷达图', fontsize=14, fontweight='bold')
        ax1.grid(True)
        
        # 子图2: 柱状图对比
        ax2 = axes[1]
        x = np.arange(len(metrics_labels))
        width = 0.25
        
        # 修复：确保每个bar的数据长度与x一致
        ax2.bar(x - width, overall_data, width, label='总体', color='blue', alpha=0.7)
        ax2.bar(x, cataract_data, width, label='白内障', color='red', alpha=0.7)
        ax2.bar(x + width, normal_data, width, label='正常', color='green', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_labels)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('分数', fontsize=12)
        ax2.set_title('各类别指标对比', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, val in enumerate(overall_data):
            if not np.isnan(val):
                ax2.text(i - width, val + 0.01, f'{val:.2%}', ha='center', va='bottom', fontsize=8)
        
        for i, val in enumerate(cataract_data):
            if not np.isnan(val):
                ax2.text(i, val + 0.01, f'{val:.2%}', ha='center', va='bottom', fontsize=8)
        
        for i, val in enumerate(normal_data):
            if not np.isnan(val):
                ax2.text(i + width, val + 0.01, f'{val:.2%}', ha='center', va='bottom', fontsize=8)
        
        # 子图3: 热力图
        ax3 = axes[2]
        heatmap_data = np.array([overall_data, cataract_data, normal_data])
        
        im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax3.set_xticks(range(len(metrics_labels)))
        ax3.set_xticklabels(metrics_labels, rotation=45)
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(['总体', '白内障', '正常'])
        ax3.set_title('指标热力图', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                if not np.isnan(heatmap_data[i, j]):
                    ax3.text(j, i, f'{heatmap_data[i, j]:.2%}', 
                            ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax3, label='分数')
        
        plt.tight_layout()
        save_path = os.path.join(self.charts_dir, '05_性能指标对比.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 性能指标对比图已保存: {save_path}")
    
    def plot_confidence_distribution(self, y_true, y_pred, y_probs):
        """绘制置信度分布"""
        if len(y_probs.shape) > 1 and y_probs.shape[1] > 1:
            # 获取预测置信度
            pred_confidences = np.max(y_probs, axis=1)
            
            # 区分正确和错误预测
            correct_mask = (y_pred == y_true)
            incorrect_mask = (y_pred != y_true)
            
            correct_confidences = pred_confidences[correct_mask]
            incorrect_confidences = pred_confidences[incorrect_mask]
            
            plt.figure(figsize=(14, 6))
            
            # 子图1: 置信度分布直方图
            plt.subplot(1, 2, 1)
            
            if len(correct_confidences) > 0:
                plt.hist(correct_confidences, bins=20, alpha=0.7, color='green', 
                        label=f'正确预测 ({len(correct_confidences)})', density=True)
            
            if len(incorrect_confidences) > 0:
                plt.hist(incorrect_confidences, bins=20, alpha=0.7, color='red', 
                        label=f'错误预测 ({len(incorrect_confidences)})', density=True)
            
            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='阈值=0.5')
            plt.xlabel('模型置信度', fontsize=12)
            plt.ylabel('密度', fontsize=12)
            plt.title('预测置信度分布', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2: 箱线图对比
            plt.subplot(1, 2, 2)
            
            data_to_plot = []
            labels = []
            
            if len(correct_confidences) > 0:
                data_to_plot.append(correct_confidences)
                labels.append(f'正确预测\n(n={len(correct_confidences)})')
            
            if len(incorrect_confidences) > 0:
                data_to_plot.append(incorrect_confidences)
                labels.append(f'错误预测\n(n={len(incorrect_confidences)})')
            
            if data_to_plot:
                box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # 设置颜色
                colors = ['lightgreen', 'lightcoral']
                for patch, color in zip(box['boxes'], colors[:len(data_to_plot)]):
                    patch.set_facecolor(color)
                
                # 添加均值点
                for i, data in enumerate(data_to_plot):
                    mean_val = np.mean(data)
                    plt.scatter(i+1, mean_val, color='blue', s=100, zorder=3, label='均值' if i == 0 else "")
            
            plt.ylabel('模型置信度', fontsize=12)
            plt.title('置信度统计对比', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            if len(data_to_plot) > 0:
                plt.legend()
            
            plt.tight_layout()
            save_path = os.path.join(self.charts_dir, '06_置信度分布.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 置信度分布图已保存: {save_path}")
    
    def plot_threshold_analysis(self, y_true, y_probs):
        """绘制阈值分析图"""
        if len(np.unique(y_true)) > 1 and len(y_probs.shape) > 1:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            thresholds = np.linspace(0, 1, 101)
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for threshold in thresholds:
                # 根据阈值重新分类
                y_pred_thresh = (y_probs[:, 1] >= threshold).astype(int)
                
                # 计算指标
                accuracies.append(accuracy_score(y_true, y_pred_thresh))
                precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
                recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
                f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
            
            plt.figure(figsize=(12, 8))
            
            plt.plot(thresholds, accuracies, label='准确率', linewidth=2)
            plt.plot(thresholds, precisions, label='精确率', linewidth=2)
            plt.plot(thresholds, recalls, label='召回率', linewidth=2)
            plt.plot(thresholds, f1_scores, label='F1分数', linewidth=2)
            
            # 标记默认阈值0.5
            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='默认阈值=0.5')
            
            # 找到最佳F1分数对应的阈值
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_f1_idx]
            plt.axvline(x=best_threshold, color='red', linestyle=':', 
                       alpha=0.7, label=f'最佳F1阈值={best_threshold:.2f}')
            
            plt.xlabel('分类阈值', fontsize=12)
            plt.ylabel('指标分数', fontsize=12)
            plt.title('阈值对模型性能的影响', fontsize=16, fontweight='bold')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            plt.tight_layout()
            save_path = os.path.join(self.charts_dir, '07_阈值分析.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 阈值分析图已保存: {save_path}")
            
            # 输出最佳阈值建议
            print(f"\n📊 阈值分析结果:")
            print(f"  默认阈值(0.5): F1分数 = {f1_scores[50]:.4f}")
            print(f"  最佳阈值({best_threshold:.3f}): F1分数 = {f1_scores[best_f1_idx]:.4f}")
            print(f"  提升: {(f1_scores[best_f1_idx] - f1_scores[50])*100:.2f}%")
    
    def plot_training_history(self):
        """绘制训练历史（如果可用）"""
        if hasattr(self, 'history') and self.history:
            history = self.history
            
            if 'train_loss' in history and 'val_loss' in history:
                plt.figure(figsize=(14, 6))
                
                # 子图1: 损失曲线
                plt.subplot(1, 2, 1)
                epochs = range(1, len(history['train_loss']) + 1)
                
                plt.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
                plt.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
                plt.title('训练和验证损失', fontsize=14, fontweight='bold')
                plt.xlabel('训练轮次 (Epoch)', fontsize=12)
                plt.ylabel('损失值', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 子图2: 准确率曲线
                plt.subplot(1, 2, 2)
                
                plt.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
                plt.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
                plt.title('训练和验证准确率', fontsize=14, fontweight='bold')
                plt.xlabel('训练轮次 (Epoch)', fontsize=12)
                plt.ylabel('准确率', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                save_path = os.path.join(self.charts_dir, '08_训练历史.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✅ 训练历史图已保存: {save_path}")
    
    def plot_misclassified_samples(self, y_true, y_pred, y_probs, y_paths, num_samples=12):
        """绘制错误分类样本示例"""
        # 找出错误分类的样本
        misclassified_indices = np.where(y_pred != y_true)[0]
        
        if len(misclassified_indices) == 0:
            print("⚠️ 没有错误分类的样本可展示")
            return
        
        # 随机选择一些样本
        if len(misclassified_indices) > num_samples:
            selected_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
        else:
            selected_indices = misclassified_indices
        
        # 创建子图
        num_rows = int(np.ceil(len(selected_indices) / 4))
        fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4*num_rows))
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (ax, sample_idx) in enumerate(zip(axes.flatten(), selected_indices)):
            if idx >= len(selected_indices):
                ax.axis('off')
                continue
            
            try:
                # 加载图片
                img_path = y_paths[sample_idx]
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((150, 150))
                
                # 显示图片
                ax.imshow(img_resized)
                
                # 添加标题信息
                true_label = self.classes[y_true[sample_idx]]
                pred_label = self.classes[y_pred[sample_idx]]
                confidence = np.max(y_probs[sample_idx])
                
                title_color = 'red'  # 错误分类用红色
                ax.set_title(f"真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.3f}", 
                           color=title_color, fontsize=9)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"无法加载图片\n{str(e)[:30]}...", 
                       ha='center', va='center', fontsize=8)
                ax.axis('off')
        
        plt.suptitle('错误分类样本示例', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.charts_dir, '09_错误分类样本.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 错误分类样本图已保存: {save_path}")
    
    def plot_performance_matrix_fixed(self, y_true, y_pred, y_probs):
        """绘制性能对比矩阵 - 修复版"""
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (1, 1):
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
        
        # 计算各种性能指标
        total = tp + tn + fp + fn
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # 创建性能矩阵 - 确保形状正确
        performance_matrix = np.array([
            [accuracy, precision, recall, specificity, f1],
            [tp/total if total > 0 else 0, fp/total if total > 0 else 0, 
             fn/total if total > 0 else 0, tn/total if total > 0 else 0, 
             (fp+fn)/total if total > 0 else 0]
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 子图1: 性能指标雷达图
        ax1 = axes[0, 0]
        metrics = ['准确率', '精确率', '召回率', '特异度', 'F1分数']
        values = [accuracy, precision, recall, specificity, f1]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_radar = values + [values[0]]  # 闭合雷达图
        angles_radar = angles + [angles[0]]
        
        ax1.plot(angles_radar, values_radar, 'o-', linewidth=2)
        ax1.fill(angles_radar, values_radar, alpha=0.25)
        ax1.set_xticks(angles)
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1)
        ax1.set_title('性能指标雷达图', fontsize=14, fontweight='bold')
        ax1.grid(True)
        
        # 子图2: 错误类型分布
        ax2 = axes[0, 1]
        error_labels = ['真阳性', '假阳性', '假阴性', '真阴性', '总错误']
        error_values = [tp, fp, fn, tn, fp+fn]
        error_colors = ['green', 'orange', 'red', 'blue', 'purple']
        
        ax2.bar(error_labels, error_values, color=error_colors, alpha=0.7)
        ax2.set_title('预测结果分布', fontsize=14, fontweight='bold')
        ax2.set_ylabel('样本数量', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值标签
        for i, v in enumerate(error_values):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
        
        # 子图3: 性能指标热力图
        ax3 = axes[1, 0]
        im = ax3.imshow(performance_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax3.set_xticks(range(len(metrics)))
        ax3.set_xticklabels(metrics, rotation=45)
        ax3.set_yticks(range(2))
        ax3.set_yticklabels(['指标值', '样本比例'])
        ax3.set_title('性能指标热力图', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i in range(performance_matrix.shape[0]):
            for j in range(performance_matrix.shape[1]):
                ax3.text(j, i, f'{performance_matrix[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax3, label='分数')
        
        # 子图4: 模型表现总结
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""模型性能总结:
{'='*30}
总样本数: {total}
正确预测: {tp + tn} ({accuracy*100:.2f}%)
错误预测: {fp + fn} ({(fp+fn)/total*100:.2f}% if total > 0 else 0)

详细指标:
准确率: {accuracy:.4f} ({accuracy*100:.2f}%)
精确率: {precision:.4f} ({precision*100:.2f}%)
召回率: {recall:.4f} ({recall*100:.2f}%)
特异度: {specificity:.4f} ({specificity*100:.2f}%)
F1分数: {f1:.4f} ({f1*100:.2f}%)

分类结果:
真阳性(TP): {tp} (正确识别的白内障)
真阴性(TN): {tn} (正确识别的正常)
假阳性(FP): {fp} (正常误判为白内障)
假阴性(FN): {fn} (白内障漏诊)
"""
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
                verticalalignment='center', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title('模型表现总结', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.charts_dir, '10_性能对比矩阵.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 性能对比矩阵图已保存: {save_path}")
    
    def analyze_errors(self, y_true, y_pred, y_probs, y_paths):
        """分析错误分类"""
        # 找出错误分类的样本
        error_indices = np.where(y_pred != y_true)[0]
        
        if len(error_indices) == 0:
            print("✅ 没有错误分类的样本")
            return
        
        print(f"\n错误分析:")
        print(f"总错误数: {len(error_indices)}")
        print(f"错误率: {len(error_indices)/len(y_true)*100:.2f}%")
        
        # 分析错误类型
        false_positives = []
        false_negatives = []
        
        for idx in error_indices:
            if y_true[idx] == 0 and y_pred[idx] == 1:  # 正常被误判为白内障
                false_positives.append(idx)
            elif y_true[idx] == 1 and y_pred[idx] == 0:  # 白内障被漏诊
                false_negatives.append(idx)
        
        print(f"假阳性 (FP, 正常误判): {len(false_positives)}")
        print(f"假阴性 (FN, 白内障漏诊): {len(false_negatives)}")
        
        # 分析错误样本的置信度
        if len(error_indices) > 0 and len(y_probs.shape) > 1:
            error_confidences = np.max(y_probs[error_indices], axis=1)
            print(f"错误样本平均置信度: {np.mean(error_confidences):.3f}")
            print(f"错误样本置信度范围: [{np.min(error_confidences):.3f}, {np.max(error_confidences):.3f}]")
            
            # 统计低置信度错误
            low_confidence_errors = np.sum(error_confidences < 0.7)
            print(f"低置信度错误 (<0.7): {low_confidence_errors}")
    
    def analyze_confidence(self, y_true, y_pred, y_probs):
        """分析模型置信度"""
        if len(y_probs.shape) > 1:
            # 获取预测置信度
            pred_confidences = np.max(y_probs, axis=1)
            
            print(f"\n置信度分析:")
            print(f"平均置信度: {np.mean(pred_confidences):.3f}")
            print(f"置信度中位数: {np.median(pred_confidences):.3f}")
            print(f"置信度标准差: {np.std(pred_confidences):.3f}")
            
            # 置信度分布
            print(f"\n置信度分布:")
            bins = [0, 0.5, 0.7, 0.9, 1.0]
            bin_labels = ['低 (<0.5)', '中 (0.5-0.7)', '高 (0.7-0.9)', '很高 (>0.9)']
            
            for i in range(len(bins)-1):
                count = np.sum((pred_confidences >= bins[i]) & (pred_confidences < bins[i+1]))
                percentage = count / len(pred_confidences) * 100
                print(f"{bin_labels[i]}: {count}个样本 ({percentage:.1f}%)")
            
            # 正确和错误预测的置信度对比
            correct_mask = (y_pred == y_true)
            if np.sum(correct_mask) > 0:
                correct_mean = np.mean(pred_confidences[correct_mask])
                print(f"正确预测平均置信度: {correct_mean:.3f}")
            
            incorrect_mask = (y_pred != y_true)
            if np.sum(incorrect_mask) > 0:
                incorrect_mean = np.mean(pred_confidences[incorrect_mask])
                print(f"错误预测平均置信度: {incorrect_mean:.3f}")
                
                if np.sum(correct_mask) > 0:
                    print(f"置信度差异: {correct_mean - incorrect_mean:.3f}")
    
    def generate_comprehensive_report(self, y_true, y_pred, y_probs, cm):
        """生成综合评估报告"""
        from sklearn.metrics import (accuracy_score, precision_score, 
                                   recall_score, f1_score, classification_report)
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        if cm.shape == (1, 1):
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
        
        total = len(y_true)
        error_rate = (fp + fn) / total if total > 0 else 0
        
        # 置信度统计
        if len(y_probs.shape) > 1:
            confidences = np.max(y_probs, axis=1)
            avg_confidence = np.mean(confidences)
            median_confidence = np.median(confidences)
        else:
            avg_confidence = median_confidence = 0
        
        report = f"""白内障筛查模型综合评估报告
{'='*80}

一、基本信息
评估时间: {np.datetime64('now')}
模型路径: {self.model_path}
数据样本: {total} 个测试样本

二、性能指标汇总
{'='*40}
准确率 (Accuracy):      {accuracy:.4f} ({accuracy*100:.2f}%)
精确率 (Precision):     {precision:.4f} ({precision*100:.2f}%)
召回率 (Recall):        {recall:.4f} ({recall*100:.2f}%)
F1分数 (F1-Score):      {f1:.4f} ({f1*100:.2f}%)
错误率 (Error Rate):    {error_rate:.4f} ({error_rate*100:.2f}%)

三、混淆矩阵分析
{'='*40}
                  预测正常    预测白内障
真实正常        {tn:>6}        {fp:>6}
真实白内障      {fn:>6}        {tp:>6}

真阳性(TP): {tp} (正确识别的白内障)
真阴性(TN): {tn} (正确识别的正常)
假阳性(FP): {fp} (正常误判为白内障)
假阴性(FN): {fn} (白内障漏诊)

四、置信度分析
{'='*40}
平均置信度: {avg_confidence:.3f}
置信度中位数: {median_confidence:.3f}

五、模型评价
{'='*40}
"""
        
        # 根据性能给出评价
        if accuracy >= 0.95:
            report += "✅ 模型性能优秀，非常适合临床应用\n"
        elif accuracy >= 0.90:
            report += "✅ 模型性能良好，适合作为辅助诊断工具\n"
        elif accuracy >= 0.85:
            report += "⚠️ 模型性能一般，建议进一步优化\n"
        else:
            report += "❌ 模型性能较差，需要重新训练或调整\n"
        
        # 建议
        report += f"""
六、改进建议
{'='*40}
1. 关注假阴性({fn}个): 这些是漏诊的白内障病例，临床风险较高
2. 关注假阳性({fp}个): 这些是误判的正常病例，可能导致不必要的检查
"""
        
        if fp > fn:
            report += "3. 模型倾向于过度诊断(假阳性较多)，可适当提高分类阈值\n"
        elif fn > fp:
            report += "3. 模型倾向于保守诊断(假阴性较多)，可适当降低分类阈值\n"
        
        report += f"""
七、图表文件
{'='*40}
所有分析图表已保存至: {self.charts_dir}
包含以下图表:
01_混淆矩阵.png           - 分类结果可视化
02_ROC曲线.png            - 模型区分能力
03_精确率召回率曲线.png    - 模型精确性
04_预测概率分布.png        - 预测置信度分布
05_性能指标对比.png        - 各类指标对比
06_置信度分布.png          - 正确/错误预测的置信度
07_阈值分析.png            - 阈值对性能的影响
08_训练历史.png            - 模型训练过程
09_错误分类样本.png        - 典型错误示例
10_性能对比矩阵.png        - 综合性能展示

{'='*80}
报告生成完成
{'='*80}
"""
        
        report_file = os.path.join(self.results_dir, '综合评估报告.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 综合评估报告已保存: {report_file}")
        
        # 显示报告摘要
        print("\n📋 报告摘要:")
        print(f"   准确率: {accuracy*100:.1f}%")
        print(f"   精确率: {precision*100:.1f}%")
        print(f"   召回率: {recall*100:.1f}%")
        print(f"   错误数: {fp+fn} ({error_rate*100:.1f}%)")
        print(f"   图表数: 10个详细分析图表")

def main():
    """主函数"""
    print("="*80)
    print("修复复杂图表版模型评估模块")
    print("="*80)
    
    # 获取脚本目录
    base_dir = get_script_dir()
    print(f"脚本目录: {base_dir}")
    
    # 检查依赖
    try:
        import sklearn
    except ImportError:
        print("❌ 错误: 缺少 scikit-learn 模块")
        print("请运行: pip install scikit-learn")
        response = input("是否尝试自动安装? (y/n): ")
        if response.lower() == 'y':
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
                print("✅ scikit-learn 安装成功")
            except:
                print("❌ 安装失败，请手动安装")
                return
        else:
            return
    
    # 检查seaborn（可选）
    try:
        import seaborn
        print("✅ seaborn 已安装")
    except ImportError:
        print("⚠️ 注意: seaborn 未安装，部分图表可能不够美观")
        print("   建议安装: pip install seaborn")
    
    # 创建评估器
    evaluator = EnhancedCataractModelEvaluator()
    
    # 加载模型
    if not evaluator.load_model():
        return
    
    try:
        # 执行全面评估
        y_true, y_pred, y_probs, y_paths = evaluator.evaluate_model_comprehensive()
        
        print("\n" + "="*80)
        print("🎉 全面评估完成!")
        print("="*80)
        print(f"\n📊 生成了10个详细的评估图表:")
        print(f"   保存位置: {evaluator.charts_dir}")
        print(f"   1. 混淆矩阵")
        print(f"   2. ROC曲线")
        print(f"   3. 精确率-召回率曲线")
        print(f"   4. 预测概率分布")
        print(f"   5. 性能指标对比")
        print(f"   6. 置信度分布")
        print(f"   7. 阈值分析")
        print(f"   8. 训练历史")
        print(f"   9. 错误分类样本")
        print(f"   10. 性能对比矩阵")
        print(f"\n📋 生成了综合评估报告:")
        print(f"   保存位置: {evaluator.results_dir}/综合评估报告.txt")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()