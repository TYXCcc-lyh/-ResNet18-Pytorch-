#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白内障筛查系统 - 本地离线版
修复 float32 JSON 序列化问题
"""

import os
import sys
import json
import random
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# 自定义 JSON 编码器，处理 NumPy 类型
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyJSONEncoder, self).default(obj)

def create_app():
    """创建 Flask 应用"""
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(current_dir, 'app', 'templates')
    static_dir = os.path.join(current_dir, 'app', 'static')
    upload_dir = os.path.join(current_dir, 'app', 'uploads')
    
    # 创建必要的目录
    for directory in [template_dir, static_dir, upload_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 创建 Flask 应用
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # 使用自定义 JSON 编码器
    app.json_encoder = NumpyJSONEncoder
    
    # 配置
    app.config['UPLOAD_FOLDER'] = upload_dir
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def convert_numpy_types(obj):
        """递归转换 NumPy 类型为 Python 原生类型"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # 白内障检测函数（演示用）
    def detect_cataract(image_path):
        """分析眼底图像，检测白内障（演示模式）"""
        try:
            # 生成随机结果（演示用）
            probability = random.uniform(0.3, 0.95)
            has_cataract = probability > 0.5
            
            if probability < 0.3:
                severity = "正常"
            elif probability < 0.6:
                severity = "轻度"
            elif probability < 0.8:
                severity = "中度"
            else:
                severity = "重度"
            
            if not has_cataract:
                recommendation = "未发现明显白内障迹象，建议定期复查。"
            else:
                if severity == "轻度":
                    recommendation = "发现轻度白内障迹象，建议3-6个月后复查，注意眼部健康。"
                elif severity == "中度":
                    recommendation = "发现中度白内障，建议就医进行详细检查，考虑治疗方案。"
                else:
                    recommendation = "发现重度白内障，建议尽快就医治疗，可能需要手术干预。"
            
            # 创建结果字典，确保所有值都是 Python 原生类型
            result = {
                'success': True,
                'is_cataract': bool(has_cataract),
                'confidence': float(probability * 100),  # 转换为 Python float
                'probability_cataract': float(probability * 100),
                'probability_normal': float((1 - probability) * 100),
                'severity': severity,
                'recommendation': recommendation,
                'prediction': '疑似白内障' if has_cataract else '正常'
            }
            
            # 再次确保转换所有类型
            result = convert_numpy_types(result)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'is_cataract': False,
                'confidence': 0.0,
                'probability_cataract': 0.0,
                'probability_normal': 0.0,
                'severity': '未知',
                'recommendation': '分析过程中出现错误'
            }
    
    # 路由定义
    @app.route('/')
    def index():
        """首页"""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """处理单张图片上传"""
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            })
        
        if file and allowed_file(file.filename):
            try:
                # 保存文件
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # 进行白内障检测
                result = detect_cataract(filepath)
                
                # 添加文件名和时间戳
                result['filename'] = filename
                result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 确保所有值都是 Python 原生类型
                result = convert_numpy_types(result)
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'处理文件时出错: {str(e)}'
                })
        
        return jsonify({
            'success': False,
            'error': '不支持的文件类型'
        })
    
    @app.route('/batch_upload', methods=['POST'])
    def batch_upload():
        """处理批量图片上传"""
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            })
        
        files = request.files.getlist('files')
        if len(files) == 0:
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            })
        
        results = []
        success_count = 0
        cataract_count = 0
        normal_count = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # 保存文件
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # 进行白内障检测
                    result = detect_cataract(filepath)
                    
                    # 添加文件名
                    result['filename'] = filename
                    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 统计
                    if result['success']:
                        success_count += 1
                        if result['is_cataract']:
                            cataract_count += 1
                        else:
                            normal_count += 1
                    
                    # 确保所有值都是 Python 原生类型
                    result = convert_numpy_types(result)
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        'success': False,
                        'filename': file.filename,
                        'error': str(e)
                    })
        
        summary = {
            'total': len(files),
            'success': success_count,
            'failed': len(files) - success_count,
            'cataract': cataract_count,
            'normal': normal_count
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': results
        })
    
    @app.route('/api/health')
    def health_check():
        """健康检查接口"""
        try:
            return jsonify({
                'status': 'healthy',
                'model_loaded': False,  # 演示模式
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """提供静态文件"""
        return send_from_directory(static_dir, filename)
    
    return app

if __name__ == '__main__':
    import webbrowser
    from threading import Timer
    
    def open_browser():
        """自动打开浏览器"""
        webbrowser.open('http://localhost:5000')
    
    # 创建应用实例
    app = create_app()
    
    # 在启动后1秒自动打开浏览器
    Timer(1, open_browser).start()
    
    print("=" * 50)
    print("白内障筛查系统 - 本地离线版")
    print("=" * 50)
    print("服务地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("-" * 50)
    
    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  # 开发环境设为True以便调试
        threaded=True
    )