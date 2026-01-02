from flask import Blueprint, render_template, request, jsonify, send_file
from .predictor import predictor
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io

main_bp = Blueprint('main', __name__)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

@main_bp.route('/')
def index():
    """首页"""
    return render_template('index.html')

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和预测"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        # 安全保存文件名
        filename = secure_filename(file.filename)
        filepath = os.path.join('app/uploads', filename)
        
        try:
            # 保存文件
            file.save(filepath)
            
            # 进行预测
            result = predictor.predict(filepath)
            
            # 添加文件名到结果中
            result['filename'] = filename
            
            # 返回结果
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
        finally:
            # 清理文件（可选，可以保留用于记录）
            # if os.path.exists(filepath):
            #     os.remove(filepath)
            pass
    
    return jsonify({'success': False, 'error': '不支持的文件类型'})

@main_bp.route('/batch_upload', methods=['POST'])
def batch_upload():
    """批量上传和预测"""
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('app/uploads', filename)
            
            try:
                file.save(filepath)
                result = predictor.predict(filepath)
                result['filename'] = filename
                results.append(result)
            except Exception as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
    
    # 统计结果
    total = len(results)
    success_count = sum(1 for r in results if r.get('success', False))
    cataract_count = sum(1 for r in results if r.get('is_cataract', False))
    
    return jsonify({
        'success': True,
        'results': results,
        'summary': {
            'total': total,
            'success': success_count,
            'cataract': cataract_count,
            'normal': success_count - cataract_count
        }
    })

@main_bp.route('/api/health')
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'model_path': predictor.model_path
    })

@main_bp.route('/api/model_info')
def model_info():
    """模型信息接口"""
    if predictor.model is None:
        return jsonify({'error': '模型未加载'})
    
    return jsonify({
        'classes': predictor.classes,
        'device': str(predictor.device),
        'model_loaded': True
    })

# 静态文件路由
@main_bp.route('/static/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_file(f'app/static/{filename}')