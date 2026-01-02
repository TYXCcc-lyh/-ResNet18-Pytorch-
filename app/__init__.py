from flask import Flask
import os

def create_app():
    """创建Flask应用实例"""
    app = Flask(__name__)
    
    # 基础配置
    app.config['SECRET_KEY'] = 'cataract-screening-local-2023'
    app.config['UPLOAD_FOLDER'] = 'app/uploads'
    app.config['MODEL_FOLDER'] = 'app/models'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
    
    # 允许的文件扩展名
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    
    # 创建必要目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    
    # 注册蓝图
    from .views import main_bp
    app.register_blueprint(main_bp)
    
    return app