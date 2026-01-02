fro@echo off
chcp 65001 >nul
echo.
echo ============================================
echo    白内障筛查系统 - 本地离线版
echo ============================================
echo.
echo 正在启动本地服务器...
echo 请勿关闭此窗口，在浏览器中访问：
echo http://localhost:5000
echo.
echo 按 Ctrl+C 可以停止服务
echo.

:: 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Python，请先安装Python 3.8+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 删除旧的要求文件
if exist "requirements.txt" del requirements.txt
if exist "setup_log.txt" del setup_log.txt

echo 📦 正在检查Python环境...
python -c "import sys; print(f'Python版本: {sys.version}')"

:: 创建最新的 requirements.txt
echo 正在创建依赖配置文件...
(
echo flask==2.3.3
echo torch==2.9.1
echo torchvision==0.20.1
echo pillow==10.0.0
echo numpy==1.24.3
echo matplotlib==3.7.2
echo scikit-learn==1.3.0
) > requirements.txt

echo ✅ requirements.txt 已创建
type requirements.txt
echo.

:: 升级pip
echo 📦 正在升级pip...
python -m pip install --upgrade pip --user > pip_upgrade.log 2>&1
if errorlevel 1 (
    echo ⚠️ pip升级失败，但继续尝试安装
)

:: 分步安装依赖
echo 📦 正在安装依赖...
echo 1. 安装Flask和其他基础依赖...
pip install flask==2.3.3 pillow==10.0.0 numpy==1.24.3 matplotlib==3.7.2 scikit-learn==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple > install_basic.log 2>&1

if errorlevel 1 (
    echo ⚠️ 基础依赖安装失败，尝试安装最新版本...
    pip install flask pillow numpy matplotlib scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple >> install_basic.log 2>&1
)

:: 安装PyTorch（使用最新的稳定版）
echo.
echo 2. 安装PyTorch（CPU版本）...
echo 这可能需要几分钟，请稍候...

:: 尝试多种PyTorch安装方式
echo 尝试方式1：安装CPU版本...
pip install torch==2.9.1 torchvision==0.20.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu > install_torch.log 2>&1

if errorlevel 1 (
    echo ⚠️ PyTorch 2.9.1 安装失败，尝试最新版本...
    echo 尝试方式2：安装最新CPU版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > install_torch2.log 2>&1
)

if errorlevel 1 (
    echo ⚠️ PyTorch CPU版本安装失败，尝试安装指定版本...
    echo 尝试方式3：安装PyTorch 2.8.0...
    pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu > install_torch3.log 2>&1
)

if errorlevel 1 (
    echo ⚠️ 所有PyTorch版本安装失败，系统将运行在演示模式
    echo 检查PyTorch安装状态...
    python -c "try: import torch; print('✅ PyTorch已安装:', torch.__version__); except: print('❌ PyTorch未安装')" > check_torch.log
    type check_torch.log
) else (
    echo ✅ PyTorch安装成功！
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
)

:: 检查所有依赖
echo.
echo 📋 检查所有依赖...
python -c "
try:
    import flask
    print('✅ Flask:', flask.__version__)
except: print('❌ Flask未安装')

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
except: print('❌ PyTorch未安装')

try:
    import PIL
    print('✅ PIL: 已安装')
except: print('❌ PIL未安装')

try:
    import numpy
    print('✅ NumPy:', numpy.__version__)
except: print('❌ NumPy未安装')

try:
    import sklearn
    print('✅ scikit-learn:', sklearn.__version__)
except: print('❌ scikit-learn未安装')
"

:: 创建必要目录
echo.
echo 📁 创建项目目录结构...
if not exist "app" mkdir app
if not exist "app\uploads" mkdir app\uploads
if not exist "app\models" mkdir app\models
if not exist "app\static" mkdir app\static
if not exist "app\templates" mkdir app\templates

:: 检查模型文件
if not exist "app\models\cataract_resnet18.pth" (
    echo ⚠️ 警告：找不到模型文件
    echo 演示模式下将使用随机预测
    echo 要使用真实模型，请：
    echo 1. 运行模型训练脚本 train_model.py
    echo 2. 或将训练好的模型复制到 app\models\ 目录
    echo.
)

:: 检查是否有 run.py 文件
if not exist "run.py" (
    echo 正在创建基本的 Flask 应用文件...
    
    :: 创建 run.py
    (
echo from flask import Flask, render_template, request, jsonify
echo import os
echo from werkzeug.utils import secure_filename
echo import random
echo 
echo app = Flask(__name__)
echo 
echo # 配置
echo app.config['UPLOAD_FOLDER'] = 'app/uploads'
echo app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
echo ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
echo 
echo def allowed_file(filename):
echo     return '.' in filename and \
echo            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
echo 
echo # 白内障检测函数（演示用）
echo def detect_cataract(image_path):
echo     try:
echo         # 返回随机结果作为演示
echo         probability = random.uniform(0.3, 0.95)
echo         has_cataract = probability > 0.5
echo         severity = random.choice(['轻度', '中度', '重度'])
echo         
echo         return {
echo             'has_cataract': has_cataract,
echo             'probability': round(probability, 2),
echo             'severity': severity,
echo             'recommendation': '建议定期复查' if not has_cataract else '建议就医检查',
echo             'mode': 'demo'
echo         }
echo     except Exception as e:
echo         return {'error': str(e)}
echo 
echo @app.route('/')
echo def index():
echo     return render_template('index.html')
echo 
echo @app.route('/upload', methods=['POST'])
echo def upload_file():
echo     if 'file' not in request.files:
echo         return jsonify({'error': '没有选择文件'})
echo     
echo     file = request.files['file']
echo     if file.filename == '':
echo         return jsonify({'error': '没有选择文件'})
echo     
echo     if file and allowed_file(file.filename):
echo         filename = secure_filename(file.filename)
echo         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
echo         file.save(filepath)
echo         
echo         # 进行白内障检测
echo         result = detect_cataract(filepath)
echo         result['filename'] = filename
echo         
echo         return jsonify(result)
echo     
echo     return jsonify({'error': '不支持的文件类型'})
echo 
echo if __name__ == '__main__':
echo     print('白内障筛查系统启动中...')
echo     print('请访问 http://localhost:5000')
echo     app.run(debug=True, host='0.0.0.0', port=5000)
    ) > run.py
    echo ✅ run.py 已创建
)

:: 检查 templates/index.html
if not exist "app\templates\index.html" (
    if not exist "app\templates" mkdir app\templates
    echo 正在创建前端页面...
    :: ... 这里放之前提供的HTML代码 ...
    echo ⚠️ 前端页面未创建，需要手动创建
)

:: 启动服务
echo.
echo 🚀 正在启动本地Web服务...
echo.
python run.py

pause