from flask import Flask, render_template_string
import os

app = Flask(__name__)

# 创建一个最简单的HTML页面
SIMPLE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>测试页面</title>
    <style>
        body { background: #667eea; color: white; font-family: Arial; padding: 50px; }
        h1 { font-size: 48px; }
    </style>
</head>
<body>
    <h1>🎉 Flask 正常运行！</h1>
    <p>如果看到这个页面，说明Flask服务正常启动。</p>
    <p>当前时间：{{ timestamp }}</p>
    <p>工作目录：{{ workdir }}</p>
</body>
</html>
'''

@app.route('/')
def index():
    import time
    import os
    return render_template_string(
        SIMPLE_HTML,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        workdir=os.getcwd()
    )

if __name__ == '__main__':
    print("=" * 60)
    print("测试服务启动...")
    print("访问: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)