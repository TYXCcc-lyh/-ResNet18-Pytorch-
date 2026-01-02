from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_pdf_report(result, image_path, output_path):
    """生成PDF报告"""
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # 标题
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 750, "白内障筛查报告")
    
    # 基本信息
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, f"分析时间: {result['timestamp']}")
    c.drawString(100, 680, f"文件名: {os.path.basename(image_path)}")
    
    # 结果
    c.setFont("Helvetica-Bold", 16)
    if result['is_cataract']:
        c.drawString(100, 640, f"诊断结果: ⚠️ 疑似白内障")
        c.setFillColorRGB(1, 0, 0)  # 红色
    else:
        c.drawString(100, 640, f"诊断结果: ✅ 未见异常")
        c.setFillColorRGB(0, 0.5, 0)  # 绿色
    
    # 恢复颜色
    c.setFillColorRGB(0, 0, 0)
    
    # 概率
    c.drawString(100, 600, f"正常概率: {result['probability_normal']}%")
    c.drawString(100, 580, f"白内障概率: {result['probability_cataract']}%")
    c.drawString(100, 560, f"置信度: {result['confidence']}%")
    
    # 建议
    c.drawString(100, 520, "医疗建议:")
    c.drawString(100, 500, result['recommendation'])
    
    # 图片
    try:
        img = ImageReader(image_path)
        c.drawImage(img, 100, 300, width=200, height=150, preserveAspectRatio=True)
    except:
        pass
    
    # 免责声明
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(100, 100, "⚠️ 本报告为AI辅助筛查结果，不能替代专业医生诊断。")
    
    c.save()