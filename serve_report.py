#!/usr/bin/env python3
"""
用于在本地启动HTTP服务器，以便正确查看HTML报告中的图表。
由于浏览器的安全限制，直接在本地打开HTML文件可能会导致某些资源加载失败，
通过HTTP服务器可以解决这个问题。
"""
import http.server
import socketserver
import os
import webbrowser
import time

# 设置端口号
PORT = 8000

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 创建HTTP请求处理器
Handler = http.server.SimpleHTTPRequestHandler

# 更改工作目录到脚本所在目录
ios.chdir(script_dir)

# 创建TCP服务器
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"本地HTTP服务器已启动，端口: {PORT}")
    print(f"请在浏览器中访问: http://localhost:{PORT}/msl_anomaly_detection_report.html")
    
    try:
        # 尝试自动打开浏览器
        webbrowser.open(f"http://localhost:{PORT}/msl_anomaly_detection_report.html")
        
        # 持续运行服务器，直到用户按下Ctrl+C
        print("服务器运行中...按Ctrl+C停止")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        httpd.server_close()