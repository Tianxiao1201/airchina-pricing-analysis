#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-R1 API简单测试脚本 - 使用最基本的API调用来测试
"""

import os
import json
import requests
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(".env")

# 获取API密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

# 设置请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 设置请求体
payload = {
    "model": "deepseek-reasoner",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Please provide a short response."}
    ],
    "temperature": 0.2,
    "max_tokens": 100,
    "stream": False
}

# 打印请求信息
print(f"API密钥: {api_key[:5]}...{api_key[-5:]}")
print(f"API URL: {base_url}/chat/completions")
print(f"请求体: {json.dumps(payload, ensure_ascii=False)}")

# 发送请求
print("\n正在发送API请求...")
start_time = time.time()

try:
    # 设置超时时间为30秒
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    end_time = time.time()
    print(f"请求耗时: {end_time - start_time:.2f}秒")
    
    # 打印响应状态码
    print(f"响应状态码: {response.status_code}")
    
    # 如果请求成功，打印响应内容
    if response.status_code == 200:
        response_json = response.json()
        print("\n响应内容:")
        print(json.dumps(response_json, ensure_ascii=False, indent=2))
        
        # 提取回答
        if "choices" in response_json and len(response_json["choices"]) > 0:
            content = response_json["choices"][0]["message"]["content"]
            print("\n回答内容:")
            print(content)
    else:
        print("\n请求失败:")
        print(response.text)
except requests.exceptions.Timeout:
    print("\n请求超时! API调用可能需要较长时间，或者服务器可能无响应。")
except requests.exceptions.RequestException as e:
    print(f"\n请求异常: {e}")
except Exception as e:
    print(f"\n其他错误: {e}") 