#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-R1 API测试脚本 - 用于测试DeepSeek-R1 API的调用
"""

import json
import time
from api_client import DeepSeekR1Client


def test_simple_query():
    """测试简单查询"""
    client = DeepSeekR1Client()
    
    # 简单查询
    query = "请分析航空公司定价策略中，客座率与票价之间的关系。"
    
    # 构建简单的测试数据
    test_data = {
        "description": "这是一个简单的测试查询，不需要复杂的数据。"
    }
    
    print("=" * 80)
    print(f"测试简单查询: {query}")
    print("=" * 80)
    
    # 调用API
    response = client.analyze_pricing_data(test_data, query)
    
    # 提取推理过程和最终答案
    reasoning, answer = client.extract_reasoning_and_answer(response)
    
    # 打印结果
    print("\n=== 推理过程 ===")
    print(reasoning if reasoning else "无推理过程")
    
    print("\n=== 最终答案 ===")
    print(answer)
    
    # 打印完整响应（可选）
    print("\n=== 完整响应 ===")
    print(json.dumps(response, ensure_ascii=False, indent=2))


def test_with_real_data():
    """使用真实数据测试"""
    client = DeepSeekR1Client()
    
    # 加载数据摘要
    try:
        with open("data_summary.json", "r", encoding="utf-8") as f:
            data_summary = json.load(f)
        print(f"已加载数据摘要，包含{len(data_summary.get('route_stats', []))}条航线统计信息")
    except FileNotFoundError:
        print("未找到data_summary.json文件，将使用简化数据")
        # 使用简化数据
        data_summary = {
            "overall_stats": {
                "avg_load_factor": 0.85,
                "avg_ticket_price": 1500.0,
                "avg_profit_rate": 0.15
            },
            "route_stats": [
                {
                    "route_name": "北京-上海",
                    "avg_load_factor": 0.9,
                    "avg_ticket_price": 1800.0,
                    "avg_profit_rate": 0.25
                },
                {
                    "route_name": "北京-广州",
                    "avg_load_factor": 0.85,
                    "avg_ticket_price": 2200.0,
                    "avg_profit_rate": 0.2
                },
                {
                    "route_name": "上海-成都",
                    "avg_load_factor": 0.75,
                    "avg_ticket_price": 1500.0,
                    "avg_profit_rate": 0.1
                }
            ]
        }
    
    # 测试查询
    query = "基于这些航班数据，请分析当前定价策略的优缺点，并提出改进建议。"
    
    print("=" * 80)
    print(f"使用真实数据测试: {query}")
    print("=" * 80)
    
    # 调用API
    response = client.analyze_pricing_data(data_summary, query)
    
    # 提取推理过程和最终答案
    reasoning, answer = client.extract_reasoning_and_answer(response)
    
    # 打印结果
    print("\n=== 推理过程 ===")
    print(reasoning if reasoning else "无推理过程")
    
    print("\n=== 最终答案 ===")
    print(answer)


if __name__ == "__main__":
    # 测试简单查询
    test_simple_query()
    
    print("\n" + "=" * 80 + "\n")
    
    # 等待一段时间，避免API限流
    time.sleep(5)
    
    # 使用真实数据测试
    test_with_real_data() 