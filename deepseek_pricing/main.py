#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek-R1航班定价分析工具 - 主程序
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from data_processor import DataProcessor
from api_client import DeepSeekR1Client
from visualizer import PricingVisualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DeepSeek-R1航班定价分析工具")
    
    parser.add_argument("--data", type=str, default="../航班定价模拟数据.csv",
                        help="数据文件路径，默认为'../航班定价模拟数据.csv'")
    
    parser.add_argument("--output", type=str, default="../定价分析结果",
                        help="输出目录路径，默认为'../定价分析结果'")
    
    parser.add_argument("--mock", action="store_true",
                        help="使用模拟数据，不调用实际API")
    
    parser.add_argument("--query", type=str,
                        help="单个查询，如果提供则只执行这一个查询")
    
    parser.add_argument("--timeout", type=int, default=180,
                        help="API请求超时时间（秒），默认为180秒")
    
    return parser.parse_args()


def get_default_queries() -> List[str]:
    """获取默认的分析查询列表"""
    return [
        "分析当前航线的定价策略，识别哪些航线的定价过高或过低，并提供具体的调价建议。",
        "分析客座率与票价之间的关系，找出最佳的价格点以最大化收入。",
        "分析季节性因素对航线定价的影响，并提供不同季节的差异化定价策略。",
        "分析价格弹性对不同航线的影响，识别哪些航线对价格变化更敏感。",
        "提供一个综合的动态定价优化方案，考虑所有相关因素。"
    ]


def execute_analysis(processor: DataProcessor, client: DeepSeekR1Client, 
                    use_mock: bool = False, single_query: Optional[str] = None,
                    timeout: int = 180) -> Dict[str, Any]:
    """
    执行定价分析
    
    参数:
        processor: 数据处理器
        client: API客户端
        use_mock: 是否使用模拟数据（不调用实际API）
        single_query: 单个查询，如果提供则只执行这一个查询
        timeout: API请求超时时间（秒）
        
    返回:
        分析结果字典
    """
    print("\n[3/4] 执行分析")
    
    # 准备数据摘要
    data_summary = processor.data_summary
    
    # 简化数据用于API请求
    simplified_data = processor.simplify_data_for_api(max_routes=5)
    
    # 确定要执行的查询
    if single_query:
        queries = [single_query]
        print(f"执行单个查询: {single_query}")
    else:
        queries = get_default_queries()
        print(f"执行默认查询列表 ({len(queries)}个查询)")
    
    # 执行分析
    if use_mock:
        print("使用模拟数据，不调用实际API...")
        # 生成模拟结果
        results = {}
        for query in queries:
            results[query] = {
                "reasoning": f"这是针对查询 '{query}' 的模拟推理过程。\n\n在实际使用中，这里会显示DeepSeek-R1的详细推理过程，包括数据分析、趋势识别和策略制定的思考过程。",
                "answer": f"这是针对查询 '{query}' 的模拟回答。\n\n在实际使用中，这里会显示DeepSeek-R1的最终建议和结论。",
                "full_response": {"mock": True}
            }
            time.sleep(0.5)  # 模拟API调用延迟
    else:
        print("调用DeepSeek-R1 API进行分析...")
        # 调用API进行批量分析
        results = client.batch_analyze(simplified_data, queries, timeout=timeout)
    
    # 处理结果
    analysis_results = {
        "queries": queries,
        "results": results,
        "data_summary": data_summary
    }
    
    # 输出分析结果摘要
    print("\n分析结果摘要:")
    for i, query in enumerate(queries):
        result = results[query]
        print(f"查询 {i+1}: {query[:50]}...")
        if "error" in result:
            print(f"  错误: {result['error']}")
        else:
            answer_preview = result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"]
            print(f"  回答: {answer_preview}")
    
    return analysis_results


def generate_visualizations(processor: DataProcessor, analysis_results: Dict[str, Any], 
                           output_dir: str) -> None:
    """
    生成可视化和HTML报告
    
    参数:
        processor: 数据处理器
        analysis_results: 分析结果
        output_dir: 输出目录
    """
    print("\n[4/4] 生成可视化和报告")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化可视化器
    visualizer = PricingVisualizer(processor.df, output_dir)
    
    # 生成各种图表
    visualizer.create_profit_rate_chart()
    visualizer.create_load_factor_price_scatter()
    visualizer.create_price_elasticity_chart()
    visualizer.create_seasonal_index_chart()
    visualizer.create_revenue_cost_chart()
    
    # 生成HTML报告
    html_path = visualizer.generate_html_report(analysis_results)
    
    print(f"已生成HTML报告: {html_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("DeepSeek-R1航班定价分析工具")
    print("=" * 80)
    
    # 解析命令行参数
    args = parse_args()
    
    # 初始化数据处理器
    print("\n[1/4] 数据处理")
    processor = DataProcessor(args.data)
    
    # 加载数据
    processor.load_data(use_mock=args.mock)
    
    # 预处理数据
    df = processor.preprocess_data()
    
    # 计算统计数据
    processor.calculate_statistics()
    
    # 准备数据摘要
    processor.prepare_data_summary()
    
    # 初始化API客户端
    print("\n[2/4] 初始化DeepSeek-R1客户端")
    client = DeepSeekR1Client()
    
    # 执行简单测试查询
    if not args.mock:
        print("执行测试查询...")
        test_response = client.simple_query("你好，请简要介绍一下自己。", timeout=60)
        if "error" in test_response:
            print(f"测试查询失败: {test_response['error']}")
            print("继续执行分析，但可能会遇到API问题...")
        else:
            print("测试查询成功")
    
    # 执行分析
    analysis_results = execute_analysis(
        processor=processor,
        client=client,
        use_mock=args.mock,
        single_query=args.query,
        timeout=args.timeout
    )
    
    # 生成可视化和报告
    generate_visualizations(
        processor=processor,
        analysis_results=analysis_results,
        output_dir=args.output
    )
    
    print("\n分析完成！报告已保存到:", args.output)
    print(f"HTML报告: {os.path.join(args.output, 'DeepSeek-R1航班定价分析报告.html')}")


if __name__ == "__main__":
    main() 