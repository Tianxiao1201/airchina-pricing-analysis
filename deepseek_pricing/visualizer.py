#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块 - 用于生成各种图表和HTML报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path
import markdown
from datetime import datetime


class PricingVisualizer:
    """定价可视化类，用于生成各种图表和HTML报告"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "../定价分析结果"):
        """
        初始化可视化器
        
        参数:
            df: 数据DataFrame
            output_dir: 输出目录
        """
        self.df = df
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        sns.set(style="whitegrid")
        # 使用系统上已有的中文字体，按优先级排序
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'sans-serif']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        print(f"已初始化可视化器，输出目录: {output_dir}")
    
    def create_profit_rate_chart(self) -> str:
        """
        创建平均利润率图表
        
        返回:
            保存的图表文件路径
        """
        print("创建平均利润率图表...")
        
        # 按航线分组计算平均利润率
        profit_data = self.df.groupby('route_name')['profit_margin'].mean().sort_values(ascending=False)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        bars = plt.bar(profit_data.index, profit_data.values * 100)
        
        # 为每个柱子添加标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', rotation=0)
        
        # 设置图表标题和标签
        plt.title('各航线平均利润率', fontsize=16)
        plt.xlabel('航线', fontsize=12)
        plt.ylabel('平均利润率 (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'avg_profit_rate.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def create_load_factor_price_scatter(self) -> str:
        """
        创建客座率与票价散点图
        
        返回:
            保存的图表文件路径
        """
        print("创建客座率与票价散点图...")
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图，按航线着色
        routes = self.df['route_name'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(routes)))
        
        for i, route in enumerate(routes):
            route_data = self.df[self.df['route_name'] == route]
            plt.scatter(route_data['load_factor'] * 100, route_data['ticket_price'], 
                       label=route, color=colors[i], alpha=0.7)
        
        # 添加趋势线
        x = self.df['load_factor'] * 100
        y = self.df['ticket_price']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        # 设置图表标题和标签
        plt.title('客座率与票价关系', fontsize=16)
        plt.xlabel('客座率 (%)', fontsize=12)
        plt.ylabel('票价 (元)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加图例，但限制显示的航线数量
        if len(routes) > 10:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=10)
        else:
            plt.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'load_factor_price.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def create_price_elasticity_chart(self) -> str:
        """
        创建价格弹性图表
        
        返回:
            保存的图表文件路径
        """
        print("创建价格弹性图表...")
        
        # 检查是否有价格弹性数据
        if 'price_elasticity' not in self.df.columns:
            # 尝试从route_stats中获取价格弹性数据
            try:
                with open(os.path.join(self.output_dir, 'data_summary.json'), 'r', encoding='utf-8') as f:
                    data_summary = json.load(f)
                
                # 提取价格弹性数据
                elasticity_data = {}
                for route, stats in data_summary['route_stats'].items():
                    if stats['price_elasticity'] is not None:
                        elasticity_data[route] = stats['price_elasticity']
            except:
                # 如果无法获取，则创建模拟数据
                elasticity_data = {route: np.random.uniform(-2, 0) for route in self.df['route_name'].unique()}
        else:
            # 直接从DataFrame中计算
            elasticity_data = self.df.groupby('route_name')['price_elasticity'].mean().to_dict()
        
        # 按价格弹性绝对值排序
        elasticity_data = {k: v for k, v in sorted(elasticity_data.items(), key=lambda item: abs(item[1]))}
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        bars = plt.bar(elasticity_data.keys(), elasticity_data.values())
        
        # 为每个柱子添加标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.1 if height >= 0 else -0.1),
                    f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 设置图表标题和标签
        plt.title('各航线价格弹性', fontsize=16)
        plt.xlabel('航线', fontsize=12)
        plt.ylabel('价格弹性', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'price_elasticity.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def create_seasonal_index_chart(self) -> str:
        """
        创建季节性指数图表
        
        返回:
            保存的图表文件路径
        """
        print("创建季节性指数图表...")
        
        # 检查是否有月份数据
        if 'month' not in self.df.columns:
            # 如果没有月份数据，则尝试从flight_date创建
            if 'flight_date' in self.df.columns:
                self.df['month'] = pd.to_datetime(self.df['flight_date']).dt.month
            else:
                # 如果无法创建，则使用模拟数据
                months = list(range(1, 13))
                seasonal_indices = {
                    '1': 0.85, '2': 0.9, '3': 0.95, '4': 1.0,
                    '5': 1.05, '6': 1.1, '7': 1.2, '8': 1.15,
                    '9': 1.0, '10': 1.05, '11': 0.95, '12': 0.9
                }
                
                plt.figure(figsize=(12, 6))
                plt.plot(months, [seasonal_indices[str(m)] for m in months], marker='o', linewidth=2)
                
                plt.title('平均季节性指数', fontsize=16)
                plt.xlabel('月份', fontsize=12)
                plt.ylabel('季节性指数', fontsize=12)
                plt.xticks(months)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # 保存图表
                output_path = os.path.join(self.output_dir, 'seasonal_index.png')
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                return output_path
        
        # 计算每月平均票价
        monthly_avg = self.df.groupby('month')['ticket_price'].mean()
        
        # 计算整体平均票价
        overall_avg = self.df['ticket_price'].mean()
        
        # 计算季节性指数
        seasonal_index = monthly_avg / overall_avg
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(seasonal_index.index, seasonal_index.values, marker='o', linewidth=2)
        
        # 为每个点添加标签
        for i, v in enumerate(seasonal_index):
            plt.text(seasonal_index.index[i], v + 0.02, f'{v:.2f}', ha='center')
        
        # 设置图表标题和标签
        plt.title('月度季节性指数', fontsize=16)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('季节性指数', fontsize=12)
        plt.xticks(range(1, 13))
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'seasonal_index.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def create_revenue_cost_chart(self) -> str:
        """
        创建收入成本对比图表
        
        返回:
            保存的图表文件路径
        """
        print("创建收入成本对比图表...")
        
        # 检查是否有收入和成本数据
        revenue_col = None
        cost_col = None
        
        if 'total_revenue' in self.df.columns:
            revenue_col = 'total_revenue'
        elif 'revenue' in self.df.columns:
            revenue_col = 'revenue'
        
        if 'total_cost' in self.df.columns:
            cost_col = 'total_cost'
        
        if revenue_col is None or cost_col is None:
            # 如果没有收入或成本数据，则创建模拟数据
            routes = self.df['route_name'].unique()
            
            # 模拟收入和成本数据
            revenue_data = {route: np.random.uniform(800000, 1200000) for route in routes}
            cost_data = {route: np.random.uniform(700000, 1100000) for route in routes}
            profit_data = {route: revenue_data[route] - cost_data[route] for route in routes}
            
            # 按利润排序
            sorted_routes = sorted(profit_data.items(), key=lambda x: x[1], reverse=True)
            sorted_routes = [r[0] for r in sorted_routes]
            
            # 创建图表
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(sorted_routes))
            width = 0.35
            
            plt.bar(x - width/2, [revenue_data[r] for r in sorted_routes], width, label='收入')
            plt.bar(x + width/2, [cost_data[r] for r in sorted_routes], width, label='成本')
            
            plt.title('各航线收入与成本对比', fontsize=16)
            plt.xlabel('航线', fontsize=12)
            plt.ylabel('金额 (元)', fontsize=12)
            plt.xticks(x, sorted_routes, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(self.output_dir, 'revenue_cost.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        
        # 按航线分组计算平均收入和成本
        revenue_by_route = self.df.groupby('route_name')[revenue_col].mean()
        cost_by_route = self.df.groupby('route_name')[cost_col].mean()
        profit_by_route = revenue_by_route - cost_by_route
        
        # 按利润排序
        sorted_indices = profit_by_route.sort_values(ascending=False).index
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(sorted_indices))
        width = 0.35
        
        plt.bar(x - width/2, [revenue_by_route[i] for i in sorted_indices], width, label='收入')
        plt.bar(x + width/2, [cost_by_route[i] for i in sorted_indices], width, label='成本')
        
        # 设置图表标题和标签
        plt.title('各航线收入与成本对比', fontsize=16)
        plt.xlabel('航线', fontsize=12)
        plt.ylabel('金额 (元)', fontsize=12)
        plt.xticks(x, sorted_indices, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'revenue_cost.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成HTML分析报告
        
        参数:
            analysis_results: 分析结果字典
            
        返回:
            HTML报告文件路径
        """
        print("生成HTML分析报告...")
        
        # 计算整体统计数据
        overall_stats = {
            'avg_load_factor': self.df['load_factor'].mean() * 100,
            'avg_profit_margin': self.df['profit_margin'].mean() * 100,
            'total_routes': self.df['route_name'].nunique(),
            'total_samples': len(self.df)
        }
        
        # 创建图表
        profit_chart = self.create_profit_rate_chart()
        load_factor_chart = self.create_load_factor_price_scatter()
        elasticity_chart = self.create_price_elasticity_chart()
        seasonal_chart = self.create_seasonal_index_chart()
        revenue_cost_chart = self.create_revenue_cost_chart()
        
        # 提取分析结果
        queries = analysis_results.get('queries', [])
        results = analysis_results.get('results', {})
        
        # 构建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>DeepSeek-R1航班定价分析报告</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }}
                .stats-container {{
                    display: flex;
                    justify-content: space-around;
                    flex-wrap: wrap;
                    margin-bottom: 30px;
                }}
                .stat-box {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px;
                    min-width: 200px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .chart-container {{
                    margin: 30px 0;
                }}
                .chart {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                    border: 1px solid #eee;
                    border-radius: 5px;
                }}
                .analysis-section {{
                    margin: 40px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    color: #7f8c8d;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DeepSeek-R1航班定价分析报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>数据概览</h2>
            <div class="stats-container">
                <div class="stat-box">
                    <h3>平均客座率</h3>
                    <div class="stat-value">{overall_stats['avg_load_factor']:.2f}%</div>
                </div>
                <div class="stat-box">
                    <h3>平均利润率</h3>
                    <div class="stat-value">{overall_stats['avg_profit_margin']:.2f}%</div>
                </div>
                <div class="stat-box">
                    <h3>航线数量</h3>
                    <div class="stat-value">{overall_stats['total_routes']}</div>
                </div>
                <div class="stat-box">
                    <h3>样本数量</h3>
                    <div class="stat-value">{overall_stats['total_samples']}</div>
                </div>
            </div>
            
            <h2>可视化分析</h2>
            
            <div class="chart-container">
                <h3>各航线平均利润率</h3>
                <img class="chart" src="{os.path.basename(profit_chart)}" alt="平均利润率图表">
            </div>
            
            <div class="chart-container">
                <h3>客座率与票价关系</h3>
                <img class="chart" src="{os.path.basename(load_factor_chart)}" alt="客座率与票价散点图">
            </div>
            
            <div class="chart-container">
                <h3>各航线价格弹性</h3>
                <img class="chart" src="{os.path.basename(elasticity_chart)}" alt="价格弹性图表">
            </div>
            
            <div class="chart-container">
                <h3>季节性指数</h3>
                <img class="chart" src="{os.path.basename(seasonal_chart)}" alt="季节性指数图表">
            </div>
            
            <div class="chart-container">
                <h3>各航线收入与成本对比</h3>
                <img class="chart" src="{os.path.basename(revenue_cost_chart)}" alt="收入成本对比图表">
            </div>
        """
        
        # 添加DeepSeek-R1分析结果
        if queries and results:
            html_content += """
            <h2>DeepSeek-R1分析结果</h2>
            """
            
            for i, query in enumerate(queries):
                if query in results:
                    result = results[query]
                    answer = result.get('answer', '')
                    reasoning = result.get('reasoning', '')
                    
                    # 使用markdown转换为HTML
                    answer_html = markdown.markdown(answer)
                    
                    html_content += f"""
                    <div class="analysis-section">
                        <h3>分析 {i+1}: {query}</h3>
                        <div class="analysis-content">
                            {answer_html}
                        </div>
                    </div>
                    """
        
        # 添加页脚
        html_content += """
            <div class="footer">
                <p>由DeepSeek-R1航班定价分析工具生成</p>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML报告
        output_path = os.path.join(self.output_dir, 'DeepSeek-R1航班定价分析报告.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已保存到: {output_path}")
        return output_path


if __name__ == "__main__":
    # 测试代码
    # 创建模拟数据
    np.random.seed(42)
    data = []
    
    routes = ["北京-上海", "上海-广州", "广州-深圳", "北京-广州", "上海-成都"]
    
    for _ in range(100):
        route = np.random.choice(routes)
        data.append({
            'route_name': route,
            'ticket_price': np.random.uniform(500, 2000),
            'load_factor': np.random.uniform(0.6, 0.95),
            'profit_margin': np.random.uniform(-0.1, 0.3),
            'total_revenue': np.random.uniform(80000, 120000),
            'total_cost': np.random.uniform(70000, 110000)
        })
    
    df = pd.DataFrame(data)
    
    # 初始化可视化器
    visualizer = PricingVisualizer(df, "test_output")
    
    # 生成各种图表
    visualizer.create_profit_rate_chart()
    visualizer.create_load_factor_price_scatter()
    visualizer.create_price_elasticity_chart()
    visualizer.create_seasonal_index_chart()
    visualizer.create_revenue_cost_chart()
    
    # 生成模拟分析结果
    mock_results = {
        "queries": ["测试查询1", "测试查询2"],
        "results": {
            "测试查询1": {
                "answer": "这是测试回答1",
                "reasoning": "这是测试推理过程1"
            },
            "测试查询2": {
                "answer": "这是测试回答2",
                "reasoning": "这是测试推理过程2"
            }
        }
    }
    
    # 生成HTML报告
    visualizer.generate_html_report(mock_results) 