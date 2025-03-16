#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理模块 - 用于加载、预处理和分析航班定价数据
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CustomJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理Pandas的Timestamp类型和其他不可JSON序列化的数据类型
    """
    def default(self, obj):
        # 处理Pandas的Timestamp类型
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        # 处理numpy的数据类型
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        # 处理datetime类型
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        # 处理其他类型
        return super(CustomJSONEncoder, self).default(obj)


class DataProcessor:
    """数据处理类，用于加载、预处理和分析航班定价数据"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化数据处理器
        
        参数:
            data_path: 数据文件路径，如果为None则使用模拟数据
        """
        self.data_path = data_path
        self.df = None
        self.route_stats = None
        self.data_summary = None
        
        print("数据处理模块已初始化")
    
    def load_data(self, use_mock: bool = False) -> pd.DataFrame:
        """
        加载数据
        
        参数:
            use_mock: 是否使用模拟数据
            
        返回:
            加载的数据DataFrame
        """
        if use_mock:
            print("使用模拟数据...")
            self.df = self._generate_mock_data()
        elif self.data_path:
            print(f"从文件加载数据: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            print(f"已加载 {len(self.df)} 行, {len(self.df.columns)} 列")
        else:
            raise ValueError("未指定数据路径且未使用模拟数据")
        
        return self.df
    
    def _generate_mock_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        生成模拟航班定价数据
        
        参数:
            n_samples: 样本数量
            
        返回:
            模拟数据DataFrame
        """
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 定义航线
        routes = [
            "北京-上海", "上海-广州", "广州-深圳", "北京-广州", "上海-成都",
            "成都-重庆", "北京-西安", "上海-杭州", "广州-海口", "北京-哈尔滨",
            "上海-青岛", "广州-昆明", "北京-成都", "上海-厦门", "广州-长沙",
            "北京-沈阳", "上海-南京", "广州-桂林", "北京-大连", "上海-武汉"
        ]
        
        # 生成日期范围（过去一年）
        end_date = pd.Timestamp.now().floor('D')
        start_date = end_date - pd.Timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 初始化数据列表
        data = []
        
        for _ in range(n_samples):
            # 随机选择航线和日期
            route = np.random.choice(routes)
            date = np.random.choice(dates)
            
            # 提取出发地和目的地
            origin, destination = route.split('-')
            
            # 生成基础票价（根据航线不同而不同）
            base_price = np.random.uniform(500, 2000)
            
            # 添加季节性因素
            month = date.month
            if month in [1, 2, 7, 8]:  # 寒暑假，高峰期
                seasonal_factor = np.random.uniform(1.2, 1.5)
            elif month in [5, 10]:  # 五一、十一，次高峰
                seasonal_factor = np.random.uniform(1.1, 1.3)
            else:  # 平季
                seasonal_factor = np.random.uniform(0.8, 1.1)
            
            # 计算实际票价
            ticket_price = base_price * seasonal_factor
            
            # 生成座位数和售出座位数
            total_seats = np.random.choice([120, 150, 180, 200, 250])
            load_factor = np.random.beta(5, 2)  # 使用Beta分布生成0-1之间的负载因子
            sold_seats = int(total_seats * load_factor)
            
            # 计算收入
            revenue = sold_seats * ticket_price
            
            # 生成成本（固定成本+可变成本）
            fixed_cost = np.random.uniform(30000, 50000)  # 固定成本
            variable_cost_per_seat = np.random.uniform(100, 300)  # 每座位可变成本
            total_cost = fixed_cost + variable_cost_per_seat * total_seats
            
            # 计算利润和利润率
            profit = revenue - total_cost
            profit_margin = profit / revenue if revenue > 0 else 0
            
            # 添加到数据列表
            data.append({
                'date': date,
                'route_name': route,
                'origin': origin,
                'destination': destination,
                'ticket_price': round(ticket_price, 2),
                'base_price': round(base_price, 2),
                'seasonal_factor': round(seasonal_factor, 2),
                'total_seats': total_seats,
                'sold_seats': sold_seats,
                'load_factor': round(load_factor, 2),
                'revenue': round(revenue, 2),
                'fixed_cost': round(fixed_cost, 2),
                'variable_cost_per_seat': round(variable_cost_per_seat, 2),
                'total_cost': round(total_cost, 2),
                'profit': round(profit, 2),
                'profit_margin': round(profit_margin, 2)
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        print(f"已生成 {len(df)} 行模拟数据")
        
        return df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        预处理数据
        
        返回:
            预处理后的数据DataFrame
        """
        if self.df is None:
            raise ValueError("数据尚未加载，请先调用load_data方法")
        
        print("正在预处理数据...")
        
        # 确保日期列为日期类型
        if 'flight_date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['flight_date'])
        elif 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 添加月份和星期几列，用于分析季节性和周内模式
        if 'date' in self.df.columns:
            self.df['month'] = self.df['date'].dt.month
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
        elif 'day_of_week' not in self.df.columns and 'flight_date' in self.df.columns:
            # 如果没有date列但有flight_date列，直接使用flight_date
            self.df['month'] = pd.to_datetime(self.df['flight_date']).dt.month
            self.df['day_of_week'] = pd.to_datetime(self.df['flight_date']).dt.dayofweek
        
        # 检查并处理缺失值
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"发现缺失值:\n{missing_values[missing_values > 0]}")
            
            # 对于数值列，用中位数填充缺失值
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            
            # 对于分类列，用众数填充缺失值
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # 确保所有必要的列都存在
        # 检查并处理profit_margin列
        if 'profit_margin' not in self.df.columns and 'profit_rate' in self.df.columns:
            self.df['profit_margin'] = self.df['profit_rate']  # profit_rate已经是小数形式，不需要除以100
        
        # 检查并处理ticket_price列
        if 'ticket_price' not in self.df.columns and 'avg_ticket_price' in self.df.columns:
            self.df['ticket_price'] = self.df['avg_ticket_price']
        
        # 检查并处理异常值（使用IQR方法）
        numeric_cols = ['ticket_price', 'load_factor', 'profit_margin']
        for col in numeric_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 统计异常值数量
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"列 '{col}' 中发现 {outliers} 个异常值")
                    
                    # 将异常值替换为边界值
                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
        
        # 确保所有必要的列都存在
        required_cols = ['route_name', 'ticket_price', 'load_factor', 'profit_margin']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"警告: 数据中缺少必要的列: {missing_cols}")
            # 尝试从其他列创建缺失的列
            if 'route_name' not in self.df.columns and 'departure_city' in self.df.columns and 'arrival_city' in self.df.columns:
                self.df['route_name'] = self.df['departure_city'] + '-' + self.df['arrival_city']
                print("已创建route_name列")
        
        print("数据预处理完成")
        return self.df
    
    def calculate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        计算各航线的统计数据
        
        返回:
            航线统计数据字典
        """
        if self.df is None:
            raise ValueError("数据尚未加载，请先调用load_data方法")
        
        print("正在计算统计数据...")
        
        # 按航线分组计算统计数据
        self.route_stats = {}
        
        # 确保使用正确的列名
        route_col = 'route_name' if 'route_name' in self.df.columns else 'route'
        
        # 检查是否有sold_seats列，如果没有则使用booked_seats列
        if 'sold_seats' not in self.df.columns and 'booked_seats' in self.df.columns:
            self.df['sold_seats'] = self.df['booked_seats']
        
        for route, group in self.df.groupby(route_col):
            # 基本统计量
            avg_price = group['ticket_price'].mean()
            avg_load_factor = group['load_factor'].mean()
            avg_profit_margin = group['profit_margin'].mean()
            
            # 计算价格弹性（如果有足够数据）
            price_elasticity = None
            if len(group) >= 10 and 'sold_seats' in self.df.columns:
                # 简化的价格弹性计算
                # 按价格排序并分组
                sorted_group = group.sort_values('ticket_price')
                n = len(sorted_group)
                mid = n // 2
                
                low_price_group = sorted_group.iloc[:mid]
                high_price_group = sorted_group.iloc[mid:]
                
                avg_low_price = low_price_group['ticket_price'].mean()
                avg_high_price = high_price_group['ticket_price'].mean()
                
                avg_low_demand = low_price_group['sold_seats'].mean()
                avg_high_demand = high_price_group['sold_seats'].mean()
                
                # 计算价格变化百分比和需求变化百分比
                price_change_pct = (avg_high_price - avg_low_price) / avg_low_price
                demand_change_pct = (avg_high_demand - avg_low_demand) / avg_low_demand
                
                if price_change_pct != 0:
                    price_elasticity = demand_change_pct / price_change_pct
            elif 'price_elasticity' in self.df.columns:
                # 如果数据中已经有价格弹性列，直接使用
                price_elasticity = group['price_elasticity'].mean()
            
            # 计算季节性指数
            if 'month' in self.df.columns:
                monthly_avg = group.groupby('month')['ticket_price'].mean()
                overall_avg = group['ticket_price'].mean()
                seasonal_index = {month: (avg / overall_avg) for month, avg in monthly_avg.items()}
            elif 'seasonal_index' in self.df.columns:
                # 如果数据中已经有季节性指数列，直接使用
                seasonal_index = {'avg': group['seasonal_index'].mean()}
            else:
                seasonal_index = {'avg': 1.0}
            
            # 存储统计数据
            self.route_stats[route] = {
                'avg_price': round(avg_price, 2),
                'avg_load_factor': round(avg_load_factor, 2),
                'avg_profit_margin': round(avg_profit_margin, 2),
                'price_elasticity': round(price_elasticity, 2) if price_elasticity is not None else None,
                'seasonal_index': {str(k): round(v, 2) for k, v in seasonal_index.items()},
                'sample_count': len(group)
            }
        
        print(f"已计算 {len(self.route_stats)} 条航线的统计数据")
        return self.route_stats
    
    def prepare_data_summary(self) -> Dict[str, Any]:
        """
        准备数据摘要，用于API分析
        
        返回:
            数据摘要字典
        """
        if self.df is None or self.route_stats is None:
            raise ValueError("数据尚未完全处理，请先调用load_data、preprocess_data和calculate_statistics方法")
        
        print("正在准备数据摘要...")
        
        # 计算整体统计数据
        overall_stats = {
            'avg_price': round(self.df['ticket_price'].mean(), 2),
            'avg_load_factor': round(self.df['load_factor'].mean(), 2),
            'avg_profit_margin': round(self.df['profit_margin'].mean(), 2),
            'total_routes': len(self.route_stats),
            'total_samples': len(self.df),
            'date_range': {
                'start': self.df['date'].min().strftime('%Y-%m-%d'),
                'end': self.df['date'].max().strftime('%Y-%m-%d')
            }
        }
        
        # 找出表现最好和最差的航线
        route_performance = [(route, stats['avg_profit_margin']) 
                            for route, stats in self.route_stats.items()]
        
        best_routes = sorted(route_performance, key=lambda x: x[1], reverse=True)[:3]
        worst_routes = sorted(route_performance, key=lambda x: x[1])[:3]
        
        # 准备数据摘要
        self.data_summary = {
            'overall_stats': overall_stats,
            'best_performing_routes': [{'route': route, 'profit_margin': round(margin, 2)} 
                                     for route, margin in best_routes],
            'worst_performing_routes': [{'route': route, 'profit_margin': round(margin, 2)} 
                                      for route, margin in worst_routes],
            'route_stats': self.route_stats,
            # 添加一些样本数据，但限制数量以避免API请求过大
            'sample_data': self.df.head(10).to_dict(orient='records')
        }
        
        # 保存数据摘要到JSON文件
        output_dir = Path("../定价分析结果")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "data_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.data_summary, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        
        print(f"数据摘要已准备完成并保存到 {output_dir / 'data_summary.json'}")
        return self.data_summary
    
    def simplify_data_for_api(self, max_routes: int = 20, keep_all_data: bool = False) -> Dict[str, Any]:
        """
        简化数据以便于API调用，减少数据量
        
        参数:
            max_routes: 最大航线数量，默认为20条
            keep_all_data: 是否保留所有数据，如果为True则忽略max_routes参数
            
        返回:
            简化后的数据字典
        """
        if self.data_summary is None:
            raise ValueError("数据摘要尚未准备，请先调用prepare_data_summary方法")
        
        # 复制数据摘要
        simplified_data = self.data_summary.copy()
        
        if keep_all_data:
            print("保留所有航线数据...")
            # 保留所有航线数据，只移除样本数据以减少数据量
            if 'sample_data' in simplified_data:
                del simplified_data['sample_data']
            
            print(f"数据已简化，保留了所有 {len(simplified_data['route_stats'])} 条航线的数据")
            return simplified_data
        
        print(f"正在简化数据，最多保留 {max_routes} 条航线...")
        
        # 保留表现最好和最差的航线，以及一些中间表现的航线
        best_routes = [item['route'] for item in simplified_data['best_performing_routes']]
        worst_routes = [item['route'] for item in simplified_data['worst_performing_routes']]
        
        # 找出中间表现的航线
        all_routes = list(simplified_data['route_stats'].keys())
        middle_routes = [route for route in all_routes 
                        if route not in best_routes and route not in worst_routes]
        
        # 如果中间航线太多，随机选择一些
        import random
        random.seed(42)  # 设置随机种子以确保可重复性
        
        if len(middle_routes) > max_routes - len(best_routes) - len(worst_routes):
            middle_routes = random.sample(
                middle_routes, 
                max(0, max_routes - len(best_routes) - len(worst_routes))
            )
        
        # 合并所有选定的航线
        selected_routes = best_routes + middle_routes + worst_routes
        selected_routes = selected_routes[:max_routes]  # 确保不超过最大数量
        
        # 只保留选定航线的统计数据
        simplified_data['route_stats'] = {
            route: stats for route, stats in simplified_data['route_stats'].items()
            if route in selected_routes
        }
        
        # 移除样本数据以减少数据量
        if 'sample_data' in simplified_data:
            del simplified_data['sample_data']
        
        print(f"数据已简化，保留了 {len(simplified_data['route_stats'])} 条航线的数据")
        return simplified_data


if __name__ == "__main__":
    # 测试代码
    processor = DataProcessor()
    
    # 加载模拟数据
    df = processor.load_data(use_mock=True)
    
    # 预处理数据
    df = processor.preprocess_data()
    
    # 计算统计数据
    route_stats = processor.calculate_statistics()
    
    # 准备数据摘要
    data_summary = processor.prepare_data_summary()
    
    # 简化数据
    simplified_data = processor.simplify_data_for_api(max_routes=5)
    
    # 打印简化后的数据
    print("\n简化后的数据:")
    print(f"航线数量: {len(simplified_data['route_stats'])}")
    print(f"最佳表现航线: {simplified_data['best_performing_routes']}")
    print(f"最差表现航线: {simplified_data['worst_performing_routes']}") 