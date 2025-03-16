#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
航班定价分析系统 - 交互式Web应用
功能1: 航班数据整体分析 - 上传历史数据，生成综合分析报告
功能2: 单个航班定价建议 - 输入特定航班信息，获取定价建议
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# 设置页面配置 - 必须是第一个Streamlit命令
st.set_page_config(
    page_title="国航内蒙古营业部航班定价分析系统",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "deepseek_pricing"))

# 导入自定义模块
try:
    from deepseek_pricing.data_processor import DataProcessor
    from deepseek_pricing.api_client import ModelClientFactory
    from deepseek_pricing.visualizer import PricingVisualizer
    st.sidebar.success("成功导入自定义模块")
except Exception as e:
    st.sidebar.error(f"导入自定义模块失败: {e}")
    st.stop()

# 设置全局变量
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = os.path.join(st.session_state.temp_dir, "定价分析结果")
    os.makedirs(st.session_state.output_dir, exist_ok=True)
if 'use_mock' not in st.session_state:
    st.session_state.use_mock = False  # 默认使用实际API
if 'model_type' not in st.session_state:
    st.session_state.model_type = "deepseek"  # 默认使用DeepSeek模型


def reset_analysis():
    """重置分析状态"""
    st.session_state.data_uploaded = False
    st.session_state.analysis_started = False
    st.session_state.analysis_completed = False
    st.session_state.current_step = 0
    st.session_state.df = None
    st.session_state.processor = None
    st.session_state.analysis_results = None
    
    # 清理临时目录
    if os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)
    
    # 创建新的临时目录
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.output_dir = os.path.join(st.session_state.temp_dir, "定价分析结果")
    os.makedirs(st.session_state.output_dir, exist_ok=True)


def load_and_process_data(uploaded_file):
    """加载和处理上传的数据文件"""
    try:
        # 保存上传的文件到临时目录
        temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.sidebar.info(f"文件已保存到: {temp_file_path}")
        
        # 初始化数据处理器
        processor = DataProcessor(temp_file_path)
        
        # 加载数据
        df = processor.load_data()
        st.sidebar.info(f"数据加载成功: {len(df)} 行, {len(df.columns)} 列")
        
        # 预处理数据
        df = processor.preprocess_data()
        st.sidebar.info("数据预处理完成")
        
        # 计算统计数据
        processor.calculate_statistics()
        st.sidebar.info("统计数据计算完成")
        
        # 准备数据摘要
        processor.prepare_data_summary()
        st.sidebar.info("数据摘要准备完成")
        
        # 保存到会话状态
        st.session_state.df = df
        st.session_state.processor = processor
        st.session_state.data_uploaded = True
        
        return df, processor
    except Exception as e:
        st.sidebar.error(f"数据处理失败: {e}")
        raise e


def execute_analysis(processor, timeout=180):
    """执行定价分析"""
    try:
        # 初始化API客户端
        if not st.session_state.use_mock:
            try:
                # 指定环境变量文件的绝对路径
                env_path = os.path.join(current_dir, "deepseek_pricing", ".env")
                
                # 使用工厂类创建选定的模型客户端
                client = ModelClientFactory.create_client(st.session_state.model_type, env_path=env_path)
                
                # 打印API密钥信息用于调试
                api_key = client.api_key
                if api_key:
                    masked_key = f"{api_key[:5]}...{api_key[-5:]}"
                    st.sidebar.info(f"API密钥已加载: {masked_key}")
                else:
                    st.sidebar.error("API密钥为空")
                
                st.session_state.client = client
                
                # 显示模型信息
                model_info = f"模型: {client.model_id}"
                if hasattr(client, 'api_provider'):
                    model_info = f"提供商: {client.api_provider}, {model_info}"
                st.sidebar.info(f"API客户端初始化成功，使用{model_info}")
            except Exception as e:
                st.sidebar.error(f"API客户端初始化失败: {e}")
                st.error(f"无法连接到{st.session_state.model_type.upper()}API。请检查API密钥或启用模拟模式继续。")
                st.session_state.use_mock = True
                st.warning("已自动切换到模拟模式")
        else:
            # 在模拟模式下，不需要实际的API客户端
            client = None
            st.session_state.client = None
            st.sidebar.info("模拟模式：不初始化API客户端")
        
        # 准备数据摘要
        data_summary = processor.data_summary
        
        # 简化数据用于API请求，保留所有航线数据
        simplified_data = processor.simplify_data_for_api(keep_all_data=True)
        st.sidebar.info("数据已准备，保留了所有航线数据进行分析")
        
        # 获取默认查询列表
        queries = [
            "分析当前航线的定价策略，识别哪些航线的定价过高或过低，并提供具体的调价建议。",
            "分析客座率与票价之间的关系，找出最佳的价格点以最大化收入。",
            "分析季节性因素对航线定价的影响，并提供不同季节的差异化定价策略。",
            "分析价格弹性对不同航线的影响，识别哪些航线对价格变化更敏感。",
            "提供一个综合的动态定价优化方案，考虑所有相关因素。"
        ]
        
        # 执行分析
        results = {}
        
        # 检查是否使用模拟模式
        if st.session_state.use_mock:
            st.sidebar.warning("使用模拟模式，不调用实际API")
            
            # 生成模拟结果
            for i, query in enumerate(queries):
                # 更新当前步骤
                st.session_state.current_step = i + 1
                st.sidebar.info(f"正在模拟查询 {i+1}/{len(queries)}")
                
                # 模拟API调用延迟
                time.sleep(2)
                
                # 生成模拟结果
                mock_reasoning = f"""
                ## 模拟思考过程 - 查询{i+1}
                
                这是针对查询"{query}"的模拟思考过程。
                
                ### 数据分析
                1. 首先，我分析了航线的整体表现
                2. 然后，我研究了客座率与票价的关系
                3. 接着，我考虑了季节性因素的影响
                4. 最后，我评估了价格弹性对不同航线的影响
                
                ### 关键发现
                - 部分航线的定价明显偏离市场均衡点
                - 客座率与票价呈现负相关关系，但不同航线的敏感度不同
                - 季节性因素对票价有显著影响，尤其是在假期期间
                - 价格弹性在不同航线间差异较大
                
                ### 分析方法
                我使用了回归分析、时间序列分析和弹性计算等方法进行深入研究。
                """
                
                mock_answer = f"""
                # 模拟分析结果 - 查询{i+1}
                
                ## 针对"{query}"的分析结果
                
                ### 主要发现
                1. **航线表现差异**：不同航线的利润率和客座率存在显著差异，需要差异化定价策略
                2. **价格敏感度**：部分航线对价格变化非常敏感，建议谨慎调整票价
                3. **季节性影响**：假期和旅游旺季对票价有明显影响，应实施动态定价
                
                ### 具体建议
                - **北京-上海线**：当前定价合理，建议维持现状
                - **广州-深圳线**：定价偏低，可提高5-10%
                - **成都-重庆线**：客座率较低，建议降低票价8-12%以刺激需求
                - **北京-广州线**：价格弹性较高，建议实施更灵活的动态定价策略
                
                ### 实施步骤
                1. 先在客流量较小的航线测试新定价策略
                2. 收集反馈并调整
                3. 逐步推广到其他航线
                
                ### 预期效果
                预计新定价策略可提升整体利润率3-5个百分点，同时保持或略微提高客座率。
                """
                
                results[query] = {
                    "reasoning": mock_reasoning,
                    "answer": mock_answer,
                    "full_response": {"mock": True}
                }
                
                st.sidebar.success(f"模拟查询 {i+1} 完成")
        else:
            # 使用实际API
            for i, query in enumerate(queries):
                # 更新当前步骤
                st.session_state.current_step = i + 1
                st.sidebar.info(f"正在执行查询 {i+1}/{len(queries)}")
                
                # 调用API进行分析
                response = client.analyze_pricing_data(simplified_data, query, timeout=timeout)
                
                # 提取推理过程和最终答案
                reasoning, answer = client.extract_reasoning_and_answer(response)
                
                results[query] = {
                    "reasoning": reasoning,
                    "answer": answer,
                    "full_response": response
                }
                
                st.sidebar.success(f"查询 {i+1} 完成")
                
                # 避免API限流，添加短暂延迟
                if i < len(queries) - 1:
                    time.sleep(2)
        
        # 保存分析结果
        analysis_results = {
            "queries": queries,
            "results": results,
            "data_summary": data_summary
        }
        
        st.session_state.analysis_results = analysis_results
        st.session_state.analysis_completed = True
        st.sidebar.success("所有分析完成")
        
        return analysis_results
    except Exception as e:
        st.sidebar.error(f"分析执行失败: {e}")
        raise e


def generate_visualizations(df, analysis_results, output_dir):
    """生成可视化和HTML报告"""
    try:
        # 初始化可视化器
        visualizer = PricingVisualizer(df, output_dir)
        st.sidebar.info("可视化器初始化成功")
        
        # 生成各种图表
        st.sidebar.info("正在生成利润率图表...")
        profit_chart = visualizer.create_profit_rate_chart()
        
        st.sidebar.info("正在生成客座率与票价散点图...")
        load_factor_chart = visualizer.create_load_factor_price_scatter()
        
        st.sidebar.info("正在生成价格弹性图表...")
        elasticity_chart = visualizer.create_price_elasticity_chart()
        
        st.sidebar.info("正在生成季节性指数图表...")
        seasonal_chart = visualizer.create_seasonal_index_chart()
        
        st.sidebar.info("正在生成收入成本对比图表...")
        revenue_cost_chart = visualizer.create_revenue_cost_chart()
        
        # 生成HTML报告
        st.sidebar.info("正在生成HTML报告...")
        html_path = visualizer.generate_html_report(analysis_results)
        
        st.sidebar.success(f"HTML报告已生成: {html_path}")
        
        return {
            "profit_chart": profit_chart,
            "load_factor_chart": load_factor_chart,
            "elasticity_chart": elasticity_chart,
            "seasonal_chart": seasonal_chart,
            "revenue_cost_chart": revenue_cost_chart,
            "html_report": html_path
        }
    except Exception as e:
        st.sidebar.error(f"可视化生成失败: {e}")
        raise e


def predict_single_flight_price(flight_info, processor, client):
    """预测单个航班的价格"""
    try:
        # 准备查询
        query = f"""
        基于历史航班数据，为以下航班提供最优定价建议：
        
        航线: {flight_info['route']}
        日期: {flight_info['date']}
        是否假期: {'是' if flight_info['is_holiday'] else '否'}
        星期几: {flight_info['day_of_week']}
        机型: {flight_info['aircraft_type']}
        座位容量: {flight_info['seat_capacity']}
        价格弹性: {flight_info['price_elasticity']}
        季节性指数: {flight_info['seasonal_index']}
        预期客座率: {flight_info['load_factor']}
        竞争对手客座率: {flight_info['competitor_load_factor']}
        历史客座率: {flight_info['historical_load_factor']}
        竞争对手价格: {flight_info['competitor_price']}元
        
        请分析该航班的最优定价区间，并给出具体的定价建议。考虑季节性因素、周内模式、假期影响、价格弹性、竞争情况等因素。
        提供详细的分析过程和定价建议，包括最优价格点和价格区间。
        """
        
        st.sidebar.info("正在准备单个航班定价查询")
        
        # 获取数据摘要
        data_summary = processor.data_summary
        
        # 简化数据用于API请求，保留所有航线数据
        simplified_data = processor.simplify_data_for_api(keep_all_data=True)
        
        # 检查是否使用模拟模式
        if st.session_state.use_mock or client is None:
            if not st.session_state.use_mock and client is None:
                # 尝试重新初始化客户端
                try:
                    env_path = os.path.join(current_dir, "deepseek_pricing", ".env")
                    client = ModelClientFactory.create_client(st.session_state.model_type, env_path=env_path)
                    st.session_state.client = client
                    st.sidebar.info("API客户端重新初始化成功")
                except Exception as e:
                    st.warning(f"API客户端初始化失败: {e}，自动切换到模拟模式")
                    st.session_state.use_mock = True
            
            if st.session_state.use_mock:
                st.sidebar.warning("使用模拟模式，不调用实际API")
            
            # 模拟API调用延迟
            time.sleep(3)
            
            # 生成模拟结果
            mock_reasoning = f"""
            ## 模拟思考过程 - 单个航班定价
            
            我正在分析航线"{flight_info['route']}"在{flight_info['date']}（{flight_info['day_of_week']}）的最优定价。
            
            ### 考虑因素
            1. **航线特性**：分析该航线的历史表现和竞争情况
            2. **日期因素**：{flight_info['date']}是{flight_info['day_of_week']}，{'是' if flight_info['is_holiday'] else '不是'}假期
            3. **机型影响**：{flight_info['aircraft_type']}型飞机，座位容量{flight_info['seat_capacity']}
            4. **季节性影响**：季节性指数为{flight_info['seasonal_index']}，{'高于' if flight_info['seasonal_index'] > 1 else '低于'}平均水平
            5. **价格弹性**：该航线的价格弹性系数为{flight_info['price_elasticity']}，表示需求对价格{'非常敏感' if flight_info['price_elasticity'] > 1.2 else '较为敏感' if flight_info['price_elasticity'] > 0.8 else '不太敏感'}
            6. **客座率分析**：
               - 预期客座率：{flight_info['load_factor']:.2f}
               - 历史客座率：{flight_info['historical_load_factor']:.2f}
               - 竞争对手客座率：{flight_info['competitor_load_factor']:.2f}
            7. **竞争分析**：竞争对手当前平均票价为¥{flight_info['competitor_price']}
            
            ### 数据分析
            - 历史平均票价：¥{flight_info['competitor_price'] * 0.95:.0f}-{flight_info['competitor_price'] * 1.05:.0f}
            - 历史平均客座率：{(flight_info['historical_load_factor'] * 100):.0f}%-{(flight_info['historical_load_factor'] * 100 + 5):.0f}%
            - 价格弹性系数：{flight_info['price_elasticity']}（{'高' if flight_info['price_elasticity'] > 1.2 else '中' if flight_info['price_elasticity'] > 0.8 else '低'}敏感度）
            - 季节性指数：{flight_info['seasonal_index']}（{'高于' if flight_info['seasonal_index'] > 1 else '低于'}平均水平）
            
            ### 竞争分析
            该航线的主要竞争对手客座率为{flight_info['competitor_load_factor']:.2f}，当前平均票价约¥{flight_info['competitor_price']}。
            """
            
            # 根据输入的因素计算价格建议
            # 基础价格计算
            base_price = flight_info['competitor_price']
            
            # 根据各种因素调整价格
            # 1. 假期调整
            if flight_info['is_holiday']:
                holiday_factor = 1.15
                holiday_reason = "假期期间需求增加"
            else:
                holiday_factor = 1.0
                holiday_reason = "非假期期间"
                
            # 2. 星期几调整
            if flight_info['day_of_week'] in ["周五", "周日"]:
                weekday_factor = 1.08
                weekday_reason = "周末前后需求较高"
            elif flight_info['day_of_week'] in ["周六"]:
                weekday_factor = 1.12
                weekday_reason = "周末需求高峰"
            else:
                weekday_factor = 0.95
                weekday_reason = "平日需求较低"
                
            # 3. 季节性调整
            seasonal_factor = flight_info['seasonal_index']
            if seasonal_factor > 1.1:
                seasonal_reason = "处于旅游旺季"
            elif seasonal_factor < 0.9:
                seasonal_reason = "处于淡季"
            else:
                seasonal_reason = "季节性因素适中"
                
            # 4. 客座率调整
            if flight_info['load_factor'] > 0.85:
                load_factor = 1.1
                load_reason = "预期客座率高，需求强劲"
            elif flight_info['load_factor'] < 0.7:
                load_factor = 0.9
                load_reason = "预期客座率低，需求疲软"
            else:
                load_factor = 1.0
                load_reason = "预期客座率适中"
                
            # 5. 竞争对手调整
            if flight_info['competitor_load_factor'] > 0.85:
                competition_factor = 1.05
                competition_reason = "竞争对手客座率高，市场需求强劲"
            elif flight_info['competitor_load_factor'] < 0.7:
                competition_factor = 0.95
                competition_reason = "竞争对手客座率低，市场需求疲软"
            else:
                competition_factor = 1.0
                competition_reason = "竞争对手客座率适中"
                
            # 6. 价格弹性调整
            if flight_info['price_elasticity'] > 1.2:
                elasticity_factor = 0.95
                elasticity_reason = "价格弹性高，需谨慎定价"
            elif flight_info['price_elasticity'] < 0.8:
                elasticity_factor = 1.05
                elasticity_reason = "价格弹性低，可适当提高价格"
            else:
                elasticity_factor = 1.0
                elasticity_reason = "价格弹性适中"
            
            # 综合所有因素计算建议价格
            price_factors = holiday_factor * weekday_factor * seasonal_factor * load_factor * competition_factor * elasticity_factor
            suggested_price = round(base_price * price_factors)
            price_range_low = round(suggested_price * 0.9)
            price_range_high = round(suggested_price * 1.1)
            
            # 计算预期利润率
            expected_profit_rate = 0.15 + (load_factor - 0.8) * 0.5  # 基础利润率15%，根据客座率调整
            expected_profit_rate = max(0.05, min(0.3, expected_profit_rate))  # 确保在5%-30%之间
            
            # 生成价格建议
            mock_answer = f"""
            # 航班定价建议
            
            ## 航线: {flight_info['route']}
            ## 日期: {flight_info['date']} ({flight_info['day_of_week']})
            
            ### 定价建议
            
            根据综合分析，我建议将此航班的票价设定在 **¥{price_range_low} - ¥{price_range_high}** 之间，最优价格点为 **¥{suggested_price}**。
            
            ### 建议理由
            
            1. **市场因素**: 
               - {holiday_reason}
               - {weekday_reason}
               - {seasonal_reason}
            
            2. **竞争情况**: 
               - 竞争对手当前平均票价为¥{flight_info['competitor_price']}
               - {competition_reason}
            
            3. **需求分析**:
               - {load_reason}
               - {elasticity_reason}
               - 历史客座率为{flight_info['historical_load_factor']:.2f}，{('高于' if flight_info['historical_load_factor'] > flight_info['load_factor'] else '低于')}当前预期
            
            4. **价格敏感度**: 价格弹性系数为{flight_info['price_elasticity']}，表明乘客对价格变化的敏感程度{('高' if flight_info['price_elasticity'] > 1.2 else '中等' if flight_info['price_elasticity'] > 0.8 else '低')}
            
            ### 实施建议
            
            1. 可采用动态定价策略，起始价格设为¥{price_range_low}
            2. 随着预订率提高，逐步提升至¥{price_range_high}
            3. 建议设置10%的折扣用于早鸟预订（提前30天）
            4. 对常旅客提供额外5%的优惠以提高忠诚度
            5. 密切监控竞争对手价格变化，保持适当的价格竞争力
            
            ### 预期效果
            
            在建议的价格区间内，预计可实现{expected_profit_rate:.0%}-{expected_profit_rate+0.05:.0%}的利润率，同时保持{flight_info['load_factor']*100:.0f}%-{min(98, flight_info['load_factor']*100+5):.0f}%的客座率。
            """
            
            return {
                "reasoning": mock_reasoning,
                "answer": mock_answer,
                "full_response": {"mock": True}
            }
        else:
            # 调用API进行分析
            st.sidebar.info(f"正在发送API请求到{st.session_state.model_type.upper()}...")
            response = client.analyze_pricing_data(simplified_data, query, timeout=180)
            
            # 提取推理过程和最终答案
            reasoning, answer = client.extract_reasoning_and_answer(response)
            
            st.sidebar.success("定价建议生成成功")
            
            return {
                "reasoning": reasoning,
                "answer": answer,
                "full_response": response
            }
    except Exception as e:
        st.sidebar.error(f"定价预测失败: {e}")
        raise e


def render_data_analysis_ui():
    """渲染数据分析界面"""
    st.title("✈️ 航班定价分析系统")
    
    # 侧边栏
    with st.sidebar:
        st.header("功能选择")
        selected_function = st.radio(
            "选择功能",
            ["航班数据整体分析", "单个航班定价建议"],
            index=0
        )
        
        st.header("系统设置")
        
        # 添加模型选择
        model_type = st.selectbox(
            "选择分析模型",
            ["DeepSeek", "通义千问"],
            index=0
        )
        
        # 将选择转换为小写并存储在会话状态中
        model_type_lower = model_type.lower().replace("通义千问", "qwen")
        if model_type_lower != st.session_state.model_type:
            st.session_state.model_type = model_type_lower
            # 如果模型类型改变，重置客户端
            if st.session_state.client is not None:
                st.session_state.client = None
                st.info(f"已切换到{model_type}模型")
        
        use_mock = st.checkbox("使用模拟模式（不调用实际API）", value=st.session_state.use_mock)
        if use_mock != st.session_state.use_mock:
            st.session_state.use_mock = use_mock
            # 如果从非模拟模式切换到模拟模式，重置客户端
            if use_mock and st.session_state.client is not None:
                st.session_state.client = None
        
        if use_mock:
            st.info("模拟模式已启用，系统将使用模拟数据而不调用实际API")
        else:
            st.warning(f"实际API模式已启用，请确保已配置正确的{model_type}API密钥")
        
        if st.button("重置分析"):
            reset_analysis()
            st.rerun()
    
    if selected_function == "航班数据整体分析":
        st.header("航班数据整体分析")
        
        if not st.session_state.data_uploaded:
            # 文件上传界面
            st.write("请上传航班历史数据文件（CSV格式）:")
            uploaded_file = st.file_uploader("选择文件", type=["csv"])
            
            if uploaded_file is not None:
                with st.spinner("正在加载和处理数据..."):
                    df, processor = load_and_process_data(uploaded_file)
                    st.success(f"数据加载成功！共 {len(df)} 行, {len(df.columns)} 列")
                    st.session_state.analysis_started = True
                    st.rerun()
        
        elif st.session_state.data_uploaded and st.session_state.analysis_started and not st.session_state.analysis_completed:
            # 分析进行中界面
            df = st.session_state.df
            processor = st.session_state.processor
            
            # 显示数据概览
            st.subheader("数据概览")
            st.write(f"数据范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
            st.write(f"航线数量: {df['route_name'].nunique()}")
            st.write(f"平均客座率: {df['load_factor'].mean()*100:.2f}%")
            st.write(f"平均利润率: {df['profit_margin'].mean()*100:.2f}%")
            
            # 显示数据样本
            with st.expander("查看数据样本"):
                st.dataframe(df.head())
            
            # 执行分析
            if st.session_state.current_step == 0:
                st.subheader("正在进行分析...")
                progress_bar = st.progress(0)
                
                # 开始分析
                with st.spinner(f"正在连接{st.session_state.model_type.upper()}API..."):
                    analysis_results = execute_analysis(processor)
                    
                    # 生成可视化
                    visualizations = generate_visualizations(
                        df, 
                        analysis_results, 
                        st.session_state.output_dir
                    )
                    
                    st.session_state.visualizations = visualizations
                    st.rerun()
            
            else:
                # 显示分析进度
                queries = [
                    "分析当前航线的定价策略，识别哪些航线的定价过高或过低，并提供具体的调价建议。",
                    "分析客座率与票价之间的关系，找出最佳的价格点以最大化收入。",
                    "分析季节性因素对航线定价的影响，并提供不同季节的差异化定价策略。",
                    "分析价格弹性对不同航线的影响，识别哪些航线对价格变化更敏感。",
                    "提供一个综合的动态定价优化方案，考虑所有相关因素。"
                ]
                
                progress_bar = st.progress(st.session_state.current_step / (len(queries) + 1))
                st.subheader(f"正在分析 ({st.session_state.current_step}/{len(queries)})")
                st.write(f"当前查询: {queries[st.session_state.current_step-1]}")
                
                # 显示思考过程动画
                thinking_placeholder = st.empty()
                for i in range(3):
                    thinking_placeholder.write("思考中" + "." * (i + 1))
                    time.sleep(0.5)
        
        elif st.session_state.analysis_completed:
            # 分析完成界面
            df = st.session_state.df
            analysis_results = st.session_state.analysis_results
            
            st.subheader("数据概览")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("航线数量", f"{df['route_name'].nunique()}")
            with col2:
                st.metric("平均客座率", f"{df['load_factor'].mean()*100:.2f}%")
            with col3:
                st.metric("平均利润率", f"{df['profit_margin'].mean()*100:.2f}%")
            with col4:
                st.metric("数据样本数", f"{len(df)}")
            
            # 显示可视化分析
            st.subheader("可视化分析")
            
            # 显示图表
            if 'visualizations' in st.session_state:
                visualizations = st.session_state.visualizations
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "利润率分析", "客座率与票价", "价格弹性", "季节性指数", "收入成本对比"
                ])
                
                with tab1:
                    st.image(visualizations["profit_chart"])
                    st.write("各航线平均利润率分析")
                
                with tab2:
                    st.image(visualizations["load_factor_chart"])
                    st.write("客座率与票价关系分析")
                
                with tab3:
                    st.image(visualizations["elasticity_chart"])
                    st.write("各航线价格弹性分析")
                
                with tab4:
                    st.image(visualizations["seasonal_chart"])
                    st.write("月度季节性指数分析")
                
                with tab5:
                    st.image(visualizations["revenue_cost_chart"])
                    st.write("各航线收入与成本对比分析")
            
            # 显示分析结果，根据选择的模型类型显示不同的标题
            model_display_name = "通义千问" if st.session_state.model_type == "qwen" else "DeepSeek-R1"
            st.subheader(f"{model_display_name}分析结果")
            
            if analysis_results and "queries" in analysis_results and "results" in analysis_results:
                queries = analysis_results["queries"]
                results = analysis_results["results"]
                
                for i, query in enumerate(queries):
                    if query in results:
                        with st.expander(f"分析 {i+1}: {query}"):
                            result = results[query]
                            
                            # 显示推理过程
                            if result.get("reasoning"):
                                st.subheader("思考过程")
                                st.write(result["reasoning"])
                            
                            # 显示最终答案
                            st.subheader("分析结果")
                            st.write(result["answer"])
            
            # 提供下载报告选项
            if 'visualizations' in st.session_state and 'html_report' in st.session_state.visualizations:
                html_path = st.session_state.visualizations["html_report"]
                with open(html_path, "rb") as file:
                    st.download_button(
                        label="下载完整分析报告",
                        data=file,
                        file_name="航班定价分析报告.html",
                        mime="text/html"
                    )
    
    elif selected_function == "单个航班定价建议":
        st.header("单个航班定价建议")
        
        if not st.session_state.data_uploaded:
            st.warning("请先上传航班历史数据并完成整体分析，然后再使用此功能。")
            st.write("请返回到'航班数据整体分析'功能，上传数据并完成分析。")
        
        elif not st.session_state.analysis_completed:
            st.info("正在进行整体数据分析，请等待分析完成后再使用此功能。")
            st.write("请等待'航班数据整体分析'完成。")
        
        else:
            st.write("请输入需要定价的航班信息:")
            
            # 获取可用的航线列表
            available_routes = sorted(st.session_state.df['route_name'].unique())
            
            # 创建输入表单
            with st.form("flight_info_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    route = st.selectbox("航线", available_routes)
                    flight_date = st.date_input("航班日期", datetime.now())
                    is_holiday = st.checkbox("是否假期")
                    price_elasticity = st.slider("价格弹性", min_value=0.5, max_value=1.5, value=1.0, step=0.1, 
                                               help="价格弹性表示需求对价格变化的敏感度，值越大表示越敏感")
                
                with col2:
                    day_of_week = st.selectbox("星期几", ["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
                    aircraft_type = st.selectbox("机型", ["737", "738", "73F", "ARJ", "320", "321", "330"])
                    seat_capacity = st.number_input("座位容量", min_value=50, max_value=300, value=180)
                    seasonal_index = st.slider("季节性指数", min_value=0.7, max_value=1.3, value=1.0, step=0.05,
                                             help="季节性指数表示当前季节的需求强度，大于1表示旺季，小于1表示淡季")
                
                with col3:
                    load_factor = st.slider("预期客座率", min_value=0.4, max_value=0.98, value=0.8, step=0.01,
                                          format="%.2f", help="预期客座率是指预计的座位预订比例")
                    competitor_load_factor = st.slider("竞争对手客座率", min_value=0.4, max_value=0.98, value=0.8, step=0.01,
                                                     format="%.2f", help="主要竞争对手的平均客座率")
                    historical_load_factor = st.slider("历史客座率", min_value=0.4, max_value=0.98, value=0.8, step=0.01,
                                                     format="%.2f", help="过去30天同航线平均客座率")
                    competitor_price = st.number_input("竞争对手价格(元)", min_value=500, max_value=5000, value=1500,
                                                     help="主要竞争对手的平均票价")
                
                submit_button = st.form_submit_button("获取定价建议")
            
            if submit_button:
                # 准备航班信息
                flight_info = {
                    "route": route,
                    "date": flight_date.strftime("%Y-%m-%d"),
                    "is_holiday": is_holiday,
                    "day_of_week": day_of_week,
                    "aircraft_type": aircraft_type,
                    "seat_capacity": seat_capacity,
                    "price_elasticity": price_elasticity,
                    "seasonal_index": seasonal_index,
                    "load_factor": load_factor,
                    "competitor_load_factor": competitor_load_factor,
                    "historical_load_factor": historical_load_factor,
                    "competitor_price": competitor_price
                }
                
                # 显示思考过程
                st.subheader("正在分析...")
                thinking_placeholder = st.empty()
                
                for i in range(5):  # 模拟思考过程
                    # 根据选择的模型类型显示不同的消息
                    model_name = "通义千问" if st.session_state.model_type == "qwen" else "DeepSeek-R1"
                    thinking_placeholder.write(f"{model_name}正在思考" + "." * (i % 4 + 1))
                    time.sleep(1)
                
                # 预测价格
                with st.spinner("正在生成定价建议..."):
                    prediction_result = predict_single_flight_price(
                        flight_info,
                        st.session_state.processor,
                        st.session_state.client
                    )
                
                # 清除思考占位符
                thinking_placeholder.empty()
                
                # 显示结果
                st.subheader("定价分析结果")
                
                # 显示推理过程
                if prediction_result.get("reasoning"):
                    with st.expander("查看思考过程", expanded=True):
                        st.write(prediction_result["reasoning"])
                
                # 显示最终建议
                st.success("定价建议已生成")
                st.write(prediction_result["answer"])


# 主函数
def main():
    render_data_analysis_ui()


if __name__ == "__main__":
    main() 