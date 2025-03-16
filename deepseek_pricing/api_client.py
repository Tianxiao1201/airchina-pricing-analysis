#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API客户端模块 - 用于与大语言模型API进行交互
支持多种模型: DeepSeek-R1、通义千问
支持多种API提供商: DeepSeek原生API、OpenRouter、Together AI、阿里云通义千问API
"""

import os
import json
import requests
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import abc

# 导入自定义JSON编码器
from data_processor import CustomJSONEncoder


class BaseModelClient(abc.ABC):
    """大语言模型API客户端基类"""
    
    @abc.abstractmethod
    def create_prompt(self, data: Dict[str, Any], query: str) -> str:
        """创建优化的提示"""
        pass
    
    @abc.abstractmethod
    def simple_query(self, query: str, timeout: int = 180) -> Dict[str, Any]:
        """发送简单查询"""
        pass
    
    @abc.abstractmethod
    def analyze_pricing_data(self, data: Dict[str, Any], query: str, 
                            temperature: Optional[float] = None,
                            timeout: int = 180) -> Dict[str, Any]:
        """分析航班定价数据"""
        pass
    
    @abc.abstractmethod
    def extract_reasoning_and_answer(self, response: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """从响应中提取推理过程和最终答案"""
        pass
    
    @abc.abstractmethod
    def batch_analyze(self, data: Dict[str, Any], queries: List[str], 
                     timeout: int = 180) -> Dict[str, Dict[str, Any]]:
        """批量分析多个查询"""
        pass


class DeepSeekR1Client(BaseModelClient):
    """DeepSeek-R1 API客户端类"""
    
    def __init__(self, env_path: str = ".env"):
        """
        初始化API客户端
        
        参数:
            env_path: 环境变量文件路径
        """
        # 加载环境变量
        load_dotenv(env_path)
        
        # 获取API配置
        self.api_provider = os.getenv("API_PROVIDER", "deepseek")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model_id = os.getenv("MODEL_ID", "deepseek-reasoner")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        
        # 根据提供商设置API基础URL
        if self.api_provider == "deepseek":
            self.base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
        elif self.api_provider == "openrouter":
            self.base_url = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
        elif self.api_provider == "together":
            self.base_url = os.getenv("TOGETHER_API_BASE_URL", "https://api.together.xyz/v1")
        else:
            raise ValueError(f"不支持的API提供商: {self.api_provider}")
        
        # 设置API请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 如果是OpenRouter，添加额外的请求头
        if self.api_provider == "openrouter":
            self.headers.update({
                "HTTP-Referer": "https://airchina-pricing-analysis.com",
                "X-Title": "航班定价分析"
            })
        
        print(f"已初始化DeepSeek-R1客户端，使用提供商: {self.api_provider}，模型: {self.model_id}")
    
    def create_prompt(self, data: Dict[str, Any], query: str) -> str:
        """
        创建优化的提示，引导DeepSeek-R1进行更有效的定价分析
        
        参数:
            data: 要分析的数据
            query: 具体的分析请求
            
        返回:
            优化的提示字符串
        """
        # 使用自定义JSON编码器序列化数据
        data_json = json.dumps(data, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        
        prompt = f"""
        作为一位航空公司定价专家，你需要分析以下航班数据并提供定价策略建议。
        
        # 数据概述
        ```json
        {data_json}
        ```
        
        # 分析任务
        {query}
        
        # 分析要求
        1. 请先理解数据结构和各字段含义
        2. 分析当前定价策略的优缺点
        3. 识别影响定价的关键因素（如客座率、成本、竞争等）
        4. 提出具体的定价优化建议，包括具体的调价幅度
        5. 考虑不同航线的特点，提供差异化定价策略
        
        # 分析步骤
        1. 首先，分析整体数据趋势和模式
        2. 其次，识别表现最好和最差的航线
        3. 然后，分析影响定价的关键因素
        4. 接着，提出针对性的定价策略
        5. 最后，总结建议并预测实施效果
        
        请详细展示你的思考过程，然后给出最终的定价建议。
        """
        return prompt
    
    def simple_query(self, query: str, timeout: int = 180) -> Dict[str, Any]:
        """
        发送简单查询到DeepSeek-R1
        
        参数:
            query: 查询内容
            timeout: 请求超时时间（秒）
            
        返回:
            模型的响应
        """
        # 构建API请求
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": self.temperature,
            "max_tokens": 200,  # 使用较小的max_tokens以加快响应速度
            "stream": False
        }
        
        # 发送请求
        try:
            print(f"正在发送简单查询到 {self.api_provider}...")
            start_time = time.time()
            
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=payload,
                timeout=timeout
            )
            
            end_time = time.time()
            print(f"API请求完成，耗时: {end_time - start_time:.2f}秒")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"API请求超时（{timeout}秒）")
            return {"error": f"请求超时（{timeout}秒）"}
        except Exception as e:
            print(f"API请求错误: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            return {"error": str(e)}
    
    def analyze_pricing_data(self, data: Dict[str, Any], query: str, 
                            temperature: Optional[float] = None,
                            timeout: int = 180) -> Dict[str, Any]:
        """
        使用DeepSeek-R1分析航班定价数据
        
        参数:
            data: 要分析的数据
            query: 具体的分析请求
            temperature: 控制输出的随机性，较低的值使输出更确定性
            timeout: 请求超时时间（秒）
            
        返回:
            模型的分析结果
        """
        # 创建优化的提示
        prompt = self.create_prompt(data, query)
        
        # 使用提供的temperature或默认值
        temp = temperature if temperature is not None else self.temperature
        
        # 构建API请求
        if self.api_provider == "deepseek":
            # DeepSeek原生API
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个航空公司定价专家，擅长分析航班数据并提供定价策略建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temp,
                "max_tokens": self.max_tokens,
                "stream": False
            }
        elif self.api_provider == "openrouter":
            # OpenRouter API
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个航空公司定价专家，擅长分析航班数据并提供定价策略建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temp,
                "max_tokens": self.max_tokens
            }
        elif self.api_provider == "together":
            # Together AI
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个航空公司定价专家，擅长分析航班数据并提供定价策略建议。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temp,
                "max_tokens": self.max_tokens
            }
        
        # 发送请求
        try:
            print(f"正在发送API请求到 {self.api_provider}...")
            start_time = time.time()
            
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=payload,
                timeout=timeout
            )
            
            end_time = time.time()
            print(f"API请求完成，耗时: {end_time - start_time:.2f}秒")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"API请求超时（{timeout}秒）")
            return {"error": f"请求超时（{timeout}秒）"}
        except Exception as e:
            print(f"API请求错误: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            return {"error": str(e)}
    
    def extract_reasoning_and_answer(self, response: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """
        从DeepSeek-R1的响应中提取推理过程和最终答案
        
        参数:
            response: API响应
            
        返回:
            (推理过程, 最终答案)元组
        """
        if "error" in response:
            return None, f"错误: {response['error']}"
        
        if "choices" not in response or len(response["choices"]) == 0:
            return None, "无法从响应中提取内容"
        
        # 获取内容
        message = response["choices"][0]["message"]
        content = message["content"]
        
        # 尝试从reasoning_content字段获取推理过程
        reasoning = None
        if "reasoning_content" in message:
            reasoning = message["reasoning_content"]
            print("从reasoning_content字段获取到推理过程")
        else:
            # 如果没有reasoning_content字段，尝试从<think>标签中提取
            if "<think>" in content and "</think>" in content:
                think_start = content.find("<think>") + len("<think>")
                think_end = content.find("</think>")
                reasoning = content[think_start:think_end].strip()
                content = content[:think_start - len("<think>")] + content[think_end + len("</think>"):]
                print("从<think>标签中提取到推理过程")
        
        # 检查是否有reasoning_tokens字段（DeepSeek-R1特有）
        if "usage" in response and "completion_tokens_details" in response["usage"] and "reasoning_tokens" in response["usage"]["completion_tokens_details"]:
            reasoning_tokens = response["usage"]["completion_tokens_details"]["reasoning_tokens"]
            print(f"推理过程使用了 {reasoning_tokens} 个tokens")
        
        return reasoning, content.strip()
    
    def batch_analyze(self, data: Dict[str, Any], queries: List[str], 
                     timeout: int = 180) -> Dict[str, Dict[str, Any]]:
        """
        批量分析多个查询
        
        参数:
            data: 要分析的数据
            queries: 查询列表
            timeout: 请求超时时间（秒）
            
        返回:
            查询结果字典
        """
        results = {}
        
        for i, query in enumerate(queries):
            print(f"正在处理查询 {i+1}/{len(queries)}: {query[:50]}...")
            
            # 调用API进行分析
            response = self.analyze_pricing_data(data, query, timeout=timeout)
            
            # 提取推理过程和最终答案
            reasoning, answer = self.extract_reasoning_and_answer(response)
            
            results[query] = {
                "reasoning": reasoning,
                "answer": answer,
                "full_response": response
            }
            
            # 避免API限流，添加短暂延迟
            if i < len(queries) - 1:
                time.sleep(2)
        
        return results


class QwenClient(BaseModelClient):
    """通义千问 API客户端类"""
    
    def __init__(self, env_path: str = ".env"):
        """
        初始化API客户端
        
        参数:
            env_path: 环境变量文件路径
        """
        # 加载环境变量
        load_dotenv(env_path)
        
        # 获取API配置
        self.api_key = os.getenv("QWEN_API_KEY")
        self.model_id = os.getenv("QWEN_MODEL_ID", "qwen-max")
        self.max_tokens = int(os.getenv("QWEN_MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
        
        # 设置API基础URL - 使用兼容模式的URL
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 设置API请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        print(f"已初始化通义千问客户端，使用模型: {self.model_id}")
    
    def create_prompt(self, data: Dict[str, Any], query: str) -> str:
        """
        创建优化的提示，引导通义千问进行更有效的定价分析
        
        参数:
            data: 要分析的数据
            query: 具体的分析请求
            
        返回:
            优化的提示字符串
        """
        # 使用自定义JSON编码器序列化数据
        data_json = json.dumps(data, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        
        prompt = f"""
        作为一位航空公司定价专家，你需要分析以下航班数据并提供定价策略建议。
        
        # 数据概述
        ```json
        {data_json}
        ```
        
        # 分析任务
        {query}
        
        # 分析要求
        1. 请先理解数据结构和各字段含义
        2. 分析当前定价策略的优缺点
        3. 识别影响定价的关键因素（如客座率、成本、竞争等）
        4. 提出具体的定价优化建议，包括具体的调价幅度
        5. 考虑不同航线的特点，提供差异化定价策略
        
        # 分析步骤
        1. 首先，分析整体数据趋势和模式
        2. 其次，识别表现最好和最差的航线
        3. 然后，分析影响定价的关键因素
        4. 接着，提出针对性的定价策略
        5. 最后，总结建议并预测实施效果
        
        请详细展示你的思考过程，然后给出最终的定价建议。
        """
        return prompt
    
    def simple_query(self, query: str, timeout: int = 180) -> Dict[str, Any]:
        """
        发送简单查询到通义千问
        
        参数:
            query: 查询内容
            timeout: 请求超时时间（秒）
            
        返回:
            模型的响应
        """
        # 构建API请求 - 使用OpenAI兼容格式
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": self.temperature,
            "max_tokens": 200  # 使用较小的max_tokens以加快响应速度
        }
        
        # 发送请求
        try:
            print(f"正在发送简单查询到通义千问...")
            start_time = time.time()
            
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=payload,
                timeout=timeout
            )
            
            end_time = time.time()
            print(f"API请求完成，耗时: {end_time - start_time:.2f}秒")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"API请求超时（{timeout}秒）")
            return {"error": f"请求超时（{timeout}秒）"}
        except Exception as e:
            print(f"API请求错误: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            return {"error": str(e)}
    
    def analyze_pricing_data(self, data: Dict[str, Any], query: str, 
                            temperature: Optional[float] = None,
                            timeout: int = 180) -> Dict[str, Any]:
        """
        使用通义千问分析航班定价数据
        
        参数:
            data: 要分析的数据
            query: 具体的分析请求
            temperature: 控制输出的随机性，较低的值使输出更确定性
            timeout: 请求超时时间（秒）
            
        返回:
            模型的分析结果
        """
        # 创建优化的提示
        prompt = self.create_prompt(data, query)
        
        # 使用提供的temperature或默认值
        temp = temperature if temperature is not None else self.temperature
        
        # 构建API请求 - 使用OpenAI兼容格式
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个航空公司定价专家，擅长分析航班数据并提供定价策略建议。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temp,
            "max_tokens": self.max_tokens
        }
        
        # 发送请求
        try:
            print(f"正在发送API请求到通义千问...")
            start_time = time.time()
            
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=payload,
                timeout=timeout
            )
            
            end_time = time.time()
            print(f"API请求完成，耗时: {end_time - start_time:.2f}秒")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"API请求超时（{timeout}秒）")
            return {"error": f"请求超时（{timeout}秒）"}
        except Exception as e:
            print(f"API请求错误: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"错误响应: {e.response.text}")
            return {"error": str(e)}
    
    def extract_reasoning_and_answer(self, response: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """
        从通义千问的响应中提取推理过程和最终答案
        
        参数:
            response: API响应
            
        返回:
            (推理过程, 最终答案)元组
        """
        # 打印完整响应用于调试
        print("通义千问API响应:")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        
        if "error" in response:
            return None, f"错误: {response['error']}"
        
        # 检查响应格式 - 使用OpenAI兼容格式的响应结构
        if "choices" not in response or len(response["choices"]) == 0:
            print("响应中没有'choices'字段或为空")
            return None, "无法从响应中提取内容: 缺少'choices'字段或为空"
        
        # 获取内容
        message = response["choices"][0]["message"]
        if "content" not in message:
            print("响应中没有'content'字段")
            return None, "无法从响应中提取内容: 缺少'content'字段"
        
        content = message["content"]
        
        # 尝试从<think>标签中提取推理过程
        reasoning = None
        if "<think>" in content and "</think>" in content:
            think_start = content.find("<think>") + len("<think>")
            think_end = content.find("</think>")
            reasoning = content[think_start:think_end].strip()
            content = content[:think_start - len("<think>")] + content[think_end + len("</think>"):]
            print("从<think>标签中提取到推理过程")
        
        # 如果没有找到推理过程，尝试其他可能的格式
        if reasoning is None and "思考过程" in content:
            # 尝试基于"思考过程"关键词分割
            parts = content.split("思考过程", 1)
            if len(parts) > 1:
                # 找到下一个标题的位置
                next_title_pos = parts[1].find("#")
                if next_title_pos > 0:
                    reasoning = parts[1][:next_title_pos].strip()
                    print("基于'思考过程'关键词提取到推理过程")
        
        return reasoning, content.strip()
    
    def batch_analyze(self, data: Dict[str, Any], queries: List[str], 
                     timeout: int = 180) -> Dict[str, Dict[str, Any]]:
        """
        批量分析多个查询
        
        参数:
            data: 要分析的数据
            queries: 查询列表
            timeout: 请求超时时间（秒）
            
        返回:
            查询结果字典
        """
        results = {}
        
        for i, query in enumerate(queries):
            print(f"正在处理查询 {i+1}/{len(queries)}: {query[:50]}...")
            
            # 调用API进行分析
            response = self.analyze_pricing_data(data, query, timeout=timeout)
            
            # 提取推理过程和最终答案
            reasoning, answer = self.extract_reasoning_and_answer(response)
            
            results[query] = {
                "reasoning": reasoning,
                "answer": answer,
                "full_response": response
            }
            
            # 避免API限流，添加短暂延迟
            if i < len(queries) - 1:
                time.sleep(2)
        
        return results


class ModelClientFactory:
    """模型客户端工厂类，用于创建不同的模型客户端"""
    
    @staticmethod
    def create_client(model_type: str, env_path: str = ".env") -> BaseModelClient:
        """
        创建模型客户端
        
        参数:
            model_type: 模型类型，支持 "deepseek" 和 "qwen"
            env_path: 环境变量文件路径
            
        返回:
            模型客户端实例
        """
        if model_type.lower() == "deepseek":
            return DeepSeekR1Client(env_path)
        elif model_type.lower() == "qwen":
            return QwenClient(env_path)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


if __name__ == "__main__":
    # 测试代码
    client = DeepSeekR1Client()
    
    # 简单测试 - 使用非常简单的查询
    print("执行简单测试...")
    response = client.simple_query("Hello! Please provide a very short response.", timeout=60)
    
    if "error" in response:
        print(f"错误: {response['error']}")
    else:
        # 打印响应
        print("\n=== 响应内容 ===")
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0]["message"]
            content = message["content"]
            print(f"回答: {content}")
            
            if "reasoning_content" in message:
                reasoning = message["reasoning_content"]
                print(f"\n推理过程: {reasoning}")
        
        # 检查是否有reasoning_tokens
        if "usage" in response and "completion_tokens_details" in response["usage"] and "reasoning_tokens" in response["usage"]["completion_tokens_details"]:
            reasoning_tokens = response["usage"]["completion_tokens_details"]["reasoning_tokens"]
            print(f"\n推理过程使用了 {reasoning_tokens} 个tokens") 