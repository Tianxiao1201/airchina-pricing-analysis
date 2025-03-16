# DeepSeek-R1 航班定价分析工具

基于DeepSeek-R1推理模型的航班定价分析工具，用于分析航班定价数据并提供优化的定价策略建议。

## 项目概述

本项目利用DeepSeek-R1推理模型的强大能力，对航班定价数据进行深入分析，并提供具有可解释性的定价策略建议。DeepSeek-R1模型能够展示详细的思考过程，这对于理解定价决策的逻辑非常有价值。

主要功能包括：

- 航班定价数据的预处理和分析
- 利用DeepSeek-R1模型进行定价策略分析
- 生成可视化图表和HTML分析报告
- 支持多种API提供商（DeepSeek原生API、OpenRouter、Together AI）

## 系统架构

系统由以下几个主要模块组成：

1. **数据处理模块**（`data_processor.py`）：负责加载、预处理和分析航班定价数据
2. **API客户端模块**（`api_client.py`）：负责与DeepSeek-R1 API进行交互
3. **可视化模块**（`visualizer.py`）：负责生成可视化图表和HTML分析报告
4. **主程序**（`main.py`）：整合所有模块并执行完整的分析流程

## 安装指南

### 环境要求

- Python 3.8+
- 必要的Python库：pandas, numpy, matplotlib, seaborn, requests, python-dotenv

### 安装步骤

1. 克隆项目代码：

```bash
git clone https://github.com/your-username/deepseek-pricing.git
cd deepseek-pricing
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置API密钥：

复制`.env.example`文件为`.env`，并填入您的API密钥和其他配置信息：

```bash
cp .env.example .env
# 编辑.env文件，填入您的API密钥
```

## 使用指南

### 基本用法

运行主程序进行分析：

```bash
python main.py --data ../航班定价模拟数据.csv --output ../定价分析结果
```

### 命令行参数

- `--data`：航班定价数据CSV文件路径，默认为`../航班定价模拟数据.csv`
- `--output`：分析结果输出目录，默认为`../定价分析结果`
- `--env`：环境变量配置文件路径，默认为`.env`
- `--temp`：模型温度参数，控制输出的随机性
- `--mock`：使用模拟数据，不调用实际API（用于测试）

### 示例

使用模拟数据进行测试：

```bash
python main.py --mock
```

使用自定义数据文件和输出目录：

```bash
python main.py --data /path/to/your/data.csv --output /path/to/output
```

## API提供商

本工具支持多种API提供商：

1. **DeepSeek原生API**：直接使用DeepSeek提供的API
2. **OpenRouter**：通过OpenRouter访问DeepSeek-R1模型
3. **Together AI**：通过Together AI访问DeepSeek-R1模型

在`.env`文件中设置`API_PROVIDER`参数来选择使用哪个提供商。

## 分析报告

分析完成后，系统会在输出目录生成以下文件：

- `data_summary.json`：数据摘要
- 各种可视化图表（PNG格式）
- `DeepSeek-R1航班定价分析报告.html`：完整的HTML分析报告

HTML报告包含以下内容：

- 数据概述
- 可视化图表
- DeepSeek-R1模型的分析结果，包括思考过程和最终建议

## 常见问题

### API调用失败

如果遇到API调用失败的问题，请检查：

1. API密钥是否正确
2. 网络连接是否正常
3. API提供商是否支持DeepSeek-R1模型
4. API请求参数是否正确

### 中文显示问题

如果图表中的中文显示为方块或乱码，请确保系统安装了支持中文的字体，并在`visualizer.py`中设置正确的字体：

```python
matplotlib.rcParams['font.sans-serif'] = ['Your Chinese Font']
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请联系：your-email@example.com 