# 航班定价分析系统

一个基于Streamlit和大语言模型的航班定价分析系统，支持航班数据整体分析和单个航班定价建议。

## 功能特点

- **航班数据整体分析**：上传历史数据，生成综合分析报告
- **单个航班定价建议**：输入特定航班信息，获取定价建议
- **多模型支持**：支持DeepSeek和通义千问两种大语言模型
- **可视化分析**：自动生成多种数据可视化图表
- **模拟模式**：无需API密钥也可体验系统功能

## 安装与使用

### 本地运行

1. 克隆仓库
   ```
   git clone https://github.com/yourusername/airchina-pricing-analysis.git
   cd airchina-pricing-analysis
   ```

2. 安装依赖
   ```
   pip install -r requirements.txt
   ```

3. 配置API密钥
   ```
   cp deepseek_pricing/.env.example deepseek_pricing/.env
   ```
   然后编辑`.env`文件，填入您的API密钥

4. 运行应用
   ```
   streamlit run app.py
   ```

### 使用说明

1. 在侧边栏选择分析模型（DeepSeek或通义千问）
2. 上传航班历史数据文件（CSV格式）
3. 系统会自动分析数据并生成报告
4. 可以切换到"单个航班定价建议"功能，输入具体航班信息获取定价建议

## 数据格式

上传的CSV文件应包含以下字段：
- route_name: 航线名称
- date: 日期
- flight_no: 航班号
- aircraft_type: 机型
- seat_capacity: 座位容量
- load_factor: 客座率
- avg_ticket_price: 平均票价
- cost_per_seat: 每座位成本
- profit_margin: 利润率
- is_holiday: 是否假期
- day_of_week: 星期几
- ...

## API密钥获取

- DeepSeek API: [https://platform.deepseek.com](https://platform.deepseek.com)
- 通义千问 API: [https://dashscope.aliyun.com](https://dashscope.aliyun.com)

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT 