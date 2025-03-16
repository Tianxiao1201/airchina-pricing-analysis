import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from datetime import datetime
import os

# 设置中文字体
try:
    # 尝试使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'STHeiti', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告: 未找到合适的中文字体，图表中的中文可能无法正确显示")

# 创建输出目录
output_dir = "航班定价分析结果"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取数据
print("正在读取数据...")
df = pd.read_csv("航班定价模拟数据.csv")

# 数据概览
print(f"数据集包含 {len(df)} 条记录和 {len(df.columns)} 个字段")
print(f"航线数量: {df['route_name'].nunique()}")
print(f"航班号数量: {df['flight_number'].nunique()}")
print(f"日期范围: {df['flight_date'].min()} 至 {df['flight_date'].max()}")

# 1. 基本统计分析
print("\n正在进行基本统计分析...")

# 计算关键指标的平均值
avg_stats = df.agg({
    'load_factor': 'mean',
    'avg_ticket_price': 'mean',
    'profit_rate': 'mean',
    'marginal_contribution_rate': 'mean'
}).to_frame().T

print("整体平均指标:")
print(f"平均客座率: {avg_stats['load_factor'].values[0]:.2%}")
print(f"平均票价: {avg_stats['avg_ticket_price'].values[0]:.2f}元")
print(f"平均利润率: {avg_stats['profit_rate'].values[0]:.2%}")
print(f"平均边际贡献率: {avg_stats['marginal_contribution_rate'].values[0]:.2%}")

# 2. 航线分析
print("\n正在进行航线分析...")

# 按航线分组计算平均指标
route_stats = df.groupby('route_name').agg({
    'load_factor': 'mean',
    'avg_ticket_price': 'mean',
    'profit_rate': 'mean',
    'marginal_contribution_rate': 'mean',
    'total_revenue': 'sum',
    'profit': 'sum',
    'flight_id': 'count'  # 航班数量
}).reset_index()

route_stats = route_stats.rename(columns={'flight_id': 'flight_count'})
route_stats = route_stats.sort_values('profit_rate', ascending=False)

# 输出航线排名
print("\n航线利润率排名:")
for i, row in route_stats.head(5).iterrows():
    print(f"Top {i+1}: {row['route_name']} - 利润率: {row['profit_rate']:.2%}, 客座率: {row['load_factor']:.2%}, 平均票价: {row['avg_ticket_price']:.2f}元")

print("\n航线利润率倒数排名:")
for i, row in route_stats.tail(5).iterrows():
    print(f"Bottom {len(route_stats)-i}: {row['route_name']} - 利润率: {row['profit_rate']:.2%}, 客座率: {row['load_factor']:.2%}, 平均票价: {row['avg_ticket_price']:.2f}元")

# 3. 客座率与票价关系分析
print("\n正在分析客座率与票价关系...")

# 计算客座率与票价的相关系数
corr = df['load_factor'].corr(df['avg_ticket_price'])
print(f"客座率与票价的相关系数: {corr:.4f}")

# 4. 定价建议
print("\n正在生成定价建议...")

# 为每条航线生成定价建议
pricing_recommendations = []

for _, route_data in df.groupby('route_name'):
    route_name = route_data['route_name'].iloc[0]
    avg_load_factor = route_data['load_factor'].mean()
    avg_price = route_data['avg_ticket_price'].mean()
    avg_profit_rate = route_data['profit_rate'].mean()
    avg_marginal_rate = route_data['marginal_contribution_rate'].mean()
    
    # 根据客座率和利润率确定定价策略
    if avg_load_factor > 0.85 and avg_profit_rate < 0:
        # 高客座率但利润率为负，说明票价过低
        strategy = "提高票价"
        price_adjustment = 0.15  # 建议提高15%
        reason = "客座率高但利润率为负，说明票价过低，无法覆盖成本"
    elif avg_load_factor > 0.85 and avg_profit_rate > 0:
        # 高客座率且利润率为正，可以适当提高票价
        strategy = "小幅提高票价"
        price_adjustment = 0.08  # 建议提高8%
        reason = "客座率高且利润率为正，市场需求强劲，可以适当提高票价"
    elif avg_load_factor < 0.6 and avg_profit_rate < 0:
        # 低客座率且利润率为负，需要降低票价刺激需求
        strategy = "降低票价"
        price_adjustment = -0.12  # 建议降低12%
        reason = "客座率低且利润率为负，需要降低票价刺激需求"
    elif 0.6 <= avg_load_factor <= 0.85 and avg_profit_rate < 0:
        # 中等客座率但利润率为负，需要优化成本结构
        strategy = "优化成本结构"
        price_adjustment = 0.05  # 建议小幅提高票价5%
        reason = "客座率适中但利润率为负，需要优化成本结构并小幅调整票价"
    else:
        # 其他情况，保持现有价格
        strategy = "维持现价"
        price_adjustment = 0
        reason = "客座率和利润率处于合理水平，建议维持现有票价"
    
    # 计算建议票价
    recommended_price = avg_price * (1 + price_adjustment)
    
    # 添加到建议列表
    pricing_recommendations.append({
        'route_name': route_name,
        'avg_load_factor': avg_load_factor,
        'avg_price': avg_price,
        'avg_profit_rate': avg_profit_rate,
        'avg_marginal_rate': avg_marginal_rate,
        'strategy': strategy,
        'price_adjustment': price_adjustment,
        'recommended_price': recommended_price,
        'reason': reason
    })

# 转换为DataFrame并排序
pricing_df = pd.DataFrame(pricing_recommendations)
pricing_df = pricing_df.sort_values('avg_profit_rate', ascending=False)

# 输出定价建议
print("\n航线定价建议:")
for i, row in pricing_df.iterrows():
    print(f"{row['route_name']}:")
    print(f"  当前平均票价: {row['avg_price']:.2f}元, 客座率: {row['avg_load_factor']:.2%}, 利润率: {row['avg_profit_rate']:.2%}")
    print(f"  建议策略: {row['strategy']}, 调整幅度: {row['price_adjustment']*100:.1f}%, 建议票价: {row['recommended_price']:.2f}元")
    print(f"  原因: {row['reason']}")
    print()

# 5. 可视化分析
print("\n正在生成可视化图表...")

# 设置图表风格
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# 5.1 客座率与票价散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='load_factor', y='avg_ticket_price', hue='route_name', alpha=0.7)
plt.title('客座率与票价关系散点图')
plt.xlabel('客座率')
plt.ylabel('平均票价(元)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/客座率与票价关系散点图.png", dpi=300)

# 5.2 航线利润率条形图
plt.figure(figsize=(12, 8))
route_profit = route_stats.sort_values('profit_rate')
sns.barplot(data=route_profit, x='profit_rate', y='route_name')
plt.title('各航线平均利润率')
plt.xlabel('利润率')
plt.ylabel('航线')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig(f"{output_dir}/各航线平均利润率.png", dpi=300)

# 5.3 客座率与边际贡献率散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='load_factor', y='marginal_contribution_rate', hue='route_name', alpha=0.7)
plt.title('客座率与边际贡献率关系散点图')
plt.xlabel('客座率')
plt.ylabel('边际贡献率')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/客座率与边际贡献率关系散点图.png", dpi=300)

# 5.4 航线票价箱线图
plt.figure(figsize=(12, 8))
route_price = df.sort_values('avg_ticket_price', ascending=False)
sns.boxplot(data=route_price, x='avg_ticket_price', y='route_name')
plt.title('各航线票价分布')
plt.xlabel('票价(元)')
plt.ylabel('航线')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig(f"{output_dir}/各航线票价分布.png", dpi=300)

# 5.5 星期几与客座率关系
plt.figure(figsize=(10, 6))
day_load = df.groupby('day_of_week')['load_factor'].mean().reset_index()
day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
day_load['day_name'] = day_load['day_of_week'].apply(lambda x: day_names[x-1])
sns.barplot(data=day_load, x='day_name', y='load_factor')
plt.title('星期几与平均客座率关系')
plt.xlabel('星期')
plt.ylabel('平均客座率')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/星期几与平均客座率关系.png", dpi=300)

# 6. 保存分析结果
print("\n正在保存分析结果...")

# 保存航线统计数据
route_stats.to_csv(f"{output_dir}/航线统计数据.csv", index=False, encoding='utf-8-sig')

# 保存定价建议
pricing_df.to_csv(f"{output_dir}/航线定价建议.csv", index=False, encoding='utf-8-sig')

# 生成HTML报告
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>航班定价分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .chart {{ margin: 20px 0; max-width: 100%; }}
    </style>
</head>
<body>
    <h1>航班定价分析报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>1. 整体统计</h2>
    <p>数据集包含 {len(df)} 条记录，涵盖 {df['route_name'].nunique()} 条航线。</p>
    <p>平均客座率: {avg_stats['load_factor'].values[0]:.2%}</p>
    <p>平均票价: {avg_stats['avg_ticket_price'].values[0]:.2f}元</p>
    <p>平均利润率: <span class="{'positive' if avg_stats['profit_rate'].values[0] > 0 else 'negative'}">{avg_stats['profit_rate'].values[0]:.2%}</span></p>
    <p>平均边际贡献率: <span class="{'positive' if avg_stats['marginal_contribution_rate'].values[0] > 0 else 'negative'}">{avg_stats['marginal_contribution_rate'].values[0]:.2%}</span></p>
    
    <h2>2. 航线分析</h2>
    <h3>利润率最高的航线</h3>
    <table>
        <tr>
            <th>航线</th>
            <th>平均客座率</th>
            <th>平均票价(元)</th>
            <th>利润率</th>
            <th>边际贡献率</th>
            <th>航班数量</th>
        </tr>
"""

# 添加利润率最高的5条航线
for _, row in route_stats.head(5).iterrows():
    html_report += f"""
        <tr>
            <td>{row['route_name']}</td>
            <td>{row['load_factor']:.2%}</td>
            <td>{row['avg_ticket_price']:.2f}</td>
            <td class="{'positive' if row['profit_rate'] > 0 else 'negative'}">{row['profit_rate']:.2%}</td>
            <td class="{'positive' if row['marginal_contribution_rate'] > 0 else 'negative'}">{row['marginal_contribution_rate']:.2%}</td>
            <td>{row['flight_count']}</td>
        </tr>
    """

html_report += """
    </table>
    
    <h3>利润率最低的航线</h3>
    <table>
        <tr>
            <th>航线</th>
            <th>平均客座率</th>
            <th>平均票价(元)</th>
            <th>利润率</th>
            <th>边际贡献率</th>
            <th>航班数量</th>
        </tr>
"""

# 添加利润率最低的5条航线
for _, row in route_stats.tail(5).iterrows():
    html_report += f"""
        <tr>
            <td>{row['route_name']}</td>
            <td>{row['load_factor']:.2%}</td>
            <td>{row['avg_ticket_price']:.2f}</td>
            <td class="{'positive' if row['profit_rate'] > 0 else 'negative'}">{row['profit_rate']:.2%}</td>
            <td class="{'positive' if row['marginal_contribution_rate'] > 0 else 'negative'}">{row['marginal_contribution_rate']:.2%}</td>
            <td>{row['flight_count']}</td>
        </tr>
    """

html_report += """
    </table>
    
    <h2>3. 定价建议</h2>
    <table>
        <tr>
            <th>航线</th>
            <th>当前平均票价(元)</th>
            <th>客座率</th>
            <th>利润率</th>
            <th>建议策略</th>
            <th>调整幅度</th>
            <th>建议票价(元)</th>
            <th>原因</th>
        </tr>
"""

# 添加所有航线的定价建议
for _, row in pricing_df.iterrows():
    html_report += f"""
        <tr>
            <td>{row['route_name']}</td>
            <td>{row['avg_price']:.2f}</td>
            <td>{row['avg_load_factor']:.2%}</td>
            <td class="{'positive' if row['avg_profit_rate'] > 0 else 'negative'}">{row['avg_profit_rate']:.2%}</td>
            <td>{row['strategy']}</td>
            <td>{row['price_adjustment']*100:.1f}%</td>
            <td>{row['recommended_price']:.2f}</td>
            <td>{row['reason']}</td>
        </tr>
    """

html_report += """
    </table>
    
    <h2>4. 可视化分析</h2>
    <h3>客座率与票价关系</h3>
    <img src="客座率与票价关系散点图.png" class="chart" alt="客座率与票价关系散点图">
    
    <h3>各航线平均利润率</h3>
    <img src="各航线平均利润率.png" class="chart" alt="各航线平均利润率">
    
    <h3>客座率与边际贡献率关系</h3>
    <img src="客座率与边际贡献率关系散点图.png" class="chart" alt="客座率与边际贡献率关系散点图">
    
    <h3>各航线票价分布</h3>
    <img src="各航线票价分布.png" class="chart" alt="各航线票价分布">
    
    <h3>星期几与客座率关系</h3>
    <img src="星期几与平均客座率关系.png" class="chart" alt="星期几与平均客座率关系">
    
    <h2>5. 结论与建议</h2>
    <p>根据分析结果，我们提出以下建议：</p>
    <ol>
        <li>对于客座率高但利润率为负的航线，应提高票价以改善盈利能力</li>
        <li>对于客座率低且利润率为负的航线，应降低票价刺激需求</li>
        <li>对于中等客座率但利润率为负的航线，应优化成本结构并适当调整票价</li>
        <li>考虑星期几的客座率差异，对不同日期的航班实施差异化定价</li>
        <li>重点关注利润率最低的航线，考虑调整运力配置或寻求更多补贴支持</li>
    </ol>
</body>
</html>
"""

# 保存HTML报告
with open(f"{output_dir}/航班定价分析报告.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print(f"\n分析完成！结果已保存到 {output_dir} 目录") 