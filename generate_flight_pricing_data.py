import pandas as pd
import numpy as np
import datetime
import random
import csv

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)

# 定义常量
TOTAL_FLIGHTS = 1000  # 总航班数
START_DATE = datetime.date(2025, 2, 1)  # 开始日期
END_DATE = datetime.date(2025, 4, 30)  # 结束日期

# 定义航线数据 - 提高部分航线的基准票价
routes = [
    {"route": "北京-通辽线", "departure": "北京", "arrival": "通辽", "distance": 500, "base_price": 1000, "elasticity": 1.2, "seasonal_index": 1.0, "profitability": "medium"},
    {"route": "北京-鄂尔多斯线", "departure": "北京", "arrival": "鄂尔多斯", "distance": 600, "base_price": 1200, "elasticity": 1.1, "seasonal_index": 1.05, "profitability": "high"},
    {"route": "北京-呼和浩特线", "departure": "北京", "arrival": "呼和浩特", "distance": 400, "base_price": 900, "elasticity": 1.3, "seasonal_index": 0.95, "profitability": "medium"},
    {"route": "北京-巴彦淖尔线", "departure": "北京", "arrival": "巴彦淖尔", "distance": 650, "base_price": 1050, "elasticity": 1.0, "seasonal_index": 0.9, "profitability": "low"},
    {"route": "北京-包头线", "departure": "北京", "arrival": "包头", "distance": 550, "base_price": 1050, "elasticity": 1.2, "seasonal_index": 1.0, "profitability": "medium"},
    {"route": "北京-乌兰浩特线", "departure": "北京", "arrival": "乌兰浩特", "distance": 700, "base_price": 1300, "elasticity": 0.9, "seasonal_index": 1.1, "profitability": "high"},
    {"route": "北京-海拉尔线", "departure": "北京", "arrival": "海拉尔", "distance": 900, "base_price": 1500, "elasticity": 0.8, "seasonal_index": 1.2, "profitability": "high"},
    {"route": "北京-锡林浩特线", "departure": "北京", "arrival": "锡林浩特", "distance": 600, "base_price": 1050, "elasticity": 1.0, "seasonal_index": 1.0, "profitability": "medium"},
    {"route": "北京-乌海线", "departure": "北京", "arrival": "乌海", "distance": 750, "base_price": 1150, "elasticity": 0.9, "seasonal_index": 0.95, "profitability": "low"},
    {"route": "呼和浩特-巴彦淖尔线", "departure": "呼和浩特", "arrival": "巴彦淖尔", "distance": 250, "base_price": 650, "elasticity": 1.5, "seasonal_index": 0.85, "profitability": "low"},
    {"route": "呼和浩特-通辽线", "departure": "呼和浩特", "arrival": "通辽", "distance": 450, "base_price": 850, "elasticity": 1.3, "seasonal_index": 0.9, "profitability": "low"},
    {"route": "呼和浩特-成都线", "departure": "呼和浩特", "arrival": "成都", "distance": 1200, "base_price": 1900, "elasticity": 0.7, "seasonal_index": 1.15, "profitability": "high"},
    {"route": "呼和浩特-重庆线", "departure": "呼和浩特", "arrival": "重庆", "distance": 1300, "base_price": 2000, "elasticity": 0.7, "seasonal_index": 1.2, "profitability": "high"},
    {"route": "呼和浩特-广州线", "departure": "呼和浩特", "arrival": "广州", "distance": 2000, "base_price": 2400, "elasticity": 0.6, "seasonal_index": 1.25, "profitability": "high"},
    {"route": "呼和浩特-海口线", "departure": "呼和浩特", "arrival": "海口", "distance": 2500, "base_price": 2600, "elasticity": 0.5, "seasonal_index": 1.3, "profitability": "high"},
    {"route": "呼和浩特-乌兰巴托线", "departure": "呼和浩特", "arrival": "乌兰巴托", "distance": 600, "base_price": 2100, "elasticity": 0.8, "seasonal_index": 1.1, "profitability": "high"},
    {"route": "呼和浩特-满洲里线", "departure": "呼和浩特", "arrival": "满洲里", "distance": 700, "base_price": 1200, "elasticity": 0.9, "seasonal_index": 1.0, "profitability": "medium"},
    {"route": "呼和浩特-石家庄线", "departure": "呼和浩特", "arrival": "石家庄", "distance": 500, "base_price": 950, "elasticity": 1.1, "seasonal_index": 0.95, "profitability": "medium"},
    {"route": "呼和浩特-西安线", "departure": "呼和浩特", "arrival": "西安", "distance": 800, "base_price": 1300, "elasticity": 0.9, "seasonal_index": 1.05, "profitability": "medium"},
    {"route": "呼和浩特-杭州线", "departure": "呼和浩特", "arrival": "杭州", "distance": 1500, "base_price": 2100, "elasticity": 0.7, "seasonal_index": 1.15, "profitability": "high"}
]

# 定义航班号
flight_numbers = {
    "北京-通辽线": ["CA1123", "CA1124", "CA8689", "CA8690"],
    "北京-鄂尔多斯线": ["CA1141", "CA1142", "CA1149", "CA1150"],
    "北京-呼和浩特线": ["CA1111", "CA1112"],
    "北京-巴彦淖尔线": ["CA1147", "CA1148", "CA8363", "CA8364"],
    "北京-包头线": ["CA1121", "CA1122"],
    "北京-乌兰浩特线": ["CA1135", "CA1136"],
    "北京-海拉尔线": ["CA1131", "CA1132"],
    "北京-锡林浩特线": ["CA1109", "CA1110", "CA8625", "CA8626"],
    "北京-乌海线": ["CA8607", "CA8608"],
    "呼和浩特-巴彦淖尔线": ["CA8173", "CA8174", "CA8175", "CA8176"],
    "呼和浩特-通辽线": ["CA8177", "CA8178"],
    "呼和浩特-成都线": ["CA8141", "CA8142", "CA8143", "CA8144"],
    "呼和浩特-重庆线": ["CA8145", "CA8146"],
    "呼和浩特-广州线": ["CA8147", "CA8148"],
    "呼和浩特-海口线": ["CA8149", "CA8150"],
    "呼和浩特-乌兰巴托线": ["CA8151", "CA8152"],
    "呼和浩特-满洲里线": ["CA8153", "CA8154"],
    "呼和浩特-石家庄线": ["CA8155", "CA8156"],
    "呼和浩特-西安线": ["CA8157", "CA8158"],
    "呼和浩特-杭州线": ["CA8159", "CA8160"]
}

# 定义机型及其座位数
aircraft_types = {
    "738": 166,  # 波音737-800
    "73W": 166,  # 波音737-700
    "73E": 175,  # 波音737-300
    "73F": 166,  # 波音737-400
    "73B": 158,  # 波音737-500
    "73K": 158,  # 波音737-600
    "ARJ": 89,   # ARJ21
    "7S7": 127,  # 波音787-7
    "737": 127   # 波音737
}

# 定义中国法定节假日（2025年预估）
holidays = [
    datetime.date(2025, 1, 1),   # 元旦
    datetime.date(2025, 1, 28),  # 春节前一天
    datetime.date(2025, 1, 29),  # 春节
    datetime.date(2025, 1, 30),  # 春节假期
    datetime.date(2025, 1, 31),  # 春节假期
    datetime.date(2025, 2, 1),   # 春节假期
    datetime.date(2025, 2, 2),   # 春节假期
    datetime.date(2025, 2, 3),   # 春节假期
    datetime.date(2025, 4, 5),   # 清明节
    datetime.date(2025, 4, 6),   # 清明节假期
    datetime.date(2025, 4, 7),   # 清明节假期
    datetime.date(2025, 5, 1),   # 劳动节
    datetime.date(2025, 5, 2),   # 劳动节假期
    datetime.date(2025, 5, 3),   # 劳动节假期
    datetime.date(2025, 5, 4),   # 劳动节假期
    datetime.date(2025, 5, 5),   # 劳动节假期
]

# 生成日期范围
date_range = []
current_date = START_DATE
while current_date <= END_DATE:
    date_range.append(current_date)
    current_date += datetime.timedelta(days=1)

# 生成航班数据
flight_data = []
flight_id = 1

for _ in range(TOTAL_FLIGHTS):
    # 随机选择航线
    route_info = random.choice(routes)
    route_name = route_info["route"]
    departure_city = route_info["departure"]
    arrival_city = route_info["arrival"]
    distance = route_info["distance"]
    base_price = route_info["base_price"]
    price_elasticity = route_info["elasticity"]
    seasonal_index = route_info["seasonal_index"]
    profitability = route_info["profitability"]
    
    # 随机选择航班号
    flight_number = random.choice(flight_numbers[route_name])
    
    # 随机选择日期
    flight_date = random.choice(date_range)
    day_of_week = flight_date.weekday() + 1  # 1-7表示周一至周日
    is_holiday = flight_date in holidays
    
    # 根据日期调整季节性指数
    month = flight_date.month
    if month in [1, 2, 12]:  # 冬季
        adjusted_seasonal_index = seasonal_index * 0.9
    elif month in [6, 7, 8]:  # 夏季
        adjusted_seasonal_index = seasonal_index * 1.2
    elif month in [3, 4, 5]:  # 春季
        adjusted_seasonal_index = seasonal_index * 1.1
    else:  # 秋季
        adjusted_seasonal_index = seasonal_index * 1.0
    
    # 根据星期几调整需求
    if day_of_week in [1, 4]:  # 周一和周四需求较低
        day_factor = 0.85
    elif day_of_week in [5, 6]:  # 周五和周六需求较高
        day_factor = 1.15
    elif day_of_week == 7:  # 周日需求高
        day_factor = 1.2
    else:  # 其他日子正常
        day_factor = 1.0
    
    # 如果是节假日，进一步调整需求
    holiday_factor = 1.3 if is_holiday else 1.0
    
    # 随机选择机型
    aircraft_type = random.choice(list(aircraft_types.keys()))
    seat_capacity = aircraft_types[aircraft_type]
    
    # 计算飞行小时（假设平均速度为800公里/小时）
    flight_hours = round(distance / 800, 2)
    
    # 根据航线盈利能力调整成本
    if profitability == "high":
        cost_factor = 0.6  # 高盈利航线成本较低
    elif profitability == "medium":
        cost_factor = 0.7  # 中等盈利航线成本适中
    else:
        cost_factor = 0.8  # 低盈利航线成本较高
    
    # 计算成本 - 进一步降低变动成本和固定成本
    variable_cost_per_hour = random.uniform(10, 18) * cost_factor  # 每小时变动成本（万元）
    variable_cost = round(variable_cost_per_hour * flight_hours, 2)
    
    fixed_cost_per_flight = random.uniform(5, 10) * cost_factor  # 每航班固定成本（万元）
    fixed_cost = round(fixed_cost_per_flight, 2)
    
    maintenance_cost = round(random.uniform(0.5, 2.0) * flight_hours * cost_factor, 2)  # 机务相关成本（万元）
    
    total_cost = round(variable_cost + fixed_cost + maintenance_cost, 2)
    
    # 计算客座率（受多种因素影响）
    base_load_factor = random.uniform(0.65, 0.9)  # 基础客座率
    load_factor = min(0.98, base_load_factor * day_factor * holiday_factor * adjusted_seasonal_index)
    
    # 确保客座率在合理范围内
    load_factor = max(0.4, min(0.98, load_factor))
    
    # 计算订座人数
    booked_seats = int(seat_capacity * load_factor)
    
    # 计算平均折扣
    if load_factor > 0.85:
        discount_rate = random.uniform(0.9, 1.0)  # 高客座率，折扣较小
    elif load_factor > 0.7:
        discount_rate = random.uniform(0.8, 0.95)  # 中等客座率，中等折扣
    else:
        discount_rate = random.uniform(0.7, 0.85)  # 低客座率，折扣较大
    
    # 计算平均票价 - 调整票价计算公式
    avg_ticket_price = round(base_price * discount_rate * (1 + (adjusted_seasonal_index - 1) * 0.5), 2)
    
    # 根据航线盈利能力进一步调整票价
    if profitability == "high":
        avg_ticket_price = round(avg_ticket_price * 1.15, 2)  # 高盈利航线票价上浮15%
    elif profitability == "medium":
        avg_ticket_price = round(avg_ticket_price * 1.1, 2)  # 中等盈利航线票价上浮10%
    else:
        avg_ticket_price = round(avg_ticket_price * 1.05, 2)  # 低盈利航线票价上浮5%
    
    # 计算客运收入（万元）
    passenger_revenue = round(avg_ticket_price * booked_seats / 10000, 2)
    
    # 计算补贴收入（增加补贴航线和补贴金额）
    if route_name in ["北京-通辽线", "北京-鄂尔多斯线", "北京-巴彦淖尔线", "呼和浩特-巴彦淖尔线", "呼和浩特-通辽线", "北京-锡林浩特线", "北京-乌海线"]:
        subsidy_revenue = round(random.uniform(3, 8), 2)  # 补贴收入（万元）
    elif route_name in ["北京-呼和浩特线", "北京-包头线", "呼和浩特-石家庄线", "呼和浩特-西安线", "呼和浩特-满洲里线"]:
        subsidy_revenue = round(random.uniform(1, 4), 2)  # 补贴收入（万元）
    else:
        subsidy_revenue = 0
    
    # 计算总收入
    total_revenue = round(passenger_revenue + subsidy_revenue, 2)
    
    # 计算边际贡献和边际贡献率
    marginal_contribution = round(total_revenue - variable_cost, 2)
    if total_revenue > 0:
        marginal_contribution_rate = round(marginal_contribution / total_revenue, 4)
    else:
        marginal_contribution_rate = 0
    
    # 计算利润和利润率
    profit = round(total_revenue - total_cost, 2)
    if total_revenue > 0:
        profit_rate = round(profit / total_revenue, 4)
    else:
        profit_rate = 0
    
    # 计算RASK（单位可用座公里收入）
    ask = round(seat_capacity * distance / 10000, 2)  # 可用座公里（万）
    rask = round(total_revenue / ask, 4) if ask > 0 else 0
    
    # 竞争对手数据
    competitor_load_factor = min(0.95, max(0.3, load_factor + random.uniform(-0.1, 0.1)))
    competitor_price = round(avg_ticket_price * random.uniform(0.9, 1.1), 2)
    
    # 历史客座率（假设与当前客座率相近但有波动）
    historical_load_factor = min(0.95, max(0.3, load_factor + random.uniform(-0.05, 0.05)))
    
    # 计算最优票价区间
    price_lower_bound = round(variable_cost * 10000 / (seat_capacity * 0.7), 2)  # 基于70%客座率的成本覆盖
    price_upper_bound = round(base_price * 1.2, 2)  # 基准价格的1.2倍
    
    # 计算建议最优票价
    if load_factor > 0.85:  # 高需求
        optimal_price = round(min(price_upper_bound, avg_ticket_price * 1.1), 2)
    elif load_factor < 0.6:  # 低需求
        optimal_price = round(max(price_lower_bound, avg_ticket_price * 0.9), 2)
    else:  # 中等需求
        optimal_price = round(avg_ticket_price, 2)
    
    # 添加到数据集
    flight_data.append({
        "flight_id": flight_id,
        "flight_number": flight_number,
        "departure_city": departure_city,
        "arrival_city": arrival_city,
        "route_name": route_name,
        "flight_date": flight_date.strftime("%Y-%m-%d"),
        "day_of_week": day_of_week,
        "is_holiday": 1 if is_holiday else 0,
        "aircraft_type": aircraft_type,
        "seat_capacity": seat_capacity,
        "booked_seats": booked_seats,
        "load_factor": round(load_factor, 4),
        "flight_hours": flight_hours,
        "variable_cost": variable_cost,
        "fixed_cost": fixed_cost,
        "maintenance_cost": maintenance_cost,
        "total_cost": total_cost,
        "passenger_revenue": passenger_revenue,
        "subsidy_revenue": subsidy_revenue,
        "total_revenue": total_revenue,
        "marginal_contribution": marginal_contribution,
        "marginal_contribution_rate": marginal_contribution_rate,
        "profit": profit,
        "profit_rate": profit_rate,
        "avg_ticket_price": avg_ticket_price,
        "discount_rate": round(discount_rate, 4),
        "rask": rask,
        "ask": ask,
        "competitor_load_factor": round(competitor_load_factor, 4),
        "competitor_price": competitor_price,
        "historical_load_factor": round(historical_load_factor, 4),
        "seasonal_index": round(adjusted_seasonal_index, 4),
        "price_elasticity": price_elasticity,
        "optimal_price": optimal_price,
        "price_lower_bound": price_lower_bound,
        "price_upper_bound": price_upper_bound
    })
    
    flight_id += 1

# 将数据转换为DataFrame
df = pd.DataFrame(flight_data)

# 保存为CSV文件
df.to_csv("航班定价模拟数据.csv", index=False, encoding="utf-8-sig")

print(f"已生成{len(flight_data)}条航班定价模拟数据，并保存为'航班定价模拟数据.csv'")

# 生成数据统计信息
print("\n数据统计信息:")
print(f"航线数量: {len(routes)}")
print(f"日期范围: {START_DATE} 至 {END_DATE}")
print(f"机型数量: {len(aircraft_types)}")

# 计算每条航线的平均客座率和平均票价
route_stats = df.groupby("route_name").agg({
    "load_factor": "mean",
    "avg_ticket_price": "mean",
    "profit_rate": "mean"
}).reset_index()

print("\n各航线平均客座率和票价:")
for _, row in route_stats.iterrows():
    print(f"{row['route_name']}: 客座率 {row['load_factor']:.2%}, 平均票价 {row['avg_ticket_price']:.2f}元, 利润率 {row['profit_rate']:.2%}") 