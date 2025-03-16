import pandas as pd
import os

# 要转换的Excel文件列表
excel_files = [
    'HET进出港客座27FEB.xlsx',
    '区内预估28FEB-4MAR25.xlsx',
    'T-2(27FEB25).xlsx'
]

def convert_excel_to_csv(excel_file):
    """将Excel文件转换为CSV格式"""
    try:
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(excel_file)[0]
        
        # 读取Excel文件中的所有工作表
        excel = pd.ExcelFile(excel_file)
        sheet_names = excel.sheet_names
        
        # 为每个工作表创建一个CSV文件
        for sheet in sheet_names:
            # 读取工作表
            df = pd.read_excel(excel_file, sheet_name=sheet)
            
            # 创建CSV文件名（包含工作表名）
            csv_file = f"{file_name}_{sheet}.csv"
            
            # 将数据保存为CSV
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"已将 {excel_file} 的 {sheet} 工作表转换为 {csv_file}")
    except Exception as e:
        print(f"转换 {excel_file} 时出错: {e}")

# 转换所有Excel文件
for excel_file in excel_files:
    if os.path.exists(excel_file):
        print(f"正在处理 {excel_file}...")
        convert_excel_to_csv(excel_file)
    else:
        print(f"文件 {excel_file} 不存在")

print("转换完成！") 