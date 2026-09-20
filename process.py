# monthly_summary.py - 股票数据月度汇总与可视化（支持中文）

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# 设置中文字体显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 设置支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

try:
    from collect import stock_code, stock_name, start_date, end_date
except ImportError:
    # 如果从collect.py导入失败，使用默认值
    stock_code = '002826'
    stock_name = '易明医药'
    start_date = '2024-06-01'
    end_date = '2025-05-31'
    print("注意：使用默认股票参数，因为从collect.py导入失败")


def get_monthly_summary(input_file=None):
    """生成月度汇总数据"""
    if input_file is None:
        input_file = f"{stock_name}_{stock_code}_{start_date}_至_{end_date}.csv"

    try:
        df = pd.read_csv(input_file, parse_dates=['日期'])
        print(f"成功读取日数据文件: {input_file}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    required_cols = ['代码', '简称', '日期', '开盘价(元)', '收盘价(元)', '成交金额(元)']
    if not all(col in df.columns for col in required_cols):
        print("错误：CSV文件中缺少必要列")
        return None

    # 数据处理和汇总
    df['月份'] = df['日期'].dt.to_period('M')
    monthly_data = df.groupby(['代码', '简称', '月份'], observed=True).agg({
        '开盘价(元)': 'mean',
        '收盘价(元)': 'mean',
        '成交金额(元)': 'sum'
    }).reset_index()

    monthly_data.columns = ['代码', '简称', '月份', '平均开盘价(元)', '平均收盘价(元)', '总成交金额(元)']
    monthly_data['月份'] = monthly_data['月份'].astype(str)
    for col in ['平均开盘价(元)', '平均收盘价(元)']:
        monthly_data[col] = monthly_data[col].round(2)
    monthly_data['总成交金额(元)'] = monthly_data['总成交金额(元)'].round(2)

    return monthly_data


def plot_price_trend(monthly_data):
    """绘制价格趋势图（支持中文）"""
    if monthly_data is None or len(monthly_data) == 0:
        print("无有效数据可绘制")
        return

    plt.figure(figsize=(14, 7))

    # 转换月份为日期格式（每月第一天）
    monthly_data['月份日期'] = pd.to_datetime(monthly_data['月份'] + '-01')

    # 绘制两条曲线
    plt.plot(monthly_data['月份日期'], monthly_data['平均开盘价(元)'],
             label='平均开盘价', marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.plot(monthly_data['月份日期'], monthly_data['平均收盘价(元)'],
             label='平均收盘价', marker='s', color='#ff7f0e', linewidth=2.5, markersize=8)

    # 设置图表标题和标签
    plt.title(f'{stock_name}({stock_code}) 股价月度趋势分析', fontsize=16, pad=20)
    plt.xlabel('月份', fontsize=12, labelpad=10)
    plt.ylabel('价格 (元)', fontsize=12, labelpad=10)

    # 设置x轴格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')

    # 添加数值标签
    for i, row in monthly_data.iterrows():
        plt.text(row['月份日期'], row['平均开盘价(元)'], f"{row['平均开盘价(元)']:.2f}",
                 ha='center', va='bottom', fontsize=9, color='#1f77b4')
        plt.text(row['月份日期'], row['平均收盘价(元)'], f"{row['平均收盘价(元)']:.2f}",
                 ha='center', va='top', fontsize=9, color='#ff7f0e')

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')

    # 调整边距
    plt.subplots_adjust(bottom=0.15, top=0.9)

    # 保存图片
    image_file = f"{stock_name}_{stock_code}_月度价格趋势.png"
    plt.savefig(image_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"趋势图已保存到: {image_file}")

    # 显示图表
    plt.show()


def save_monthly_summary(monthly_data, output_file=None):
    """保存月度汇总数据"""
    if output_file is None:
        output_file = f"{stock_name}_{stock_code}_月度汇总_{start_date}_至_{end_date}.csv"

    monthly_data.to_csv(output_file, index=False, encoding='utf_8_sig')
    print(f"月度汇总数据已保存到: {output_file}")
    print("\n前5个月汇总数据:")
    print(monthly_data.head())


if __name__ == '__main__':
    print(f"正在处理 {stock_name}({stock_code}) 的月度数据...")

    # 获取月度汇总数据
    monthly_data = get_monthly_summary()

    if monthly_data is not None:
        # 保存CSV
        save_monthly_summary(monthly_data)

        # 绘制趋势图
        plot_price_trend(monthly_data)