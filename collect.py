import requests
import pandas as pd
from datetime import datetime


def get_stock_data(stock_code, stock_name, start_date, end_date):
    # 东方财富API
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        'secid': f"0.{stock_code}",  # 0-深市，1-沪市
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',  # 日K线
        'fqt': '1',
        'beg': start_date.replace('-', ''),
        'end': end_date.replace('-', ''),
    }

    try:
        print(f"正在获取 {stock_name}({stock_code}) 数据...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get('data'):
            print("未获取到有效数据")
            return pd.DataFrame()

        # 解析数据
        records = []
        for line in data['data']['klines']:
            items = line.split(',')
            records.append({
                '代码': stock_code,
                '简称': stock_name,
                '日期': items[0],
                '开盘价(元)': float(items[1]),
                '收盘价(元)': float(items[2]),
                '成交金额(元)': float(items[6])
            })

        df = pd.DataFrame(records)
        df['日期'] = pd.to_datetime(df['日期'])
        return df

    except Exception as e:
        print(f"获取数据时出错: {e}")
        return pd.DataFrame()


def fill_missing_dates(df, start_date, end_date):
    """对齐日期范围，并剔除无行情的周末/节假日（只保留真实交易日）"""
    date_range = pd.date_range(start=start_date, end=end_date)
    full_df = pd.DataFrame({'日期': date_range})
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    if not df.empty:
        merged_df = pd.merge(full_df, df, on='日期', how='left')
    else:
        merged_df = full_df
        merged_df['代码'] = None
        merged_df['简称'] = None
        merged_df['开盘价(元)'] = None
        merged_df['收盘价(元)'] = None
        merged_df['成交金额(元)'] = None

    # 填充股票代码和简称
    merged_df['代码'] = merged_df['代码'].ffill().bfill()
    merged_df['简称'] = merged_df['简称'].ffill().bfill()

    # 剔除无行情的日期（周末/节假日），只保留真实交易日，避免下游产生假数据
    merged_df = merged_df.dropna(subset=['开盘价(元)', '收盘价(元)']).reset_index(drop=True)

    return merged_df


def save_to_csv(df, filename):
    """保存数据到CSV"""
    df.to_csv(filename, index=False, encoding='utf_8_sig')
    print(f"数据已保存到: {filename}")
    print("\n数据示例:")
    print(df.head(10))


if __name__ == '__main__':
    # 参数设置
    stock_code = '002826'
    stock_name = '易明医药'
    start_date = '2024-06-01'
    end_date = '2025-05-31'

    # 获取数据
    stock_df = get_stock_data(stock_code, stock_name, start_date, end_date)

    # 填充完整日期范围
    full_df = fill_missing_dates(stock_df, start_date, end_date)

    # 保存文件
    filename = f"{stock_name}_{stock_code}_{start_date}_至_{end_date}.csv"
    save_to_csv(full_df, filename)

    # 数据质量检查
    missing_data = full_df[full_df['开盘价(元)'].isna()]
    if not missing_data.empty:
        print("\n以下日期缺少数据:")
        print(missing_data[['日期', '开盘价(元)']].to_string(index=False))