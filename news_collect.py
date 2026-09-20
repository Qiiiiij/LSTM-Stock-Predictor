# -*- coding: utf-8 -*-
"""news_collect.py - 多源新闻采集（东方财富 + 新浪财经）

按公司关键词（易明医药/002826）与行业关键词（医药/生物医药）分别检索，
合并去重后过滤时间窗，输出统一结构的 CSV。
"""

import json
import re

import pandas as pd
import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
}

# ====== 可配置参数 ======
STOCK_NAME = '易明医药'
STOCK_CODE = '002826'
COMPANY_KEYWORDS = ['易明医药', '002826']
INDUSTRY_KEYWORDS = ['医药', '生物医药']
DEFAULT_DAYS = 30
PAGE_SIZE = 50

NEWS_FILE = f'新闻数据_{STOCK_NAME}_{STOCK_CODE}.csv'


def _clean_html(text):
    """去除 <em> 等 HTML 标签与多余空白"""
    if not text:
        return ''
    return re.sub(r'<[^>]+>', '', str(text)).strip()


def fetch_eastmoney(keyword, days=DEFAULT_DAYS):
    """东方财富资讯搜索接口（JSONP）"""
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    inner = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "time",
                "pageIndex": 1,
                "pageSize": PAGE_SIZE,
                "preTag": "",
                "postTag": "",
            }
        },
    }
    params = {"cb": "jsonpCallback", "param": json.dumps(inner, ensure_ascii=False)}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        m = re.search(r'\((.*)\)', resp.text.strip(), re.S)
        if not m:
            print(f"  [东方财富] '{keyword}': 返回格式异常")
            return pd.DataFrame()
        data = json.loads(m.group(1))
        items = (data.get('result') or {}).get('cmsArticleWebOld') or []
        records = []
        for it in items:
            media = _clean_html(it.get('mediaName'))
            records.append({
                '日期': pd.to_datetime(it.get('date'), errors='coerce'),
                '来源': f"东方财富·{media}" if media else '东方财富',
                '标题': _clean_html(it.get('title')),
                '摘要': _clean_html(it.get('content'))[:200],
                '链接': it.get('url', ''),
            })
        df = pd.DataFrame(records)
        print(f"  [东方财富] '{keyword}': {len(df)} 条")
        return df
    except Exception as e:
        print(f"  [东方财富] '{keyword}' 抓取失败: {e}")
        return pd.DataFrame()


def fetch_sina(keyword, days=DEFAULT_DAYS):
    """新浪搜索新闻接口（JSON）"""
    url = "https://search.sina.com.cn/api/news"
    headers = {**HEADERS, 'Referer': 'https://search.sina.com.cn/'}
    params = {'q': keyword, 'page': 1, 'size': PAGE_SIZE}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') != 0:
            print(f"  [新浪财经] '{keyword}': 接口返回异常 {data.get('message')}")
            return pd.DataFrame()
        items = (data.get('data') or {}).get('list') or []
        records = []
        for it in items:
            ts = it.get('ctime')
            dt = pd.to_datetime(ts, unit='s') if ts else \
                pd.to_datetime(it.get('dataTime'), errors='coerce')
            media = _clean_html(it.get('media_show') or it.get('media'))
            records.append({
                '日期': dt,
                '来源': f"新浪财经·{media}" if media else '新浪财经',
                '标题': _clean_html(it.get('title')),
                '摘要': _clean_html(it.get('intro'))[:200],
                '链接': it.get('url', ''),
            })
        df = pd.DataFrame(records)
        print(f"  [新浪财经] '{keyword}': {len(df)} 条")
        return df
    except Exception as e:
        print(f"  [新浪财经] '{keyword}' 抓取失败: {e}")
        return pd.DataFrame()


def collect_news(days=DEFAULT_DAYS):
    """公司 + 行业关键词 × 两个新闻源，合并去重并按时间窗过滤"""
    tasks = [(k, '公司') for k in COMPANY_KEYWORDS] + \
            [(k, '行业') for k in INDUSTRY_KEYWORDS]
    parts = []
    for keyword, category in tasks:
        for fetcher in (fetch_eastmoney, fetch_sina):
            part = fetcher(keyword, days)
            if not part.empty:
                part['类别'] = category
                part['关键词'] = keyword
                parts.append(part)

    if not parts:
        print("未抓取到任何新闻。")
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期', '标题'])
    df = df[df['标题'].str.len() > 0]

    # 时间窗过滤（基于新闻发布日期）
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    df = df[df['日期'] >= cutoff]

    # 按 标题+日期 去重
    before = len(df)
    df = df.drop_duplicates(subset=['标题', '日期'])
    df = df.sort_values('日期', ascending=False).reset_index(drop=True)

    cols = ['日期', '来源', '类别', '关键词', '标题', '摘要', '链接']
    df = df[cols]
    df.to_csv(NEWS_FILE, index=False, encoding='utf_8_sig')

    print(f"\n采集完成：共 {len(df)} 条（去重前 {before} 条），已保存到 {NEWS_FILE}")
    print("分类统计:")
    print(df.groupby(['类别', '来源']).size().to_string())
    if not df.empty:
        print(f"日期范围: {df['日期'].min():%Y-%m-%d} ~ {df['日期'].max():%Y-%m-%d}")
    return df


if __name__ == '__main__':
    collect_news(days=DEFAULT_DAYS)
