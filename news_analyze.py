# -*- coding: utf-8 -*-
"""news_analyze.py - 新闻情绪分析

两档设计：
- DictSentimentAnalyzer：本地金融情感词典规则打分（默认，免费离线）
- LLMSentimentAnalyzer：预留 LLM API 接口（DeepSeek/通义等，填入 Key 后启用）

输出：逐条情绪明细 CSV、每日情绪得分 CSV（供后续与 LSTM 融合）、
      新闻情绪 × 股价走势叠加图 PNG。
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd

from sentiment_dict import DEGREE_WORDS, NEGATION_WORDS, NEGATIVE_WORDS, POSITIVE_WORDS

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====== 可配置参数 ======
STOCK_NAME = '易明医药'
STOCK_CODE = '002826'
NEWS_FILE = f'新闻数据_{STOCK_NAME}_{STOCK_CODE}.csv'
PRICE_FILE = f'{STOCK_NAME}_{STOCK_CODE}_2024-06-01_至_2025-05-31.csv'
DETAIL_FILE = '新闻情绪分析明细.csv'
DAILY_SCORE_FILE = '新闻情绪每日得分.csv'
FIGURE_FILE = '新闻情绪与股价对比.png'

POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5


class SentimentAnalyzer(ABC):
    """情绪分析器抽象接口：analyze(text) -> (score: float, label: str)"""

    @abstractmethod
    def analyze(self, text):
        """对一段文本打分。score > 0 偏利好，< 0 偏利空；label ∈ {利好, 利空, 中性}"""
        raise NotImplementedError


class DictSentimentAnalyzer(SentimentAnalyzer):
    """基于金融情感词典的规则分析器：正词加分、负词减分，
    程度副词加权、否定词反转。词典见 sentiment_dict.py，可自行扩充。"""

    def analyze(self, text):
        score = self._score(text)
        if score >= POSITIVE_THRESHOLD:
            return score, '利好'
        if score <= NEGATIVE_THRESHOLD:
            return score, '利空'
        return score, '中性'

    def _score(self, text):
        if not text:
            return 0.0
        total = 0.0
        for word in POSITIVE_WORDS:
            total += self._weighted_count(text, word, sign=1)
        for word in NEGATIVE_WORDS:
            total += self._weighted_count(text, word, sign=-1)
        return round(total, 3)

    @staticmethod
    def _weighted_count(text, word, sign):
        count = text.count(word)
        if count == 0:
            return 0.0
        weight = 1.0
        prefix = text[max(0, text.find(word) - 3):text.find(word)]
        for deg, mult in DEGREE_WORDS.items():
            if deg in prefix:
                weight *= mult
                break
        if any(neg in prefix for neg in NEGATION_WORDS):
            weight = -weight
        return sign * weight * count


class LLMSentimentAnalyzer(SentimentAnalyzer):
    """预留实现：接入 DeepSeek / 通义千问等 LLM API 做新闻情绪分析。

    使用步骤：
        1. 在 __init__ 传入 api_key 与 endpoint；
        2. 实现 _call_api(text) 调用 LLM 并解析情绪得分；
        3. 在 main() 中把 analyzer 替换为本类实例。
    """

    def __init__(self, api_key='', endpoint=''):
        self.api_key = api_key
        self.endpoint = endpoint

    def analyze(self, text):
        raise NotImplementedError(
            'LLMSentimentAnalyzer 为预留接口：请填入 API Key 并实现 API 调用后使用。'
        )


def analyze_news(df, analyzer):
    """逐条新闻打分：以 标题+摘要 作为分析文本"""
    df = df.copy()
    df['摘要'] = df['摘要'].fillna('')
    results = [analyzer.analyze(f"{row['标题']}。{row['摘要']}")
               for _, row in df.iterrows()]
    df['情绪得分'] = [r[0] for r in results]
    df['情绪标签'] = [r[1] for r in results]
    return df


def aggregate_daily(df):
    """按日聚合：日均情绪得分 + 当日新闻数量"""
    d = df.copy()
    d['日期'] = pd.to_datetime(d['日期']).dt.normalize()
    daily = (d.groupby('日期')
               .agg(情绪得分=('情绪得分', 'mean'),
                    新闻数量=('情绪得分', 'count'))
               .reset_index()
               .sort_values('日期'))
    daily['情绪得分'] = daily['情绪得分'].round(3)
    return daily


def print_summary(df):
    """终端报告：正负占比 + TOP3 利好/利空"""
    total = len(df)
    counts = df['情绪标签'].value_counts()
    print(f"\n===== 情绪分析汇总（共 {total} 条） =====")
    for label in ('利好', '中性', '利空'):
        n = counts.get(label, 0)
        print(f"  {label}: {n} 条 ({n / total * 100:.1f}%)")

    for label, ascending in (('利好', False), ('利空', True)):
        top = df[df['情绪标签'] == label].sort_values(
            '情绪得分', ascending=ascending).head(3)
        print(f"\nTOP {len(top)} {label} 新闻:")
        for _, row in top.iterrows():
            print(f"  [{row['情绪得分']:+.2f}] {row['日期']:%Y-%m-%d} {row['标题']}")


def plot_sentiment_vs_price(daily):
    """上下双子图：历史收盘价曲线 × 近 30 天每日新闻情绪得分（红涨绿跌）。

    注：历史行情与实时新闻时间窗通常不重叠，故分面板展示而非强行叠加。"""
    try:
        price = pd.read_csv(PRICE_FILE, parse_dates=['日期'])
    except Exception as e:
        print(f"读取行情文件失败（仅绘制情绪得分）: {e}")
        price = None

    colors = ['#d62728' if s >= 0 else '#2ca02c' for s in daily['情绪得分']]

    if price is not None:
        fig, (ax_price, ax_sent) = plt.subplots(
            2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})
        ax_price.plot(price['日期'], price['收盘价(元)'],
                      color='#1f77b4', linewidth=2, label='收盘价')
        ax_price.set_title(f'{STOCK_NAME}（{STOCK_CODE}）历史收盘价与近期新闻情绪',
                           fontsize=15, pad=12)
        ax_price.set_ylabel('收盘价 (元)', fontsize=12)
        ax_price.grid(True, linestyle='--', alpha=0.4)
        ax_price.legend(fontsize=11, loc='upper left')
    else:
        fig, ax_sent = plt.subplots(figsize=(14, 5))

    ax_sent.bar(daily['日期'], daily['情绪得分'], color=colors, alpha=0.7,
                width=0.8, label='每日新闻情绪得分')
    ax_sent.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_sent.set_ylabel('每日新闻情绪得分', fontsize=12)
    ax_sent.set_xlabel('日期', fontsize=12)
    ax_sent.set_title('近 30 天每日新闻情绪得分', fontsize=12)
    ax_sent.grid(True, linestyle='--', alpha=0.4)
    ax_sent.legend(fontsize=11, loc='upper left')

    fig.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n对比图已保存到: {FIGURE_FILE}")
    plt.show()


def main():
    try:
        df = pd.read_csv(NEWS_FILE, parse_dates=['日期'])
    except FileNotFoundError:
        print(f"未找到 {NEWS_FILE}，请先运行: python news_collect.py")
        return

    if df.empty:
        print("新闻数据为空，无法进行情绪分析。")
        return

    analyzer = DictSentimentAnalyzer()
    df = analyze_news(df, analyzer)
    df.to_csv(DETAIL_FILE, index=False, encoding='utf_8_sig')
    print(f"逐条情绪明细已保存到: {DETAIL_FILE}")

    print_summary(df)

    daily = aggregate_daily(df)
    daily.to_csv(DAILY_SCORE_FILE, index=False, encoding='utf_8_sig')
    print(f"\n每日情绪得分已保存到: {DAILY_SCORE_FILE}（可作为后续 LSTM 融合的输入特征）")

    plot_sentiment_vs_price(daily)


if __name__ == '__main__':
    main()
