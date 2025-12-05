#!/usr/bin/env python3
"""
数据时间范围推断与历史数据对比分析
通过对比数据集中的特征与历史真实数据，推断训练集的实际时间范围
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# 创建输出目录
output_dir = Path('../analysis')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("数据时间范围推断分析")
print("=" * 80)

# 1. 加载数据
print("\n[1/5] 加载训练数据...")
df = pd.read_csv('../data/hull-tactical-market-prediction/train_cleaned.csv')
print(f"✓ 数据集大小: {df.shape}")
print(f"✓ 时间跨度: {len(df)} 个交易日 (约 {len(df)/252:.1f} 年)")

# 2. 利率分析
print("\n[2/5] 分析无风险利率...")
df['annual_rf_rate'] = df['risk_free_rate'] * 252 * 100  # 转换为年化百分比

print(f"\n利率统计:")
print(f"  平均年化利率: {df['annual_rf_rate'].mean():.2f}%")
print(f"  最小年化利率: {df['annual_rf_rate'].min():.2f}%")
print(f"  最大年化利率: {df['annual_rf_rate'].max():.2f}%")

print(f"\n利率区间分布:")
ultra_low = (df['annual_rf_rate'] < 1.0).sum()
low = ((df['annual_rf_rate'] >= 1.0) & (df['annual_rf_rate'] < 3.0)).sum()
medium = ((df['annual_rf_rate'] >= 3.0) & (df['annual_rf_rate'] < 5.0)).sum()
high = ((df['annual_rf_rate'] >= 5.0) & (df['annual_rf_rate'] < 7.0)).sum()
ultra_high = (df['annual_rf_rate'] >= 7.0).sum()

print(f"  超低利率(<1%): {ultra_low} 天 ({ultra_low/252:.1f} 年)")
print(f"  低利率(1-3%): {low} 天 ({low/252:.1f} 年)")
print(f"  中等利率(3-5%): {medium} 天 ({medium/252:.1f} 年)")
print(f"  高利率(5-7%): {high} 天 ({high/252:.1f} 年)")
print(f"  超高利率(>7%): {ultra_high} 天 ({ultra_high/252:.1f} 年)")

# 3. 波动率分析
print("\n[3/5] 计算市场波动率...")
window = 20
df['rolling_volatility'] = df['market_forward_excess_returns'].rolling(window=window).std() * np.sqrt(252) * 100

print(f"✓ 20日滚动波动率计算完成")
print(f"  平均波动率: {df['rolling_volatility'].mean():.2f}%")
print(f"  最大波动率: {df['rolling_volatility'].max():.2f}%")

# 找出高波动率时期
top_vol = df.nlargest(10, 'rolling_volatility')[['date_id', 'rolling_volatility', 'annual_rf_rate']]
print(f"\nTop 10 最高波动率时期:")
print(top_vol.to_string(index=False))

# 4. 时间范围推断
print("\n[4/5] 推断时间范围...")

# 关键时期识别
high_rate_end = df[df['annual_rf_rate'] > 7.0]['date_id'].max() if (df['annual_rf_rate'] > 7.0).any() else 0
ultra_low_df = df[df['annual_rf_rate'] < 0.5]

print(f"\n关键时期:")
if high_rate_end > 0:
    print(f"  高利率期(>7%)结束于 date_id {high_rate_end}")
    print(f"    → 可能对应: 1990年代初 或 2006-2007年")

if len(ultra_low_df) > 0:
    ultra_low_start = ultra_low_df['date_id'].min()
    ultra_low_end = ultra_low_df['date_id'].max()
    print(f"  超低利率期(<0.5%)范围: {ultra_low_start} - {ultra_low_end}")
    print(f"    持续时长: {(ultra_low_end - ultra_low_start)/252:.1f} 年")
    print(f"    → 明确对应: 2009-2021年 (量化宽松时期)")

# 最高波动率时期
max_vol_idx = df['rolling_volatility'].idxmax()
max_vol_date_id = df.loc[max_vol_idx, 'date_id']
max_vol_rate = df.loc[max_vol_idx, 'annual_rf_rate']
print(f"  最高波动率时期 date_id: {max_vol_date_id}")
print(f"    当时利率: {max_vol_rate:.2f}%")
if max_vol_rate < 1.0:
    print(f"    → 可能是: 2020年3月 (COVID-19)")
elif 1.0 < max_vol_rate < 3.0:
    print(f"    → 可能是: 2008-2009年 (金融危机)")

# 测试不同起始年份假设
print(f"\n时间映射推断:")
for start_year in [1987, 1990, 1995]:
    end_year = start_year + len(df) / 252
    print(f"\n  假设起始年份 {start_year}:")
    print(f"    数据范围: {start_year} - {end_year:.0f}")
    if len(ultra_low_df) > 0:
        ultra_low_year = start_year + ultra_low_start / 252
        error = abs(ultra_low_year - 2009)
        print(f"    超低利率期起始: {ultra_low_year:.0f} (实际应为2009)")
        print(f"    误差: {error:.1f} 年 {'✓' if error < 3 else '✗'}")

# 最佳推断
ESTIMATED_START_YEAR = 1990
df['estimated_year'] = ESTIMATED_START_YEAR + df['date_id'] / 252

print(f"\n最佳推断: {ESTIMATED_START_YEAR} - {ESTIMATED_START_YEAR + len(df)/252:.0f}年")

# 5. 生成可视化
print("\n[5/5] 生成可视化图表...")

# 美联储利率历史参考数据
fed_rate_history = pd.DataFrame([
    {'year': 1990, 'rate': 8.0}, {'year': 1991, 'rate': 5.5},
    {'year': 1992, 'rate': 3.5}, {'year': 1995, 'rate': 5.8},
    {'year': 2000, 'rate': 6.5}, {'year': 2001, 'rate': 3.8},
    {'year': 2004, 'rate': 1.4}, {'year': 2005, 'rate': 3.2},
    {'year': 2006, 'rate': 5.0}, {'year': 2007, 'rate': 5.0},
    {'year': 2008, 'rate': 1.9}, {'year': 2009, 'rate': 0.15},
    {'year': 2010, 'rate': 0.15}, {'year': 2015, 'rate': 0.35},
    {'year': 2016, 'rate': 0.65}, {'year': 2018, 'rate': 2.4},
    {'year': 2019, 'rate': 2.2}, {'year': 2020, 'rate': 0.38},
    {'year': 2021, 'rate': 0.08}, {'year': 2022, 'rate': 1.7},
    {'year': 2023, 'rate': 5.1},
])

# 图1: 利率对比
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

axes[0].plot(df['date_id'], df['annual_rf_rate'], linewidth=1, alpha=0.8, label='数据集无风险利率')
axes[0].axhline(y=7.5, color='r', linestyle='--', alpha=0.5, label='7.5% (高利率参考)')
axes[0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1.0% (超低利率参考)')
axes[0].fill_between(df['date_id'], 0, df['annual_rf_rate'], alpha=0.3)
axes[0].set_xlabel('Date ID', fontsize=12)
axes[0].set_ylabel('年化利率 (%)', fontsize=12)
axes[0].set_title('数据集无风险利率时间序列', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].bar(fed_rate_history['year'], fed_rate_history['rate'], width=0.8, alpha=0.7, color='steelblue')
axes[1].axhline(y=7.5, color='r', linestyle='--', alpha=0.5)
axes[1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
axes[1].set_xlabel('年份', fontsize=12)
axes[1].set_ylabel('联邦基金利率 (%)', fontsize=12)
axes[1].set_title('美联储联邦基金利率历史 (1990-2023)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

events = [(2000, 'dot-com泡沫'), (2008, '金融危机'), (2020, 'COVID-19'), (2022, '激进加息')]
for year, event in events:
    axes[1].annotate(event, xy=(year, 0.5), xytext=(year, -1.5),
                    fontsize=9, ha='center', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

plt.tight_layout()
plot1_path = output_dir / 'rate_comparison.png'
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"✓ 利率对比图已保存: {plot1_path}")
plt.close()

# 图2: 波动率分析
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

axes[0].plot(df['date_id'], df['market_forward_excess_returns'] * 100, linewidth=0.5, alpha=0.7, color='navy')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0].fill_between(df['date_id'], 0, df['market_forward_excess_returns'] * 100, 
                     where=(df['market_forward_excess_returns'] > 0), alpha=0.3, color='green', label='正收益')
axes[0].fill_between(df['date_id'], 0, df['market_forward_excess_returns'] * 100, 
                     where=(df['market_forward_excess_returns'] < 0), alpha=0.3, color='red', label='负收益')
axes[0].set_ylabel('日收益率 (%)', fontsize=12)
axes[0].set_title('市场超额收益率时间序列', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['date_id'], df['rolling_volatility'], linewidth=1, color='orange', label='20日滚动波动率')
axes[1].fill_between(df['date_id'], 0, df['rolling_volatility'], alpha=0.3, color='orange')
axes[1].set_ylabel('年化波动率 (%)', fontsize=12)
axes[1].set_title('市场波动率时间序列 (20日滚动)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

scatter = axes[2].scatter(df['annual_rf_rate'], df['rolling_volatility'], 
                         c=df['date_id'], cmap='viridis', alpha=0.4, s=10)
axes[2].set_xlabel('年化无风险利率 (%)', fontsize=12)
axes[2].set_ylabel('年化波动率 (%)', fontsize=12)
axes[2].set_title('利率 vs 波动率关系 (颜色代表时间顺序)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[2], label='Date ID')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot2_path = output_dir / 'volatility_analysis.png'
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"✓ 波动率分析图已保存: {plot2_path}")
plt.close()

# 图3: 推断时间轴
fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

axes[0].plot(df['estimated_year'], df['annual_rf_rate'], linewidth=1.5, color='steelblue', label='数据集利率')
axes[0].fill_between(df['estimated_year'], 0, df['annual_rf_rate'], alpha=0.3, color='steelblue')
axes[0].set_ylabel('年化利率 (%)', fontsize=12, fontweight='bold')
axes[0].set_title(f'推断时间轴: {ESTIMATED_START_YEAR} - {ESTIMATED_START_YEAR + len(df)/252:.0f}年', 
                 fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

historical_events = [
    (2000, 'dot-com\n泡沫', 'red'),
    (2008, '金融\n危机', 'darkred'),
    (2020, 'COVID-19\n疫情', 'purple'),
    (2022, '激进\n加息', 'orange'),
]

for year, event, color in historical_events:
    if ESTIMATED_START_YEAR <= year <= (ESTIMATED_START_YEAR + len(df)/252):
        axes[0].axvline(x=year, color=color, linestyle='--', alpha=0.6, linewidth=2)
        axes[0].text(year, axes[0].get_ylim()[1] * 0.9, event, 
                    rotation=0, ha='center', fontsize=10, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

axes[1].plot(df['estimated_year'], df['rolling_volatility'], linewidth=1.5, color='orange', label='20日滚动波动率')
axes[1].fill_between(df['estimated_year'], 0, df['rolling_volatility'], alpha=0.3, color='orange')
axes[1].set_xlabel('年份', fontsize=12, fontweight='bold')
axes[1].set_ylabel('年化波动率 (%)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

for year, event, color in historical_events:
    if ESTIMATED_START_YEAR <= year <= (ESTIMATED_START_YEAR + len(df)/252):
        axes[1].axvline(x=year, color=color, linestyle='--', alpha=0.6, linewidth=2)

plt.tight_layout()
plot3_path = output_dir / 'time_axis_inference.png'
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
print(f"✓ 时间轴推断图已保存: {plot3_path}")
plt.close()

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
print(f"\n📊 生成的图表:")
print(f"  1. {plot1_path}")
print(f"  2. {plot2_path}")
print(f"  3. {plot3_path}")
print(f"\n🎯 推断结论: 数据集覆盖约 {ESTIMATED_START_YEAR}-{ESTIMATED_START_YEAR + len(df)/252:.0f}年")
print("=" * 80)
