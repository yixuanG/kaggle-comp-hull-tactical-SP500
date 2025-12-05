"""
这段代码可以直接复制粘贴到 eda.ipynb 的末尾
它会生成一个交互式的 t-SNE 3D 可视化
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

# 假设 df 已经加载（如果没有，取消下面的注释）
# df = pd.read_csv('../data/hull-tactical-market-prediction/train.csv')

# 选择数值特征（排除 date_id 和目标变量）
feature_cols = [c for c in df.columns if c not in ['date_id', 'forward_returns', 'market_forward_excess_returns']]

# 由于 t-SNE 计算开销大，我们先采样 2000 个样本
sample_size = min(2000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

# 准备数据：填充 NaN 并标准化
from sklearn.preprocessing import StandardScaler
X = df_sample[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 执行 t-SNE（3D）
print("Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)

# 创建可视化数据框
df_viz = pd.DataFrame({
    'tSNE_1': X_tsne[:, 0],
    'tSNE_2': X_tsne[:, 1],
    'tSNE_3': X_tsne[:, 2],
    'Target': df_sample['market_forward_excess_returns'].values,
    'Date_ID': df_sample['date_id'].values
})

# 创建目标变量的分类（正收益 vs 负收益）
df_viz['Target_Sign'] = df_viz['Target'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

# 方法1：按目标变量符号着色
fig1 = px.scatter_3d(
    df_viz, 
    x='tSNE_1', y='tSNE_2', z='tSNE_3',
    color='Target_Sign',
    title='t-SNE 3D Visualization (Colored by Return Sign)',
    hover_data=['Target', 'Date_ID'],
    opacity=0.7,
    color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B'}
)
fig1.update_traces(marker=dict(size=3))
fig1.show()

# 方法2：按目标变量连续值着色
fig2 = px.scatter_3d(
    df_viz, 
    x='tSNE_1', y='tSNE_2', z='tSNE_3',
    color='Target',
    title='t-SNE 3D Visualization (Colored by Return Magnitude)',
    hover_data=['Date_ID'],
    opacity=0.7,
    color_continuous_scale='RdYlGn'  # 红（负）-> 黄（零）-> 绿（正）
)
fig2.update_traces(marker=dict(size=3))
fig2.show()

# 方法3：按时间着色（看是否有时间聚类）
fig3 = px.scatter_3d(
    df_viz, 
    x='tSNE_1', y='tSNE_2', z='tSNE_3',
    color='Date_ID',
    title='t-SNE 3D Visualization (Colored by Time)',
    hover_data=['Target'],
    opacity=0.7,
    color_continuous_scale='Viridis'
)
fig3.update_traces(marker=dict(size=3))
fig3.show()

print("\n✓ 生成了 3 个 t-SNE 可视化：")
print("  1. 按收益正负着色")
print("  2. 按收益大小着色")
print("  3. 按时间着色")
