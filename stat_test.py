import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle('./data/MEGA_PICKLE/MEGA.pkl')
df['next_1y_pct_change'] = df['next_1y_pct_change'].apply(lambda x: 1 if x > 0 else -1)
corr_matrix = df.corr('kendall')
corr_matrix = corr_matrix['next_1y_pct_change']
print(corr_matrix.sort_values(ascending=False))

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()