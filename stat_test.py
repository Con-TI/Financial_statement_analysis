import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

number_str = '9.8325e+24'

# Split the string at 'e+' to get the exponent part
exponent_part = number_str.split('e+')[1]

print(exponent_part)
# df = pd.read_pickle('./data/MEGA_PICKLE/MEGAMEGA.pkl')
# df['next_1y_pct_change'] = df['next_1y_pct_change'].apply(lambda x: 1 if x > 0 else -1)
# corr_matrix = df.corr('pearson')
# corr_matrix = corr_matrix['next_1y_pct_change']
# print(corr_matrix.sort_values(ascending=False))

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()