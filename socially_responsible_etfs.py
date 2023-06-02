import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)

# Data Import and Processing

etfs = 'socially_responsible_etfs.csv'
df = pd.read_csv(etfs, na_values='--')

# Display the first 5 rows
print(df.head())

#output 2
df['AUM'] = df['AUM'].apply(lambda x: float(x[1:-1]) * (1000 if x[-1] == 'B' else 1))

df.head()

#output 3
temp_df = df['SEGMENT'].str.extract(r'(.+):\s+(.+)\s+-\s+(.+)|(.+):\s+(.+?)\s+(.+)')
temp_df.columns = ['Class1', 'Market1', 'Segment1', 'Class2', 'Market2', 'Segment2']

temp_df.head(10)

#output 4
df.insert(7, 'Class', np.where(temp_df['Class1'].notnull(), temp_df['Class1'], temp_df['Class2']))
df.insert(8, 'Market', np.where(temp_df['Market1'].notnull(), temp_df['Market1'], temp_df['Market2']))
df.insert(9, 'Segment', np.where(temp_df['Segment1'].notnull(), temp_df['Segment1'], temp_df['Segment2']))

# Drop the original SEGMENT column
df = df.drop(columns=['SEGMENT'])

df.head()

#output 5
df = df.set_index(['Issuer', 'Ticker']).sort_index()
df.head()

#output 6
issuer_counts = df.groupby('Issuer')['Fund Name'].count().sort_values(ascending=False)
print(issuer_counts)

#output 7
issuer_aum = df.groupby('Issuer')['AUM'].sum().sort_values(ascending=False)
print(issuer_aum)

#output 8
top_10_etfs = df.sort_values(by='AUM', ascending=False).head(10)
print(top_10_etfs[['Fund Name', 'AUM', 'Expense Ratio', '3-Mo TR', 'Class', 'Market', 'Segment']])

#output 9
average_expense_ratio = df.groupby('Issuer')['Expense Ratio'].mean()
print(average_expense_ratio.sort_values())

#output 10
class_distribution = df['Class'].value_counts(normalize=True)
print(class_distribution)

#output 11
market_distribution = df['Market'].value_counts(normalize=True)
print(market_distribution)

#output 12
segment_distribution = df['Segment'].value_counts(normalize=True)
print(segment_distribution)

#output 13
average_returns = df.groupby('Class')['3-Mo TR'].mean()
print(average_returns)

#output 14
mean_returns_by_class_market = df.groupby(['Class', 'Market'])['3-Mo TR'].mean()
print(mean_returns_by_class_market)


#output 15
mean_returns_by_class_market_segment = df.groupby(['Class', 'Market', 'Segment'])['3-Mo TR'].mean()
print(mean_returns_by_class_market_segment)
