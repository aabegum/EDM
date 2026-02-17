import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Read just header
df = pd.read_csv('data/input/aydin.csv', encoding='cp1254', nrows=1, on_bad_lines='skip')
print(f'Total columns: {len(df.columns)}')
print('\nFirst 30 column names:')
for i, col in enumerate(df.columns[:30]):
    print(f'{i}: {repr(col)}')

# Check for duplicates
dup_cols = df.columns[df.columns.duplicated()].tolist()
if dup_cols:
    print(f'\nDuplicate column names found:')
    for col in set(dup_cols):
        count = (df.columns == col).sum()
        print(f'  "{col}": appears {count} times')
else:
    print('\nNo duplicate column names')
