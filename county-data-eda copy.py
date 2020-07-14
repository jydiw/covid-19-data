import pandas as pd
from urllib.request import urlopen


def optimize(df):
    '''
    Optimizes the data types in a pandas dataframe.
    '''
    dft = df.copy()
    # converts to datetime if possible
    dft = dft.apply(lambda col: pd.to_datetime(col, errors='ignore') if col.dtypes=='object' else col)
    # if there are less than half as many unique values as there are rows, convert to category
    for col in dft.select_dtypes(include='object'):
        if len(dft[col].unique()) / len(df[col]) < 0.5:
            dft[col] = dft[col].astype('category')
    # downcasts numeric columns if possible
    dft = dft.apply(lambda col: pd.to_numeric(col, downcast='integer') if col.dtypes=='int64' else col)
    dft = dft.apply(lambda col: pd.to_numeric(col, downcast='float') if col.dtypes=='float64' else col)
    return dft


with urlopen('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv') as response:
    nyt_df = optimize(pd.read_csv(response, dtype={'fips':'str'}))


pop_df = pd.read_csv('data/pop_df.csv')

df = nyt_df.merge(pop_df[['fips', 'area', 'population', 'lat', 'lon']], on='fips')

# per capita and per area cols
df[['cases_per_100k', 'deaths_per_100k']] = df[['cases', 'deaths']].div(df['population'], axis=0) * 100_000
df[['case_density', 'death_density']] = df[['cases', 'deaths']].div(df['population'], axis=0).div(df['area'], axis=0) * 100_000
df = df.sort_values(by=['date', 'fips'])

cols = ['cases', 'deaths', 'cases_per_100k', 'deaths_per_100k', 'case_density', 'death_density']
new_cols = ['new_' + c for c in cols]

# making new_cols
df[new_cols] = df[cols] - df.groupby(by='fips')[cols].shift()
df[new_cols] = df[new_cols].fillna(0)
df[new_cols] = df[new_cols].clip(lower=0)

# new_cols_7d
new_cols_7d = [c + '_7d' for c in new_cols]
df[new_cols_7d] = df.groupby(by='fips')[new_cols].apply(lambda x: x.rolling(7, min_periods=1).mean())

df = optimize(df)

df.to_csv('data/df.csv', index=False)