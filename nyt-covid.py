import pandas as pd
from urllib.request import urlopen


def optimize(df):
    '''
    Optimizes the data types in a pandas dataframe.
    '''
    dft = df.copy()
    # converts to datetime if possible
    dft = dft.apply(lambda col:
        pd.to_datetime(col, errors='ignore') if col.dtypes=='object' else col)
    
    # if there are less than half as many unique values as there are rows
    # convert to category
    for col in dft.select_dtypes(include='object'):
        if len(dft[col].unique()) / len(df[col]) < 0.5:
            dft[col] = dft[col].astype('category')
            
    # downcasts numeric columns if possible
    dft = dft.apply(lambda col: 
        pd.to_numeric(col, downcast='integer') if col.dtypes=='int64' else col)
    dft = dft.apply(lambda col: 
        pd.to_numeric(col, downcast='float') if col.dtypes=='float64' else col)
    
    return dft


def add_change_cols(df, cols, pre='new_', clip=False):
    df = df.sort_values(by=['date', 'fips'])
    new_cols = [pre + c for c in cols]
    df[new_cols] = df[cols] - df.groupby(by='fips')[cols].shift()
    df[new_cols] = df[new_cols].fillna(0)
    if clip:
        df[new_cols] = df[new_cols].clip(lower=0)
    return (df, new_cols)


def add_savgol_cols(df, cols, window=7, clip=False):
    def my_savgol(x, w):
        if len(x) >= w:
            return savgol_filter(x, w, 1)
        else:
            new_window = int(np.ceil(len(x) / 2) * 2 - 1)
            if new_window <= 1:
                return x
            else:
                return savgol_filter(x, new_window, 1)
    df = df.sort_values(by=['date', 'fips'])
    cols_d = [c + '_' + str(window) + 'sg' for c in cols]
    df[cols_d] = df.groupby(by='fips')[cols].transform(lambda x: my_savgol(x, window))
    if clip:
        df[cols_d] = df[cols_d].clip(lower=0)
    return (df, cols_d)


def update_nyt(cluster=False):
    
    # load newest info from nyt
    with urlopen('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')\
    as response:
        nyt_df_raw = optimize(pd.read_csv(response, dtype={'fips':'str'}))

    nyt_df_raw['fips'] = nyt_df_raw['fips'].astype('object')
    nyt_df_raw.loc[nyt_df_raw['county'] == 'New York City','fips'] = '36NYC'
    nyt_df_raw.loc[nyt_df_raw['county'] == 'Kansas City','fips'] = '29KCM'
    nyt_df_raw.loc[nyt_df_raw['county'] == 'Joplin','fips'] = '29JOP'
    nyt_df_raw['fips'] = nyt_df_raw['fips'].astype('category')
    
    # load demographic information to get population info
    dem_df_to_merge = optimize(
        pd.read_csv(
            './data/processed/dem_df_to_merge.csv', 
            dtype={'fips':'str', 'cluster':'int'}
        )
    )
    
    merge_cols = ['fips', 'total_pop', 'county']
    if cluster:
        merge_cols += ['cluster']
        
    # merge
    df = nyt_df_raw.merge(
        dem_df_to_merge[merge_cols], 
        on='fips', 
        suffixes=('_x','')
    )
    drop_cols = [c for c in df.columns if c[-2:]=='_x']
    df = df.drop(columns=drop_cols)
    
    if cluster:
        df = df.groupby(by=['state', 'cluster', 'date']).agg(
            cases=('cases', sum),
            deaths=('deaths', sum),
            total_pop=('total_pop', sum)
        ).dropna().reset_index().astype({'cases': 'int', 'deaths':'int'})

    df[['cases_per_100k', 'deaths_per_100k']] = (
        df[['cases', 'deaths']].div(df['total_pop'], axis=0) * 100_000
    )
    df = df.drop(columns=['total_pop'])
    df = df.sort_values(by=['date', 'fips'])
    
    cols = ['cases', 'deaths', 'cases_per_100k', 'deaths_per_100k']

    df, new_cols = add_change_cols(df, cols, pre='new_', clip=True)
    df, new_cols_7sg = add_savgol_cols(df, new_cols, clip=True)
    df, new_cols_15sg = add_savgol_cols(df, new_cols, window=15, clip=True)
    df, delta_new_cols = add_change_cols(df, new_cols, pre='delta_')
    df, delta_new_cols_7sg = add_savgol_cols(df, delta_new_cols)
    df, delta_new_cols_15sg = add_savgol_cols(df, delta_new_cols, window=15)

    df['days'] = (
        (df['date'] - df['date'].min()) / np.timedelta64(1, 'D')
    ).astype('int')
    df['mortality_rate'] = df['deaths'] / df['cases']
    
    name = 'nyt_df'
    if cluster:
        name += '_cluster'
    df.to_csv(f'./data/{name}.csv', index=False)
    
    return df