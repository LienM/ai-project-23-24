import numpy as np


def print_na(df, df_name, column):
    na_users = df[column].isna().sum()
    print(f"Total number of {df_name} is {df.shape[0]}. {na_users / df.shape[0] * 100}% have no {column}.")


def inconsistency(df, col1, col2, print_all=False):
    inconsistency_helper(df, col1, col2, print_all)
    inconsistency_helper(df, col2, col1, print_all)


def inconsistency_helper(df, col1, col2, print_all):
    inconsistencies = df.groupby(col1)[col2].nunique().max()
    if inconsistencies == 1:
        print(f'No inconsistencies from {col1} to {col2}')
    else:
        inconsistencies = df.groupby(col1)[col2].unique()
        inconsistencies = inconsistencies[inconsistencies.map(len) > 1]
        print(f'Found {len(inconsistencies)} inconsistencies from {col1} to {col2}')
        if print_all:
            print(inconsistencies[inconsistencies.map(len) > 1])


def AinB(df_a, col_a, df_b, col_b):
    ainb = df_a[col_a].isin(df_b[col_b]).astype(int)
    ainb = df_a.assign(InDf2=ainb)
    bina = df_b[col_b].isin(df_a[col_a]).astype(int)
    bina = df_b.assign(InDf2=bina)
    print(ainb[ainb.InDf2 == 0])
    print(bina[bina.InDf2 == 0])


def check_regex(df, column, regex, unique=True):
    column_check = df[column].str.fullmatch(regex)
    print("Rows not matching the regex:")
    print(df[~column_check])
    if unique:
        if df[column].nunique() == df.shape[0]:
            print(f"All values for column {column} are unique")
        else:
            print(f'Column {column} has non unique values')


def verify_submission(df, customers, articles):
    columns_with_nan = df.columns[df.isna().any()].tolist()
    if len(columns_with_nan) == 0:
        print('No columns with nan values')
    else:
        print(f'Following columns contain nan values: {columns_with_nan}')
    AinB(df, "customer_id", customers, "customer_id")

    items = df["prediction"].str.split(" ", expand=True)
    print(f'All are 12 items: {items.shape[1] == 12 and items.notna().all().all()}')

    items_unique = np.unique(items.values)
    articles_unique = articles['article_id'].unique()
    mask = np.isin(articles_unique, items_unique)
    print(articles_unique[~mask])
