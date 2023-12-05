import pandas as pd

index_group_name_gender_score_mapping = {
    'Ladieswear': 1,
    'Menswear': -1,
    'Baby/Children': 0.2,
    'Sport': 0,
    'Divided': 0,
}


def _get_gender_score_for_index_group(index_group: str) -> float:
    """
    Get gender score for index group.
    :param index_group: Index group to get gender score for
    :return: gender score for index group
    """
    return index_group_name_gender_score_mapping.get(index_group, 0)


def _get_gender_str_for_score(gender_score: float) -> str:
    """
    Get gender str for gender score. Ranges are [-1, -0.25), [-0.25, 0.25], and (0.25, 1] respectively for m, u, and f.
    :param gender_score: gender score to get gender str for
    :return: gender str for gender score
    """
    if gender_score <= -0.25:
        return 'm'
    elif gender_score < 0.25:
        return 'u'
    else:
        return 'f'


def add_gender_scores_to_articles(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate gender score for each article in articles_df.

    :param articles_df: DataFrame containing article information
    :return: Modified articles_df with a gender score column
    """
    return articles_df['index_group_name'].apply(_get_gender_score_for_index_group)


def add_gender(customers_df: pd.DataFrame, transactions_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gender to customers df based on category of products purchased.

    Gender is added as a str and as a numerical column.
    - numerical column signifies -1 as male, and 1 as female
    - the str has m, u, and f as values, with ranges [-1, -0.25), [-0.25, 0.25], and (0.25, 1] respectively

    :param customers_df: DataFrame containing customer information
    :param transactions_df: DataFrame containing transaction information
    :param articles_df: DataFrame containing article information
    :return: Customers DataFrame containing new gender enum and gender numerical column
    """
    # Add gender score to articles_df
    temp_articles_df = articles_df.copy()
    temp_articles_df['gender_score'] = add_gender_scores_to_articles(articles_df)

    # We match transactions to articles based on article_id, adding gender_score
    temp_transactions_df = transactions_df.merge(temp_articles_df[['article_id', 'gender_score']], on='article_id')

    # We create a copy which only has the customer_id and gender_score columns
    transactions_df_gender = temp_transactions_df[['customer_id', 'gender_score']].copy()

    # We group by customer_id, and calculate the mean gender_score for each customer
    transactions_df_gender = transactions_df_gender.groupby('customer_id').mean()

    # We add the gender_score column and the calculated gender column to the customers_df
    temp_customers_df = customers_df.merge(transactions_df_gender, on='customer_id')
    return temp_customers_df['gender_score'].apply(_get_gender_str_for_score)
