import pandas as pd
from typing import List, Tuple

from features.add_gender import add_gender_scores_to_articles
from features.add_article_total_sales_popularity import add_article_total_sales_popularity
from features.add_article_unique_customers_popularity import add_article_unique_customers_popularity


def get_most_popular_gendered_items(articles_df: pd.DataFrame, transactions_df: pd.DataFrame, item_amount: int = 12, alt_popularity: bool = False) -> Tuple[List[str], List[str]]:
    """
    Selects the most popular gendered items from the articles dataframe.
    :param articles_df: The articles dataframe.
    :param transactions_df: The transactions dataframe.
    :param item_amount: The amount of items to select.
    :param alt_popularity: Whether to use the alternative popularity metric.
    :return: The most popular gendered items.
    """
    temp_articles_df = articles_df.copy()

    if "gender_score" not in temp_articles_df.columns:
        temp_articles_df['gender_score'] = add_gender_scores_to_articles(temp_articles_df)
    if "popularity" not in temp_articles_df.columns:
        if alt_popularity:
            temp_articles_df['popularity'] = add_article_unique_customers_popularity(temp_articles_df, transactions_df)
        else:
            temp_articles_df['popularity'] = add_article_total_sales_popularity(temp_articles_df, transactions_df)

    male_top = temp_articles_df.sort_values(by=['gender_score', 'popularity'], ascending=[False, True]).head(item_amount)['article_id'].tolist()
    female_top = temp_articles_df.sort_values(by=['gender_score', 'popularity'], ascending=[True, True]).head(item_amount)['article_id'].tolist()

    return male_top, female_top
