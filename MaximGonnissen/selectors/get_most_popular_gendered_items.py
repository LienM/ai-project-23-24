import pandas as pd
from typing import List

from features.add_gender import add_gender_scores_to_articles


def get_most_popular_gendered_items(articles_df: pd.DataFrame, for_men: bool, item_amount: int = 12) -> List[str]:
    """
    Selects the most popular gendered items from the articles dataframe.
    :param articles_df: The articles dataframe.
    :param for_men: Whether to select the most popular items for men or women customers.
    :param item_amount: The amount of items to select.
    :return: The most popular gendered items.
    """
    articles_df['gender_score'] = add_gender_scores_to_articles(articles_df)

    if for_men:
        return articles_df.sort_values(by=['gender_score'], ascending=False).head(item_amount)['article_id'].tolist()

    return articles_df.sort_values(by=['gender_score'], ascending=True).head(item_amount)['article_id'].tolist()
