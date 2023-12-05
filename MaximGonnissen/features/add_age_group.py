import pandas as pd
import time


class AgeGroup:
    """
    Static class to hold age group constants.
    """
    YOUNG = '0-20'
    YOUNG_ADULT = '20-30'
    ADULT = '30-40'
    MIDDLE_AGED = '40-60'
    OLD = '60+'


def add_age_group(customers_df: pd.DataFrame):
    """
    Adds age group features to the dataframe, using the age column as origin.
    :param customers_df: Customers dataframe to add features to.
    :return: Customers dataframe with added features.
    """
    return pd.cut(customers_df['age'], bins=[0, 20, 30, 40, 60, 100],
                  labels=[AgeGroup.YOUNG, AgeGroup.YOUNG_ADULT, AgeGroup.ADULT, AgeGroup.MIDDLE_AGED, AgeGroup.OLD])
