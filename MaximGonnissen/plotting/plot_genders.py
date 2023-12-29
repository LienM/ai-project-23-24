from features.add_gender import add_gender
from utils.utils import load_data_from_hnm, DataFileNames


def plot_genders():
    import matplotlib.pyplot as plt

    articles_df = load_data_from_hnm(DataFileNames.ARTICLES.replace('.csv', '.parquet'), True,
                                     dtype={'article_id': str})
    transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN.replace('.csv', '.parquet'), True,
                                         dtype={'article_id': str})
    customers_df = load_data_from_hnm(DataFileNames.CUSTOMERS.replace('.csv', '.parquet'))

    customers_df['gender'] = add_gender(customers_df, transactions_df, articles_df)

    fig, ax = plt.subplots()
    male_count = len(customers_df.loc[customers_df.gender == 'm'])
    female_count = len(customers_df.loc[customers_df.gender == 'f'])
    unknown_count = len(customers_df.loc[customers_df.gender == 'u'])

    male_percentage = round(male_count / len(customers_df) * 100, 2)
    female_percentage = round(female_count / len(customers_df) * 100, 2)
    unknown_percentage = round(unknown_count / len(customers_df) * 100, 2)

    bars = ax.bar(
        [f"Male\n({male_percentage}%)", f"Female\n({female_percentage}%)", f"Unknown\n({unknown_percentage}%)"],
        [male_count, female_count, unknown_count])
    ax.bar_label(bars)
    ax.set_ylabel('Customer count')
    ax.set_title('Customer counts by predicted gender')
    plt.show()
