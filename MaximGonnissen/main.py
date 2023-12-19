import multiprocessing as mp
import zipfile
from io import BytesIO

from analysis.seasonality_analysis import run_seasonal_analysis_parallel, run_seasonal_analysis
from features.add_gender import add_gender
from pruning.prune_no_purchases import prune_no_purchases
from pruning.prune_outdated_items import prune_outdated_items
from selection.get_most_popular_gendered_items import get_most_popular_gendered_items
from utils.convert_to_parquet import convert_to_parquet
from utils.utils import load_data_from_hnm, DataFileNames, get_data_path


def get_seasonality_combinations():
    max_score_offset_range = range(-60, 10, 1)
    max_score_day_range_range = range(7, 30, 1)

    combinations = [(max_score_offset, max_score_day_range) for max_score_offset in max_score_offset_range for
                    max_score_day_range in max_score_day_range_range]
    return combinations


def seasonal_analysis():
    use_mp = False

    if use_mp:
        mp_pool_count = max(mp.cpu_count() - 1, 1)
        print(f'Using {mp_pool_count} cores for multiprocessing.')

        run_seasonal_analysis_parallel(mp_pool_count, get_seasonality_combinations())

    else:
        for combination in get_seasonality_combinations():
            run_seasonal_analysis(*combination)


def generate_gender_recommendations():
    articles_df = load_data_from_hnm(DataFileNames.ARTICLES.replace('.csv', '.parquet'), True,
                                     dtype={'article_id': str})
    transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN.replace('.csv', '.parquet'), True,
                                         dtype={'article_id': str})
    sample_submission_df = load_data_from_hnm(DataFileNames.SAMPLE_SUBMISSION.replace('.csv', '.parquet'))
    customers_df = load_data_from_hnm(DataFileNames.CUSTOMERS.replace('.csv', '.parquet'))

    customers_df['gender'] = add_gender(customers_df, transactions_df, articles_df)

    articles_df, transactions_df = prune_outdated_items(articles_df, transactions_df, cutoff_days=365)

    most_popular_m_items, most_popular_f_items, most_popular_u_items = get_most_popular_gendered_items(articles_df, transactions_df)

    submission_df = sample_submission_df.copy()

    submission_df = submission_df.merge(customers_df[['customer_id', 'gender']], on='customer_id')

    submission_df.loc[submission_df.gender == 'm', 'prediction'] = " ".join(most_popular_m_items[:9] + most_popular_u_items[:3])
    submission_df.loc[submission_df.gender == 'f', 'prediction'] = " ".join(most_popular_f_items[:9] + most_popular_u_items[:3])
    submission_df.loc[submission_df.gender == 'u', 'prediction'] = " ".join(most_popular_u_items[:6] + most_popular_m_items[:3] + most_popular_f_items[:3])

    submission_df.drop('gender', axis=1, inplace=True)

    output_path = get_data_path() / DataFileNames.OUTPUT_DIR / DataFileNames.ZIP_DIR

    submission_bytes = BytesIO()

    submission_df.to_csv(submission_bytes, index=False)

    with zipfile.ZipFile(output_path / 'submission.zip', 'w') as m_zip:
        m_zip.writestr('submission.csv', submission_bytes.getvalue())


def convert_all_to_parquet():
    data_path = get_data_path() / DataFileNames.HNM_DIR

    convert_to_parquet(data_path / DataFileNames.CUSTOMERS)
    convert_to_parquet(data_path / DataFileNames.ARTICLES)
    convert_to_parquet(data_path / DataFileNames.TRANSACTIONS_TRAIN)
    convert_to_parquet(data_path / DataFileNames.SAMPLE_SUBMISSION)


def plot_pruning():
    import matplotlib.pyplot as plt

    original_customer_df = load_data_from_hnm(DataFileNames.CUSTOMERS.replace('.csv', '.parquet'))
    original_articles_df = load_data_from_hnm(DataFileNames.ARTICLES.replace('.csv', '.parquet'),
                                              dtype={'article_id': str})
    original_transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN.replace('.csv', '.parquet'),
                                                  dtype={'article_id': str})

    # Prune customers & check how many were pruned
    pruned_customers_df = original_customer_df.copy()
    pruned_customers_df = prune_no_purchases(pruned_customers_df, original_transactions_df)

    print(f"Original customer count: {len(original_customer_df)}")
    print(f"Pruned customer count: {len(pruned_customers_df)}")
    print(f'Pruned {len(original_customer_df) - len(pruned_customers_df)} customers.')
    print(
        f'Pruned {((len(original_customer_df) - len(pruned_customers_df)) / len(original_customer_df)) * 100}% of customers.')

    fig, ax = plt.subplots()
    ax.bar(['Original', 'Pruned'], [len(original_customer_df), len(pruned_customers_df)])
    ax.set_ylabel('Customer count')
    ax.set_title('Original vs. pruned customer count')
    plt.show()

    # Prune articles & check how many were pruned
    pruned_articles_df = original_articles_df.copy()
    pruned_transactions_df = original_transactions_df.copy()
    pruned_articles_df, pruned_transactions_df = prune_outdated_items(pruned_articles_df, pruned_transactions_df)

    print(f"Original article count: {len(original_articles_df)}")
    print(f"Pruned article count: {len(pruned_articles_df)}")
    print(f'Pruned {len(original_articles_df) - len(pruned_articles_df)} articles.')
    print(
        f'Pruned {((len(original_articles_df) - len(pruned_articles_df)) / len(original_articles_df)) * 100}% of articles.')

    print(f"Original transaction count: {len(original_transactions_df)}")
    print(f"Pruned transaction count: {len(pruned_transactions_df)}")
    print(f'Pruned {len(original_transactions_df) - len(pruned_transactions_df)} transactions.')
    print(
        f'Pruned {((len(original_transactions_df) - len(pruned_transactions_df)) / len(original_transactions_df)) * 100}% of transactions.')

    fig, ax = plt.subplots()
    ax.bar(['Original', 'Pruned'], [len(original_articles_df), len(pruned_articles_df)])
    ax.set_ylabel('Article count')
    ax.set_title('Original vs. pruned article count')
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(['Original', 'Pruned'], [len(original_transactions_df), len(pruned_transactions_df)])
    ax.set_ylabel('Transaction count')
    ax.set_title('Original vs. pruned transaction count')
    plt.show()


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

    male_percentage = round(male_count/len(customers_df) * 100, 2)
    female_percentage = round(female_count/len(customers_df) * 100, 2)
    unknown_percentage = round(unknown_count/len(customers_df) * 100, 2)

    bars = ax.bar([f"Male\n({male_percentage}%)", f"Female\n({female_percentage}%)", f"Unknown\n({unknown_percentage}%)"], [male_count, female_count, unknown_count])
    ax.bar_label(bars)
    ax.set_ylabel('Customer count')
    ax.set_title('Customer counts by predicted gender')
    plt.show()


if __name__ == '__main__':
    generate_gender_recommendations()
