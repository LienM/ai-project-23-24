import multiprocessing as mp

from analysis.seasonality_analysis import run_seasonal_analysis_parallel, run_seasonal_analysis
from pruning.prune_outdated_items import prune_outdated_items
from selectors.get_most_popular_gendered_items import get_most_popular_gendered_items
from utils.utils import load_data_from_hnm, DataFileNames, get_data_path
from utils.convert_to_parquet import convert_to_parquet
from features.add_gender import add_gender

from io import BytesIO
import zipfile


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
    articles_df = load_data_from_hnm(DataFileNames.ARTICLES.replace('.csv', '.parquet'), True, dtype={'article_id': str})
    transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN.replace('.csv', '.parquet'), True, dtype={'article_id': str})
    sample_submission_df = load_data_from_hnm(DataFileNames.SAMPLE_SUBMISSION.replace('.csv', '.parquet'))
    customers_df = load_data_from_hnm(DataFileNames.CUSTOMERS.replace('.csv', '.parquet'))

    customers_df['gender'] = add_gender(customers_df, transactions_df, articles_df)

    articles_df, transactions_df = prune_outdated_items(articles_df, transactions_df)

    most_popular_m_items = get_most_popular_gendered_items(articles_df, True)
    most_popular_f_items = get_most_popular_gendered_items(articles_df, False)

    submission_df = sample_submission_df.copy()

    submission_df = submission_df.merge(customers_df[['customer_id', 'gender']], on='customer_id')
    submission_df.loc[submission_df.gender == 'm', 'prediction'] = " ".join(most_popular_m_items)
    submission_df.loc[submission_df.gender == 'u', 'prediction'] = " ".join(most_popular_m_items)
    submission_df.loc[submission_df.gender == 'f', 'prediction'] = " ".join(most_popular_f_items)

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


if __name__ == '__main__':
    pass
