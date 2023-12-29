import zipfile
from io import BytesIO

from features.add_gender import add_gender
from pruning.prune_outdated_items import prune_outdated_items
from selection.get_most_popular_gendered_items import get_most_popular_gendered_items
from utils.utils import DataFileNames, get_data_path, get_all_dataframes


def generate_gender_recommendations():
    """
    Generate top-12 gendered recommendations for customers, based on predicted article and customer gender information.
    """
    articles_df, customers_df, transactions_df, submissions_df = get_all_dataframes()

    customers_df['gender'] = add_gender(customers_df, transactions_df, articles_df)

    articles_df, transactions_df = prune_outdated_items(articles_df, transactions_df, cutoff_days=365)

    most_popular_m_items, most_popular_f_items, most_popular_u_items = get_most_popular_gendered_items(articles_df,
                                                                                                       transactions_df)

    submission_df = submissions_df.copy()

    submission_df = submission_df.merge(customers_df[['customer_id', 'gender']], on='customer_id')

    submission_df.loc[submission_df.gender == 'm', 'prediction'] = " ".join(
        most_popular_m_items[:9] + most_popular_u_items[:3])
    submission_df.loc[submission_df.gender == 'f', 'prediction'] = " ".join(
        most_popular_f_items[:9] + most_popular_u_items[:3])
    submission_df.loc[submission_df.gender == 'u', 'prediction'] = " ".join(
        most_popular_u_items[:6] + most_popular_m_items[:3] + most_popular_f_items[:3])

    submission_df.drop('gender', axis=1, inplace=True)

    output_path = get_data_path() / DataFileNames.OUTPUT_DIR / DataFileNames.ZIP_DIR

    submission_bytes = BytesIO()

    submission_df.to_csv(submission_bytes, index=False)

    with zipfile.ZipFile(output_path / 'submission.zip', 'w') as m_zip:
        m_zip.writestr('submission.csv', submission_bytes.getvalue())
