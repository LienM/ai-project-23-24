import time
import pandas as pd


def get_candidates(customer_chunks, unique_articles, t_merged, mean_age_per_article, top_x_price=50, top_x_age=25):

    result_candidates_3feat = pd.DataFrame()  # DataFrame to store final candidates
    result_candidates_3feat_chunks = []

    for idx, customer_chunk in enumerate(customer_chunks):
        start = time.time()
        # Cartesian product of unique articles and customers, since we want to choose candidates out of all unique articles for each customer
        candidate_articles = pd.merge(
            unique_articles,
            pd.DataFrame({'customer_id': customer_chunk}),
            how='cross'
        )
        # get the necessary columns to filter out the candidates
        candidate_articles = pd.merge(candidate_articles, t_merged, on='customer_id', how='left')
        candidate_articles = pd.merge(candidate_articles, mean_age_per_article, on='article_id', how='left')

        # Select all candidates per customer_id where highest_count_ign_per_c is equal to index_group_name
        candidate_articles = candidate_articles[
            candidate_articles['highest_count_ign_per_c'] == candidate_articles['index_group_name']]

        # Calculate price difference for each combination
        candidate_articles['price_difference'] = abs(
            candidate_articles['mean_price_per_c'] - candidate_articles['price'])
        # Rank articles within each customer group based on price difference
        candidate_articles['price_rank'] = (
            candidate_articles
            .groupby(['week', 'customer_id'])['price_difference']
            .rank(ascending=True, method='min')
        )
        # Select the top 50 candidates for each customer
        top_candidates = (
            candidate_articles
            .sort_values(by=['customer_id', 'week', 'price_rank'])
            .groupby(['week', 'customer_id'])
            .head(top_x_price)
        )

        # Calculate age difference for each combination
        top_candidates['age_difference'] = abs(top_candidates['age'] - top_candidates['mean_age_per_a'])
        # Rank articles within each customer group based on age difference
        top_candidates['age_rank'] = (
            top_candidates
            .groupby(['week', 'customer_id'])['age_difference']
            .rank(ascending=True, method='min')
        )
        # Select the top 12 candidates for each customer based on age difference
        top_candidates = (
            top_candidates
            .sort_values(by=['customer_id', 'week', 'age_rank'])
            .groupby(['week', 'customer_id'])
            .head(top_x_age)
        )

        result_candidates_3feat_chunks.append(top_candidates)  # Append current chunk's candidates to result

        print(f'Chunk {idx} processed in {time.time() - start:.2f} seconds')

    # Concatenate all chunks into the final result
    result_candidates_3feat = pd.concat(result_candidates_3feat_chunks, ignore_index=True)

    top_candidates_3feat = result_candidates_3feat.drop(columns=['price_difference', 'age_difference'])
    return top_candidates_3feat
