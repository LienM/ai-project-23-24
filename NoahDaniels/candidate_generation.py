import pandas as pd
from fastbm25 import fastbm25
import re
from tqdm.notebook import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def recall(predictions, test_data):
    """
    Takes a dataframe of predictions and a dataframe of test data and calculates the recall.

    Both dataframes are expected to have the columns "customer_id" and "article_id", and have no duplicate rows.
    (pandas.DataFrame.drop_duplicates() should be called on both before being passed to this function)

    Customers not in the test data are ignored; predictions for those customers are not used.
    Customers missing in the predictions (but present in the test data) have recall of zero.
    """
    joined = pd.merge(test_data, predictions, how="inner").drop_duplicates()
    relevant_selected = joined.groupby("customer_id").count()
    relevant_total = test_data.groupby("customer_id").count()

    recall = relevant_selected.divide(relevant_total, fill_value=0)
    return recall.mean().values[0]


def add_relative_week(data):
    """
    Adds a feature "week_rel" to the dataframe indicating in how many weeks the customer purchased something after the given week.
    For example, if a customer has bought an article in weeks 5, 7 and 10, then the week_rel values for those purchases will be 2, 1 and 0 respectively.

    Useful for getting repurchase candidates in the last x weeks that the customer bought something (simply filter week_rel < x)
    """
    a = (
        data.groupby("customer_id")
        .week.apply(lambda x: pd.factorize(-x, sort=True)[0])
        .explode()
    )
    b = data.copy()
    b.sort_values(["customer_id", "week"], inplace=True)
    b.reset_index(drop=True, inplace=True)
    b["week_rel"] = a.values.astype("int64")

    return b


def baskets(train_data, test_week, test_customers=None):
    """
    Generate a dataframe filled with (customer, week) pairs for which we want to generate candidates:
    a) All unique (customer, week) pairs in the training data are included.
    b) If test_customers are given, then each of those customers is included for the given test_week. Else, all customers in the training data are included for the given test_week.

    If training data is None, only baskets for the test weel are generated and test_customers must be given.
    """
    if test_customers is None:
        test_customers = pd.Series(train_data.customer_id.unique(), name="customer_id")
    else:
        test_customers = pd.Series(test_customers, name="customer_id", dtype="uint64")

    test_baskets = pd.merge(
        test_customers, pd.Series([test_week], name="week"), how="cross"
    )

    result = (
        pd.concat([train_data, test_baskets])
        if train_data is not None
        else test_baskets
    )
    return (
        result.drop_duplicates(["customer_id", "week"])[["customer_id", "week"]]
        .sort_values(["customer_id", "week"])
        .reset_index(drop=True)
        .astype({"week": "int8"})
    )


def by_week(wks, data, d, f, relative=False):
    """
    For each week t in wks, apply f to the data in weeks [t-d, t[ and return the result as a dataframe with an additional column "week" containing t.

    For example, if wks = [3, 4, 7], d = 2 and f is a function that returns the most popular item in the given data, then the result will be a dataframe with 3 rows:
    a) for week 7: the most popular item in weeks 5 and 6
    b) for week 4: the most popular item in weeks 2 and 3
    c) for week 3: the most popular item in weeks 1 and 2

    If relative is True, then 'rel_week' is used instead of 'week' in the above example.
    This means that the considered data is not the previous d weeks, but the previous d weeks that the customer bought something.
    """
    if relative:
        test_week = wks.max()
        result = pd.DataFrame()
        for wk in wks:
            mask = (data.week_rel >= test_week - wk) & (
                data.week_rel < test_week - wk + d
            )
            a = f(data[mask])
            a["week"] = wk
            result = pd.concat((result, a))
        return result
    else:
        result = pd.DataFrame()
        for wk in wks:
            mask = (data.week < wk) & (data.week >= wk - d)
            a = f(data[mask])
            a["week"] = wk
            result = pd.concat((result, a))
        return result.reset_index(drop=True)


def candidates_popularity(baskets, train_data, k=3, d=1):
    """
    Generate candidates for each basket (customer, week) by taking the k most popular items in the previous d weeks.
    """

    def get_popular(df):
        return (
            df.drop_duplicates(["customer_id", "article_id"])
            .article_id.value_counts()
            .rank(method="dense", ascending=False)
            .head(k)
            .astype(int)
            .to_frame("rank")
            .reset_index()
        )

    pops = by_week(baskets.week.unique(), train_data, d, get_popular, False)

    return pd.merge(baskets, pops, on=["week"])


def candidates_repurchase(baskets, train_data, d=1, relative=True):
    """
    Generate candidates for each basket (customer, week) by taking the items the given customer purchased in the previous d weeks.
    If relative is True, only look in the d previous weeks that the customer actually bought something
    """

    def get_repurchased(df):
        return df.sort_values(by="week", ascending=False).drop_duplicates(
            ["customer_id", "article_id"]
        )[["customer_id", "article_id"]]

    rep = by_week(baskets.week.unique(), train_data, d, get_repurchased, relative)

    return pd.merge(baskets, rep, on=["customer_id", "week"])


def candidates_customer_feature(baskets, train_data, customers, feature, k=3, d=1):
    """
    Generate candidates for each basket (customer, week) by taking the k most popular items in the previous d weeks within the group of customers that match on the given feature.
    """

    tt = pd.merge(train_data, customers, how="left", on="customer_id")[
        ["customer_id", "week", "week_rel", "article_id", feature]
    ]

    def get_popular_by_feature(df):
        return (
            df.groupby(feature, observed=False)
            .article_id.value_counts()
            .groupby(feature, observed=False)
            .rank(method="dense", ascending=False)
            .astype(int)
            .to_frame("rank")
            .groupby(feature, observed=False)
            .head(k)
            .reset_index()
        )

    pops = by_week(baskets.week.unique(), tt, d, get_popular_by_feature, False)

    bb = pd.merge(baskets, customers, how="left", on="customer_id")[
        ["customer_id", "week", feature]
    ]

    return pd.merge(bb, pops, on=["week", feature]).drop(columns=feature)


def candidates_article_feature(
    baskets,
    train_data,
    articles,
    feature,
    k,
    d,
    h,
    d2,
    relative,
):
    """
    Generate candidates for each basket (customer, week)
    by taking the h most common feature values in the customer's history in the previous d2 weeks
    for each of those values, take the k most popular items that have that value in the previous d weeks.
    If relative is True, only look in the d2 previous weeks that the customer bought something
    """

    tt = pd.merge(train_data, articles, how="left", on="article_id")

    def get_common_features(df):
        return (
            df.groupby("customer_id", as_index=False)[feature]
            .value_counts()
            .groupby("customer_id")
            .head(h)[["customer_id", feature]]
        )

    # histories contains for each (customer, week) basket the h most common values for the feature in customer's history in [week-d2, week[
    histories = by_week(baskets.week.unique(), tt, d2, get_common_features, relative)
    histories = pd.merge(histories, baskets, on=["customer_id", "week"])

    def get_popular_by_feature(df):
        return (
            df.groupby(feature, observed=False)
            .article_id.value_counts()
            .groupby(feature, observed=False)
            .rank(method="dense", ascending=False)
            .astype(int)
            .to_frame("rank")
            .groupby(feature, observed=False)
            .head(k)
            .reset_index()
        )

    # popularities contains for each (week, feature) the k most popular items in [week-d, week[ that match on the feature
    popularities = by_week(baskets.week.unique(), tt, d, get_popular_by_feature, False)

    return pd.merge(histories, popularities, on=["week", feature]).drop(columns=feature)


def candidate_article_customer_feature(
    baskets, train_data, articles, customers, fa, fc, k, d, h, d2, relative
):
    """
    Combination of candidates_article_feature and candidates_customer_feature
    """

    tt = train_data
    tt = pd.merge(tt, articles, how="left", on="article_id")[
        ["customer_id", "week", "week_rel", "article_id", fa]
    ]
    tt = pd.merge(tt, customers, how="left", on="customer_id")[
        ["customer_id", "week", "week_rel", "article_id", fa, fc]
    ]

    def get_common_features(df):
        return (
            df.groupby("customer_id", as_index=False)[fa]
            .value_counts()
            .groupby("customer_id")
            .head(h)[["customer_id", fa]]
        )

    def get_popular_by_features(df):
        return (
            df.groupby([fa, fc], observed=False)
            .article_id.value_counts()
            .groupby([fa, fc], observed=False)
            .rank(method="dense", ascending=False)
            .astype(int)
            .to_frame("rank")
            .groupby([fa, fc], observed=False)
            .head(k)
            .reset_index()
        )

    # for each customer, get the k2 most common values for the feature fa
    hists = by_week(baskets.week.unique(), tt, d2, get_common_features, relative)

    pops = by_week(baskets.week.unique(), tt, d, get_popular_by_features, False)

    bb = baskets
    bb = pd.merge(bb, hists, how="left", on=["customer_id", "week"])[
        ["customer_id", "week", fa]
    ]
    bb = pd.merge(bb, customers, how="left", on="customer_id")[
        ["customer_id", "week", fa, fc]
    ]

    return pd.merge(bb, pops, on=["week", fa, fc]).drop(columns=[fa, fc])


def candidate_article_similarity(baskets, train_data, article_similarities, k, h, d):
    """
    Similar to candidates_article_feature, but instead of taking the k most popular items that have the same value for the feature, take k most similar items according to the article_similarities dataframe.
    """

    def get_history(df):
        return (
            df.groupby("customer_id", as_index=False)
            .article_id.value_counts()
            .groupby("customer_id")
            .head(h)[["customer_id", "article_id"]]
        )

    # histories contains for each (customer, week) basket the h most common article ids in customer's history in [week-d2, week[
    histories = by_week(baskets.week.unique(), train_data, d, get_history, False)
    histories = pd.merge(histories, baskets, on=["customer_id", "week"])

    # articles contains for each article_id the k_sim most similar articles
    articles = article_similarities.groupby("article_id").head(k)

    return (
        pd.merge(histories, articles, on="article_id")
        .drop(columns=["article_id", "score"])
        .rename(columns={"similar_article_id": "article_id"})
    )


def candidates_filter_popularity(train_data, candidates, k, d):
    """
    Filter generated candidates by retaining only the k most popular (previous d weeks) inclusions for each basket.
    """

    def get_popular(df):
        return (
            df.drop_duplicates(["customer_id", "article_id"])
            .article_id.value_counts()
            .rank(method="dense", ascending=False)
            .astype(int)
            .to_frame("rank")
            .reset_index()
        )

    a = by_week(candidates.week.unique(), train_data, d, get_popular)

    return (
        pd.merge(candidates, a, on=["week", "article_id"])
        .sort_values(["customer_id", "week", "rank"])
        .drop_duplicates(["customer_id", "week", "article_id"])
        .groupby(["customer_id", "week"], as_index=False)
        .head(k)
        .drop(columns="rank")
    )


def candidates_repurchase_bm25(baskets, train_data, articles, d=1, k=3, relative=True):
    """
    Generate candidates for each basket (customer, week) by taking the items that were purchased in the previous d weeks.
    For each of those items, get the k most similar items according to a BM25 search engine.
    If relative is True, only look in the d previous weeks that the customer actually bought something
    """

    # setup BM25
    def tokenize(string):
        return re.sub(r"[^0-9a-zA-Z ]", "", string.lower()).split(" ")

    popular_articles = pd.merge(
        train_data.article_id.value_counts()
        .head(10000)
        .index.to_frame("article_id")
        .reset_index(drop=True),
        articles[["article_id", "descriptor"]],
    )
    corpus = list(popular_articles.descriptor.apply(tokenize).values)
    bm25 = fastbm25(corpus)

    def get_repurchased(df):
        repurchase_data = df.drop_duplicates(["customer_id", "article_id"])[
            ["customer_id", "article_id"]
        ]
        a = pd.merge(repurchase_data, articles, how="left", on="article_id")[
            ["customer_id", "descriptor"]
        ].drop_duplicates(["customer_id", "descriptor"])

        results = []
        for d in tqdm(a.itertuples(index=False), total=len(a)):
            user = d.customer_id
            desc = d.descriptor

            c = bm25.top_k_sentence(tokenize(desc), k=k)
            results.extend(
                [(user, popular_articles.iloc[i].article_id) for _, i, _ in c]
            )

        return pd.DataFrame(results, columns=["customer_id", "article_id"])

    rep = by_week(baskets.week.unique(), train_data, d, get_repurchased, relative)

    return pd.merge(baskets, rep, on=["customer_id", "week"])


def candidates_gpt4rec(baskets, train_data, articles, d=1, k=3, relative=True):
    """
    Generate candidates for each basket (customer, week) by taking the items that were purchased in the previous d weeks.
    Using those items, generate a prompt for GPT4Rec and let it choose k relevant items.
    If relative is True, only look in the d previous weeks that the customer bought something
    """

    def generate_prompt(history):
        previous_items = history[-6:-1]
        return f"Previously, a customer has bought the following items: <{'>, <'.join(previous_items)}>. In the future, this customer will want to buy <"

    # setup GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "../LLM/output5", pad_token_id=tokenizer.eos_token_id
    )

    def generate(p):
        prompt = tokenizer.encode(
            p, return_tensors="pt", truncation=True, max_length=999
        )
        results = model.generate(
            prompt, max_new_tokens=100, num_return_sequences=3, num_beams=5
        )
        for q in results:
            query = tokenizer.decode(q[len(prompt[0]) :], skip_special_tokens=True)[:-1]
            yield query

    # setup BM25
    def tokenize(string):
        return re.sub(r"[^0-9a-zA-Z ]", "", string.lower()).split(" ")

    popular_articles = pd.merge(
        train_data.article_id.value_counts()
        .head(10000)
        .index.to_frame("article_id")
        .reset_index(drop=True),
        articles[["article_id", "descriptor"]],
    )
    corpus = list(popular_articles.descriptor.apply(tokenize).values)
    bm25 = fastbm25(corpus)

    def get_repurchased(df):
        repurchase_data = df.drop_duplicates(["customer_id", "article_id"])[
            ["customer_id", "article_id"]
        ]
        a = pd.merge(repurchase_data, articles, how="left", on="article_id")[
            ["customer_id", "descriptor"]
        ].drop_duplicates(["customer_id", "descriptor"])

        prompts = a.groupby("customer_id").descriptor.apply(list).apply(generate_prompt)

        results = []
        for customer_id, prompt in tqdm(prompts.items(), total=len(prompts)):
            for q in generate(prompt):
                art = bm25.top_k_sentence(tokenize(q), k=k)
                results.extend(
                    [
                        (customer_id, popular_articles.iloc[idx].article_id)
                        for (_, idx, _) in art
                    ]
                )

        return pd.DataFrame(results, columns=["customer_id", "article_id"])

    histories = pd.merge(train_data, baskets, on=["customer_id"], how="inner")
    rep = by_week(baskets.week.unique(), histories, d, get_repurchased, relative)

    return pd.merge(baskets, rep, on=["customer_id", "week"])


#############################################################################################
### The three methods below were used to generate the candidates for the final submission ###
#############################################################################################


def generate_candidates(train_data, test_week, test_customers, c, a):
    """
    Generate candidates for each basket (customer, week) by taking the union of the candidates generated by various candidate generation schemes.

    train_data is used as input to the candidate generation schemes.
    For each basket in the training data, candidate are generated + for the given test week.
        If test_customers are given, they are used for the test week. Else, all customers in the training data are used for the test week.

    c and a are dataframes containing customer and article features respectively.

    Results is a dataframe containing (customer_id, week, article_id, indicator) tuples, where indicator is a string indicating which candidate generation scheme generated the candidate.
    """
    # no canidates for the first third of the training data because canidates depend on a couple of weeks of history
    num_weeks_spare = (test_week - train_data.week.min()) // 3

    b = baskets(train_data, test_week, test_customers)
    b = b[b.week >= test_week - num_weeks_spare]

    def add_indicator_features(df, label):
        df["indicator"] = label
        return df

    candidates = [
        add_indicator_features(
            candidates_repurchase(b, train_data, 10, False), "c_repurchase"
        ),
        add_indicator_features(
            candidates_popularity(b, train_data, 12, 1), "c_popularity1"
        ),
        add_indicator_features(
            candidates_popularity(b, train_data, 5, 3), "c_popularity2"
        ),
        add_indicator_features(
            candidates_customer_feature(b, train_data, c, "postal_code", 100, 4),
            "c_cf_postal_code",
        ),
        add_indicator_features(
            candidates_customer_feature(b, train_data, c, "age", 10, 2),
            "c_cf_age",
        ),
        add_indicator_features(
            candidates_customer_feature(b, train_data, c, "FN", 10, 1), "c_cf_FN"
        ),
        add_indicator_features(
            candidates_article_feature(
                b, train_data, a, "product_code", 8, 1, 100, 5, True
            ),
            "c_af_product_code",
        ),
        add_indicator_features(
            candidates_article_feature(
                b, train_data, a, "department_name", 8, 1, 100, 5, True
            ),
            "c_af_department_name",
        ),
        add_indicator_features(
            candidates_article_feature(
                b, train_data, a, "colour_group_name", 8, 1, 100, 5, True
            ),
            "c_af_colour_group_name",
        ),
        add_indicator_features(
            candidates_article_feature(
                b, train_data, a, "graphical_appearance_name", 8, 1, 100, 5, True
            ),
            "c_af_graphical_appearance_name",
        ),
        add_indicator_features(
            candidates_article_feature(
                b, train_data, a, "index_name", 8, 1, 100, 5, True
            ),
            "c_af_index_name",
        ),
    ]

    result = pd.concat(candidates)

    return result[result.week >= test_week - num_weeks_spare]


def get_examples_candidates(train_data, test_week, test_customers, c, a):
    candidates = generate_candidates(train_data, test_week, test_customers, c, a)
    candidates.reset_index(drop=True, inplace=True)
    candidates = candidates.join(pd.get_dummies(candidates.indicator)).drop(
        columns="indicator"
    )

    # create training examples by taking ground truth data (positive examples) and adding the candidates (negative examples) for non-test weeks
    actual = train_data.copy()[["week", "customer_id", "article_id"]]
    actual["purchased"] = True

    train_examples = pd.concat([actual, candidates[candidates.week != test_week]])
    train_examples["purchased"].fillna(False, inplace=True)

    # remove duplicate examples, but aggregate the rank and indicator columns
    train_examples_a = (
        train_examples.groupby(["customer_id", "week", "article_id"], as_index=False)
        .any()
        .drop(columns="rank")
    )
    train_examples_b = train_examples.groupby(
        ["customer_id", "week", "article_id"], as_index=False
    )["rank"].min()
    train_examples = pd.merge(
        train_examples_a, train_examples_b, on=["customer_id", "week", "article_id"]
    )
    train_examples.sort_values(["customer_id", "week"], inplace=True)
    train_examples.reset_index(drop=True, inplace=True)

    # create testing candidates by filtering on test week
    test_candidates = candidates[candidates.week == test_week].drop(columns="week")
    test_candidates_a = (
        test_candidates.groupby(["customer_id", "article_id"], as_index=False)
        .any()
        .drop(columns="rank")
    )
    test_candidates_b = test_candidates.groupby(
        ["customer_id", "article_id"], as_index=False
    )["rank"].min()
    test_candidates = pd.merge(
        test_candidates_a, test_candidates_b, on=["customer_id", "article_id"]
    )
    test_candidates.sort_values("customer_id", inplace=True)
    test_candidates.reset_index(drop=True, inplace=True)

    return train_examples, test_candidates


def add_features(data, t, c, a):
    columns_to_exclude = [
        "rank",
        "week",
        "purchased",
        "customer_id",
        "article_id",
        "colour_group_name",
        "department_name",
        "detail_desc",
        "garment_group_name",
        "graphical_appearance_name",
        "index_group_name",
        "index_name",
        "perceived_colour_master_name",
        "perceived_colour_value_name",
        "prod_name",
        "product_code",
        "product_group_name",
        "product_type_name",
        "section_name",
    ]

    result = data
    result = pd.merge(result, c, how="left", on="customer_id")
    result = pd.merge(result, a, how="left", on="article_id")

    # features from assignment 2 could go here
    article_max_price = t.groupby("article_id").price.max().to_frame("article_price")
    customer_avg_price = (
        t.groupby("customer_id").price.mean().to_frame("preferred_price")
    )
    buys_for_kids = (
        pd.merge(
            t[["customer_id", "article_id"]],
            a[["article_id", "index_group_name"]],
            on="article_id",
        )
        .groupby("customer_id")
        .index_group_name.agg(lambda x: 1 in x.values)
        .to_frame("buys_for_kids")
    )
    result = pd.merge(result, customer_avg_price, on="customer_id", how="left")
    result = pd.merge(result, article_max_price, on="article_id", how="left")
    result = pd.merge(result, buys_for_kids, on="customer_id", how="left")

    return result.drop(columns=columns_to_exclude, errors="ignore")
