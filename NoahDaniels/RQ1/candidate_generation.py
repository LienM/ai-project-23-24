import pandas as pd
from fastbm25 import fastbm25
import re
from tqdm.notebook import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# def get_shifted_weeks(t):
#     def list_min_first(arg):
#         return list(arg)[1:]

#     def list_min_last(arg):
#         return list(arg)[:-1]

#     b = t.drop_duplicates(["customer_id", "week"]).sort_values("week")[
#         ["customer_id", "week"]
#     ]
#     c = (
#         b.groupby("customer_id")
#         .week.apply(list_min_first)
#         .explode()
#         .dropna()
#         .to_frame()
#     )
#     c["week2"] = b.groupby("customer_id").week.apply(list_min_last).explode().dropna()
#     d = (
#         t.groupby(["customer_id", "week"], as_index=False)
#         .article_id.apply(list)
#         .rename(columns={"week": "week2"})
#     )

#     result = (
#         pd.merge(c, d, on=["customer_id", "week2"])
#         .drop(columns=["week2"])
#         .explode("article_id")
#     )
#     result["article_id"] = result.article_id.astype("uint64")
#     return result


def add_relative_week(data, test_week):
    a = (
        data.groupby("customer_id")
        .week.apply(lambda x: pd.factorize(test_week - x, sort=True)[0])
        .explode()
    )
    b = data.copy()
    b.sort_values(["customer_id", "week"], inplace=True)
    b.reset_index(drop=True, inplace=True)
    b["week_rel"] = a.values

    return b


def baskets(train_data, test_week, test_customers=None, only_test=False):
    if test_customers is None:
        test_customers = pd.Series(train_data.customer_id.unique(), name="customer_id")
    else:
        test_customers = pd.Series(test_customers, name="customer_id", dtype="uint64")

    test_baskets = pd.merge(
        test_customers, pd.Series([test_week], name="week"), how="cross"
    )

    result = pd.concat([train_data, test_baskets]) if not only_test else test_baskets
    return (
        result.drop_duplicates(["customer_id", "week"])[["customer_id", "week"]]
        .sort_values(["customer_id", "week"])
        .reset_index(drop=True)
        .astype({"week": "int8"})
    )


def by_week(wks, data, d, f, relative=False):
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
        return result


def candidates_popularity(baskets, train_data, k=3, d=1):
    """
    Generate candidates for each basket (customer, week) by taking the k most popular items in the previous d weeks.
    """

    def get_popular(df):
        return df.article_id.value_counts().head(k).index.to_frame("article_id")

    pops = by_week(baskets.week.unique(), train_data, d, get_popular)

    return pd.merge(baskets, pops, on=["week"])


def candidates_repurchase(baskets, train_data, d=1, relative=True):
    """
    Generate candidates for each basket (customer, week) by taking the items that were purchased in the previous d weeks.
    If relative is True, only look in the d previous weeks that the customer bought something
    """

    def get_repurchased(df):
        return df.drop_duplicates(["customer_id", "article_id"])[
            ["customer_id", "article_id"]
        ]

    rep = by_week(baskets.week.unique(), train_data, d, get_repurchased, relative)

    return pd.merge(baskets, rep, on=["customer_id", "week"])


def candidates_repurchase_bm25(baskets, train_data, articles, d=1, k=3, relative=True):
    """
    Generate candidates for each basket (customer, week) by taking the items that were purchased in the previous d weeks.
    If relative is True, only look in the d previous weeks that the customer bought something
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
                art = bm25.top_k_sentence(tokenize(q), k=10)
                results.extend(
                    [
                        (customer_id, popular_articles.iloc[idx].article_id)
                        for (_, idx, _) in art
                    ]
                )

        return pd.DataFrame(results, columns=["customer_id", "article_id"])

        # results = []
        # for d in tqdm(a.itertuples(index=False), total=len(a)):
        #     user = d.customer_id
        #     desc = d.descriptor

        #     c = bm25.top_k_sentence(tokenize(desc), k=k)
        #     results.extend([(user, articles.iloc[i].article_id) for _, i, _ in c])

        # return pd.DataFrame(results, columns=["customer_id", "article_id"])

    histories = pd.merge(train_data, baskets, on=["customer_id"], how="inner")
    rep = by_week(baskets.week.unique(), histories, d, get_repurchased, relative)

    return pd.merge(baskets, rep, on=["customer_id", "week"])


# popular k1 items (previous l1 weeks), within group of similar customers (customer feature)
def candidates_customer_feature(
    baskets, train_data, customers, feature, k=3, d=1, relative=False
):
    """
    Generate candidates for each basket (customer, week) by taking the k most popular items in the previous d weeks within the group of customers that match on the given feature.
    If relative is True, only look in the d previous weeks that the customer bought something
    """

    tt = pd.merge(train_data, customers, how="left", on="customer_id")[
        ["customer_id", "week", "week_rel", "article_id", feature]
    ]
    bb = pd.merge(baskets, customers, how="left", on="customer_id")[
        ["customer_id", "week", feature]
    ]

    def get_popular_by_feature(df):
        return (
            df.groupby(feature, observed=False, as_index=False)
            .article_id.value_counts()
            .groupby(feature, observed=False)
            .head(k)[[feature, "article_id"]]
        )

    pops = by_week(baskets.week.unique(), tt, d, get_popular_by_feature, relative)

    return pd.merge(bb, pops, on=["week", feature])[
        ["customer_id", "week", "article_id"]
    ]


def candidates_article_feature(
    baskets,
    train_data,
    articles,
    feature,
    k1=3,
    k2=1,
    d1=1,
    d2=1,
    rel1=False,
    rel2=True,
):
    """
    Generate candidates for each basket (customer, week)
    by taking the k2 most common items in the customer's history in the previous d2 weeks
    for each of those items, take the k1 most popular items that match on the given feature in the previous d1 weeks.
    If rel1 is True, only look in the d1 previous weeks that the customer bought something
    If rel2 is True, only look in the d2 previous weeks that the customer bought something
    """

    tt = pd.merge(train_data, articles, how="left", on="article_id")

    def get_popular_by_feature(df):
        return (
            df.groupby(feature, observed=False, as_index=False)
            .article_id.value_counts()
            .groupby(feature, observed=False)
            .head(k1)[[feature, "article_id"]]
        )

    def get_common_features(df):
        return (
            df.groupby("customer_id", as_index=False)[feature]
            .value_counts()
            .groupby("customer_id")
            .head(k2)[["customer_id", feature]]
        )

    pops = by_week(baskets.week.unique(), tt, d1, get_popular_by_feature, rel1)
    hists = by_week(baskets.week.unique(), tt, d2, get_common_features, rel2)
    histories = pd.merge(hists, baskets, on=["customer_id", "week"])

    return pd.merge(histories, pops, on=["week", feature])[
        ["customer_id", "week", "article_id"]
    ]


def generate_candidates(train_data, test_week, test_customers, c, a):
    num_weeks_spare = 5

    b = baskets(train_data, test_week, test_customers)
    b = b[b.week >= test_week - num_weeks_spare]

    candidates = [
        candidates_repurchase(b, train_data, 5),
        candidates_article_feature(b, train_data, a, "prod_name", 10, 20, 3, 5),
        candidates_customer_feature(b, train_data, c, "postal_code", 10, 2),
        candidates_customer_feature(b, train_data, c, "age", 10, 2),
    ]

    result = pd.concat(candidates).drop_duplicates(
        ["customer_id", "week", "article_id"]
    )
    return result[result.week >= test_week - num_weeks_spare]


def get_examples_candidates(train_data, test_week, test_customers, c, a):
    candidates = generate_candidates(train_data, test_week, test_customers, c, a)

    # create training examples by taking ground truth data (positive examples) and adding the candidates (negative examples) for non-test weeks
    actual = train_data.copy()[["week", "customer_id", "article_id"]]
    actual = actual[actual.week.isin(candidates.week.unique())]
    actual["purchased"] = True
    train_examples = pd.concat([actual, candidates[candidates.week != test_week]])
    train_examples.purchased.fillna(False, inplace=True)
    train_examples.drop_duplicates(
        ["customer_id", "article_id", "week"], keep="first", inplace=True
    )
    train_examples.sort_values(["customer_id", "week"], inplace=True)
    train_examples.reset_index(drop=True, inplace=True)

    # create testing candidates by filtering on test week
    test_candidates = candidates[candidates.week == test_week].drop(columns="week")
    test_candidates.drop_duplicates(["customer_id", "article_id"], inplace=True)
    test_candidates.sort_values("customer_id", inplace=True)
    test_candidates.reset_index(drop=True, inplace=True)

    return train_examples, test_candidates


def add_features(data, t, c, a):
    columns_to_use = [
        # "article_id",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "department_no",
        "index_code",
        "index_group_no",
        "section_no",
        "garment_group_no",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "postal_code",
        "preferred_price",
        "article_price",
        "buys_for_kids",
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

    return result[columns_to_use]
