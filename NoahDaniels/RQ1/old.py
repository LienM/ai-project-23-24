import pandas as pd


def merge_candidates(cs):
    c = pd.concat(cs)
    c.drop_duplicates(["customer_id", "article_id"], inplace=True)
    return c[["customer_id", "article_id"]]


def popular_global(t, c, k=10):
    popular_articles = pd.Series(
        t.article_id.value_counts().head(k).index, name="article_id"
    )
    return pd.merge(c, popular_articles, how="cross")


def repurchase(t, c):
    return t[t.customer_id.isin(c)][["customer_id", "article_id"]]


# get `k` most popular items among users who match on user `feature`
def popular_by_feature(t, c, customers, feature, k=10):
    cc = pd.merge(c, customers, on="customer_id", how="left")
    tt = pd.merge(t, customers, on="customer_id", how="left")
    tt = tt.groupby(feature, observed=False, as_index=False).article_id.value_counts()
    tt = tt.groupby(feature, observed=False).head(k)
    return pd.merge(cc, tt, on=feature)[["customer_id", "article_id"]]


# get top `k2` items in user history (with at least `threshold` occurences in that history)
# for each of those items, get `k1` most popular articles which match on item `feature`
def popular_similar_items(t, c, articles, feature, k1=10, k2=1, threshold=2):
    tt = pd.merge(t, articles[["article_id", feature]], on="article_id", how="left")

    # k1 popular representatives per group
    a = (
        tt.groupby(feature, observed=False, as_index=False)
        .article_id.value_counts()
        .groupby(feature, observed=False)
        .head(k1)
    )
    a["pop_rank"] = (
        a.groupby(feature, observed=False)["count"]
        .rank(method="dense", ascending=False)
        .astype("int8")
    )
    a.drop(columns="count")

    # k2 common items in user history
    b = (
        tt[tt.customer_id.isin(c)]
        .groupby("customer_id", as_index=False)[feature]
        .value_counts()
    )
    b = (
        b[b["count"] >= threshold]
        .groupby("customer_id")
        .head(k2)[["customer_id", feature]]
    )

    return pd.merge(a, b, on=feature, how="inner")[
        ["customer_id", "article_id", "pop_rank"]
    ]


def popular_similar_items_by_week(data, articles, feature, c=None, k1=10, k2=1, t=2):
    tt = pd.merge(data, articles[["article_id", feature]], on="article_id", how="left")

    # k1 popular representatives per group
    a = (
        tt.groupby(feature, observed=False, as_index=False)
        .article_id.value_counts()
        .groupby(feature, observed=False)
        .head(k1)
    )
    a["pop_rank"] = (
        a.groupby(feature, observed=False)["count"]
        .rank(method="dense", ascending=False)
        .astype("int8")
    )
    a.drop(columns="count")

    # k2 common items in user history
    if c is None:
        b = tt
    else:
        b = tt[tt.customer_id.isin(c)]
    b = b.groupby(["customer_id", "week"], as_index=False)[feature].value_counts()
    b = (
        b[b["count"] >= t]
        .drop(columns=["count"])
        .groupby(["customer_id", "week"], as_index=False)
        .value_counts()
        .groupby(["customer_id", "week"])
        .head(k2)[["customer_id", "week", feature]]
    )

    return pd.merge(a, b, on=feature, how="inner")[
        ["customer_id", "week", "article_id", "pop_rank"]
    ]
