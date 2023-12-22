base_feature_notation = {
    "article_id": "aid",
    "product_type_no": "ptn",
    "graphical_appearance_no": "gan",
    "colour_group_code": "cgc",
    "perceived_colour_value_id": "pcv",
    "perceived_colour_master_id": "pcm",
    "department_no": "dn",
    "index_code": "ic",
    "index_group_no": "ign",
    "section_no": "sn",
    "garment_group_no": "ggn",
    "FN": "fn",
    "Active": "a",
    "club_member_status": "cms",
    "fashion_news_frequency": "fnf",
    "age": "a",
    "postal_code": "pc",
}

added_feature_notation = {
    "has_promotion": "hp",
    "weekly_rank": "wr",
    "all_time_rank": "atr",
    "price_sensitivity": "ps",
    "bought": "b",
}

# Short feature/candidates description
feature_notation = {
    **base_feature_notation,
    **added_feature_notation,
}

# Features to train on (destruct on keys)
all_base_features = list(base_feature_notation.keys())

# Custom features that were engineered
all_added_features = list(added_feature_notation.keys())



candidate_notation = {
    "weekly_bestsellers": "wb",
    "all_time_bestsellers": "atb",
    "age_group_bestsellers": "agb",
    "new_arrivals": "na",
    "previous_purchases": "pp",
}

all_candidate_methods = list(candidate_notation.keys())
