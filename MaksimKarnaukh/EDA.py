import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def load_dataset(file, data_types: dict = None):
    """
    Loads a dataset.
    :param file: the file path of the dataset
    :param data_types: the data types of the columns in the dataset
    :return: the loaded dataset
    """
    return pd.read_csv(file, dtype=data_types)

def perform_basic_eda(df) -> None:
    """
    Performs a basic EDA on a dataset
    :param df: the dataframe
    :return:
    """

    # Get a summary of the dataset's structure
    print(df.info())
    # Summary statistics of numerical columns
    print(df.describe())
    # Number of missing values in each column
    print(df.isna().sum())

def verify_ids(df, id_column: str, column_print_name: str, regex: str, check_unique: bool = True, column_type: str = "string") -> None:
    """
    Verifies the ids/codes (in an id/code column) of a dataframe.
    :param df: the dataframe.
    :param id_column: the name of the column in the dataset containing the ids.
    :param column_print_name: the name of the column for printing purposes.
    :param regex: regex to match the ids against. Example: '^[0-9]{10}$' for 10-digit ids. None for no regex matching.
    :param check_unique: whether to check if the ids are unique. Default: True.
    :param column_type: the type of the column. Default: "string".
    :return:
    """

    na_ids = df[id_column].isna().sum()
    # Are there any missing ids?
    print(f"There are {na_ids} missing {column_print_name}.")
    # Check unique ids
    if check_unique:
        print(f"{column_print_name} contain unique ids: {df[id_column].shape[0] == df[id_column].nunique()}")
    # Check if all ids have a length specified in the given regex
    if regex:
        id_matches = None
        if column_type == "string":
            id_matches = df[id_column].str.fullmatch(regex)
        elif column_type == "numeric":
            id_matches = df[id_column].astype(str).str.fullmatch(regex)
        print(
            f"All {column_print_name} have the same format: {df[~id_matches].shape[0] == 0}")


# Verifying articles #

def verify_article_ids(articles) -> None:
    """
    Verifies the article id column in the articles' dataset.
    :param articles: the article ids in the articles dataset.
    """

    verify_ids(articles, "article_id", "article ids", '^[0-9]{10}$')


def verify_product_code(articles) -> None:
    """
    Verifies the product codes in the articles' dataset.
    :param articles: articles dataframe.
    """

    verify_ids(articles, "product_code", "product codes", '^[0-9]{7}$', False)

def verify_prod_name(articles) -> None:
    """
    Verifies the product name in the articles' dataset.
    :param articles: articles dataframe.
    """

    # check for amount of missing values
    print(f"There are {articles['prod_name'].isna().sum()} missing product names.")

    # check the (top) value counts of the product name
    print(f"Product name value counts:\n{articles['prod_name'].value_counts()}")


def verify_product_type_no(articles) -> None:
    """
    Verifies the product type number in the articles' dataset.
    :param articles: articles dataframe.
    """

    # check for amount of missing values
    print(f"There are {articles['product_type_no'].isna().sum()} missing product type numbers.")

    # check the (top) value counts of the product type number
    print(f"Product type number value counts:\n{articles['product_type_no'].value_counts()}")


def verify_product_type_name(articles) -> None:
    """
    Verifies the product type name in the articles' dataset.
    :param articles: articles dataframe.
    """

    # check for unique product type names
    print(f"Product type names are unique:{articles['product_type_name'].shape[0] == articles['product_type_name'].nunique()}")

    # check for amount of missing values
    print(f"There are {articles['product_type_name'].isna().sum()} missing product type names.")

    # check the (top) value counts of the product type name
    print(f"Product type name value counts:\n{articles['product_type_name'].value_counts()}")


def verify_graphical_appearance_name(articles) -> None:
    """
    Verifies the graphical appearance name in the articles' dataset.
    :param articles: articles dataframe.
    """

    # check for amount of missing values
    print(f"There are {articles['graphical_appearance_name'].isna().sum()} missing graphical appearance names.")

    # check the (top) value counts of the graphical appearance name
    print(f"Graphical appearance name value counts:\n{articles['graphical_appearance_name'].value_counts()}")


def verify_colour_group_name(articles) -> None:
    """
    Verifies the colour group name in the articles' dataset.
    :param articles: articles dataframe.
    """

    # check for amount of missing values
    print(f"There are {articles['colour_group_name'].isna().sum()} missing colour group names.")

    # check the (top) value counts of the colour group name
    print(f"Colour group name value counts:\n{articles['colour_group_name'].value_counts()}")


def check_correlation_between_columns_articles(articles) -> None:
    """
    Checks for correlation between the columns of the articles dataset.
    :param articles: articles dataframe.
    :return:
    """

    # Check if each product_code is uniquely mapped to a product_type_no
    presumably_unique_pairs = articles.groupby(['product_code', 'product_type_no']).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False)
    print(f"Each product_code is uniquely mapped to a product_type_no: {presumably_unique_pairs['product_code'].shape[0] == presumably_unique_pairs['product_code'].nunique() & presumably_unique_pairs['product_type_no'].shape[0] == presumably_unique_pairs['product_type_no'].nunique()}")

    # Check if each product_type_no is uniquely mapped to a product_type_name
    presumably_unique_pairs = articles.groupby(['product_type_no', 'product_type_name']).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False)
    print(f"Each product_type_no is uniquely mapped to a product_type_name: {presumably_unique_pairs['product_type_no'].shape[0] == presumably_unique_pairs['product_type_no'].nunique() & presumably_unique_pairs['product_type_name'].shape[0] == presumably_unique_pairs['product_type_name'].nunique()}")

    # find all non unique product_type_name values
    non_unique_product_type_name = presumably_unique_pairs[presumably_unique_pairs.duplicated(subset=['product_type_name'], keep=False)]
    print(f"Non unique product_type_name values:\n {non_unique_product_type_name}")


# Verifying customers #

def verify_customer_ids(customers) -> None:
    """
    Verifies the customer id column in the customers dataset.
    :param customers: customers dataframe
    """

    verify_ids(customers, "customer_id", "customer ids", '^[a-f0-9]{64}$')


def verify_ages(customers) -> None:
    """
    Verifies the age of each customer in the customers dataset.
    :param customers: customers dataframe
    """

    # Are there missing ages?
    na_ages = customers["age"].isna().sum()
    print(f"{na_ages} ({na_ages/customers.shape[0]*100}%) customers have no age.")

    # Check if all ages are between 12 and 120
    valid_ages = (customers["age"] >= 12) & (customers["age"] <= 120)
    print(f"All ages are within the valid range: {len(valid_ages) == len(customers)}")
    print(f"There are {len(customers[~valid_ages])} customers with invalid ages.")
    try: # if we have the right python and pandas versions, we can use the apply function for clearer output.
        print(f"{customers['age'].describe().apply('{:.2f}'.format)}") # actual type is float64.
    except (Exception, ):
        print(f"{customers['age'].describe()}")


def verify_postal_codes(customers) -> None:
    """
    Verifies the postal code of each customer in the customers dataset.
    :param customers: customers dataframe
    """

    verify_ids(customers, "postal_code", "postal codes", '^[a-f0-9]{64}$', False)

    # check the (top) value counts of the postal codes
    print(f"Postal code value counts:\n{customers['postal_code'].value_counts()}")


def verify_fn_status(customers) -> None:
    """
    Verifies the FN status of each customer in the customers dataset.
    :param customers: customers dataframe
    :return:
    """

    # Check and count all the different FN status values
    print(f"FN value counts:\n{customers['FN'].value_counts(dropna=False)}")

    # Percentage of missing FN status values
    print(f"Percentage of missing FN values: {customers['FN'].isna().sum()/customers.shape[0]*100}%")


def verify_active_status(customers) -> None:
    """
    Verifies the active status of each customer in the customers' dataset.
    :param customers: customers dataframe
    :return:
    """

    # Check and count all the different active status values
    print(f"Active status:\n{customers['Active'].value_counts(dropna=False)}")

    # Percentage of missing active status values
    print(f"Percentage of missing Active values: {customers['Active'].isna().sum()/customers.shape[0]*100}%")


def verify_club_member_status(customers) -> None:
    """
    Verifies the club member status of each customer in the customers' dataset.
    :param customers: customers dataframe
    :return:
    """

    # Check and count all the different club member status values
    print(f"Club member status:\n{customers['club_member_status'].value_counts(dropna=False)}")

    # Percentage of missing club member status values
    print(f"Percentage of missing club member status values: {customers['club_member_status'].isna().sum()/customers.shape[0]*100}%")


def verify_fashion_news_frequency(customers) -> None:
    """
    Verifies the fashion news frequency of each customer in the customers' dataset.
    :param customers: customers dataframe
    :return:
    """

    # Check and count all the different fashion news frequencies values
    print(f"Fashion news frequencies:\n{customers['fashion_news_frequency'].value_counts()}")

    # Percentage of missing fashion news frequency values
    print(f"Percentage of missing fashion news frequency values: {customers['fashion_news_frequency'].isna().sum()/customers.shape[0]*100}%")


def check_correlation_between_columns_customers(customers) -> None:
    """
    Checks for correlation between the columns of the customers dataset.
    :param customers: customers dataframe
    :return:
    """

    # Check for correlation between fashion news frequency values and FN status but including rows with NaN values
    print(f"Fashion news frequency and FN status correlation (including NaN values and sorted by count):\n {customers.groupby(['FN', 'fashion_news_frequency'], dropna=False).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False)}")

    # Check for correlation between FN and Active status values but including rows with NaN values
    print(f"Fashion news frequency and FN status correlation (including NaN values and sorted by count):\n {customers.groupby(['FN', 'Active'], dropna=False).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False)}")

    # same but with columns FN, Active, club_member_status, and fashion_news_frequency
    print(f"Fashion news frequency and FN status correlation (including NaN values and sorted by count):\n {customers.groupby(['FN', 'fashion_news_frequency', 'Active', 'club_member_status'], dropna=False).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False)}")


def create_grouped_barplot(df, original_ng_column, x, hue, dropna, graph_title, xlabel, ylabel, legend_title, percentages_to_mark: list = None) -> None:
    """
    Creates a grouped barplot with age groups on the x-axis, per age group the percentage of customers that have the different hue statuses represented by the different colors. The y-axis is the percentage of customers per (age) group.
    :param df: dataframe with (age) groups included.
    :param original_ng_column: name of the original column with non-grouped values.
    :param x: factor to group by on the x-axis.
    :param hue: different 'statuses' (= values) to represent by the different colors.
    :param dropna: whether to drop hue values that are NaN. If False, also filters rows where original_ng_column is NaN.
    :param graph_title: graph title.
    :param xlabel: x-axis label.
    :param ylabel: y-axis label.
    :param legend_title: legend title.
    :param percentages_to_mark: list of percentages to mark with a horizontal line. Usually received hardcoded.
    :return:
    """

    df_valid_ages = df
    # Filter rows where 'age' is not NaN (if dropna is False, since we don't want an age group with value NaN)
    if not dropna:
        df_valid_ages = df[~df[original_ng_column].isna()]

    # Group the data by 'age_group' and 'fashion_news_frequency', and calculate the percentage within each age group
    grouped = df_valid_ages.groupby([x, hue], dropna=dropna).size().unstack()
    grouped_percentage = grouped.div(grouped.sum(axis=1),
                                     axis=0) * 100  # this final result is a dataframe where the rows are the age groups and the columns are the fashion_news_frequency statuses, and the values are the percentages of customers in that age group that have that fashion_news_frequency status.

    # Plot the data
    p = grouped_percentage.plot.bar(figsize=(10, 6))

    # Customize the plot
    plt.title(graph_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add horizontal lines at percentage levels
    if percentages_to_mark:
        for percentage in percentages_to_mark:
            p.axhline(percentage, color='gray', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()

def graphs_by_age_group(customers) -> None:
    """
    Creates several graphs with data per age group.
    Source for the inspiration for the bar chart:
    general link: https://www.kaggle.com/code/ludovicocuoghi/h-m-sales-and-customers-deep-analysis/notebook
    link to chart: https://www.kaggle.com/code/ludovicocuoghi/h-m-sales-and-customers-deep-analysis?scriptVersionId=88797706&cellId=10
    :param customers: customers dataframe
    :return:
    """

    # create age groups
    customers["age_group"] = pd.cut(customers["age"], bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120], labels=["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100", ">100"])

    # create a barplot with age groups on the x-axis, per age group the percentage of customers that have the different FN statuses represented by the different colors. The y-axis is the percentage of customers per age group.
    create_grouped_barplot(customers, 'age', 'age_group', 'FN', False, 'Percentage Distribution of FN Status by Age Group', 'Age Group', 'Percentage (%)', 'FN Status', [10, 20, 30, 40, 50, 60, 70])
    # create a barplot with age groups on the x-axis, per age group the percentage of customers that have the different Active statuses represented by the different colors. The y-axis is the percentage of customers per age group.
    create_grouped_barplot(customers, 'age', 'age_group', 'Active', False, 'Percentage Distribution of Active Status by Age Group', 'Age Group', 'Percentage (%)', 'Active Status', [10, 20, 30, 40, 50, 60, 70, 80])
    # create a barplot with age groups on the x-axis, per age group the percentage of customers that have the different club member statuses represented by the different colors. The y-axis is the percentage of customers per age group.
    create_grouped_barplot(customers, 'age', 'age_group', 'club_member_status', False, 'Percentage Distribution of Club Member Status by Age Group', 'Age Group', 'Percentage (%)', 'Club Member Status', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # create a barplot with age groups on the x-axis, per age group the percentage of customers that have the different fashion news frequencies represented by the different colors. The y-axis is the percentage of customers per age group.
    create_grouped_barplot(customers, 'age', 'age_group', 'fashion_news_frequency', False, 'Percentage Distribution of Fashion News Frequency by Age Group', 'Age Group', 'Percentage (%)', 'Fashion News Frequency', [10, 20, 30, 40, 50, 60, 70])


# Verifying transactions #

def perform_transactions_eda(transactions) -> None:
    """
    Verifies the transactions dataset.
    :param transactions: transactions dataframe
    :return:
    """

    # Check for missing values
    print(f"Missing values per column:\n{transactions.isna().sum()}")

    verify_purchase_dates(transactions)
    verify_transaction_customer_id(transactions)
    verify_transaction_article_id(transactions)
    verify_prices(transactions)
    verify_sales_channel_ids(transactions)


def verify_purchase_dates(transactions) -> None:
    """
    Verifies the purchase date of each transaction.
    :param transactions: transactions dataframe
    """

    # Get the transaction dates
    t_dates = transactions['t_dat']

    # Are there missing transaction dates?
    print(f"There are {t_dates.isna().sum()} missing transactions dates")

    # Check if all dates have the same ISO format
    date_matches = t_dates.str.fullmatch('^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
    print(f"All dates have the same ISO format: {t_dates[~date_matches].shape[0] == 0}")

    # Check if the transaction date is after 1947-01-01 (founding year) and before now
    print(f"Earliest transaction date: {t_dates.min()}")
    print(f"Latest transaction date: {t_dates.max()}")
    valid_dates = (t_dates >= "1947-01-01") & (t_dates <= datetime.now().strftime("%Y-%m-%d"))
    print(f"All transactions dates are within the valid range: {len(valid_dates) == len(t_dates)}")
    print(f"There are {len(t_dates[~valid_dates])} transactions with invalid dates.")


def verify_transaction_customer_id(transactions) -> None:
    """
    Verifies the customer id that performed a transaction.
    :param transactions: transactions dataframe
    """

    verify_ids(transactions, "customer_id", "customer ids", '^[a-f0-9]{64}$', False)


def verify_transaction_article_id(transactions) -> None:
    """
    Verifies the article id that is linked to a performed transaction.
    :param transactions: transactions dataframe
    """

    verify_ids(transactions, "article_id", "article ids", '^[0-9]{10}$', False)


def verify_prices(transactions) -> None:
    """
    Verifies all the price of each transaction.
    :param transactions: transactions dataframe
    """

    verify_ids(transactions, "price", "prices", r'[0-9]+(\.[0-9]+)?', False, "numeric")

    # Check prices summary statistics (also only positive prices allowed)
    print(f"{transactions['price'].describe().apply('{:.6f}'.format)}")


def verify_sales_channel_ids(transactions) -> None:
    """
    Verifies the sales channel id of each transaction.
    :param transactions: transactions dataframe
    """

    # Check for missing values
    print(f"There are {transactions['sales_channel_id'].isna().sum()} missing sales channel ids")

    # Check all value counts
    print(f"Sales channel id value counts:\n {transactions['sales_channel_id'].value_counts()}")

