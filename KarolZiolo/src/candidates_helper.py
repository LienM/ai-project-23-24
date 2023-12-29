import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_rare_customers(transactions, threshold=5):
    '''
    Returns a list of customers who had small number of transactions.
    Args:
        transactions: transactions dataframe
        threshold: number of transactions required to be considered a rare customer
    Returns:
        list of customer_ids
    '''
    grouped = transactions.groupby("customer_id")["article_id"].count()
    rare_customers = grouped[grouped<threshold]
    return list(rare_customers.index)

def assign_season(x):
    '''
    Assign seasons to transactions based on the month the transaction was done.
    Args:
        x: date of the transaction
    Returns:
        season id (1-winter, 2-spring, 3-summer, 4-fall)
    '''
    if x in [12,1,2]:
        return 1
    elif x in [3,4,5]:
        return 2
    elif x in [6,7,8]:
        return 3
    else:
        return 4 

def bestsellers_age_season(customers, transactions, rare_customers, set_threshold=0.6):
    '''
    Returns a list of customer_ids who are buying from bestsellers for given age and season. The propensity index is calculated,
    which uses article rankings in a given season to determine if a customer was following it. So it takes also into account the season.
    Args:
        customers: customers dataframe
        transactions: transactions dataframe
        rare_customers: list of customer_ids who had small number of transactions
        set_threshold: cut of index determining the customers are following bestsellers trends 
    Returns:
        list of propensity indices
        list of customer_ids 
        dictionary: 
            keys-age groups
            values-list of customer_ids who are buying from bestsellers for given age and season
    '''
    cust_age_id = {}
    #get age groups
    bins = [0,25,40,55,float("inf")]
    labels = ["young_preference","adult_preferences","middle_aged_preference","senior_preference"]
    customers["age_group"] = pd.cut(customers["age"], bins=bins, labels=labels, right=False)
    # get season 
    transactions = transactions.merge(customers[["customer_id", "age_group"]], how="left", on="customer_id")
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["month"] = transactions["t_dat"].dt.month 
    transactions["season"] = transactions["month"].apply(assign_season)
    # generate counts for each article_id given age_group and season
    grouped = transactions.groupby(["age_group","season","article_id"], observed=False)["customer_id"].count().reset_index()
    grouped.rename(columns={"customer_id":"transaction_age_count"}, inplace=True)
    # generate ranks given age_group and season
    grouped["age_season_rank"] = grouped.groupby(["age_group","season"],observed=False)["transaction_age_count"].rank(method="dense", ascending=True)
    # scale the ranking
    grouped['max_rank'] = grouped.groupby(['age_group', 'season'], observed=False)['age_season_rank'].transform('max')
    grouped['scaled_age_season_rank'] = grouped['age_season_rank'] / grouped['max_rank']
    # generate customers scores to determine who is buying from bestsellers for given age_group and season
    transactions = transactions.merge(grouped[["age_group","season","article_id","scaled_age_season_rank"]], how="left", on=["age_group","season","article_id"])
    bestseller_propensity = transactions.groupby(["customer_id","age_group"], observed=True)["scaled_age_season_rank"].mean()
    # drop rare customers
    bestseller_propensity = bestseller_propensity.drop(rare_customers, level='customer_id')
    # plot bestseller_propensity
    plt.hist(bestseller_propensity.values, bins=30, color='purple', edgecolor='black')
    plt.xlabel('Customer Bestseller Propensity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bestseller Propensity')
    plt.grid(True)
    plt.axvline(x=set_threshold, color='blue', linestyle='--', label=f'Threshold: {set_threshold}')
    plt.legend()
    plt.show()
    # get customers with high propensity score
    customers_id = np.array(list(bestseller_propensity[bestseller_propensity>0.6].index.get_level_values('customer_id')))
    for label in labels:
        cust_age_id[label] = bestseller_propensity[(bestseller_propensity>0.6)&(bestseller_propensity.index.get_level_values('age_group') == label)].index.get_level_values('customer_id')
    return bestseller_propensity, customers_id, cust_age_id
    
def index_preferences(transactions, articles, customers, rare_customers, set_threshold=0.8):
    '''
    Returns a list of customer_ids who are buying mostly products from the specific clothes segments. 
    The proportion of clothes from each segment is calculated and if any of that cross the threshold then the customer
    is considered to be prone to buy products from that segment.

    Args:
        transactions: transactions dataframe
        articles: articles dataframe
        customers: customers dataframe
        rare_customers: list of customer_ids who had small number of transactions
        set_threshold: cut of index determining the customers are buying from the specific clothes segments
    Returns:
        mens: list of customer_ids who are buying from the mens segment
        ladies: list of customer_ids who are buying from the ladies segment
        kids: list of customer_ids who are buying from the kids segment
        div: list of customer_ids who are buying from the divided segment
        sprt: list of customer_ids who are buying from the sport segment
    '''
    transactions = transactions.merge(articles[["article_id", "index_name"]], how="left", on="article_id")
    grouped = transactions.groupby(["customer_id", "index_name"])["article_id"].count()
    percentages = grouped/grouped.groupby(level=0).transform("sum")
    percentages = percentages.drop(index=rare_customers, level='customer_id')
    # get menswear 
    manswear = percentages[percentages.index.get_level_values('index_name') == 3]
    manswear = manswear.rename("manswear")
    customers = customers.merge(manswear, how="left", on="customer_id")
    customers["manswear"] = customers["manswear"].fillna(0)
    # get ledieswear
    ladieswear = percentages[percentages.index.get_level_values('index_name').isin([0,1,4])].groupby("customer_id").sum()
    ladieswear = ladieswear.rename("ladieswear")
    customers = customers.merge(ladieswear, how="left", on="customer_id")
    customers["ladieswear"] = customers["ladieswear"].fillna(0)
    # get kids 
    kids = percentages[percentages.index.get_level_values('index_name').isin([2,6,8,9])].groupby("customer_id").sum()
    kids = kids.rename("kids")
    customers = customers.merge(kids, how="left", on="customer_id")
    customers["kids"] = customers["kids"].fillna(0)
    # get divided 
    divided = percentages[percentages.index.get_level_values('index_name') == 7]
    divided = divided.rename("divided")
    customers = customers.merge(divided, how="left", on="customer_id")
    customers["divided"] = customers["divided"].fillna(0)
    # get sport 
    sport = percentages[percentages.index.get_level_values('index_name') == 5]
    sport = sport.rename("sport")
    customers = customers.merge(sport, how="left", on="customer_id")
    customers["sport"] = customers["sport"].fillna(0)

    # plot plots distribution of index preferences
    indices = ["manswear","ladieswear","kids","divided","sport"]
    for index in indices:
        plt.hist(customers[index], bins=30, color='purple', edgecolor='black')
        plt.xlabel(f'Percentage of bought {index} products')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {index} customers')
        plt.grid(True)
        plt.axvline(x=set_threshold, color='blue', linestyle='--', label=f'Threshold: {set_threshold}')
        plt.legend()
        plt.show()
    
    mens = customers["customer_id"][customers["manswear"]>set_threshold].values
    ladies = customers["customer_id"][customers["ladieswear"]>set_threshold].values
    kid = customers["customer_id"][customers["kids"]>set_threshold].values
    div = customers["customer_id"][customers["divided"]>set_threshold].values
    sprt = customers["customer_id"][customers["sport"]>set_threshold].values
    return mens, ladies, kid, div, sprt  

def get_discount_hunters(transactions, rare_customers, set_threshold=0.8):
    '''
    Generates a list of customers who are mostly interested in the discounted products.
    Args:
        transactions: transactions dataframe
        rare_customers: list of customer_ids who had small number of transactions
        set_threshold: cut of index determining the customers are buying mostly discounted products
    Returns:
        customers: list of customer_ids who are buying mostly discounted products
    '''
    # determine if the product was sold with discounted price
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["month"] = transactions["t_dat"].dt.month 
    transactions["season"] = transactions["month"].apply(assign_season)
    grouped = transactions.groupby(["article_id","season"])["price"].median()
    grouped = grouped.rename("normal_price")
    transactions = transactions.merge(grouped, on=["article_id","season"], how="left")
    transactions["price_discount"] = transactions["price"] - transactions["normal_price"]
    transactions["price_discount"] = transactions["price_discount"]<0
    # get percentages of discounted products bought by customer
    grouped = transactions.groupby(["customer_id","price_discount"])["article_id"].count()
    percentages = grouped/grouped.groupby(level=0).transform("sum")
    percentages = percentages.drop(index=rare_customers, level='customer_id')
    discounted_percentages = percentages[percentages.index.get_level_values('price_discount') == 1]
    # print distribution
    plt.hist(discounted_percentages.values, bins=30, color='purple', edgecolor='black')
    plt.xlabel('Discount Percentage')
    plt.ylabel('Frequency')
    plt.title('Distribution of Discount Customers')
    plt.axvline(x=set_threshold, color='blue', linestyle='--', label=f'Threshold: {set_threshold}')
    plt.grid(True)
    plt.legend()
    plt.show()
    # get customers indices
    discount_hunters = np.array(list(discounted_percentages[discounted_percentages>set_threshold].index.get_level_values('customer_id')))
    return discount_hunters

def seasonal_customers(transactions,rare_customers,set_threshold=0.8):
    '''
    Generates a list of customers who are mostly interested in the seasonal products.
    To determine that the proportion of the sales made by customer in a given season is calculated. 
    If any season cross the threshold the customer is considered as a this specific season customer.
    Args:
        transactions: transactions dataframe
        rare_customers: list of customer_ids who had small number of transactions
        set_threshold: cut of index determining the customers are buying mostly seasonal products
    Returns:
        customers: list of customer_ids who are buying mostly seasonal products
    '''
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["month"] = transactions["t_dat"].dt.month 
    transactions["season"] = transactions["month"].apply(assign_season)
    grouped = transactions.groupby(["customer_id","season"])["article_id"].count()
    percentages = grouped/grouped.groupby(level=0).transform("sum")
    percentages = percentages.drop(index=rare_customers, level='customer_id')
    seasons = ["winter","spring","summer","autumn"]
    cust_ids = []
    for i, season in enumerate(seasons):
        season_perc = percentages[percentages.index.get_level_values('season') == i+1]
        season_perc = season_perc.rename(f"{season}_perc")
        # plot distribution of the season perc
        plt.hist(season_perc.values, bins=30, color='purple', edgecolor='black')
        plt.xlabel(f"Percentages of bought {season} clothes")
        plt.ylabel('Frequency')
        plt.title(f'{season} customers distribution ')
        plt.axvline(x=set_threshold, color='blue', linestyle='--', label=f'Threshold: {set_threshold}')
        plt.grid(True)
        plt.legend()
        plt.show()
        # append customer_ids 
        cust_ids.append(season_perc[season_perc>set_threshold].index.get_level_values("customer_id"))
    return cust_ids

def age_article_candidates(customers, transactions, date_thershold='2020-08-22', article_threshold=500):
    '''
    Generates list of article_ids which are strongly popular among specific age group.
    Args:
        customers: customers dataframe
        transactions: transactions dataframe
        date_thershold: cut of date determining starting date of transactions that are considered
        article_threshold: threshold determining top-k articles
    Returns:
        article_age_indices: dictionary of article_ids for each age group
    '''
    #get age groups
    bins = [0,25,40,55,float("inf")]
    labels = ["young_preference","adult_preferences","middle_aged_preference","senior_preference"]
    customers["age_group"] = pd.cut(customers["age"], bins=bins, labels=labels, right=False)
    # get resent 
    transactions = transactions.merge(customers[["customer_id", "age_group"]], how="left", on="customer_id")
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["month"] = transactions["t_dat"].dt.month 
    transactions["year"] = transactions["t_dat"].dt.year 
    transactions["season"] = transactions["month"].apply(assign_season)
    transactions = transactions[transactions.t_dat>date_thershold]
    grouped = transactions.groupby(["age_group","article_id"], observed=False)["customer_id"].count().reset_index()
    grouped.rename(columns={"customer_id":"transaction_age_count"}, inplace=True)
    # generate ranks given age_group and season
    grouped["age_season_rank"] = grouped.groupby(["age_group"],observed=False)["transaction_age_count"].rank(method="dense", ascending=True)
    article_age_indices = {}
    for label in labels:
        label_grouped = grouped[grouped.age_group==label].sort_values("age_season_rank", ascending=False)
        article_age_indices[label] = label_grouped["article_id"][0:article_threshold].values
    return article_age_indices
        
def get_discounted_articles(transactions, date_threshold='2020-08-22'):
    '''
    Generates a list of article_ids which are considered as discounted products.
    Args:
        transactions: transactions dataframe
        date_threshold: cut of date determining starting date of transactions that are considered
    Returns:
        discounted_articles: list of article_ids considered as discounted products
    '''
    transactions = transactions[transactions["t_dat"]>date_threshold]
    grouped = transactions.groupby("article_id")["price"].median()
    grouped = grouped.rename("normal_price")
    transactions = transactions.merge(grouped, on="article_id", how="left")
    transactions["price_discount"] = transactions["price"] - transactions["normal_price"]
    discounted_articles = transactions[transactions["price_discount"]<0]["article_id"].values
    return list(discounted_articles)
    
def get_season_articles(transactions, season="autumn", set_threshold=0.8):
    '''
    Generates a list of article_ids which are considered as seasonal products.
    Args:
        transactions: transactions dataframe
        season: season to be considered
        set_threshold: cut of index determining the the product is considered as seasonal one
    Returns:
        season_articles: list of article_ids considered as seasonal products
    '''
    season_id = {"winter":1,"spring":2,"summer":3,"autumn":4}
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["month"] = transactions["t_dat"].dt.month 
    transactions["year"] = transactions["t_dat"].dt.year 
    transactions["season"] = transactions["month"].apply(assign_season)
    grouped = transactions.groupby(["article_id","season"])["customer_id"].count()
    percentages = grouped/grouped.groupby(level=0).transform("sum")
    season_perc = percentages[percentages.index.get_level_values('season') == season_id[season]]
    season_perc = season_perc.rename(f"{season}_perc")
    # plot distribution of the season perc
    plt.hist(season_perc.values, bins=30, color='purple', edgecolor='black')
    plt.xlabel(season)
    plt.ylabel('Frequency')
    plt.title(f'Article distribution of {season}')
    plt.axvline(x=set_threshold, color='blue', linestyle='--', label=f'Threshold: {set_threshold}')
    plt.grid(True)
    plt.legend()
    plt.show()
    # append customer_ids 
    return season_perc[season_perc>set_threshold].index.get_level_values("article_id")

