No data will be uploaded. Needs to be downloaded from kaggle (more info on main README file)

there are 3 folders for each of the datasets

    -customers
    -articles
    -transactions

inside each folder there are different variations for datasets, being:

    -articles.csv: original
    -articles.parquet: processed data
    -articles_train_sample_0.05.parquet: sample data containing 5% of processed data

additionally transactions data counts with the following:

    -transactions_train_short.parquet: shortened version of the processed data, only takes into account last 2 months of data
    -transactions_train_short-1week.parquet: does not include last week of data (15-09-2022 -> 22-09-2022)

Even though in my case i will not be using the images, i've also downloaded it into this folder

