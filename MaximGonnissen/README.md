# AI Project 23-24

## Project description

For this class, we were tasked to tackle the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) Kaggle competition.
The goal of this competition was to predict the 12 most likely articles a customer would purchase, to be evaluated against the products the customer actually bought in the next week.

To do this, we were provided with a plethora of [data](#Data), including customer data, article data, article images and transaction data.

## Research questions

For this project, we had to narrow down on one or more research questions to answer. The decision was made to focus on the following:

- What is the impact of adding a season score feature to clothing using a date offset and range to calculate season score?
- Can we calculate the gender of articles \& customers and use this to make predictions?
- What is the impact of seasonality and gender features on the performance of the Radek baseline LGBM ranker?

## Data

We received 3 datasets in csv format, along with an example submission csv. Additionally, we received a directory with images for articles.

- Customer data (1371980 rows)
  - Id
  - Age
  - Account info (Active, Newsletter, etc...)
  - Postal code
- Article data (105542 rows)
  - Id
  - Product code
  - Name & description
  - Colour information
  - Category information (Major category, sub category, appearance, etc...)
- Article images (105100 images)
- Transaction data (31788324 rows)
  - Date
  - Customer id
  - Article id
  - Price
  - Sales channel id