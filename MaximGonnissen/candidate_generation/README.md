# Candidate generation

These are scripts used to generate top-12 candidates for customers.

## [gender_recommendations.py](gender_recommendations.py)

This script generates top-12 candidates for customers based on the gender score of articles, and the predicted gender of customers.
It creates the same candidates for all customers of the same gender.

## [seasonality_recommendations.py](seasonality_recommendations.py)

This script generates top-12 candidates for customers based on the seasonality score of articles. It creates the same
candidates for all customers.