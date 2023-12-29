# Pruning

These scripts were used for pruning the data set.

## [prune_inactive.py](prune_inactive.py)

Prunes inactive customers.
- Removes customers with ACTIVE status equal to "PRE-CREATE", which seems to be a status used for customers that have not completed their registration.

## [prune_no_purchases.py](prune_no_purchases.py)

Prunes customers with no purchases.

## [prune_outdated_items.py](prune_outdated_items.py)

Prunes articles that haven't been sold in a specified amount of time. (default: 1 year)

Also prunes transactions that contain these articles.
