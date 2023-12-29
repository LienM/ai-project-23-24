You have arrived at the README.md of the second assignment, concerning Feature Engineering.
This README.md will give a brief overview of the contents of this assignment, detailing some of the
design choices that were made.

---

## Solution Structure


```bash
Project Root
├───/OtherSubmissioms/
├───data
│   │   articles.csv
│   │   customers.csv
│   │   transactions_train.csv
│   └───images
└───FelixVernieuwe
    │   "requirements.txt"
    │       Required packages to run the scripts
    │   util.py
    │       General utility functions fo​r the notebooks (loading/submitting/...)
    ├───01 ─ EDA
    ├───02 ─ Feature Engineering
    │   │   README.md <─ You are here!    
    │   │   02 ─ Feature Engineering.ipynb
    │   │       !! Notebook fo​r working out Feature Engineering !!
    │   │   Radek LGBM.ipynb
    │   │       Radek Baseline
    │   └───features
    │           Feature generation methods
    └───03 ─ Research Question 1
```

---

## Assignment Description

The goal of this assignment was to experiment with a known baseline (Radek), and try to understand
the general candidate generation process, as well as seeing what features/candidates would perform well.

I started out by trying to understand and rewrite the baseline, which was not possible to fully complete in the limited
time allotted to this assignment.

So I instead divided up the baseline into its constituent parts:
1. Data preparation
2. Candidate generation
3. Feature generation
4. LGBM Model training
5. Candidate prediction
6. Submission generation  

Each of these parts is clearly noted as a heading between two horizontal lines.

Further, I added five new features:
- `discount` (Boolean --- whether the product was discounted)
- `all_time_rank` (Integer --- the rank of the product in terms of all-time sales)
- `price_sensitivity` (Integer --- how sensitive the customer is to product price)

Finally, I also tested out how customers without predicted candidates are handled.