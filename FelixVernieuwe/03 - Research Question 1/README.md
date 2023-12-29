You have arrived at the README.md of the last assignment, concerning the research for the first and only research question (RQ1). 
This README.md will give an overview of the project structure, as well as a brief description of the research question,
and some of the thought processes.

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
    └───03 ─ Research Question 1
        │   README.md <─ You are here!
        │   03 ─ Research Question 1.ipynb
        │       Original notebook fo​r RQ1 (superceded by `/suite/`)
        │   scorers.py
        │       Evaluation metrics
        ├───"candidates"    (candidate generation methods)
        │   │   bestseller.py
        │   │   "generate.py" (Helper functi​on fo​r generating candidates)
        │   │   new_arrivals.py
        │   │   repurchase.py
        │   │   u2u_collaborative_filtering.py
        │   └   __init__.py        
        ├───data            (extract dataset information)
        │   │   candidate_customers.py
        │   │   candidate_products.py
        │   │   purchase_rank.py
        │   └   __init__.py
        ├───features       (feature generation methods)
        │   │   age_group.py
        │   │   bestseller.py
        │   │   discount.py
        │   │   new_arrivals.py
        │   │   price_sensitivity.py
        │   └   __init__.py
        ├───playgrounds     (notebooks fo​r testing out data/generation methods/...)
        │   │   age_group_analysis.ipynb
        │   │   kaggle_get_score.ipynb
        │   │   missing_customers.ipynb
        │   │   output_submission_analysis.ipynb
        │   │   purchase_probability_analysis.ipynb
        │   └   trends_analysis.ipynb
        └───"suite"
            │   "config.py"     (contains all the configuration options)
            │   constants.py    (contains all the constants)
            │   evaluate.py     (online/offline evaluation)
            │   "main.py"       (easiest file to run the experiments with)
            │   predict.py      (train rankers and predict candidates)
            │   preparation.py  (generates candidates and adds features)
            │   run.py          (contains the run scripts)
            │   util.py         (generic utility functions)
            └   __init__.py
```

---

## Assignment Description

This assignment mainly covers the research question 1 (RQ1), which is:
<center><i>
Can we quantify the importance of global, contextual and personalised candidates in evaluation and output score?
</i></center>

The most time on this part was spent for getting a framework working for running through different configs.
Refer to the paper for more details on the research question, methodology and results.


To start, I recommend running `suite/main.py` and inspecting the `suite/config.py` for seeing all the possible
run configurations.
