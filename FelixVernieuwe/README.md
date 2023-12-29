This is the assignments solutions sub-repo for the AI Project 23-24 course by Felix Vernieuwe.


---
## General

**Packages used** (see also `requirements.txt`)**:**
- **pandas:** (duh)
  - _pyarrow_: loading `.parquet` files 
  - _numpy_: (duh²)
- ~~**dask:** parallel computing and dataset loading (factor 10 speedup for loading csv from disk)~~ (solved with Parquet)
- **matplotlib:** (pretty) graphs
- **seaborn:** pretty & pretty effortless graphs
- **tqdm:** proverbially watching the paint dry
- **lightgbm:** gradient boosting framework
  - _scikit-learn_: (duh³) 
- **kaggle:** because scraping from kaggle proved too difficult
- ~~**tauricreate**: a rare Apple FOSS package, that did not work very well~~
- **jupyter/notebook:** (duh⁴)

---

## Project Structure
The project structure is given as below. Note that the `data` folder is not included with the Git repository,
but it _must_ be present in order to run the scripts. The given `parquet` files are found from the [Radek starterpack](https://www.kaggle.com/code/marcogorelli/radek-s-lgbmranker-starter-pack-warmup/output),
and were unchanged. `sample_submission.csv` was provided in the [Kaggle competition](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview).

Important folders/files are marked below with quotes, like `"important"`, and will be explained in more detail
in the respective assignment folders (i.e. `0X - .../README.md`).


```bash
Project Root
├───/OtherSubmissioms/
├───data
│   │   articles.parquet
│   │   customers.parquet
│   │   sample_submission.csv
│   │   transactions_train.parquet
│   │     !! NOTE: original kaggle .csv files required fo​r running 01 ─ EDA !!
│   ├───images
│   └───submissions
│           "scores.csv"
│               Keeps track of experiment scores
│           xxxx_num_submission_METADATA.csv.gz
└───FelixVernieuwe
    │   README.md <─ You are here!
    │   "requirements.txt"
    │       Required packages to run the scripts
    │   util.py
    │       General utility functions fo​r the notebooks (loading/submitting/...)
    ├───01 ─ EDA
    │       01 ─ EDA.ipynb
    │           Notebook fo​r exploratory data analysis
    ├───02 ─ Feature Engineering
    │   │   02 ─ Feature Engineering.ipynb
    │   │       Notebook fo​r working out Feature Engineering
    │   │   Radek LGBM.ipynb
    │   │       Radek Baseline
    │   └───features
    │           Feature generation methods
    └───03 ─ Research Question 1
        │   03 ─ Research Question 1.ipynb
        │       Original notebook fo​r RQ1 (superceded by /suite/)
        │   scorers.py
        │       Evaluation metrics
        ├───"candidates"
        │       Candidate generation methods
        ├───data
        │       Extract dataset information
        ├───features
        │       Feature generation methods
        ├───playgrounds
        │       Notebooks fo​r testing out different aspects of the data/generation methods/...
        └───"suite"
                Scripts fo​r running an experiment start-to-finish given a configuration
```


----
## Assignment Descriptions

## Assignment 1
Went over each dataset and made some basic observations with regards to missing and odd values, as well as some basic statistics and 
data distributions. Also made a few pretty graphs.

Recommended links:
- [How the prices were normalised](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/310496)
- [Explanations for columns](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/307001)


## Assignment 2
Refactored the Radek code to allow for easier insertion of other engineered features.  
Added a few features of my own, mainly based on reasonable and intuitive assumptions.
Also made a few pretty graphs.


## Assignment 3 (Research Question)

Investigated the following research question:

<center><i>
Can we quantify the importance of global, contextual and personalised candidates in evaluation and output score?
</i></center>


As extensively detailed by the research paper, this research question could not easily be answered, without having the
proper infrastructure in place to comparatively evaluate the different combinations of candidates.

The bulk of the time was spent on implementing the framework, which can be found under `/suite/`,
and finding interesting candidate generation strategies, generally found under `/candidates/`.

Also made a very small amount of pretty graphs.

---