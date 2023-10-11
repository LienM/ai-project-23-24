This is the assignments solutions mini-repo for the AI Project 23-24 course by Felix Vernieuwe.

## General
Dataset is located in root folder of the repository, under `./data`
Submissions will be placed under the `./data/submissions` folder, with the following naming convention.

Packages used:
- **numpy:** (duh)
- **pandas:** (duh²)
- ~~**dask:** parallel computing and dataset loading (factor 10 speedup for loading csv from disk)~~ (solved with Parquet)
- **seaborn:** pretty & pretty effortless graphs
- **tqdm:** proverbially watching the paint dry

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