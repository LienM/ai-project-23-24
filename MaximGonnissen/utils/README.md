# Utils

These scripts provide various utils, helper methods, and helper classes to be used in other scripts throughout the
project.

## [convert_to_parquet.py](convert_to_parquet.py)

- Converts a CSV file to a parquet file.
- Has a method to convert all H&M CSV files to parquet files.

## [kaggle_tool.py](kaggle_tool.py)

A class to help with Kaggle submissions for a given competition.

- Fetch list of submissions
- Submit a new submission

## [progress_bar.py](progress_bar.py)

A class to create a progress bar in the terminal.

Was mostly used as a fun way to help me keep track of the progress of the scripts during experiments.

## [season.py](season.py)

A class to represent a season, as well as calendar days.

Calendar Days are used to represent the days of the year, and are used to calculate the number of days between two
dates.
They clamp the number of days, and provide various methods and overloads to make working with them simple.

This module also contains a class that contains the 4 seasons, for easy access.

## [utils.py](utils.py)

A collection of general helper methods.

- ProjectConfig: A class to store the start & end date of the project.
- DataFileNames: A class to store the names & paths of various data files & directories used throughout the project.
- Methods to work with Pathlib paths.
- Methods to easily load the H&M dataframes as either CSV or parquet files.

