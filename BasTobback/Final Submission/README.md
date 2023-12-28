# Final Submission
The file 'FinalSubmission' is, as the name suggests, the final submission. This file is the point I stopped during my research, it covers several steps but does not cover them all. Some parts are different version, different inputs, different orders etc.
Since this is the version which gave me my best results I will go in a bit more detail in this description, but for any concrete info the documentation in the code and the report should be more helpful.

First important note is that this file was edited on Kaggle. This means that this file will probably not work by running it in an IDE. That being said, it should run just fine in the environment of the RADEK baseline.
This also means that when using Kaggle, there is no need for requirement installation.

In this file there are some important parts that are built in other part of the project, hence they can be found in the other directories and files.

In this directory there is only one file, which is my solution to the assignment to try and improve the score on Kaggle for the H&M recommendations.
To start I used the Radek solution as a baseline and took several steps to improve it.

In this notebook you can find a small set of cells, they have these purposes (in order, *italic* if relevant):
* define measure functions (Radek)
* helper functions (Radek)
* importing pandas
* loading the customer, transaction and articles data
* *defining age groups*
* checking cell
* *define average recall function*
* *define candidate generation function*
* *define add features function*
* *actual running code*

The age groups are chosen with a reason. This way gave the best results, from the ones I tested. I did not test that many distributions so this is definitely an improvement to be done.

Running the code as is will result in a submission being formed that will predict items for week 5 based on the age group bestsellers using the origin features and the age group bestsellers as filling in case customers are missing.
While running several sanity check will be displayed to show certain contents and interactions. But also feature importance and recall scores will be displayed.

To alter the 'parameters' so to say, there are several locations that should be altered which would result in different outcomes.

IMPORTANT NOTE: not every step in the proces can easily run from this file. If a certain solution is already described in a file in RQ1 or is very similar to such a solution, I strongly advise to try that file and slightly alter that instead.

Some of the easy parameter to change are:
* TEST WEEK: at the top of the last cell the test week is defined, this can be set to any value lower than or equal to 105 and this will result in the last 10 weeks before being used. Suggestions for testing weeks are 100 or 104 as I used these myself. Note that using 105 will not show recall values, as these are not available as this is the ultimate target week
* AMOUNT OF WEEKS: The amount of weeks could be altered, but I advise to use 10 to receive similar outcomes to mine

Another very useful, but slight more complex one is:
* USE POPULARITY CANDIDATES: If the candidates of bestsellers would be used several things should be changed. For one the term ', candidates_bestsellers' should be added to the list to concatenate on line 150 of the candidate generation cell. Further the line 168 of the same cell and line 7 of the add features cell should be uncommented.

---
## Changes 
This file was not the first. This worked on solution that are discovered in other files. The last file before this one is the 'age_groups_with_origin' file.

The key difference in this file is that in the candidate generation function the filler has changed. Previously it was located in the bestseller section as these were used, but now it has moved and slightly changed to the age group bestseller to contain these bestsellers per group.
In the running code cell the code also has change, in 'else' branch of the if statement towards the end of the last cell the merging of the filler has also changed. This now merges the customer information with these bestsellers.


---
This Project is finalized due to end of semester but is not finished in research and improvements.