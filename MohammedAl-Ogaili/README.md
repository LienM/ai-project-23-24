# AI Project Code submission

In this short readme file I will roughly explain the purpose of each file that is relevant to my submission.

Here is the rough outline of the project structure. Note that the folders mentioned here need to exist otherwise the program might throw an error when saving to these folders.

I have uploaded some check points and the metrics on wetransfer https://we.tl/t-GyTv2BUHeH (expires on 04/01/2024)
```
📦MohammedAl-Ogaili
 ┣ 📂data
 ┃ ┗ // Contains the data both in csv and parquet format (not pushed to repo)
 ┣ 📂final
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📂article_id
 ┃ ┃ ┃ ┗ // Contains model checkpoints, graphs, and statistics (not pushed)
 ┃ ┃ ┣ 📂article_id_104
 ┃ ┃ ┃ ┗ // Contains model checkpoints, graphs, and statistics (not pushed)
 ┃ ┃ ┣ 📂product_type_name
 ┃ ┃ ┃ ┗ // Contains model checkpoints, graphs, and statistics (not pushed)
 ┃ ┃ ┣ 📂product_type_name_104
 ┃ ┃ ┃ ┗ // Contains model checkpoints, graphs, and statistics (not pushed)
 ┃ ┃ ┣ 📂prod_name
 ┃ ┃ ┃ ┗ // Contains model checkpoints, graphs, and statistics (not pushed)
 ┃ ┃ ┗ 📂prod_name_104
 ┃ ┃   ┗ // Contains model checkpoints, graphs, and statistics (not pushed)
 ┃ ┣ 📜BaselineExperiments.ipynb
 ┃ ┣ 📜experiment_template.py
 ┃ ┣ 📜models.zip
 ┃ ┣ 📜SequenceModelExperiments.ipynb
 ┃ ┗ 📜util.ipynb
 ┗ 📜README.md
```

To answer my research questions I used the following files:

* **util.ipynb:** This file houses functions that were needed, I placed them in a separate notebook to keep the notebooks short and clean. It also contains the LSTM model definition.
* **SequenceModelExperiments.ipynb:** Notebook containing my pre-processing, training, validating, and testing of the LSTM model by itself.
* **experiment_template.py:** This is also a script containing utility functions, written by Noah's and used by his experiment template that I based my hybrid solutions on.
* **BaselineExperiments.ipynb:** Notebook where I implemented and tested my hybrid approaches.

