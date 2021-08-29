# Predict Customer Churn

- Project **Predict Customer Churn ML creationg and testing** 

## Project Description
We aim to identify credit card customers that are most likely to churn. 
We include a Python package for a machine learning project that follows coding (PEP8) 
and engineering best practices for implementing software (modular, documented, and tested). 
The package will also have the flexibility of being run interactively or from the command-line interface (CLI).


## Prerequesites

> Python >= 3.8

Recommended to create virtual environment:

> python3 -m venv /path/to/new/virtual/environment

Install required packages:

> pip install -r requirements.txt

## Running Files

For testing purpose:

> python churn_script_logging_and_test.py



### Running module:

> import churn_library as cl

> df = cl.import_data('path_to_data)

> df_encoded = encoder_helper(
        df,
        ['Gender',
         'Education_Level',
         'Marital_Status',
         'Income_Category',
         'Card_Category'])


> samples = cl.perform_feature_engineering(df_encoded)

> train_models(samples[0], samples[1], samples[2], samples[3]) 

Models stored in models folder and eda resultas stored in images folder. 
The logs results output produced by testing is stored under churn_library.log in logs folder

