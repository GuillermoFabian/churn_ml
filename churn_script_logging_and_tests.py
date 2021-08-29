'''
Tittle: Churn Tests

Author: Guillermo Figueroa

Date: August 2021
'''

import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(path):
    '''
    test data import
    '''
    try:
        df = cl.import_data(path)
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(data_frame_input):
    '''
    test perform eda function
    '''
    try:
        cl.perform_eda(data_frame_input)
        logging.info("SUCCESS: Testing perform_eda")
    except KeyError as err:
        logging.error(
            "ERROR: Testing perform_eda: column not found in dataframe")
        raise err


def test_encoder_helper(data_frame_input, column_list):
    '''
    test encoder helper
    '''
    try:
        cl.encoder_helper(data_frame_input, column_list)
        logging.info("SUCCESS: Testing encoder_helper")
    except KeyError as err:
        logging.error(
            "ERROR: Testing encoder_helper: column not found in dataframe")
        raise err


def test_perform_feature_engineering(data_frame_input):
    '''
    test perform_feature_engineering
    '''
    try:
        cl.perform_feature_engineering(data_frame_input)
        logging.info("SUCCESS: Testing perform_feature_engineering")
    except KeyError as err:
        logging.error(
            "ERROR: Testing perform_feature_engineering: The file wasn't found")
        raise err


def test_train_models(X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        cl.train_models(X_train, X_test, y_train, y_test)
        logging.info("SUCCESS: Testing train_models")
    except ValueError as err:
        logging.error(
            "ERROR: Testing train_models: inconsistent lenght of elements")
        raise err


if __name__ == "__main__":
    test_import(r"./data/bank_data.csv")
    test_eda(cl.import_data(r"./data/bank_data.csv"))
    test_encoder_helper(
        cl.import_data(r"./data/bank_data.csv"),
        ['Gender',
         'Education_Level',
         'Marital_Status',
         'Income_Category',
         'Card_Category'])

    test_perform_feature_engineering(cl.encoder_helper(

        cl.import_data(r"./data/bank_data.csv"),
        ['Gender',
         'Education_Level',
         'Marital_Status',
         'Income_Category',
         'Card_Category']))
    result = cl.perform_feature_engineering(cl.encoder_helper(

        cl.import_data(r"./data/bank_data.csv"),
        ['Gender',
         'Education_Level',
         'Marital_Status',
         'Income_Category',
         'Card_Category']))
    test_train_models(result[0], result[1], result[2], result[3])
