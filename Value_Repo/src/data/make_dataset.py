# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
os.getcwd()

def main(input_filepath):
    
    stock_data = pd.read_csv("data\\raw\\MSFT.csv")

    #Select and rename columns for prophet
    stock_data = stock_data[['Date','Close']]
    stock_data = stock_data.rename(columns = {'Date':'ds','Close':'y'})
    # Extract the 'Date' column and use it as the index in separate df
    stock_data_ = stock_data.copy()
    stock_data_.loc[:, 'ds'] = pd.to_datetime(stock_data['ds'])
    stock_data_.set_index('ds', inplace=True)
    # Split the data into training and test sets
    train_data = stock_data[0:len(stock_data_[stock_data_.index.year != 2021])]
    test_data = stock_data[len(stock_data_[stock_data_.index.year != 2021]):]

    train_data.to_csv(os.path.join("data\\interim\\train_data.csv"))
    test_data.to_csv(os.path.join("data\\interim\\test_data.csv"))
    stock_data.to_csv(os.path.join("data\\interim\\stock_data.csv"))

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    
    main("data\\raw\\MSFT.csv")
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

