#reading fromm one to another
#read the data divide to train and test 
#read the data from some specific data source can be created from cloud team/ split the data to test and split

import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #create class variables
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import dataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
#where to save data

@dataclass  #directly define class variable
class DataIngestionConfig: 
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()   #three paths will get saved

    def initiate_data_ingestion(self):  #if my data is stored somewhere else it will read the data from the database to here

        logging.info("enter the data ingestion method ir component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')  #here you can read from mongodb

            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #convert the raw dataset to csv

            logging.info("train_test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) #split the data to train and test

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modelTrainer=ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))




 

