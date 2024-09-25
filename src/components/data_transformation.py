import os, sys
from unicodedata import category

import numpy as np
from pandas.io.xml import preprocess_data

sys.path.append("/Users/alindsplayground/PycharmProjects/mltest")
import pandas as pd
import numpy
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformConfig:
    preprocesser_obj_file_path=os.path.join("artifacts","preprocesser.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformer_config=DataTransformConfig()

    def get_datatransformer_object(self):
        try:
            numerical_columns=['reading_score', 'writing_score']
            category_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline(steps=
                                  [("Impute",SimpleImputer(strategy="median")),
                                   ("scaler",StandardScaler())]
                        )

            category_pipeline=Pipeline(
                steps=[("Impute",SimpleImputer(strategy='most_frequent')),
                       ("Encoder",OneHotEncoder()),
                       ("Scaler", StandardScaler(with_mean=False))]
            )

            logging.info("Scaling for numerical columns completed")
            logging.info("Encoding for categorical columns completed")
            preprocessor=ColumnTransformer(
                [("Num Columns",num_pipeline,numerical_columns),
                 ("Cat Columns",category_pipeline,category_columns)]

            )

            return preprocessor
        except  Exception as e:
            raise CustomException(e,sys)

    def initialize_data_transformation(self,train_data,test_data):
            try:
                train_df = pd.read_csv(train_data)
                test_df = pd.read_csv(test_data)
                logging.info("Read Train and test data completed")
                logging.info("Obtaining preprocessing object")
                preprocessing_obj=self.get_datatransformer_object()
                target_column_name="math_score"
                numerical_columns = ['reading_score', 'writing_score']
                input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df=train_df[target_column_name]
                logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
                input_feature_test_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = train_df[target_column_name]

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
                train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
                logging.info("Saving Object")

                save_object(

                    file_path=self.data_transformer_config.preprocesser_obj_file_path,
                    obj=preprocessing_obj

                )

                return (train_arr,
                        test_arr)
            except Exception as e:
                raise CustomException(e,sys)


