#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import s3fs
import numpy as np
import pandas as pd
import re
import boto3
import pickle 
from datetime import timedelta, datetime
import os
import sys
from io import StringIO
from io import BytesIO
import time
from botocore.exceptions import ClientError
# model = pickle.load(open('Rf_model_version1.pkl','rb'))
# #Raw_data=pd.read_csv('test.csv')
# dirpath = os.getcwd()
# print("dirpath = ", dirpath, "\n")
var_dict={

    "result_file_location" : "ml/analytical-result-store",
    "result_file_name"    : "titanic_prediction_docker_result.csv",
    "pretrained_model_loc" : "ml/prediction_model/titanic_prediction_model.pkl",
    "inference_data_folder":"ml/prediction-data",
    "inference_file_name":"test.csv",
    "s3_bucket_name" : "-datalake-dev-bucket",
     "inference_file_backup_path" : "ml/prediction-data-backup",
    "inference_file_backup_key" : "test.csv"
    #"aws_id": "2J",
    #"aws_secret_key" :"HTh"
    
}

# file_name =  "s3://swire-analytics/training/test.csv"
# Raw_data = pd.read_csv(file_name)
def data_cleansing (source_df):
    """  
    Get the source, Drop Id column
    convert cabin feature into deck feature
    embarked,Title, Fare, Sex features are mapped and cleaned
    """
    # Drop the id column
    #data=source_df.drop(['PassengerId'], axis=1)
    # Converting Cabin feature in to Deck and drop cabin feature
    data=source_df[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data['Cabin'] = data['Cabin'].fillna("U0")
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['Deck'] = data['Deck'].map(deck).fillna(0).astype(int) 
    data = data.drop(['Cabin'], axis=1)
    # Age feature 
    mean = 3.46
    std = data["Age"].replace(np.NaN,0).std()
    is_null = data["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = data["Age"].astype(int)

    ## Embarked
    common_value = 'S'
    data['Embarked'] = data['Embarked'].fillna(common_value)

    # Fare
    data['Fare'] = data['Fare'].fillna(0).astype(int)

    # Title 
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    # extract titles
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    data['Title'] = data['Title'].map(titles)
    # filling NaN with 0, to get safe
    data['Title'] = data['Title'].fillna(0)
    # Droping the Name feature
    data = data.drop(['Name'], axis=1)
    #
    genders = {"male": 0, "female": 1}
    data['Sex'] = data['Sex'].map(genders)
    #
    data = data.drop(['Ticket'], axis=1)
    #
    ports = {"S": 0, "C": 1, "Q": 2}
    data['Embarked'] = data['Embarked'].map(ports)
    cleansed_df=data
    
    return cleansed_df


def create_new_feature_df(cleansed_df):
    ## Age--Convert numeric into categorical features
    data=cleansed_df
    data['Age'] = data['Age'].astype(int)
    data.loc[ data['Age'] <= 11, 'Age'] = 0
    data.loc[(data['Age'] > 11) & (data['Age'] <= 18), 'Age'] = 1
    data.loc[(data['Age'] > 18) & (data['Age'] <= 22), 'Age'] = 2
    data.loc[(data['Age'] > 22) & (data['Age'] <= 27), 'Age'] = 3
    data.loc[(data['Age'] > 27) & (data['Age'] <= 33), 'Age'] = 4
    data.loc[(data['Age'] > 33) & (data['Age'] <= 40), 'Age'] = 5
    data.loc[(data['Age'] > 40) & (data['Age'] <= 66), 'Age'] = 6
    data.loc[ data['Age'] > 66, 'Age'] = 6
    ## fare --Convert numeric into categorical features
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 99), 'Fare']   = 3
    data.loc[(data['Fare'] > 99) & (data['Fare'] <= 250), 'Fare']   = 4
    data.loc[ data['Fare'] > 250, 'Fare'] = 5
    data['Fare'] = data['Fare'].astype(int)
    ################
    # Relatives feature
    data['relatives'] = data['SibSp'] + data['Parch']
    data.loc[data['relatives'] > 0, 'not_alone'] = 0
    data.loc[data['relatives'] == 0, 'not_alone'] = 1
    data['not_alone'] = data['not_alone'].astype(int)
    ## Create Age class
    data['Age_Class']= data['Age']* data['Pclass']
    ## fare per person
    data['Fare_Per_Person'] = data['Fare']/(data['relatives']+1)
    data['Fare_Per_Person'] = data['Fare_Per_Person'].astype(int)
    cleaned_normalised_data_=data
    
    return cleaned_normalised_data_

def read_csv_file(bucket_name,inference_data_key): #,aws_access_id,aws_access_key
    """ Read the csv predcition file from s3 using boto3 client
    """
    client = boto3.client('s3') #, aws_access_key_id = aws_access_id , aws_secret_access_key= aws_access_key

    csv_obj = client.get_object(Bucket=bucket_name, Key=inference_data_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')

    Raw_data = pd.read_csv(StringIO(csv_string))
    
    return Raw_data

def load_pickle_data(bucket_name, model_key): # ,aws_access_id,aws_access_key
    """
    Get the stored pretrained model from S3 bucket
    """
    client = boto3.client('s3') #, aws_access_key_id = aws_access_id, aws_secret_access_key=aws_access_key
    response = client.get_object(Bucket=bucket_name, Key=model_key)
    body = response['Body'].read()
    trained_model = pickle.loads(body)

    return trained_model

def write_to_s3(bucket_name,result_key, raw_data): # ,aws_access_id,aws_access_key
    csv_buffer = StringIO()
    raw_data.to_csv(csv_buffer)
    resource = boto3.resource('s3') # , aws_access_key_id= aws_access_id ,aws_secret_access_key=aws_access_key
    
    return resource.Object(bucket_name,result_key).put(Body=csv_buffer.getvalue())

def copy_and_delete_prediction_data():
    try:
        #backup the prediction data
        s3 = boto3.resource('s3')
        copy_source = {
                       'Bucket' : bucket_name,
                     'Key'      : inference_data_key
                       }
        s3.meta.client.copy(copy_source,bucket_name,inf_backup_key)
        # deleting the object
        s3_client=boto3.client('s3')
        response =s3_client.delete_object(
                                     Bucket=bucket_name,
                                     Key=inference_data_key
                                            )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print("File not found : ",  e)

#var_dict = get_json_from_variable()

#s3_conn_id = var_dict["s3_conn_id"]
result_file_location = var_dict["result_file_location"]
result_file_name    =  var_dict["result_file_name"]
pretrained_model_loc = var_dict["pretrained_model_loc"]
bucket_name = var_dict["s3_bucket_name"]
inference_data_folder = var_dict["inference_data_folder"]
inference_file_name = var_dict["inference_file_name"]
backup_path_inf_file = var_dict["inference_file_backup_path"]
backup_filename_inf = var_dict["inference_file_backup_key"]
bucket_name=var_dict['s3_bucket_name']  
#aws_access_id= var_dict['aws_id']
#aws_access_key= var_dict['aws_secret_key']

inference_data_key = inference_data_folder + "/" + inference_file_name
model_key = pretrained_model_loc
result_key = result_file_location +"/" + result_file_name
inf_backup_key = backup_path_inf_file + "/"+ backup_filename_inf


def make_prediction():
    """Make the prediction using 
       trained Model and the inference data   
       
    """
    csv_buffer = BytesIO()
    
    # read the raw data from S3 --from utility
    # raw_data = read_csv_from_s3(s3_conn_id, bucket_name, inference_data_key)
    raw_data =read_csv_file(bucket_name,inference_data_key) # ,aws_access_id,aws_access_key

    #passenger_id=raw_data['PassengerId']
    # clean the data --custom functions
    cleansed_df = data_cleansing(raw_data)
    #print(cleansed_df.shape)
    cleaned_normalised_data_=create_new_feature_df(cleansed_df)   
    #print(cleaned_normalised_data_.shape)
    # read the  trained model from s3 --from utility  
    trained_model=load_pickle_data(bucket_name, model_key) # ,aws_access_id,aws_access_key
    #print(trained_model)
    prediction_output=trained_model[0].predict(cleaned_normalised_data_)
    print(" Shape of the prediction data : ", prediction_output.shape)
    results=pd.DataFrame()
    results['Predicted_result']=prediction_output
    #results=results.reset_index(passenger_id)
    
    # ##  store the data into S3
    #data = results.to_csv(sep='|',index=False)
    print(results.head())
    #write_to_s3(data, bucket_name, result_key)
    write_to_s3(bucket_name,result_key,results) # ,aws_access_id,aws_access_key
    print('**********Results will be stored in the Analytical result bucket********')
    copy_and_delete_prediction_data()
    print('***********inference file backedup and deleted**************************** ')

def prediction():
    
    try:
        s3 = boto3.resource('s3')
        s3.Object(bucket_name,inference_data_key).load()
        #print(time.time())

    except ClientError as e:
        print("*************Waiting for the next file to predict*********************")

    else:
        # The object does exist.
        make_prediction()

if __name__ == "__main__" :
    
    while True:
        time.sleep(5)
        prediction()
        

        
        
        
        
        