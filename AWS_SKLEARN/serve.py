import argparse
import pickle
import os
import json
import pandas as pd
import numpy as np
import time
import sys
from scipy.sparse import csr_matrix,hstack,vstack
from transactions_model import TransactionsModel, KEYWORDS, catg_dic

def input_fn(input_data,content_type ):
    
    if content_type=='application/json':
        
        data=json.loads(input_data)
        account_id=data['account_id']
        account_holder=data['account_holder']
        data=data['data']
        data=pd.DataFrame.from_dict(data)
    
        return account_id, account_holder, data

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    file_credit_model=os.path.join(model_dir, "credit_model.pkl")
    file_debit_model=os.path.join(model_dir, "debit_model.pkl")
    file_vocabulary_debits=os.path.join(model_dir, "vocabulary_debits.pkl")
    file_person=os.path.join(model_dir, "person.pkl")
    file_credit_keywords=os.path.join(model_dir, "credit_keywords.pkl")
    
    model=TransactionsModel(file_credit_model,file_debit_model,file_vocabulary_debits,file_person,file_credit_keywords,catg_dic)
    
    return model

def predict_fn(input_object, model):
    
    account_id,account_holder,input_data=input_object
    
    out_data=input_data[['transaction_id','type']].copy()
    
    # predictions from TransactionsModel object
    credit_predict, debit_predict=model.predict(input_data, account_holder)
    
    # Take into account whether the request misses Credits, Debits or both
    if credit_predict is not None:
        no_credit=False
        out_data.loc[out_data['type']=='Credit',"category"]=[model.credit_catg_dic[cat] for cat in credit_predict]
    else:
        no_credit=True
        
    if debit_predict is not None:
        no_debit=False
        out_data.loc[out_data['type']=='Debit',"category"]=[model.debit_catg_dic[cat] for cat in debit_predict]
    else:
        no_debit=True
    
    # If there are no transactions return empty list
    if no_debit and no_credit:
        out_data_dic=[]
    else:
        out_data_dic=list(out_data[['transaction_id','category']].T.to_dict().values())
    
    prediction={"account_id": account_id,
                
                "account_holder": account_holder,
         
                'predictions': out_data_dic
    }
    
    return prediction

def output_fn(output_data,content_type):
    
    if content_type=='application/json':
        out=json.dumps(output_data)
        
        return out