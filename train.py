#/usr/bin/env python
# coding: utf-8

# In[20]:


import pickle

import pandas as pd
import openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor




# In[21]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[22]:


# fumction for loadind data
def load_data():
    # Fetch dataset 43801 directly from OpenML via ID
    # 'as_frame=True' forces scikit-learn to parse the ARFF file directly into a clean Pandas DataFrame
    print("Downloading and parsing data...")
    oml_data = fetch_openml(data_id=43801, as_frame=True, parser="auto")
    # Extract the DataFrame
    df = oml_data.frame
    df['ex_showroom_price'] = df['ex_showroom_price'].fillna(df['ex_showroom_price'].mean())
    df['age'] = 2026 - df['year']
    df.drop(columns= ['year','name'], inplace = True)

    return(df)



# In[26]:


def train_model(df):
    df_full_train, df_test = train_test_split(df, test_size =0.2, random_state = 11)
    categorical = ['seller_type','owner']
    numerical = ['km_driven','ex_showroom_price','age']


    y_train = df_full_train['selling_price']
    y_test = df_test['selling_price']

    train_dicts = df_full_train[categorical + numerical].to_dict(orient ='records')
    test_dicts =df_test[categorical + numerical].to_dict(orient ='records')
    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(n_estimators = 100 , max_depth= 10 ,min_samples_leaf = 5,  random_state = 42)
    )
    pipeline.fit(train_dicts,y_train)
    return(pipeline)




# In[27]:


def save_model(pipeline,output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)




# In[28]:


df = load_data()
pipeline = train_model(df)
save_model(pipeline,'model.bin')
print('Model saved to model.bin')


# In[ ]:





# In[ ]:





# In[ ]:




