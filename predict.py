#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


# In[ ]:


with open('model.bin','rb') as f_in:
    pipeline = pickle.load(f_in)


# In[ ]:


class Motorcycle_Features(BaseModel):
    seller_type:str
    owner:str
    km_driven:int
    ex_showroom_price:float
    age:int



# In[4]:


app =FastAPI(title = "motorcycle Price Prediction")
@app.post("/predict")
def predict_Selling_price(motorcycle: Motorcycle_Features):
    motorcycle_dicts = motorcycle.model_dump()
    prediction = pipeline.predict(motorcycle_dicts)[0]
    return {
        "predicted_selling_price": round(float(prediction), 2)
    }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)






# In[ ]:




