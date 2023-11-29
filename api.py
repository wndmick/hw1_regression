import os
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

print(os.getcwd())

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

def preprocess_data(data):

    for col in ['mileage', 'engine', 'max_power']:
        data[col] = data[col].str.replace(r'[^\.\d]+', '', regex=True).astype(float)
    data.drop('torque', axis=1, inplace=True)

    data[['engine', 'seats']] = data[['engine', 'seats']].astype(int)

    num_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    cat_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

    data[num_features] = scaler.transform(data[num_features])

    data = pd.concat([data.drop(columns=cat_features), 
                      pd.DataFrame(ohe.transform(data[cat_features]),
                      columns=ohe.get_feature_names_out())], axis=1)
    
    return data

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame.from_dict([item.model_dump()])
    prediction_data = preprocess_data(data.drop(columns=['name', 'selling_price']))
    res = model.predict(prediction_data)
    return res

@app.post("/predict_items")
def predict_items(file: UploadFile):
    data = pd.read_csv(file.file)
    prediction_data = preprocess_data(data.drop(columns=['name', 'selling_price']))
    res = model.predict(prediction_data)
    print(res)
    data['predicted_price'] = res
    data.to_csv('predicted_items.csv', index=False)
    return FileResponse('predicted_items.csv')