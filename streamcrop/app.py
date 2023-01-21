import pandas as pd
import numpy as np
import streamlit as s
import pickle
from PIL import Image
pickle_in=open("model.pkl","rb")
model=pickle.load(pickle_in)
def welcome():
    return 'welcome all'
def prediction(n,p,k,temperature,humidity,ph,rainfall):
    prediction=model.predict([[n,p,k,temperature,humidity,ph,rainfall]])
    print(prediction)
    return prediction
def main():
    s.title("crop recommendation")
    s.markdown("crops are recommended")
    n=s.text_input("n","type here")
    p=s.text_input("p","type here")
    k=s.text_input("k","type here")
    temperature=s.text_input("temperature","type here")
    humidity=s.text_input("humidity","type here")
    ph=s.text_input("ph","type here")
    rainfall=s.text_input("rainfall","type here")
    result=""
    if s.button("predict"):
        result=prediction(n,p,k,temperature,humidity,ph,rainfall,s.success("the output is {}".format(result)))
    if __name__=="__main__":
        main()
