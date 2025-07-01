from flask import Flask, render_template,redirect,url_for,request,jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import yeojohnson 
import numpy as np
from sklearn.preprocessing import RobustScaler 
import pickle,joblib
import seaborn as sns


#import saved models and processing steps
impute=joblib.load('impute')
encoder=joblib.load('encoder')
transformer=pickle.load(open('power_transformer.pkl', 'rb'))
scale=joblib.load('robust_scaler')
pca_model=pickle.load(open('pca_applied.pkl', 'rb'))
model1=pickle.load(open('model1.pkl','rb'))


def predict(data):
    if 'default.payment.next.month'  not in data.columns:
        
    # Handling specific values in 'EDUCATION' and 'MARRIAGE'
        #data['EDUCATION'].replace([0, 6],5, inplace=True)
        #data['MARRIAGE'].replace(0, 3, inplace=True)
            
        #features of categorical and numerical
        x_categ_features=['SEX','MARRIAGE']
        x_numeric_features = data.drop(x_categ_features,axis=1).columns
            
        # Transform the original data

        cleandata1 = pd.DataFrame(impute.transform(data), columns=data.columns) 

        cleandata2 = pd.DataFrame(encoder.transform(data), columns=encoder.get_feature_names_out(data.columns))

            #transformation
        transformed_new_data=pd.DataFrame(transformer.transform(cleandata1),columns=cleandata1.columns)

            # concatinate categ and numeric columns
        concat_data=pd.concat([cleandata2,transformed_new_data],axis=1)
        concat_data_final=concat_data.drop(x_categ_features,axis=1)

            #scaling
        scaled_data_final=pd.DataFrame(scale.transform(concat_data_final),columns=concat_data_final.columns)

            # pca applying on the dataset :#dataframe with principal components

        pca_df=pd.DataFrame(pca_model.transform(scaled_data_final)).iloc[:,:16] #95% of data retained
        
        #Prediction
        prediction = pd.DataFrame(model1.predict(pca_df),columns=['choice'])
            
        return prediction

def main():
    
    
    st.title("credit card default prediction")
    st.title("upload data")
    
    uploadedFile = st.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    
    if uploadedFile is not None :
        try:
            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:
                data = pd.DataFrame()
                st.sidebar.warning("unable to read the file")
    else:
        st.sidebar.warning("u need to upload csv or an excel file")
    if data.empty:
        st.warning("no valid data found")
        st.stop()
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data)
        
        #st.dataframe(result) 
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()








