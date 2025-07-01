from flask import Flask, render_template,redirect,url_for,request,jsonify
from flask_sqlalchemy import SQLAlchemy

from datetime import datetime


import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import yeojohnson 
import numpy as np
from sklearn.preprocessing import RobustScaler 
import pickle,joblib




#import saved models and processing steps
impute=joblib.load('impute')
encoder=joblib.load('encoder')
transformer=pickle.load(open('power_transformer.pkl', 'rb'))
scale=joblib.load('robust_scaler')
pca_model=pickle.load(open('pca_applied.pkl', 'rb'))
model1=pickle.load(open('model1.pkl','rb'))

#In the context of a file upload in Flask, request.files is a special dictionary-like object that contains the files uploaded in the request. The keys of this dictionary correspond to the names of the form fields used for the file upload
app=Flask(__name__)

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    try:
        
        #below steps do accessing the file object associated with the form field named 'file'. 
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})
        
        
        #get the file from the request
        file=request.files['file']
        
        # check if file is empty
        if file.filename=='':
            return jsonify({'error': 'Empty file provided'})
        
        #access the filename associated with file
        filename =file.filename
        
        # check if the file in a csv file
        if file.mimetype !='text/csv':
            return jsonify({'error': 'only csv files are supported'})
        
        #read the csv file
        data=pd.read_csv(file,header=None)
        
        # Handling specific values in 'EDUCATION' and 'MARRIAGE'
        data['EDUCATION'].replace([0, 6], 5, inplace=True)
        data['MARRIAGE'].replace(0, 3, inplace=True)
        
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
     

       #Predicttion

        prediction = pd.DataFrame(model1.predict(pca_df),columns=['choice'])
        
        return jsonify(prediction.to_dict(orient='records'))
    
    except Exception as e :
        return jsonify({'error':str(e)})
    
if __name__=="__main__":
    app.run(debug=True,port=5001)
                
                
                        

                
                
                
                
                





