import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
from scipy.stats import yeojohnson 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

from sklearn.preprocessing import RobustScaler  #it is insensitive to outliers as it uses median


# Read the data
data = pd.read_csv(r"C:\Users\Tippu\Downloads\UCI_Credit_Card.csv\UCI_Credit_Card.csv")

data
# Handling specific values in 'EDUCATION' and 'MARRIAGE'
data['EDUCATION']=data['EDUCATION'].replace([0, 6], 5)
data['MARRIAGE']=data['MARRIAGE'].replace(0, 3)

# Separating input and output data variables
X = data.drop(['default.payment.next.month','ID'], axis=1)
Y = data['default.payment.next.month']

 

#features of categorical and numerical
x_categ_features=['SEX','MARRIAGE']
x_numeric_features=X.drop(x_categ_features,axis=1).columns

import plotly.express as px

px.histogram(X['SEX'],nbins=10).show()
data['MARRIAGE'].unique()
px.histogram(X['MARRIAGE'],nbins=10).show()

# Imputation techniques to handle missing data
cat_pipeline1 = Pipeline(steps=[('impute_cat', SimpleImputer(strategy='most_frequent'))])
num_pipeline1 = Pipeline(steps=[('impute_num', SimpleImputer(strategy='median'))])

# ColumnTransformer directly with column names
preprocessor1 = ColumnTransformer([('mode', cat_pipeline1, x_categ_features),
                                   ('median', num_pipeline1, x_numeric_features)])

# Fit the data to train the imputation pipeline model
impute_data = preprocessor1.fit(X)

# Save the pipeline
joblib.dump(impute_data, 'impute')

# Transform the original data
X1 = pd.DataFrame(impute_data.transform(X), columns=X.columns)  # shape(30000,23)
print(X1[x_categ_features].isna().sum())


X.isna().sum()
# Encoding on categorical data
encode_pipeline = Pipeline(steps=[('encode', OneHotEncoder())])
preprocessor2 = ColumnTransformer([('encoding', encode_pipeline, x_categ_features)])

# Fit and save the encoding pipeline
encode_data = preprocessor2.fit(X)
joblib.dump(encode_data, 'encoder')

# Transform the original data with encoding
X2 = pd.DataFrame(encode_data.transform(X), columns=encode_data.get_feature_names_out(X.columns))

X2  #encoded data
#drop the categorical variables from x1
X1.drop(x_categ_features,axis=1,inplace=True)

#As there is presence of negative values in the dataset we use yeo johnson method for transformation 
#Yeo johnson tranformation 
from scipy.stats import yeojohnson
import seaborn as sns
import matplotlib.pyplot as plt 


transformed_final=pd.DataFrame()

for col in X1.columns:
    transformed_data,lambda_value=yeojohnson(X1[col])
    transformed_final[col] = transformed_data
    # Plot the original and transformed data side by side
    #plt.figure(figsize=(12, 5))
    # Original data
    #plt.subplot(1, 2, 1)
    #sns.histplot(X1[col], kde=True)
    #plt.title('Original Data')
    # Transformed data
    #plt.subplot(1, 2, 2)
    #sns.histplot(transformed_data, kde=True)
    #plt.title('yeo johnson Transformed Data')
    #plt.tight_layout()
    #plt.show()
    

#transformed and concatinated data
concat_data=pd.concat([X2,transformed_final],axis=1)


#SCALING THE CONCAT_DATA
scale_pipeline=Pipeline(steps=[('scale',RobustScaler())])
scaled_data=scale_pipeline.fit(concat_data) 
# save the scaling model\
joblib.dump(scaled_data,'robust_scaler')

scaled_data_final=pd.DataFrame(scaled_data.transform(concat_data),columns=concat_data.columns)
scaled_data_final

scaled_data_final.describe()

c1=scaled_data_final.corr()

import seaborn as sns 
import numpy as np
lower_tdata=np.tril(c1)
fig=px.imshow(c1,labels=dict(color="value"))
fig.update_layout(title="correaltion between variables")
fig.show()


#VARIATION INFLATION FACTOR-VIF

#from statsmodels.stats.outliers_influence import variance_inflation_factor

#data.shape[1]

#vif=pd.DataFrame()
#vif['features']=scaled_data_final.columns

#vif['vif']=[variance_inflation_factor(scaled_data_final.values,i) for i in range(len(scaled_data_final.columns)) ]

#pay_amts have  vif closer to 2;bill_amts have between 4-5.5 --moderate multicollinearity;whereas pay_i-scales have b/w 2-7.5;others<2 

###########  split the data into train and test   ###############################


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(scaled_data_final,Y,test_size=0.2,stratify=Y,random_state=42)


############################# MODEL  ##########################################



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score

#MODEL1--BASELINE MODELING -approach-1
simple_model = LogisticRegression()
sm1=simple_model.fit(x_train, y_train)
#prediction of model
y_pred = simple_model.predict(x_test) #binary result;y default threshold of 0.5 for binary classification used by LR

accuracy_score(y_test,y_pred) #80%

#METRICS
confu_matrix=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print(confu_matrix)

print(classification)  #f1score: 1---32% ! 0--88% ; 



#as the model predict gives the binary result, THE PREDICT_PROBA METHOD PROVIDES PREDICTED PROBABILITIES FOR EACH CLASS 
#the predicted probabilities for the positive class for each sample in the test set 
y_probabilities=sm1.predict_proba(x_test)[:,1]
len(y_probabilities)

y_test.shape

#approach-2
#ROC CURVE
fpr,tpr,thresholds=roc_curve(y_test,y_probabilities)
optimal_idx=np.argmax(tpr-fpr)
optimal_idx
optimal_threshold=thresholds[optimal_idx]
print(optimal_threshold)#0.2771

auc=roc_auc_score(y_test,y_probabilities)
print(f"AUC: {auc}")#73.46%


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label = "AUC="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.show()

y_pred2=pd.DataFrame()
y_pred2['pred']=np.where(y_probabilities>optimal_threshold,1,0)

accuracy_score(y_test,y_pred2.pred) #77%

#METRICS
confu_matrix2=confusion_matrix(y_pred2.pred,y_test)
classification2=classification_report(y_pred2.pred,y_test)
print(confu_matrix2)

print(classification2)    #f1score: 1---52% ! 0--85% ; recall(+veclass)--48%


#the recall value should be more for +ve class as increase in value represents less 
# false negative (defaulters incorrectly predicted as  not defaulters) which is a loss 


############################################.................................................############################################################

#approach---3

# pca applying on the dataset

from sklearn.decomposition import PCA
from kneed import KneeLocator


pca1=PCA(n_components=26)
processed1=pca1.fit(scaled_data_final)
df=processed1.transform(scaled_data_final)

#dataframe with principal components
pca_df=pd.DataFrame(df)

# calculate explained variance
explained_variance1=processed1.explained_variance_ratio_
cumulative_variance1=np.cumsum(explained_variance1)
print(cumulative_variance1)


# Create scree plot with variance ratio along the pcs

knee_locator=KneeLocator(range(1,len(explained_variance1)+1),explained_variance1,curve='concave',direction='decreasing')

knee_point=knee_locator.knee


# Plot cumulative variance with the knee point
plt.plot(range(1, len(explained_variance1) + 1), explained_variance1, marker='o', label='explained_variance')
plt.vlines(knee_point, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r', label='Knee Point')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('explained_variance Plot with Knee Point')
plt.legend()
plt.grid(True)
plt.show()




#about 98% of data been taken
pca_df1=pca_df.iloc[:,:16]
pca_df1

############### split the pca-data in to training and testing ##############################

x_train1,x_test1,y_train1,y_test1=train_test_split(pca_df1,Y,test_size=0.2,stratify=Y,random_state=42)



#modelling the principal components

model1=LogisticRegression()
model_pca=model1.fit(x_train1,y_train1)

#prediction of model
y_pred_pca =model_pca.predict(x_test1) #default threshold--0.5
y_pred_pca

#probabilities
y_probabilities1=model_pca.predict_proba(x_test1)[:,1]
y_probabilities1

fpr1,tpr1,thresholds1=roc_curve(y_test1,y_probabilities1)
optimal_idx1=np.argmax(tpr1-fpr1)  
print(optimal_idx1) 

optimal_threshold1=thresholds[optimal_idx1]
print('optimal_threshold : ',optimal_threshold1)  #0.2805



#auc
auc1=roc_auc_score(y_test1,y_probabilities1)
print(f"AUC: {auc1}")#73.58%

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label = "AUC="+str(auc1))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.show()

a1=pd.DataFrame()
a1['pred_pca']=np.where(y_probabilities1>optimal_threshold1,1,0)
accuracy_score(y_test1,a1.pred_pca) #77.4%
#confusion matrix
confusion_matrix(a1.pred_pca,y_test1)
#classification report
classifcation_pca=classification_report(a1.pred_pca,y_test1) 
print(classifcation_pca) 
##f1score------! 0--0.85 ! 1-- 0.51 ; recall(+veclass)--49%

########################################################################################################################################################################

#base line models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
instance_dt=DecisionTreeClassifier()
instance_dt.fit(x_train,y_train)
predict_dt=instance_dt.predict(x_test)
#METRICS
accuracy_score(y_test,predict_dt) #72.37%
confu_matrix_dt=confusion_matrix(y_test,predict_dt)
classification_dt=classification_report(y_test,predict_dt)
print(confu_matrix_dt)
print(classification_dt) #f1_score: 0:82 !! 1:40 


#knneigborclasssifier
instance_kn=KNeighborsClassifier()
instance_kn.fit(x_train,y_train)
predict_kn=instance_kn.predict(x_test)
#METRICS
accuracy_score(y_test,predict_kn) #79.27%
confu_matrix_kn=confusion_matrix(y_test,predict_kn)
classification_kn=classification_report(y_test,predict_kn)
print(confu_matrix_kn)
print(classification_kn)  #f1_score: 0:87 !! 1:43 ; recqll(+ve class):0.35


#svc
instance_svc=SVC()
instance_svc.fit(x_train,y_train)
predict_svc=instance_svc.predict(x_test)
#METRICS
accuracy_score(y_test,predict_svc) #81.54%
confu_matrix_svc=confusion_matrix(y_test,predict_svc)
classification_svc=classification_report(y_test,predict_svc)
print(confu_matrix_svc)
print(classification_svc)  #f1_score: 0:89 !! 1:45 ; recqll(+ve class):0.34

#CATBOOST
from catboost import CatBoostClassifier
instance_cb=CatBoostClassifier()
instance_cb.fit(x_train,y_train)
predict_cb=instance_cb.predict(x_test)
#METRICS
accuracy_score(y_test,predict_cb) #81.744%
confu_matrix_cb=confusion_matrix(y_test,predict_cb)
classification_cb=classification_report(y_test,predict_cb)
print(confu_matrix_cb)
print(classification_cb) #f1_score: 0:89 !! 1:47 ; recqll(+ve class):0.37

############## parameter tuning #############################
''''
#Regularization provides a means to control the complexity of the model. By discouraging the use of all available features, 
#Use Grid Search for a systematic exploration of predefined values.
#Start with a broad search over several orders of magnitude (e.g., 0.001, 0.01, 0.1, 1, 10, 100).
#If the best-performing C value is near the extremes of the search space, consider refining the search around that value.
#Use Randomized Search for a more efficient search over a continuous range or when the number of hyperparameter combinations is large
#Smaller C values might be more appropriate in the presence of outliers.
#In datasets with a large number of features, the risk of overfitting increases so use smaller c value as it leads to  poor generalization
#or small datasets, regularization is crucial to prevent overfitting.
#In situations where the features have a weak relationship with the target variable,
# there is a risk of the model fitting noise in the training data, leading to poor generalization on new, unseen data.

#By systematically varying the C values during cross-validation, you can identify the level of regularization that optimally balances bias and variance in the model
#A higher number of folds reduces the variance of the cross-validation estimate but increases its bias. A lower number of folds has the opposite effect.
#The bias-variance trade-off should be considered based on the specific characteristics of the dataset and the modeling problem.
#for  imbalanced class distribution, consider using Stratified K-Fold cross-validation
#For very small datasets, Leave-One-Out cross-validation (cv=n where n is the number of samples) provides the most accurate estimate, but it can be computationally expensive.



#If you use f1_score without specifying the pos_label, it calculates the F1 score for both classes and returns an array of two values (F1 score for class 0 and class 1).
#If you use make_scorer(f1_score) without specifying pos_label, it uses the F1 score for the positive class (class 1) as the primary metric for optimization.
#The hyperparameters selected during grid search are the ones that maximize this primary F1 score.

'''

#logistic regression parameter tuning
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
param_grid= { 'C':[0.001,0.01,0.1,1,10,100],
              'penalty':['l1','l2','elasticnet'],
              'solver':['liblinear','lbfgs','newton-cg','sag','saga'],
              'max_iter':[250,450,1200,3000,5000]
            }
            
mdl = LogisticRegression()
scorer=make_scorer(f1_score)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

grid_search = GridSearchCV(mdl,param_grid,cv=cv,scoring=scorer)
grid_search.fit(x_train1,y_train1)
grid_search.best_params_

#'C': 0.001, 'max_iter': 250, 'penalty': 'l2', 'solver': 'liblinear'
mdl1=LogisticRegression(C=0.001, max_iter= 250, penalty= 'l2', solver= 'liblinear')
mdl1.fit(x_train1,y_train1)
pred_l1=mdl1.predict(x_test1)

accuracy_score(y_test,pred_l1)  #80.14%
cr1=classification_report(y_test,pred_l1) # #f1_score: 0:88 !! 1:34 ; recqll(+ve class):0.24
print(cr1)

################  MODEL APPROACH -- DECISION TREE       ########################
#decisiontree parameter tuning
from sklearn.tree import DecisionTreeClassifier 
param_grid_dt= { 'criterion':['gini', 'entropy', 'log_loss'],
                'splitter':['best','random'] ,
                'min_samples_split':[3,5,25,50,100],
                'min_samples_leaf':[2,3,5,11,15,45],
                'max_features':['auto', 'sqrt', 'log2'],
                'random_state':[0,25,42,65,82,100]
            }

model_dt=DecisionTreeClassifier()
scorer_dt=make_scorer(f1_score)
grid_search_dt = GridSearchCV(model_dt,param_grid_dt,cv=cv,scoring=scorer_dt)
grid_search_dt.fit(x_train,y_train)
grid_search_dt.best_params_
#'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_leaf': 45, 'min_samples_split': 100, 'random_state': 65, 'splitter': 'best'
model_dt_=DecisionTreeClassifier(criterion= 'entropy', max_features= 'sqrt', min_samples_leaf= 45, min_samples_split= 100, random_state= 65, splitter= 'best')
model_dt_.fit(x_train,y_train)
dt_pred_=model_dt_.predict(x_test)   #81.74%
print("Accuracy:", accuracy_score(y_test, dt_pred_))
print("\nClassification Report:\n", classification_report(y_test,dt_pred_)) # #f1_score: 0:0.89 !! 1:0.46 ; recqll(+ve class):0.36

'''
grid_search_dt1 = GridSearchCV(model_dt,param_grid_dt,cv=cv,scoring=scorer_dt)
grid_search_dt1.fit(x_train1,y_train1)
grid_search_dt1.best_params_
#'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_leaf': 45, 'min_samples_split': 3, 'random_state': 65, 'splitter': 'best'
model_dt_1=DecisionTreeClassifier(criterion= 'entropy', max_features= 'sqrt', min_samples_leaf= 45, min_samples_split= 100, random_state= 65, splitter= 'best')
model_dt_1.fit(x_train1,y_train1)
dt_pred_1=model_dt_1.predict(x_test1)   #82%
print("Accuracy:", accuracy_score(y_test1, dt_pred_1))    #f1_score: 0:0.88 !! 1:0.42 ; recqll(+ve class):0.32
print("\nClassification Report:\n", classification_report(y_test1,dt_pred_1))

'''

'''
feature_importances = dt_train1.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(x_train1.shape[1]), feature_importances[sorted_idx], align="center")
plt.xticks(range(x_train1.shape[1]), X.columns[sorted_idx], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

from sklearn.tree import plot_tree
plt.figure(figsize=(20,16))
plot_tree(dt_train1, feature_names=x_train1.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
'''

#svc parameter tuning
param_grid_svc= {'C':[0.001,0.01,0.1,1,10,100],
                'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': [0.001,0.01,0.1,1]
                }
instance_svc=SVC()
grid_search_svc1 =RandomizedSearchCV(instance_svc,param_grid_svc,cv=cv,scoring=scorer_dt)
grid_search_svc1.fit(x_train,y_train)
grid_search_svc1.best_params_

# 
instance_svc1=SVC(C=,kernel=,gamma=)
instance_svc1.fit(x_train,y_train)
predict_svc1=instance_svc.predict(x_test)
#METRICS
confu_matrix_svc1=confusion_matrix(y_test,predict_svc1)
print(confu_matrix_svc1)
print("Accuracy:", accuracy_score(y_test,predict_svc1))
print("\nClassification Report:\n", classification_report(y_test,predict_svc1)) #81.54%

  #f1_score: 0:89 !! 1:45 ; recqll(+ve class):0.34