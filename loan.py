
# coding: utf-8

# In[66]:


# import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.preprocessing import StandardScaler



loan=pd.read_csv('C:\\Users\\deeksha.aneja\\Desktop\\jigsaw\\dsProjects\\loanPrediction\\train.csv')

# print(loan.head())
# print(loan.describe(include = 'all'))
loan['Gender']=loan['Gender'].map({'Female':0,'Male':1})
loan['Married'] = loan['Married'].map({'No':0, 'Yes':1})
loan['Education'] = loan['Education'].map({'Not Graduate':0, 'Graduate':1})
loan['Self_Employed'] = loan['Self_Employed'].map({'No':0, 'Yes':1})
loan['Property_Area'] = loan['Property_Area'].map({'Urban':3, 'Semiurban':2, 'Rural':1})
loan['Loan_Status'] = loan['Loan_Status'].map({'N':0, 'Y':1})

loan['Gender'] = loan['Gender'].fillna( loan['Gender'].dropna().mode().values[0] )
loan['Married'] = loan['Married'].fillna( loan['Married'].dropna().mode().values[0] )
loan['Dependents'] = loan['Dependents'].fillna( loan['Dependents'].dropna().mode().values[0] )
loan['Dependents'].replace('3+', 4,inplace=True)
loan['Self_Employed'] = loan['Self_Employed'].fillna( loan['Self_Employed'].dropna().mode().values[0] )
loan['LoanAmount'] = loan['LoanAmount'].fillna( loan['LoanAmount'].dropna().mean() )
loan['Loan_Amount_Term'] = loan['Loan_Amount_Term'].fillna( loan['Loan_Amount_Term'].dropna().mode().values[0] )
loan['Credit_History'] = loan['Credit_History'].fillna( loan['Credit_History'].dropna().mode().values[0] )
# loan['Dependents'] = loan['Dependents'].str.rstrip('+')
# loan['Gender'] = loan['Gender'].map({'Female':0,'Male':1}).astype(np.int)
# loan['Married'] = loan['Married'].map({'No':0, 'Yes':1}).astype(np.int)
# loan['Education'] = loan['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
# loan['Self_Employed'] = loan['Self_Employed'].map({'No':0, 'Yes':1}).astype(np.int)
# loan['Loan_Status'] = loan['Loan_Status'].map({'N':0, 'Y':1}).astype(np.int)
# loan['Dependents'] = loan['Dependents'].astype(np.int)

X_train=loan.iloc[:, 1:-1]
print(X_train)
slc= StandardScaler()
X_train_std = slc.fit_transform(X_train)

y_train=loan.iloc[:, -1]

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
rf.fit(X_train_std, y_train)
sns.barplot(x = 'Gender', y ='Loan_Status', data=loan)
sns.barplot(x = 'Education', y ='Loan_Status', data=loan)
sns.barplot(x = 'Dependents', y ='Loan_Status', data=loan)
sns.barplot(x = 'Credit_History', y ='Loan_Status', data=loan)








# In[74]:


loan_test=pd.read_csv('C:\\Users\\deeksha.aneja\\Desktop\\jigsaw\\dsProjects\\loanPrediction\\test.csv')
loan_test['Gender'] = loan_test['Gender'].map({'Female':0,'Male':1})
loan_test['Married'] = loan_test['Married'].map({'No':0, 'Yes':1}).astype(np.int)
loan_test['Education'] = loan_test['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
loan_test['Self_Employed'] = loan_test['Self_Employed'].map({'No':0, 'Yes':1})
# loan_test['Dependents'] = loan_test['Dependents'].str.rstrip('+')
loan_test['Dependents'].replace('3+', 4,inplace=True)
loan_test['Gender'] = loan_test['Gender'].fillna( loan_test['Gender'].dropna().mode().values[0]).astype(np.int)
loan_test['Dependents'] = loan_test['Dependents'].fillna( loan_test['Dependents'].dropna().mode().values[0]).astype(np.int)
loan_test['Self_Employed'] = loan_test['Self_Employed'].fillna( loan_test['Self_Employed'].dropna().mode().values[0])
loan_test['LoanAmount'] = loan_test['LoanAmount'].fillna( loan_test['LoanAmount'].dropna().mode().values[0])
loan_test['Loan_Amount_Term'] = loan_test['Loan_Amount_Term'].fillna( loan_test['Loan_Amount_Term'].dropna().mode().values[0])
loan_test['Credit_History'] = loan_test['Credit_History'].fillna( loan_test['Credit_History'].dropna().mode().values[0] )
loan_test['Property_Area'] = loan_test['Property_Area'].map({'Urban':3, 'Semiurban':2, 'Rural':1})
X_test = loan_test.iloc[:,1:]
X_test_std = slc.transform(X_test)
y_test_pred = rf.predict(X_test_std)
# print(y_test_pred)
loan_test['Loan_Status'] = y_test_pred
loan_final = loan_test.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'], axis=1)

loan_final['Loan_Status'] = loan_final['Loan_Status'].map({0:'N', 1:'Y'})
print(loan_final)
loan_final.to_csv('my_submission.csv', index=False)





