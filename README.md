# ds
Loan Prediction Problem

This is the solution for Loan Prediction Problem on Analytics Vidhya. Here training data and test data are provided in seperate CSV problem. Based on the understanding from training data, we need to predict whether loan should be given in Test Data.
Here are a few inferences, you can draw by looking at the output of describe() function:

LoanAmount has (614 – 592) 22 missing values.
Loan_Amount_Term has (614 – 600) 14 missing values.
Credit_History has (614 – 564) 50 missing values.

We can also look that about 84% applicants have a credit_history. How? The mean of Credit_History field is 0.84 (Remember, Credit_History has value 1 for those who have a credit history and 0 otherwise)
The ApplicantIncome distribution seems to be in line with expectation. Same with CoapplicantIncome

For training the model, we need the data to be inline with what machine can understand. So, for string values we need to map int values. Such as- gender->map('F':0,'M':1)
#Code
loan['Gender']=loan['Gender'].map({'Female':0,'Male':1})
Similarly, we map following columns to integers-

  Married,Education,Self_Employed,Property_Area,Loan_Status
  
In Dependent column, there's a value 3+. This cannot be interpreted by the machine. Hence, we replace 3+ by 4
#Code

loan['Dependents'].replace('3+', 4,inplace=True)

Now, for fixing NaN values, we replace NaN with either the mean or the mode of the column
#Code

  loan['Gender'] = loan['Gender'].fillna( loan['Gender'].dropna().mode().values[0] )
  loan['Married'] = loan['Married'].fillna( loan['Married'].dropna().mode().values[0] )
  loan['Dependents'] = loan['Dependents'].fillna( loan['Dependents'].dropna().mode().values[0] )
  loan['Self_Employed'] = loan['Self_Employed'].fillna( loan['Self_Employed'].dropna().mode().values[0] )
  loan['LoanAmount'] = loan['LoanAmount'].fillna( loan['LoanAmount'].dropna().mean() )
  loan['Loan_Amount_Term'] = loan['Loan_Amount_Term'].fillna( loan['Loan_Amount_Term'].dropna().mode().values[0] )
  loan['Credit_History'] = loan['Credit_History'].fillna( loan['Credit_History'].dropna().mode().values[0] )

Let's normalize the independent variables.This can be done using StandardScaler(). The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.

#Code

  X_train=loan.iloc[:, 1:-1]
  print(X_train)
  slc= StandardScaler()
  X_train_std = slc.fit_transform(X_train)
  
 Using Random Forests for curve fitting
 
  # Code
  
  rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
  rf.fit(X_train_std, y_train)
  
 Now when we have got the curve, we can predict whether a loan can be disbursed or not in test data. We need to do same transformations as above on test.csv
  
 #Code
 
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
