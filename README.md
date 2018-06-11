# ds
dsProjects
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
