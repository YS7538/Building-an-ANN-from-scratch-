import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

df= pd.read_csv(r"D:\Projects\Building-an-ANN-from-scratch-\data\heart_disease_uci.csv")


numeric_cols = ['thalch','oldpeak','chol','trestbps']
binary_cols= ['exang','fbs']
categorical_cols= ['restecg','slope']

for col in numeric_cols:
    df[col]=df[col].fillna(df[col].mean()) #filling the columns which has <50% missing values

for col in binary_cols:
    df[col]= df[col].fillna(df[col].mode()[0])
    
for col in categorical_cols:
    df[col]= df[col].fillna(df[col].mode()[0])

df= df.drop(columns=['thal','ca'])#dropping columns with more than 50% missing values

#missing_percent = df.isnull().mean() * 100
#print(missing_percent)

df['num'] = (df['num'] > 0).astype(int) #converting num to binary outputs where 1= disease and 0= no disease

obj_cols= df.select_dtypes(include=['object','string']).columns

df = pd.get_dummies(df, columns=obj_cols, drop_first=True) #using one hot encoding to encode the str and object dtypes so it doesnt break our ANN
df=df.drop('id',axis=1) # drrop the id column cuz its meaningless
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

X= df.drop('num',axis=1).values #X are the inputs
y= df['num'].values #THis is the result

mean= X.mean(axis=0) #
std= X.std(axis=0)
std[std==0]=1

X= (X-mean)/std #normalized

X=X.T
y= y.reshape(1,-1)# reshaping for ANN

print("X shape:", X.shape)
print("Y shape:", y.shape)

if __name__ == "__main__":
    print(df.head())

X= X.T
y=y.T

X_train,y_train,X_test,y_test= train_test_split(
    X,y,test_size=0.2,random_state=42
)

X_train= X_train.T
y_train= y_train.T
X_test= X_test.T
y_test= y_test.T


def get_data():
    return X_train,y_train,X_test,y_test
