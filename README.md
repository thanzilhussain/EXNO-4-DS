# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/b14720cf-d418-4ded-8a4d-8af4e6455f66)


data.isnull().sum()

![image](https://github.com/user-attachments/assets/44c80427-49d3-4188-84bd-ee63b5515289)


missing=data[data.isnull().any(axis=1)]
missing

![image](https://github.com/user-attachments/assets/b416d769-05ca-4268-b2ce-d5de3fc24cdd)


data2=data.dropna(axis=0)
data2

![image](https://github.com/user-attachments/assets/4cc4db32-feeb-4979-a0ad-0c751560f729)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/3e0b2bf1-ab4f-485f-b97a-9bce3c66056c)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/754ac952-3ec1-4791-825e-10d0465f8cca)


data2

![image](https://github.com/user-attachments/assets/7c82d87d-72bb-4a55-b66e-27577faee095)


new_data=pd.get_dummies(data2, drop_first=True)
new_data

![image](https://github.com/user-attachments/assets/15e6b84c-a067-44e3-a1cd-05fb9991fb74)


columns_list=list(new_data.columns)
print(columns_list)

![image](https://github.com/user-attachments/assets/1e2a6145-7463-431b-a0e6-a19d3d1faba1)


features=list(set(columns_list)-set(['SalStat']))
print(features)

![image](https://github.com/user-attachments/assets/d1658578-892a-4bb0-8184-76f17493f5c4)


y=new_data['SalStat'].values
print(y)

![image](https://github.com/user-attachments/assets/16cf6c2c-0c6b-49ff-ab9b-1079a38c8d8e)


x=new_data[features].values
print(x)

![image](https://github.com/user-attachments/assets/d20b7e62-8fb7-4a9c-96cd-789d4635dd35)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/059e479c-812c-49b3-be90-b54f27f7ca56)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/91105127-7f9b-488c-b61b-857e88b4757e)


accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

![image](https://github.com/user-attachments/assets/232d62de-8db0-47c6-8fc0-347a7aa1a0d9)


print("Misclassified Samples : %d" % (test_y !=prediction).sum())

![image](https://github.com/user-attachments/assets/f24637af-34b6-46f7-b7f5-b9cbe499ebea)


data.shape

![image](https://github.com/user-attachments/assets/7b317c40-ef57-4c67-888f-f9471b9bc16b)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/42d6bc2d-878f-4843-8274-7164cd3b0d8f)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/bc6ea0c7-c571-45b8-9313-e1b6e75c62cc)


tips.time.unique()

![image](https://github.com/user-attachments/assets/de0b4148-3356-40db-acba-94e983ebfc95)


contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

![image](https://github.com/user-attachments/assets/7e13d2f7-ae6d-48d2-bb3f-c805cf8e7ee8)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/b5db2b1b-a206-4153-a8b2-8afaedac7d9a)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
