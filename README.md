<H3>NAME : SRIVATSAN G </H3>
<H3>REGISTER NO : 212223230216</H3>
<H3>EX. NO.1</H3>
<H3>DATE :           </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```python
import pandas as pd                  
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
print(df)

x = df.iloc[:, :-1].values
x

y = df.iloc[:, -1].values
y

print(df.isnull().sum())

df.duplicated()

df.describe()

df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)

scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train)
print(len(x_train))

print(x_test)
print(len(x_test))

```

## OUTPUT:
#### DATASET PREVIEW:

<img width="550" height="660" alt="image" src="https://github.com/user-attachments/assets/89563b20-ef19-4691-ae9b-b7f94d589e0a" />

#### FEATURE MATRIX:

<img width="551" height="140" alt="image" src="https://github.com/user-attachments/assets/9128f493-4c08-4e4a-8b1f-db7b4495836a" />

#### TARGET VECTOR:

<img width="340" height="42" alt="image" src="https://github.com/user-attachments/assets/89b94a3c-7520-4eec-aa26-ca05ad620bb7" />

#### CHECK FOR MISSING VALUES:

<img width="210" height="258" alt="image" src="https://github.com/user-attachments/assets/462c4b17-093a-4fcc-bf05-58f72ef9d15c" />

#### CHECK FOR DUPLICATE VALUES:

<img width="206" height="223" alt="image" src="https://github.com/user-attachments/assets/71cc8d90-5289-4a0b-b5de-0d2c1344b641" />

#### DATASET STATISTICAL SUMMARY:

<img width="1012" height="219" alt="Screenshot 2025-09-14 114921" src="https://github.com/user-attachments/assets/87be2565-44f5-4707-9334-3a999e0ae083" />

#### NORMALIZED DATASET:

<img width="601" height="441" alt="image" src="https://github.com/user-attachments/assets/879f808b-a74a-4201-8070-938906b6f32d" />

#### TRAINING DATA:

<img width="397" height="138" alt="image" src="https://github.com/user-attachments/assets/fbbc1195-dddd-4ef6-8c58-a90a368c0067" />

#### TESTING DATA:

<img width="381" height="142" alt="image" src="https://github.com/user-attachments/assets/bcf5a1e5-32c0-4f85-937e-569208c1ed49" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
