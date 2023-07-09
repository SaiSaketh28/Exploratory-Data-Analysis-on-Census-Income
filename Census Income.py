#!/usr/bin/env python
# coding: utf-8

# # 1. Exploratory data analysis

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
import os
import math
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection,preprocessing,linear_model,metrics


# # Loading the data

# In[2]:


df = pd.read_csv('adult.data.csv')
df.head()


# ### Columns description
# 
# 1. age: continuous.
# ----
# 2. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# -----
# 3. fnlwgt: continuous.
# ---------
# 4. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# ----------------
# 5. education-num: continuous.
# ------------------------------
# 6. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# ---------------------
# 7. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# -------------------------
# 8. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# -------------------
# 9. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# ---------------
# 10. sex: Female, Male.
# -----------------------
# 11. capital-gain: continuous.
# ---------------------
# 12. capital-loss: continuous.
# -------------------
# 13. hours-per-week: continuous.
# -----------------------------
# 14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# ----------------------
# 15.Income(Target Column) :<=50k and >50k

# In[4]:


df_all = df.copy()
df_all.head()


# ### Observation:
#     In this data we have the columnns as age,workclass,final weight-fnlwgt, education,education number, marital-status, occupation,relationship,race,sex,capital-loss, hours-per-week,native-country,income

# # Describing the data types of each column

# In[5]:


df.dtypes


# ### Finding length of each column and types in it

# In[6]:


for i in df.columns:
  print(i," : ",df[i].unique()," Length "," : ",len(df[i].unique()))


# ### Obseravtion:
#     * In education column we have  16 types as  Bachelors,  HS-grad,  11th,  Masters,  9th,  Some-college,
#     Assoc-acdm, Assoc-voc,  7th-8th,  Doctorate,  Prof-school, 5th-6th,  10th, 1st-4th,  Preschool,  12th
#     * In education-number column we have 16 types as 13,  9,  7, 14,  5, 10, 12, 11,  4, 16, 15,  3,  6,  2,  1,  8
#     * In marital-status column we have 7 types as Never-married, Married-civ-spouse,Divorced,Married-spouse-absent,
#     Separated,  Married-AF-spouse, Widowed

# In[7]:


print("Shape of Dataset: ",df.shape)


# ### Observation:
#     Shape of the data set is (32561,15) so there are 32561 rows and 15 columns

# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.nunique()


# ### Observation:
#      It gives the no of unique values for each feature

# # Finding the Null values

# In[11]:


df.isnull().sum()


# ### Observation:
# No null values are present in data

# In[12]:


colns = df.columns
print(colns)


# In[13]:


print("*"*25,"Numerical Column Names","*"*25)
num_colns=df._get_numeric_data().columns #List of numerical columns
print(num_colns)


print("*"*25,"Categorical Column Names","*"*25)
cat_colns=list(set(colns)-set(num_colns)) #List of categorical columns
print(cat_colns)


# ### Observation:
# 
#     Numerical columns are : age,final weight - fnlwgt, education-number,capital-gain,capital-loss,hours-per-week
# ----------------------
#     Categorical columns are : sex,workclass,income,race,occupation,relationship,education,marital-status,native-country

# In[14]:


df[num_colns].describe()


# # Replacing ? with value 'Nan'

# In[15]:


df['native-country'] = df['native-country'].replace(' ?',np.NaN)
df['workclass'] = df['workclass'].replace(' ?',np.NaN)
df['occupation'] = df['occupation'].replace(' ?',np.NaN)
df.head(30)


# In[16]:


df.isnull().sum()


# ### Observation:
#     * The Type with '?' value has been changed to 'Nan'

# In[17]:


df['workclass'].value_counts()


# In[18]:


df.isnull().sum()


# # Replacing null values with mode value

# In[19]:


for i in df.columns:
    df[i].fillna(df[i].mode()[0], inplace=True)
print(df)


# In[20]:


df.isnull().sum()


# In[21]:


df["income"].value_counts()


# In[22]:


print("range of age :",df.age.max()-df.age.min())


# In[23]:


print("range of fnlwgt :",df.fnlwgt.max()-df.fnlwgt.min())


# In[24]:


print("range of education-num :",df['education-num'].max()-df['education-num'].min())


# In[25]:


print("range of capital-gain :",df['capital-gain'].max()-df['capital-gain'].min())


# In[26]:


print("range of capital-loss  :",df['capital-loss'].max()-df['capital-loss'].min())


# In[27]:


print("range of hours-per-week  :",df['hours-per-week'].max()-df['hours-per-week'].min())


# # Data visualization

# ### Uni-variate analysis

# In[28]:


for i in df:
    plt.figure(figsize=(30, 6))
    sns.histplot(x=df[i])
    plt.show()


# # Observations:
#     AGE: maximum count of people are of the age  20- 50 
# -------------------------------------------------------------
# 
#     WORKCLASS: The count of people under private workcalss are more
# -----------------------------------------------------------------
# 
#     FINAL-WEIGHT: Count of 0.2 fnl wgt are more
# ------------------------------------------------------------
# 
#     EDUCATION: We can see 16 different categories and count of people coming under intermediate,bachelors, 
#     and some-college are more
# ------------------------------------------------------------
# 
#     EDUCATION-NUMBER : We observe that it is related with education (i.e they both describe the same)
# ---------------------------------------------------
# 
#     MARITAL-STATUS: There are  7 different categories. Married-civ-spouse count are more and Married-AF-spouse are less
# ------------------------------------------
# 
#     OCCUPATION : majority poeple are from Pro-Speciality column and least are from Armed-Forces
# -----------------------------------------------
# 
#     RELATIONSHIP :  There are 6 different categories and majority of the people are from 'Husband' 
#     category and least are from 'Other-relative'.
# -------------------------------------------
# 
#     RACE :  We observe that majority of the people are from 'White', and least are from 'others' category.
# ------------------------------------------------
# 
#     SEX : We observe that Male and female ratio is like 2:1
# --------------------------------------------------------
# 
#     CAPITAL-GAIN AND CAPITAL-LOSS : We observe that in both majority of the values are set to 0
# ---------------------------------------
# 
#     HOURS-PER-WEEK : Maximum count of people work for 40 hrs
# ----------------------------------------------
# 
#     NATIVE-COUNTRY : We observe the highest count is from 'United states'
# --------------------------------------------------
# 
#     INCOME : We observe majority  of the people have <=50k income  when compared to >50k.

# ### Bi and Multi-variate Analysis

# In[29]:


plt.figure(figsize=(22,10))

sns.countplot('education',hue ='income',data=df_all)
plt.title("Education count vs income",size=40)
plt.legend(labels=[' <=50K' ,' >50K'])
plt.xlabel("education ",size=16)
plt.ylabel("count",size=16)


# ### Observation: 
#     * From the above graph we see that the  number of people who have completed their HS-grad having the income <=50k are more and are of more than 8000 people
#     * The number of people who have completed their Bachelors having the income >50k are more and are of 2000 people
#     * The number of people who have completed their preschool having the income <=50k  and >50k are  very less
#     

# In[30]:


sns.countplot(df['income'], hue='age', data=df)
plt.title("age vs income")


# # Observations:
#     * we observe that peopple in the age group of 21 and 22 having an income of <=50k are more
#     * people in the age group of 75,76 are having very less income compared to others

# In[31]:


sns.barplot(x = 'income', y = 'age', data=df)
plt.title("age vs income")


# # Observation:
#     * We observe that the age group of 22-25 are earning less income when compared with the people of age group 
#         more than them like 30-40
#     * We observe that as the age increases i.e like seniority increases the income also increases

# In[32]:


sns.scatterplot(data=df, x= 'age', y='workclass', hue='income')
plt.title("age, workclass vs income")
plt.xlabel('age')
plt.ylabel('workclass')
plt.show()


# # Observation:
#      * We observe that the Local-gov people of age group 35 to 61 are earning >50k 
# -----------------
# 
#      * The self-emp-inc category the people of 40-85 age group earns income >50k 
# ----------------------
#      * The workclass of without-pay and never-worked  people are earning very less income compared to others

# In[33]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=df, x= 'age', y='fnlwgt', hue='income')
plt.title("fnlwgt, age vs income")
plt.xlabel('age')
plt.ylabel('fnlwgt')
plt.show()


# # Observation:
#     We observe that all the age groups are earning income based on their respective perspectives like 
#     education level which impacts on this
#     But the people earning >50k income are less compared with <=50k 

# In[34]:


plt.figure(figsize=(22,10))

sns.countplot('education-num',hue ='income',data=df_all) # education number defines Years of education
plt.title("Education number count vs income",size=40)
plt.legend(labels=[' <=50K' ,' >50K'])
plt.xlabel("education-number ",size=16)
plt.ylabel("count",size=16)


# ### Observation:
#     * The education number count of 9 having <=50k  income are more i.e greater than  8000
#     * The education number count of 1 having <=50k and >50k  income are very less 
#     * The education number count of 13 having >50k  income are more 
#    
# 
#     

# In[35]:


plt.figure(figsize=(22,10))

sns.countplot('marital-status',hue ='income',data=df_all)
plt.title("marital-status count vs income",size=40)
plt.legend(labels=[' <=50K' ,' >50K'])
plt.xlabel("marital-status ",size=16)
plt.ylabel("count",size=16)


# ### Observation:
#    *  Most of the people having <=50K income  were Never-married
#    *  Count of people  having >50k income are more in Married-civ-spouse category
#    *  Income of Married-AF-spouse people is very low compared to the reamining

# In[36]:


plt.figure(figsize=[22,16])
ax = sns.barplot(data = df, x = 'marital-status', y = 'education-num', hue = 'income')
ax.legend(loc = 8, ncol = 3, framealpha = 1, title = 'Income')
plt.title('Relation between marital-status,educationlevel and income')
plt.xlabel('marital-status')
plt.ylabel('Average of education-level')


# Observation: 
#     The people in never-married category having >50k income have their education level of above 12

# In[37]:


df1=df[df['education-num']==9]
df1


# In[38]:


df['hours-per-week'].value_counts()


# In[39]:


plt.figure(figsize=[30,6])
ax = sns.countplot(data = df[df['education-num']==9], x = 'marital-status' , hue = 'income')


# In[40]:


df[df["education"]==" HS-grad"]


# In[41]:


plt.figure(figsize=(8,6))
sns.countplot('sex',hue='income',data=df)
plt.title("Gender vs income",size=20)
plt.legend(labels=['<=50k','>50k'],title = 'income')
plt.xlabel("Gender",size=15)
plt.ylabel("count",size=15)


# # Observation:
#     * We observe that count of male persons is more than female and 
#      even income earning is also more in case of male compared to female

# In[42]:


ax=sns.distplot(df['capital-gain'],bins=10,kde=False,hist_kws=dict(edgecolor="k",linewidth=3))
ax.set_title('Histogram of Capital Gain')
ax.set_xlabel('Capital Gain')
plt.show()


# # Observation:
#     In capital gain column maximum values are set to 0.The distribution is right skewed

# In[43]:


ax=sns.distplot(df['capital-loss'],bins=10,kde=False,hist_kws=dict(edgecolor="k",linewidth=3))
ax.set_title('Histogram of Capital Loss')
ax.set_xlabel('Capital Loss')
plt.show()


# # Observation:
#     This Graph is also similar to above graph which is also right skewed

# In[44]:


plt.figure(figsize=(18,8))
sns.countplot('hours-per-week',hue='income',data=df)
plt.title("Income based on Working hours",size=20)
plt.legend(labels=['<=50k','>50k'],title = 'income')
plt.xlabel("Hours-per-week",size=15)
plt.ylabel("count",size=15)


# # Observation:
#     * We observe that people working approximately 40 hrs earns more income compared to others

# In[45]:


sns.countplot(df['income'], hue='relationship',data=df)
plt.title('income vs relationship')


# # Observation:
#     * We observe that mainly husband earns income either it may be <=50k or >50k
#     * The count of  person not in the family earning <=50k is more compared with >50k
#     * The other-relative people earning income count is very much less

# In[46]:


sns.countplot(df['income'], hue='race',data=df)
plt.title('income vs race')


# # Observation:
#     * We observe that people coming into white race type earn more income  and other type people earn very less income

# In[47]:


plt.figure(figsize=(22,10))

sns.countplot('education',hue ='education-num',data=df_all)
plt.title("eucation count vs education-num",size=40)
#plt.legend(labels=[' <=50K' ,' >50K'])
plt.xlabel("education ",size=16)
plt.ylabel("education-num",size=16)


# # Observations:
# ------------------------------------
# Here we can see that education and education-number have same values. They are correlated .
# The only thing in which they differ are education gives the categorical value ,while education-   number gives the numerical value.
# 
# ----------------------------------------
#     
# 1: Preschool, 2: 1st-4th, 3: 5th-6th, 4: 7th-8th, 5: 9th, 6: 10th, 7: 11th, 8: 12th, 9: HS-grad,
# 10: Some-college, 11: Assoc-voc, 12: Assoc-acdm, 13: Bachelors, 14: Masters, 15: Prof-school,     16:Doctorate
# 
# --------------------------------------------
#  *So if we want we can  drop education column from the data

# In[48]:



df.drop('fnlwgt', axis=1, inplace=True)
df.drop('relationship', axis=1, inplace=True)
df.drop('capital-gain', axis=1, inplace=True)
df.drop('capital-loss', axis=1, inplace=True)


# In[49]:


df.columns


# # 3. Outlier detection and skewness treatment

# In[50]:


df.skew(axis = 0, skipna = True)


# # Observation:
#     
# Positive – observed when the distribution has a thicker right tail and mode<median<mean.
# -->final weight,capital-gain,capital-loss comes under this category
# 
# ----------------
# 
# Negative – observed when the distribution has a thicker left tail and mode>median>mean.
# --> education number comes under this category
# 
# --------------------------------
# Zero (or nearly zero) – observed when the distribution is symmetric about its mean and approximately mode=median=mean
# --> age,hours-per-week comes under this category

# new=['fnlwgt','capital-gain','capital-loss']
# new

# for i in new:
#     #np.random.seed(0)
# 
# #create beta distributed random variable with 200 values
#     #df = np.random.beta(a=1, b=5, size=300)
# 
# 
#     df_sqr = np.sqrt(df[i])
#    
# 
# 
#     fig, axs = plt.subplots(nrows=1, ncols=2)
# 
# #create histograms
#     axs[0].hist(df[i], edgecolor='black')
#     axs[1].hist(df_sqr, edgecolor='black')
#   
# 
# #add title to each histogram
#     axs[0].set_title('Original Data')
#     axs[1].set_title('Square Root Transformed Data')
#    
#     

# In[51]:


new=['fnlwgt','capital-gain','capital-loss']
new


# for i in new:
#     
#     df_cube = np.cbrt(df[i])
#     
#    
# 
# 
#     fig, axs = plt.subplots(nrows=1, ncols=2)
# 
# #create histograms
#     axs[0].hist(df[i], edgecolor='black')
#     axs[1].hist(df_cube, edgecolor='black')
#   
# 
# #add title to each histogram
#     axs[0].set_title('Original Data')
#     axs[1].set_title('Cube Root Transformed Data')
#    
#     

# In[52]:


for i in  new:
    df_all[i]=np.cbrt(df_all[i])
df_all


# In[53]:


IQ1 = df_all.quantile(.25)
IQ3 = df_all.quantile(.75)
IQR = IQ3 - IQ1
print(IQR)
((df_all < (IQ1 - 1.5 * IQR)) | (df_all > (IQ3 + 1.5 * IQR))).sum()


# In[ ]:





# # 4. Encoding the data — Label Encoder

# In[54]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in df_all.columns:
    if(df_all[i].dtype == "object"):
        df_all[i] = le.fit_transform(df_all[i])
    else:
        continue
df_all


# # Observation:
#      All the string/object values  are converted into discrete 

# # 5. Scaling the data — Standard scaler

# In[55]:


'''from sklearn.preprocessing import StandardScaler
 
sc= StandardScaler()
 
# Splitting the independent and dependent variables
X = df_all[num_colns]



 
# standardization 
scale = sc.fit_transform(X)
print(scale)
scale.columns()
'''


# 
# categ_colns=df['income','marital-status', 'sex', 'relationship', 'workclass', 'race', 'native-country', 'occupation']
# categ_colns

# In[56]:


from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
#transformer.fit(X_train

X1=df_all.drop(['income','marital-status', 'sex', 'relationship', 'workclass', 'race', 'native-country','education', 'occupation'],axis=1)
x_numeric_log=pd.DataFrame(data=transformer.fit_transform(X1),columns=X1.columns)
x_log=pd.merge(x_numeric_log,df_all[cat_colns],left_index=True, right_index=True)
x=x_log[['workclass','education','marital-status','occupation','race','sex','native-country','age','hours-per-week','education-num']]
y=x_log['income']


# In[57]:


x


# In[58]:


x_log


# In[59]:


IQ1 = x_log.quantile(.25)
IQ3 = x_log.quantile(.75)
IQR = IQ3 - IQ1
print(IQR)
((x_log < (IQ1 - 1.5 * IQR)) | (x_log > (IQ3 + 1.5 * IQR))).sum()


# # 6. Fitting the machine learning models
# 

# ### Using Random Forest Classifier

# In[60]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3,random_state = 10)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


# In[61]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("Accuracy Score: ",accuracy_score(y_pred,y_test))


# In[62]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
y_predict_train = clf.predict(x_train)
y_predict_test = clf.predict(x_test)


print("*"*50,"Training Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_train,y_train))

print("*"*50)
print('Train Classification Report:')
print("*"*50)
print(classification_report(y_predict_train,y_train))

print("*"*50)
print('Train Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_train,y_train))


print("*"*50,"Testing Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_test,y_test))

print("*"*50)
print('Test Classification Report:')
print("*"*50)
print(classification_report(y_predict_test,y_test))

print("*"*50)
print('Test Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_test,y_test))


# ### Using Logistic Regression

# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred=logreg.predict(x_test)


# In[64]:


print("Accuracy Score: ",accuracy_score(y_pred,y_test))


# from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# y_predict_train = logreg.predict(x_train)
# y_predict_test = logreg.predict(x_test)
# 
# 
# print("*"*50,"Training Evaluation","*"*50)
# print("Accuracy Score: ",accuracy_score(y_predict_train,y_train))
# 
# print("*"*50)
# print('Train Classification Report:')
# print("*"*50)
# print(classification_report(y_predict_train,y_train))
# 
# print("*"*50)
# print('Train Confusion Matrix:')
# print("*"*50)
# print(confusion_matrix(y_predict_train,y_train))
# 
# 
# print("*"*50,"Testing Evaluation","*"*50)
# print("Accuracy Score: ",accuracy_score(y_predict_test,y_test))
# 
# print("*"*50)
# print('Test Classification Report:')
# print("*"*50)
# print(classification_report(y_predict_test,y_test))
# 
# print("*"*50)
# print('Test Confusion Matrix:')
# print("*"*50)
# print(confusion_matrix(y_predict_test,y_test))

# ### Using DecisionTree Classifier

# In[65]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

DTC = DecisionTreeClassifier()
DTC.fit(x_train, y_train)
y_pred=DTC.predict(x_test)


# In[66]:


print("Accuracy Score: ",accuracy_score(y_pred,y_test))


# In[67]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
y_predict_train = DTC.predict(x_train)
y_predict_test = DTC.predict(x_test)


print("*"*50,"Training Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_train,y_train))

print("*"*50)
print('Train Classification Report:')
print("*"*50)
print(classification_report(y_predict_train,y_train))

print("*"*50)
print('Train Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_train,y_train))


print("*"*50,"Testing Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_test,y_test))

print("*"*50)
print('Test Classification Report:')
print("*"*50)
print(classification_report(y_predict_test,y_test))

print("*"*50)
print('Test Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_test,y_test))


# ### Using SVM

# In[68]:


from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train)  


# In[69]:


y_pred= classifier.predict(x_test)  

print("Accuracy Score: ",accuracy_score(y_pred,y_test))


# In[70]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
y_predict_train = classifier.predict(x_train)
y_predict_test = classifier.predict(x_test)


print("*"*50,"Training Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_train,y_train))

print("*"*50)
print('Train Classification Report:')
print("*"*50)
print(classification_report(y_predict_train,y_train))

print("*"*50)
print('Train Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_train,y_train))


print("*"*50,"Testing Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_test,y_test))

print("*"*50)
print('Test Classification Report:')
print("*"*50)
print(classification_report(y_predict_test,y_test))

print("*"*50)
print('Test Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_test,y_test))


# ### Using KNN

# In[71]:


from sklearn.neighbors import KNeighborsClassifier  
K= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
K.fit(x_train, y_train)  


# In[72]:


y_pred= K.predict(x_test)  

print("Accuracy Score: ",accuracy_score(y_pred,y_test))


# In[73]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
y_predict_train = K.predict(x_train)
y_predict_test = K.predict(x_test)


print("*"*50,"Training Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_train,y_train))

print("*"*50)
print('Train Classification Report:')
print("*"*50)
print(classification_report(y_predict_train,y_train))

print("*"*50)
print('Train Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_train,y_train))


print("*"*50,"Testing Evaluation","*"*50)
print("Accuracy Score: ",accuracy_score(y_predict_test,y_test))

print("*"*50)
print('Test Classification Report:')
print("*"*50)
print(classification_report(y_predict_test,y_test))

print("*"*50)
print('Test Confusion Matrix:')
print("*"*50)
print(confusion_matrix(y_predict_test,y_test))


# l=[]
# l1=[]
# for i in range(1,31):
#     if i%2!=0:
#         l1.append(i)
#         classifier = KNeighborsClassifier(n_neighbors=i)
#         classifier.fit(x_train, y_train)
#         y_predict = classifier.predict(x_test)
#         a=accuracy_score(y_test,y_predict)
#         l.append(a)
# sns.barplot(l1,l)
# plt.show()

# # OBSERVATIONS:
#     ACCURACY OF EACH ALGORITHM
#     Random forest - 0.854
#     Logistic Regression - 0.798
#     Decision Tree - 0.812
#     SVM - 0.806
#     KNN - 0.823

# from sklearn.decomposition import PCA
# pca=PCA(n_components=2)
# x_train_scaled=pca.fit_transform(x_train)
# x_test_scaled=pca.fit_transform(x_test)
# clf.fit(x_train_scaled, y_train)
# y_pred=clf.predict(x_test_scaled)
# print('Accuracy score for test data ',accuracy_score(y_test,y_pred))
# print('Accuracy score for train data ',accuracy_score(y_train,clf.predict(x_train_scaled)))
# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))

# # 7. Cross-validation of the selected model

# In[74]:


from sklearn import datasets

from sklearn.model_selection import KFold, cross_val_score



k_folds = KFold(n_splits = 5)

scores = cross_val_score(clf, x, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# # 8. Model hypertuning

# In[1]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# # ABOUT PARAMETERS
# * n_estimators ---> The number of trees in the forest.
# 
# 
# 
# * criterion{“gini”, “entropy”, “log_loss”}, default=”gini”--->The function to measure the quality of a split. 
# 
# * max_depthint, default=None
# * The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less    than min_samples_split samples.
# 
# * min_samples_splitint or float, default=2 ---->The minimum number of samples required to split an internal node
# 
# 
# 
# * max_features{“sqrt”, “log2”, None}, int or float, default=”sqrt”
# * The number of features to consider when looking for the best split:
# 
# * If int, then consider max_features features at each split.
# 
# * If float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.
# 
# * If “auto”, then max_features=sqrt(n_features).
# 
# * If “sqrt”, then max_features=sqrt(n_features).
# 
# * If “log2”, then max_features=log2(n_features).
# 
# * If None, then max_features=n_features.

# In[2]:


from sklearn.model_selection import GridSearchCV
CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
CV_clf.fit(x_train, y_train)


# In[77]:


print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",CV_clf.best_estimator_)
print("\n The best score across ALL searched params:\n",CV_clf.best_score_)
print("\n The best parameters across ALL searched params:\n",CV_clf.best_params_)


# In[78]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[79]:


def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[80]:


y_prob = clf.predict_proba(x_test)[:,1]
plot_roc_curve(y_test,y_prob)


# In[81]:


import pickle
from pickle import dump
dump(clf, open('clf.pkl', 'wb'))


# import joblib
# joblib.dump(le,open('le.joblib','wb'))

# In[82]:


with open('le.pkl', 'wb') as f:
    pickle.dump(le, f)


# In[83]:


df.columns


# In[ ]:





# In[ ]:




