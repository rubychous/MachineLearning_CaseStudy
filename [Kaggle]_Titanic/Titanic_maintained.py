# Kaggle Titanic
# Ref: https://www.kaggle.com/arthurtok/titanic/introduction-to-ensembling-stacking-in-python/notebook 

# Define the problem: predict what kinds of people were likely to survive
	# it is a binary classification problem (supervised ML problem)
	# suitable algorithms might be: Logistic Regression, KNN, SVM, Naive Bayes Classifier, Tree, LDA
	# some ensemble methods might be good to try: AdaBoost, GradientBoosting, RandomForest, ExtraTrees

# load libraries
import re
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

# load the dataset
train_data=pd.read_csv("C:/Users/LUYI/Downloads/[Kaggle]_Titanic/train.csv")
test_data=pd.read_csv("C:/Users/LUYI/Downloads/[Kaggle]_Titanic/test.csv")

# summarize the data (descriptive statistics and visualization)
train_data.head(3)

train_data.shape
train_data.dtypes
# note: 12 features are included in the train data included the targeted variable(survived)

# check the missing value
train_data.isnull().sum().sort_values(ascending=False)
test_data.isnull().sum().sort_values(ascending=False)
# note: lots of missing value in cabin and age for both train and test data
# 		considering both train and test is not a huge dataset, 
#		filling those missing value would be applied instead of getting rid of them

# join both training and testing data together in order to fill the missing value
combine=pd.concat([train_data,test_data],axis=0) # axis = 0 indicates row

combine["Embarked"].unique()
combine["Embarked"].isin(["S"]).sum() 
combine["Embarked"].isin(["C"]).sum()
combine["Embarked"].isin(["Q"]).sum()

# since there were only 2 missing value(embarked) in training set and 1 missing value(fare) in testing set
# assign embarked "S" to those missing value since most people boarded there
combine["Embarked"].fillna("S",inplace=True)

# fill fare value with median 
combine["Fare"].fillna(combine["Fare"].median(),inplace=True)

# create a new column
# using regular expression to search for a title
combine["Title"]=combine["Name"].apply(lambda x :re.search("([A-Za-z]+)\.", x).group(1))
# followed by assigning title id to different titles
# assign id 10 for royality
# assign id 7 for officer
# assign id 8 for french
combine["Title"]=combine["Title"].replace(["Dona","Countess","Lady","Jonkheer"],"10")
combine["Title"]=combine["Title"].replace(["Mlle","Mme"],"8")
combine["Title"]=combine["Title"].replace(["Major","Col","Don","Sir","Capt"],"7")
combine["Title"]=combine["Title"].replace(["Rev"],"6")
combine["Title"]=combine["Title"].replace(["Dr"],"5")
combine["Title"]=combine["Title"].replace(["Master"],"4")
combine["Title"]=combine["Title"].replace(["Mrs"],"3")
combine["Title"]=combine["Title"].replace(["Miss","Ms"],"2")
combine["Title"]=combine["Title"].replace(["Mr"],"1")

# clean the cabin info
# fill in 0 for missing value
# if there is no missing value, assign dummy codes for each distinct Deck ie.A=1 B=2, etc.
combine["Cabin"]=combine["Cabin"].fillna("0")
combine["Cabin"]=combine["Cabin"].apply(lambda x: x[0])
combine["CabinCat"]=pd.Categorical.from_array(combine["Cabin"]).codes

# assign dummy codes for embarked column as well
combine["EmbarkedCat"]=pd.Categorical.from_array(combine["Embarked"]).codes

# clean ticket info
def CleanTickets(ticket):
	ticket=ticket.replace("/","").replace(".","")
	if (len(ticket)>0) and (ticket.isdigit()==False):
		return ticket[0]
	else:
		return "XXXX"

combine["Ticket"]=combine["Ticket"].map(CleanTickets)
combine["TicketCat"]=pd.Categorical.from_array(combine["Ticket"]).codes

# assign dummy code to sex and move survive to the last column
whole_data=pd.concat([combine.drop(["Survived"],axis=1),pd.get_dummies(combine["Sex"],prefix="Sex"),combine["Survived"]],axis=1)

# feature engineering
# find the size of family for individual passenger
whole_data["FamilySize"]=whole_data["SibSp"]+whole_data["Parch"]

# find the length of name for individual passenger
whole_data["NameLength"]=whole_data["Name"].apply(lambda x: len(x))

# family mapping
# do not understand that part
# and family size

# assign the age of child to 14
child_age = 14