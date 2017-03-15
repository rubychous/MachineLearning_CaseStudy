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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# load the dataset
train_data=pd.read_csv("C:/Users/LUYI/Downloads/[Kaggle]_Titanic/train.csv",dtype={"Age": np.float64})
test_data=pd.read_csv("C:/Users/LUYI/Downloads/[Kaggle]_Titanic/test.csv",dtype={"Age": np.float64})

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


family_id_mapping = {}
def get_family_id(row):
    #get last surname from name col
    last_name = row["Name"].split(",")[0]
    #Assign family_id in following format last_namefamilysize
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    #if the above found family id is new then enter loop
    if family_id not in family_id_mapping:
        #for the very first iteration of loop
        if len(family_id_mapping) == 0:
            current_id = 1
        #from second iteration    
        else:
            #get the highest id value found in family_id_mapping and add one to it to get the new id
            #example if family_id_mapping has {1,3,4,2} pick 4 then 4+1=5 is current_id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id  # in this step, it assigns value to key. It is how it works in dict
    return family_id_mapping[family_id]

#get all family id's
family_ids = whole_data.apply(get_family_id, axis=1)
#to minimize the family id's group small families into one and assign -1 as their id
family_ids[whole_data["FamilySize"] < 3] = -1
whole_data["FamilyId"] = family_ids


# classify passengers based on age
# assign the age of child to 14
child_age = 14

def get_person(passenger):
	age, sex=passenger
	if age<child_age:
		return "child"
	elif (sex=="female"):
		return "female_adult"
	else:
		return "male_adult"

# use get_person function and store the return value to the new created person column
# convert 3 different types of classification result to dummy variable
# notice the difference between pd.Categorical.codes and get_dummies
whole_data = pd.concat([whole_data, pd.DataFrame(whole_data[["Age", "Sex"]].apply(get_person, axis=1), columns=["person"])],axis=1)
dummies = pd.get_dummies(whole_data["person"])
whole_data = pd.concat([whole_data,dummies],axis=1)

# add new feature for surname
whole_data["surname"] = whole_data["Name"].apply(lambda x: x.split(",")[0].lower())

# find family names of females who perished
perishing_female_surnames = list(set(whole_data[(whole_data["female_adult"] == 1.0) & (whole_data["Survived"] == 0.0) &((whole_data["Parch"] > 0) | (whole_data["SibSp"] > 0))]["surname"].values))

# find perishing mothers
def perishing_mother(passenger):
	surname, Pclass, person =passenger
	if (surname in perishing_female_surnames):
		return 1.0
	else:
		return 0.0

whole_data["perishing_mother"] = whole_data[["surname", "Pclass", "person"]].apply(perishing_mother, axis=1)

# find family name of males who survive
perishing_male_surnames = list(set(whole_data[(whole_data["male_adult"] == 1.0) & (whole_data["Survived"] == 1.0) &((whole_data["Parch"] > 0) | (whole_data["SibSp"] > 0))]["surname"].values))

# find survival fathers
def survival_father(passenger):
	surname, Pclass, person = passenger
	if (surname in perishing_male_surnames):
		return 1.0
	else:
		return 0.0

whole_data["survival_father"] = whole_data[["surname", "Pclass", "person"]].apply(survival_father, axis=1)

# find missing age by using Extra Tree Regressor
classers = ["Fare","Parch","Pclass","SibSp","Title","CabinCat","Sex_female","Sex_male", "EmbarkedCat", "FamilySize", "NameLength","FamilyId"]

# fill in missing value(age)
X_train_age=whole_data.ix[whole_data["Age"].notnull(),classers]
Y_train_age=whole_data.ix[whole_data["Age"].notnull(),"Age"]
X_test_age=whole_data.ix[whole_data["Age"].isnull(),classers]



# use ensemble method: ElasticNet to fill age 
# not tune yet
scaler = StandardScaler().fit(X_train_age)
rescaledX = scaler.transform(X_train_age)
model=ElasticNet()
model.fit(X_train_age,Y_train_age)
predictions = model.predict(X_test_age)
whole_data.ix[whole_data["Age"].isnull(),"Age"] = predictions


# build model
model_dummys = ['Age','male_adult','female_adult', 'child','perishing_mother','survival_father','Fare','Parch','Pclass','SibSp','Title','CabinCat','Sex_female','Sex_male', 'EmbarkedCat','FamilySize', 'NameLength', 'FamilyId']

# split data
X_data = whole_data.iloc[:891,:]
X_train = X_data.loc[:,model_dummys]

Y_data = whole_data.iloc[:891,:]
Y_train = Y_data.loc[:,"Survived"]

X_t_data = whole_data.iloc[891:,:]
X_test = X_t_data.loc[:,model_dummys]


scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model=SVC(C=0.1,kernel='linear')
model.fit(rescaledX,Y_train)
predictions = model.predict(Y_test)

predictions = [str(int(x)) for x in predictions]
submission = pd.DataFrame()
submission['PassengerId'] = X_t_data.PassengerId
submission['Survived'] = predictions
submission.set_index(['PassengerId'],inplace=True)
submission.head(3)
submission.to_csv('titanic_submission.csv')  

