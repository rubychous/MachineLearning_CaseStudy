# load libraries
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# load dataset
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
data=read_csv(URL,header=None)

# descriptive statistics
data.shape

set_option("display.max_row",100) 
data.dtypes # look data types for all attributes
data.head(30)

set_option("precision",3) # floating point output precision
data.describe() # summarize the distribution of all attributes, such as mean, std, min, max etc.

# insight:
# the data are not at the same range, especially a wide range of different average value
# the standardizing method can be applied

data.groupby(60).size() # class distribution

# insight:
# classes are balancing between R and M

# visualization
# histogram of each attributes
data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

# density plot of each attributes
data.plot(kind="density",subplots=True,layout=(8,8),sharex=False,legend=False,fontsize=1)
pyplot.show()

# box plot
data.plot(kind="box",subplots=True,layout=(8,8),sharex=False,sharey=False,fontsize=1)
pyplot.show()

# check correlations between the attribute
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(),vmin=-1,vmax=1,interpolation="none")
fig.colorbar(cax)
pyplot.show()

# split validation data to confirm the accuracy of the final model
# in this case, 80% for mmodel and 20% for validation
array = data.values
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)

# evaluate algorithms : baseline
# apply 10-fold cross validation
# evaluate algorithms by using accuracy metric

number_fold = 10
seed = 7
scoring = "accuracy"

# create a baseline performance and spot-check different algorithms
# linear algorithms: LR, LDA
# non-linear algorithms: CART,SVM, NB, KNN

models=[]
models.append(("LR",LogisticRegression()))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("CART",DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))
models.append(("SVM",SVC()))

results=[]
names=[]
for name, model in models:
	kfold = KFold(n_splits=number_fold,random_state=seed)
	cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
	print(msg)

# compare algorithms
fig = pyplot.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# insight:
# KNN has the tighest distribution implying low variance 
# and the poor result for SVM is abnormal
# perhaps varied distribution of attriutes is having an influence on the accuracy of the SVM algorithm

# evaluate algorithms: standardize data
# evaluate the same algorithms again with standardized copy of the data
# because differing scaling raw data might have a negatively impact on the skill of some algorithms
# the data is transformed with mean value of zero and std with 1
# apply pipeline to aviod data leakage when transform the data