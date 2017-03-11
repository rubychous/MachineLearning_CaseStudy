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

pipeline = []
pipeline.append(("ScaledLR",Pipeline([("Scaler",StandardScaler()),("LR",LogisticRegression())])))
pipeline.append(("ScaledLDA",Pipeline([("Scaler",StandardScaler()),("LDA",LinearDiscriminantAnalysis())])))
pipeline.append(("ScaledKNN",Pipeline([("Scaler",StandardScaler()),("KNN",KNeighborsClassifier())])))
pipeline.append(("ScaledCART",Pipeline([("Scaler",StandardScaler()),("CART",DecisionTreeClassifier())])))
pipeline.append(("ScaledNB",Pipeline([("Scaler",StandardScaler()),("NB",GaussianNB())])))
pipeline.append(("ScaledSVM",Pipeline([("Scaler",StandardScaler()),("SVC",SVC())])))

results=[]
names=[]

for name, model in pipeline:
	kfold=KFold(n_splits=number_fold,random_state=seed)
	cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg="%s %f (%f)"%(name,cv_results.mean(),cv_results.std())
	print(msg)

# plot boxplot to compare algorithms again
fig = pyplot.figure()
fig.suptitle("Scaled Algorithm Comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Algorithm tuning - KNN
# try a range of different number of neighbors(k) from 1 to 21
# each k is evaluated by using 10-fold cross validation on training standardized data

scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
neighbors=[1,3,5,7,9,11,13,15,17,19,21]
param_grid=dict(n_neighbors=neighbors)
model=KNeighborsClassifier()
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result=grid.fit(rescaledX,Y_train)
print("Best: %f using %s"%(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_["mean_test_score"]
stds=grid_result.cv_results_["std_test_score"]
params=grid_result.cv_results_["params"]
for mean, stdev, param in zip(means,stds,params):
	print("%f (%f) with %r"%(mean,stdev,param))

# Algorithm tuning -SVM
# in this section, 2 parameters will be tuned manually. Namely, c value and types of kernels
# c value = how much to relax the margin
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
c_values=[0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,1.9,2.0]
kernel_values=["linear","poly","rbf","sigmoid"]
param_grid=dict(C=c_values,kernel=kernel_values)
model=SVC()
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result=grid.fit(rescaledX,Y_train)
print("Best: %f using %s"%(grid_result.best_score_,grid_result.best_params_))

means=grid_result.cv_results_["mean_test_score"]
stds=grid_result.cv_results_["std_test_score"]
params=grid_result.cv_results_["params"]

for mean,stdev,param in zip(means,stds,params):
	print("%f (%f) with %r"%(mean,stdev,param))

# ensemble methods
# boosting methods: AdaBoost(AB), Gradient Boosting(GBM)
# bagging methods: RandomForest(RF), Extra Tree(ET)

ensembles=[]
ensembles.append(("AB",AdaBoostClassifier()))
ensembles.append(("GBM",GradientBoostingClassifier()))
ensembles.append(("RF",RandomForestClassifier()))
ensembles.append(("ET",ExtraTreesClassifier()))

results=[]
names=[]

for name, model in ensembles:
	cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg="%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
	print(msg)

# compare algorithms
fig=pyplot.figure()
fig.suptitle("Ensemble Algorithm Comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# finalize the model
# apply SVM to the training set and make predictions for validation data to confirm the finding

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model=SVC(C=1.5)
model.fit(rescaledX,Y_train)

# estimate accuracy on validation data
rescaledValidationX=scaler.transform(X_validation)
predictions=model.predict(rescaledValidationX)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))