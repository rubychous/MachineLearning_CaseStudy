# load the libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# load dataset
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data=read_csv("the file path",delim_whitespace=True,names=names)

# analyze the data - descriptive statistics
data.shape # to check the dimensions of the data

data.dtypes # to check the types of each attribute

data.head(5)

set_option('precision',2) # Floating point output precision
data.describe()

# correlation between all numeric attributes
data.corr(method='pearson')

# insight:
# many attributes have a strong correlation either greater than 0.7 or less than -0.7
# For exampl. NOX and INDUS, DIS and INDUS, TAX and INDUS, AGE and NOX, DIS and NOX
# Furthermore, LSTAT has a negative correlation with MEDV which is our output variable

# data visualization
data.hist(sharex=False, sharey=False,xlabelsize=1,ylabelsize=1)
pyplot.show() # plot histogram to check data distribution

data.plot(kind='density',subplots=True,layout=(4,4),sharex=False,sharey=False,fontsize=1,legend=False)
pyplot.show() # plot density plot to check whether those attributes may suffer from skewed Gaussian distributions

data.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,fontsize=7)
pyplot.show() # plot box plot to check the outliers

scatter_matrix(data)
pyplot.show() # plot scatter plot to get the idea of how attributes interact with each other

# plot correlation matrix
fig=pyplot.figure()
axis=fig.add_subplot(111)
cax=axis.matshow(data.corr(),vmin=-1,vmax=1,interpolation='none')
figure.colorbar=(cax)
ticks=numpy.arange(0,14,1)
axis.set_xticks(ticks)
axis.set_yticks(ticks)
axis.set_xticklabels(names)
axis.set_yticklabels(names)
pyplot.show()

# validation dataset
# split training (80%) and testing data (20%)
array=data.values
X=array[:,0:13]
Y=array[:,13]
test_size=0.2
seed=7
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=test_size,random_state=seed)

# evaluate algorithms: baseline model
# by using MSE to grasp the idea that how wrong all predictions are
# Bascially, MSE aims to measure the difference and estimator and what is estimated
# Linear Regression, Lasso Regression, and ElasticNet are selected for Linear Algorithms
# CART, Support Vector Regression, and KNN are selected for Non-Linear Regression

number_folds=10
seed=7
scoring='neg_mean_squared_error'

models=[]
models.append(('LR',LinearRegression()))
models.append(('LASSO',Lasso()))
models.append(('EN',ElasticNet()))
models.append(('KNN',KNeighborsRegressor()))
models.append(('CART',DecisionTreeRegressor()))
models.append(('SVR',SVR()))

results=[]
names=[]
for name,model in models:
    kfold=KFold(n_splits=number_folds,random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    outcome="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(outcome)

# compare algorithms by plotting box plot
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# insight:
# CART has the tightest distribution
# However, the different scales of the data might have an influence on the outcomes
# such as KNN and SVR

# evaluate algorithms with standardization
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

results=[]
names=[]

for name, model in pipelines:
    kfold=KFold(n_splits=number_folds,random_state=seed)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    outcome="%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(outcome)

# compare algorithms by plotting box plot
fig=pyplot.figure()
fig.suptitle('Scaled_Comparison')
axis=fig.add_subplot(111)
pyplot.boxplot(results)
axis.set_xticklabels(names)
pyplot.show()

# insight:
# KNN has the tightest distribution and the lowest score

# improve results bu tuning KNN models
# the default neighbors in KNN is 7, by using grid search 
# to find out the different number of neighbors to improve scores

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=number_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# use ensemble method to improve the model
# boosting methods: AdaBoost, Gradient Boosting
# Bagging methods: Random Forest, Extra Tree

ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor())])))

results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=number_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Tune Ensemble methods by increasing the boosting stage to 400
# but it will take a lot of training time

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=number_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# finalize the model
# by using gradient boosting model to evaluate the validation dataset
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)

rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))