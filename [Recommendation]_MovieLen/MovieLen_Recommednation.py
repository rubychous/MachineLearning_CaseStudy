# Movie Recommendation using Python
# the dataset were collected by the GroupLens Research Project at the University of Minnesota

import numpy as np 
import pandas as pd 

# load the dataset
url="http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
dataset=pd.read_csv(url,sep="\t",header=None,names=["userId","itemId","rating","timestamp"])
# have a quick view about the dataset
dataset.head(3)

movieInfoUrl="http://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movieInfoData=pd.read_csv(movieInfoUrl,sep="|",header=None,index_col=False,names=["itemId","title"],usecols=[0,1])
movieInfoData.head(3)

# add movie title to dataset table
# use itemId as a key
dataset=pd.merge(dataset,movieInfoData,left_on='itemId',right_on="itemId")
dataset.head()

# check the data type
# it is a Pandas series object
userId=dataset.userId
type(userIds)
# it is a Pandas dataframe object
userId2=dataset[["userId"]]
type(userIds2)
# both of them are same, but dataframe object will show column name

# give loc a list of row indices and a list of column names 
# similarily, it works as dataset.loc[0:10,"userId"]
# however, the object type would be series
dataset.loc[0:10,["userId"]]

# select the subset of Toy Story
toyStoryUsers=dataset[dataset.title=="Toy Story (1995)"]

# use sort_value function in dataframe
dataset=pd.DataFrame.sort_values(dataset,["userId","itemId"],ascending=[0,1])

# check how many movies and users
numUsers=max(dataset.userId)
numMovies=max(dataset.itemId)

# check how many movies were rated by each user
moviesPerUser=dataset.userId.value_counts()
# the number of users that rated each movie
usersPerMovie=dataset.title.value_counts()

# define a function to find top n favorite for a user
def favoriteMovie(user,n):
	topMovie=pd.DataFrame.sort_values(dataset[dataset.userId==user],["rating"],ascending=[0])[:n]
	return list(topMovie.title)

# start working on recommendation
# by using neigbour based collaborative filtering model
# this recommendation aims to find the k nearest neighbours of a user and use their rating to predict
# the user for movies they haven't rated

# initial a vector to represent each individual user
userItemRatingMatrix=pd.pivot_table(dataset,values="rating",index=["userId"],columns=["itemId"])
userItemRatingMatrix.head(3)

# define a function to find a similarity between 2 users
from scipy.spatial.distance import correlation
def similarity(user1,user2):
	# normalize user1 by the mean rating without taking nan values into consideration
	user1=np.array(user1)-np.nanmean(user1)
	user2=np.array(user2)-np.nanmean(user2)
	# set a condition to make sure user1 and user2 have in common
	commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
	if len(commonItemIds)==0:
		return 0
	else:
		user1=np.array([user1[i] for i in commonItemIds])
		user2=np.array([user2[i] for i in commonItemIds])
		return correlation(user1,user2)

# find the nearest neighbors of the user by applying similarity function that defined previously
def nearestNeighborRatings(user,k):
	# initialize an empty dataframe whose row index is userIds and 
	# the value is the similarity of that user to the other one
	similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=["similarity"])
	for i in userItemRatingMatrix.index:
		similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[user],userItemRatingMatrix.loc[i])
	# sort the similarity matrix in the descending order 
	similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,["similarity"],ascending=[0])
	# sort the similarity matrix in a descending order of similarity
	nearestNeighbours=similarityMatrix[:k]
	# take the k nearest neighbors and use their rating to predict the user rating
	neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
	predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns,columns=["Rating"])
	for i in userItemRatingMatrix.columns:
		# start with average rating of the user
		predictedRating=np.nanmean(userItemRatingMatrix.loc[user])
		for j in neighbourItemRatings.index:
			if userItemRatingMatrix.loc[j,i]>0:
				predictedRating+=(userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'similarity']
		predictItemRating.loc[i,"Rating"]=predictedRating
	return predictItemRating

# define a function to find the top n recommendation for the user
def topNRecommendation(user,n):
	# use 10 nearest neighbors to find the predicted ratings
	predictItemRating=nearestNeighborRatings(user,10)
	moviesAlreadyWatched=list(userItemRatingMatrix.loc[user].loc[userItemRatingMatrix.loc[user]>0].index)
	predictItemRating=predictItemRating.drop(moviesAlreadyWatched)
	topRecommendations=pd.DataFrame.sort_values(predictItemRating,["Rating"],ascending=[0])[:n]
	topRecommendationTitles=(movieInfoData.loc[movieInfoData.itemId.isin(topRecommendations.index)])
	return list(topRecommendationTitles.title)

# give the recommendations for a user
# to identify what sorts of factors influence a user's rating
# the factors are identified by decomposing the user item rating matrix into 
# user-factor matrix and item-factor matrix
def matrixFactorization(R, K, steps=10, gamma=0.001,lamda=0.02):
    N=len(R.index)# Number of users
    M=len(R.columns) # Number of items 
    P=pd.DataFrame(np.random.rand(N,K),index=R.index)
    
    Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
     
    for step in xrange(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0: 
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        e=0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    #Sum of squares of the errors in the rating
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        print step
    return P,Q

# call the function
(P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02, steps=100)

