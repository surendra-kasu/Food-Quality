#import libraries 
import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
# from sklearn.externals import joblib
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')

#data selection 
df=pd.read_csv('rating_final.csv')

#visulaization for ratings 
plt.figure(figsize = (6,5))
sns.countplot(df['rating'])
 
#check the uniqueness 
print('Number of unique users in Raw data = ', df['userID'].nunique())
print('Number of unique product in Raw data = ', df['name'].nunique())

#display the top 10 places 
most_rated=df.groupby('name').size().sort_values(ascending=False)[:10]
print('Top 10 place based on ratings: \n',most_rated)

#check count 
counts=df.userID.value_counts()
df1_final=df[df.userID.isin(counts[counts>=15].index)]
print('Number of users who have rated 15 or more items =', len(df1_final))
print('Number of unique users in the final data = ', df1_final['userID'].nunique())
print('Number of unique products in the final data = ', df1_final['userID'].nunique())


#constructing the pivot table
final_ratings_matrix = df1_final.pivot(index = 'userID', columns ='placeID', values = 'rating').fillna(0)
final_ratings_matrix.head()
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)


#Calucating the density of the rating marix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))


#data splitting 
train_data, test_data = train_test_split(df1_final, test_size = 0.3, random_state=0)
train_data.head()
print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)


#Count of user_id for each unique product as recommendation score 
train_data_grouped = train_data.groupby('name').agg({'userID': 'count'}).reset_index()
train_data_grouped.rename(columns = {'userID': 'score'},inplace=True)
train_data_grouped.head(40)

#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'name'], ascending = [0,1]) 
      
#Generate a recommendation rank based upon score 
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
          
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head(5) 
popularity_recommendations 

# Use popularity based recommender model to make predictions
def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userID'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations 


print("-------------Recommendation Based On Score------------------")
find_recom = [5]   # This list is user choice.
for i in find_recom:
    print("The list of recommendations for the userId: %d\n" %(i))
    print(recommend(i))    
    print("\n") 


#Building Collaborative Filtering recommender model
electronics_df_CF = pd.concat([train_data, test_data]).reset_index()
electronics_df_CF.head()   

# Matrix with row per 'user' and column per 'item' 
pivot_df = electronics_df_CF.pivot(index = 'userID', columns ='placeID', values = 'rating').fillna(0)
pivot_df.head()    

print('Shape of the pivot table: ', pivot_df.shape)

#define user index from 0 to 10
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
pivot_df.head()

pivot_df.set_index(['user_index'], inplace=True)
# Actual ratings given by users
pivot_df.head()

# Singular Value Decomposition
U, sigma, Vt = svds(pivot_df, k = 1)
print('Left singular matrix: \n',U)
print('Sigma: \n',sigma)

# Construct diagonal array in SVD
sigma = np.diag(sigma)
print('Diagonal matrix: \n',sigma)

print('Right singular matrix: \n',Vt)

#Predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
# Convert predicted ratings to dataframe
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)
preds_df.head()



from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#data splitting 
data_1 = pd.read_csv('rating_final.csv')
X =data_1.drop(["rating_1","userID","name"],axis=1)
Y = data_1['rating']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

#classification using logistic regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
print('---Training Accuracy For Logistic Regression--')
print('----------------Classification Report-----------------------------------------')
print(classification_report(y_train,y_pred))
print('---------------------------')
accuracy=accuracy_score(y_train, y_pred)*100
print(("accuracy for logisticregression",accuracy))






