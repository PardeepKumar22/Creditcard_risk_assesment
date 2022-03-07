#!/usr/bin/env python
# coding: utf-8

# In[59]:


#import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[60]:


data=pd.read_csv('desktop/Credit_default_dataset.csv')


# In[61]:


data.head()


# In[62]:


data.describe()


# In[63]:


data.info()


# In[64]:


data.tail()


# In[65]:


data.shape


# In[66]:


data=data.drop(["ID"],axis=1)


# In[67]:


#changing the name of  pay_0 column to pay_1 to make the numbering correct
data.rename(columns={'PAY_0':'PAY_1'}, inplace=True)


# In[68]:


data['EDUCATION'].value_counts()


# In[69]:


data['EDUCATION'].value_counts().plot.bar()


# In[70]:


data['MARRIAGE'].value_counts()


# In[71]:


data['MARRIAGE'].value_counts().plot.bar()


# In[72]:


data.head()


# In[73]:


#data preprocessing
data["EDUCATION"]=data["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
data["MARRIAGE"]=data["MARRIAGE"].map({0:3,1:1,2:2,3:3})


# In[74]:


data.head()


# In[75]:


from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
X=data.drop(['default.payment.next.month'],axis=1)
X=scaling.fit_transform(X)


# In[76]:


y=data['default.payment.next.month']


# In[77]:


## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[78]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[79]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[80]:


classifier=xgboost.XGBClassifier()


# In[81]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[82]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time) # timing ends here for "start_time" variable


# In[83]:


random_search.best_estimator_


# In[84]:


random_search.best_params_


# In[85]:


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[87]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)


# In[ ]:


score


# In[ ]:


score.mean()


# In[ ]:




