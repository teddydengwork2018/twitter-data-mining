from sklearn.model_selection import train_test_split
import simplejson as json
import math
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

from sklearn.metrics import mean_squared_error

import matplotlib.patches as mpatches

from sklearn.model_selection import permutation_test_score

import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std
#import data
tweets=[]
hr = []
fr = []
rt = []
mfr = []
totd = []
flag = 0
coutfr = 0
coutrt = 0
tmfr = 0
febhr1 = math.floor(1422802800/60/60)
febhr2 = math.floor(1422846000/60/60)
countsec1 = 0
countsec2 = 0
countsec3 = 0
#f = open("tweets_#nfl.txt","r")
f = open("tweets_#superbowl.txt","r")
tw_in_hr = 0
totalfollow = 0
for line in f:
    tweet = json.loads(line)
    #tweets.append(tweet)
    
    
    #calculate average # of tweets per hour
    tdate = math.floor(tweet['firstpost_date']/60/60)
    
    tusr = tweet['tweet']
    aut = tweet['author']
    usr = tusr['user']
    usrid = usr['id']
    follower = usr['followers_count']
    thas = tusr['entities']
    ret = tusr['retweet_count']
    mfollower = aut['followers']
     
    
    if(flag ==0):
        refhr = tdate
        thour = (math.floor((tweet['firstpost_date']-1409983200)/60/60))%24
        flag =1
    if (tdate == refhr):
        tw_in_hr+=1
        totalfollow = totalfollow+follower
        coutrt = coutrt + ret
        coutfr = coutfr+1
        tmfr = tmfr + mfollower
    else: 
        totd.append(thour)
        if tdate<febhr1:
            countsec1 +=1
        elif tdate>=febhr1 and tdate<febhr2:
            countsec2 +=1
        elif tdate>=febhr2:
            countsec3 +=1
        thour= thour+1
        if (thour == 25):
            thour = 0
        hr.append(tw_in_hr)
        fr.append(totalfollow/coutfr)
        rt.append(coutrt)
        mfr.append(tmfr)
        refhr+=1
        while (tdate!= refhr):
            if tdate<febhr1:
                countsec1 +=1
            elif tdate>=febhr1 and tdate<=febhr2:
                countsec2 +=1
            elif tdate>febhr2:
                countsec3 +=1
            refhr+=1
            totd.append(thour)
            thour= thour+1
            if (thour == 25):
                thour = 0
            mfr.append(0)
            hr.append(0)
            fr.append(0)
            rt.append(0)
        tw_in_hr =1
        coutfr = 1
        coutrt = ret
        totalfollow = follower
        tmfr = mfollower
#print tweet['firstpost_date']  
#print tdate
print type(tmfr)
i = 0
fea = []
t = []
while i<len(hr)-1:
    sfea = []
    sfea.append(hr[i])
    sfea.append(rt[i])
    sfea.append(mfr[i])
    sfea.append(fr[i])
    sfea.append(totd[i])
    t.append(hr[i+1])
    fea.append(sfea)
    i = i+1   
    
regr = linear_model.LinearRegression()
regr.fit(fea, t)    
model = sm.OLS(t, fea)

print (np.mean((regr.predict(fea) - t) ** 2))
print (np.mean(abs(regr.predict(fea) - t)))
print len(hr)

results = model.fit()

print (results.summary())   