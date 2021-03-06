#part 1 
import simplejson as json
import math
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
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
notweet = 0
tofollower = 0
#f = open("tweets_#nfl.txt","r")
f = open("tweets_#gopatriots.txt","r")
tw_in_hr = 0
totalfollow = 0
totalretweet = 0
for line in f:
    tweet = json.loads(line)
    #tweets.append(tweet)
    notweet+=1
    
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
    tofollower += follower 
    totalretweet+=ret
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
print float(notweet)/len(hr), 'tweetperhour'
print tofollower/float(notweet), 'tweetperfollower'
print totalretweet/float(notweet), 'retweet'
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
fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = len(hr)

## necessary variables
ind = np.arange(N)               # the x locations for the groups

width = 0.7                    # the width of the bars

## the bars
rects1 = ax.bar(ind,hr, width)
ax.set_ylabel('Number of tweet')
ax.set_xlabel('Hours ')
ax.set_title('#NFL number of tweet per hour')
plt.show()    