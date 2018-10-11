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
f = open("tweets_#nfl.txt","r")
#f = open("tweets_#superbowl.txt","r")
tw_in_hr = 0
totalfollow = 0
maxfollow = 0
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
        if maxfollow < follower:
            maxfollow = follower
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
        fr.append(totalfollow)
        rt.append(coutrt)
        mfr.append(maxfollow)
        maxfollow = 0
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
            maxfollow = 0
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
#print type(sfea)
#print type(t)
#print fea      
    #average number of followers of users posting the tweets
    
regr = linear_model.LinearRegression()
regr.fit(fea, t)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(fea) - t) ** 2))
    
print('Variance score: %.2f' % regr.score(fea, t))    
    
clf_l2_LR = LogisticRegression(C=1, penalty='l2', tol=0.01)
clf_l2_LR.fit(fea, t)
coef_l2_LR = clf_l2_LR.coef_.ravel()
sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
print("score with L2 penalty: %.4f" % clf_l2_LR.score(fea, t))
print("Mean squared error: %.2f"
      % np.mean((clf_l2_LR.predict(fea) - t) ** 2))            
countsec3 = countsec3 -1        


print "sec 1: ", countsec1
print "sec 2: ", countsec2
print "sec 3: ", countsec3
print len(t)
print t[countsec1:countsec1+countsec2]
testi = 0
meanlasso = 0
meanlinear = 0
meanpoly = 0
please = []
while testi<10:
    x_train, x_test, y_train, y_test = train_test_split(fea[:countsec1], t[:countsec1], test_size=0.1, random_state=testi+1)
    poly1 = PolynomialFeatures(2)
    x_new = poly1.fit_transform(x_train,y_train)
    xt_new = poly1.fit_transform(x_test,y_test)
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(x_train,y_train)
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    
    regr1 = linear_model.LinearRegression()
    regr1.fit(x_new,y_train)
    
    meanlasso = meanlasso + np.mean(abs(clf.predict(x_test) - y_test))
    meanlinear = meanlinear + np.mean(abs(regr.predict(x_test) - y_test))
    meanpoly = meanpoly + np.mean(abs(regr1.predict(xt_new) - y_test))
    testi +=1
    


#print clf.predict(x_test)
#print regr.predict(x_test)
#print regr1.predict(xt_new)
#print y_test
print meanlasso
print meanlinear
print meanpoly

testi = 0
tmean = 0
meanlasso = 0
meanlinear = 0
meanpoly = 0
while testi<10:
    x_train, x_test, y_train, y_test = train_test_split(fea[countsec1:countsec2+countsec1], t[countsec1:countsec2+countsec1], test_size=0.1, random_state=testi+1)
    poly1 = PolynomialFeatures(2)
    x_new = poly1.fit_transform(x_train,y_train)
    xt_new = poly1.fit_transform(x_test,y_test)
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(x_train,y_train)
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    
    regr1 = linear_model.LinearRegression()
    regr1.fit(x_new,y_train)
    
    meanlasso = meanlasso + np.mean(abs(clf.predict(x_test) - y_test))
    meanlinear = meanlinear + np.mean(abs(regr.predict(x_test) - y_test))
    meanpoly = meanpoly + np.mean(abs(regr1.predict(xt_new) - y_test))
   # print lr2.predict(x_test)
   # print y_test
    testi +=1
print "2:  ", tmean 
print meanlasso
print meanlinear
print meanpoly
testi = 0
tmean = 0
meanlasso = 0
meanlinear = 0
meanpoly = 0
while testi<10:
    x_train, x_test, y_train, y_test = train_test_split(fea[countsec2+countsec1:], t[countsec2+countsec1:], test_size=0.1, random_state=testi+1)
    poly1 = PolynomialFeatures(2)
    x_new = poly1.fit_transform(x_train,y_train)
    xt_new = poly1.fit_transform(x_test,y_test)
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(x_train,y_train)
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    
    regr1 = linear_model.LinearRegression()
    regr1.fit(x_new,y_train)
    
    meanlasso = meanlasso + np.mean(abs(clf.predict(x_test) - y_test))
    meanlinear = meanlinear + np.mean(abs(regr.predict(x_test) - y_test))
    meanpoly = meanpoly + np.mean(abs(regr1.predict(xt_new) - y_test))
    testi +=1
print "3:  ", tmean
print meanlasso
print meanlinear
print meanpoly


        
fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = len(hr)

## necessary variables
ind = np.arange(N)               # the x locations for the groups

width = 0.4                    # the width of the bars

## the bars
rects1 = ax.bar(ind,hr, width)
ax.set_ylabel('Number of tweet')
ax.set_xlabel('Hours ')
ax.set_title('#SuperBowl number of tweet per hour')




#xTickMarks = ['Topic'+str(i) for i in range(0,20)]
#ax.set_xticks(ind)
#xtickNames = ax.set_xticklabels(xTickMarks)
#plt.setp(xtickNames, rotation=90, fontsize=9)
plt.show()
  
            
                

#average # of followers of users posting the tweets
#average number of retweets
#plot "number of tweets in hour" over time for #superbow and nflm
