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
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn import linear_model
import xml.etree.ElementTree as ET
import itertools
from sklearn import svm
stemmer = SnowballStemmer("english")
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import string
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
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
tcontent = []
location = []
flag = 0
coutfr = 0
coutrt = 0
tmfr = 0
febhr1 = math.floor(1422802800/60/60)
febhr2 = math.floor(1422846000/60/60)
countsec1 = 0
countsec2 = 0
countsec3 = 0
counts = 0
f = open("tweets_#nfl.txt","r")
#f = open("tweets_#superbowl.txt","r")
tw_in_hr = 0
totalfollow = 0
for line in f:
    tweet = json.loads(line)
    #tweets.append(tweet)
    
    
    #calculate average # of tweets per hour
    #tdate = math.floor(tweet['firstpost_date']/60/60)
    
    tusr = tweet['tweet']
    usr = tusr['user']
    if usr['location'].find('WA')!=-1 or usr['location'].find('Washington')!=-1 or usr['location'].find('washington')!=-1:
        tcontent.append(tweet['title'])
        location.append("WA")
    if usr['location'].find('MA')!=-1 or usr['location'].find('Massachusetts')!=-1 or usr['location'].find('massachusetts')!=-1:
        tcontent.append(tweet['title'])
        location.append("MA")    
    counts +=1
    #if counts == 2000:
    #    break
stop_words = text.ENGLISH_STOP_WORDS

t=0
#print len(newsgroups_train.target)
def stem_data(data):
    stemdata=[]
    for doc in data:
        tokenizer = TreebankWordTokenizer()
        a = tokenizer.tokenize(doc)
        stemmed_text = [stemmer.stem(i) for i in a]
        #print stemmed_text
        s = " "
        seq = s.join(stemmed_text)
        stemdata.append(seq)
    return stemdata

stemdata = stem_data(tcontent)
vectorizer = CountVectorizer(analyzer = 'word',strip_accents = 'unicode',stop_words=stop_words)
#print tcontent[0:10]
X = vectorizer.fit_transform(stemdata)

vocab = vectorizer.get_feature_names()
#print vocab

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X.toarray())

#print tfidf.toarray()
svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
svd1 = svd.fit_transform(tfidf) 
gnb = GaussianNB()   

#print len(hi)  
x5, x2, y5, y2 = train_test_split(tfidf,location , test_size=0.2, random_state=3)
clf_l2_LR = LogisticRegression(C=1, penalty='l2', tol=0.01)
clf_l2_LR.fit(x5, y5)
l2pre = clf_l2_LR.predict(x5)

clf = svm.LinearSVC(C=100)
clf.fit(x5,y5)

clf.predict(x2)    
class_names = ['0','1']
#ROC PLOT
pre = clf.predict(x2) 
npre = []
l2npre = []
ny2 = []
ii = 0
while ii<len(y2):
    if y2[ii] == 'WA':
        ny2.append(0)
    elif y2[ii] == 'MA':
        ny2.append(1) 
        
    if pre[ii] == 'WA':
        npre.append(0)
    elif pre[ii] == 'MA':
        npre.append(1)   
    
    if l2pre[ii] == 'WA':
        l2npre.append(0)
    elif l2pre[ii] == 'MA':
        l2npre.append(1) 
    ii+=1     


fpr,tpr,thres = roc_curve(ny2,npre)
fpr2,tpr2,thres2 = roc_curve(ny2,l2npre)
roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)

plt.title('ROC plot')
plt.plot(fpr , tpr, 'b',
label='AUC for linear svc = %0.2f'% roc_auc)

plt.plot(fpr2 , tpr2, 'g',
label='AUC for logistic = %0.2f'% roc_auc2)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')








# Compute confusion matrix
#cnf_matrix = confusion_matrix(y2, clf.predict(x2))
cnf_matrix = confusion_matrix(y2, clf_l2_LR.predict(x2))
np.set_printoptions(precision=2)
 #plot
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

"""
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
"""

plt.show()    
    
 