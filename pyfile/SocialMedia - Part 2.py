#!/usr/bin/env python
# coding: utf-8

#  <table><tr><td><img src="images/dbmi_logo.png" width="75" height="73" alt="Pitt Biomedical Informatics logo"></td><td><img src="images/pitt_logo.png" width="75" height="75" alt="University of Pittsburgh logo"></td></tr></table>
#  
# 
# # Social Media and Data Science - Part 2
# 
# Data science modules developed by the University of Pittsburgh Biomedical Informatics Training Program with the support of the National Library of Medicine data science supplement to the University of Pittsburgh (Grant # T15LM007059-30S1). 
# 
# Developed by Harry Hochheiser, harryh@pitt.edu. All errors are my responsibility.
# 
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
# 

# ###  *Goal*: Use social media posts to explore the appplication of text and natural language processing to see what might be learned from online interactions.
# 
# Specifically, we will retrieve, annotate, process, and interpret Twitter data on health-related issues such as depression.

# --- 
# References:
# * [Mining Twitter Data with Python (Part 1: Collecting data)](https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/)
# * The [Tweepy Python API for Twitter](http://www.tweepy.org/)
# ---

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import jsonpickle
import json
import random
import tweepy
import time
import operator
from datetime import datetime


# ## 2.0.1 Setup
# 
# Before we dig in, we must grab a bit of code from [Part 1](SocialMedia%20-%20Part%201.ipynb):
# 
# 1. The Tweets class used to store the tweets.
# 2. Our twitter API Keys - be sure to copy the keys that you generated when you completed [Part 1](SocialMedia%20-%20Part%201.ipynb).
# 3. Configuration of our Twitter connection

# In[2]:


class Tweets:
    
    
    def __init__(self,term="",corpus_size=100):
        self.tweets={}
        if term !="":
            self.searchTwitter(term,corpus_size)
    
    def searchTwitter(self,term,corpus_size):
        searchTime=datetime.now()
        while (self.countTweets() < corpus_size):
            new_tweets = api.search(term,lang="en",tweet_mode='extended',count=corpus_size)
            for nt_json in new_tweets:
                nt = nt_json._json
                if self.getTweet(nt['id_str']) is None and self.countTweets() < corpus_size:
                    self.addTweet(nt,searchTime,term)
            time.sleep(120)
                
    def addTweet(self,tweet,searchTime,term="",count=0):
        id = tweet['id_str']
        if id not in self.tweets.keys():
            self.tweets[id]={}
            self.tweets[id]['tweet']=tweet
            self.tweets[id]['count']=0
            self.tweets[id]['searchTime']=searchTime
            self.tweets[id]['searchTerm']=term
        self.tweets[id]['count'] = self.tweets[id]['count'] +1
        
    def combineTweets(self,other):
        for otherid in other.getIds():
            tweet = other.getTweet(otherid)
            searchTerm = other.getSearchTerm(otherid)
            searchTime = other.getSearchTime(otherid)
            self.addTweet(tweet,searchTime,searchTerm)
        
    def getTweet(self,id):
        if id in self.tweets:
            return self.tweets[id]['tweet']
        else:
            return None
    
    def getTweetCount(self,id):
        return self.tweets[id]['count']
    
    def countTweets(self):
        return len(self.tweets)
    
    # return a sorted list of tupes of the form (id,count), with the occurrence counts sorted in decreasing order
    def mostFrequent(self):
        ps = []
        for t,entry in self.tweets.items():
            count = entry['count']
            ps.append((t,count))  
        ps.sort(key=lambda x: x[1],reverse=True)
        return ps
    
    # reeturns tweet IDs as a set
    def getIds(self):
        return set(self.tweets.keys())
    
    # save the tweets to a file
    def saveTweets(self,filename):
        json_data =jsonpickle.encode(self.tweets)
        with open(filename,'w') as f:
            json.dump(json_data,f)
    
    # read the tweets from a file 
    def readTweets(self,filename):
        with open(filename,'r') as f:
            json_data = json.load(f)
            incontents = jsonpickle.decode(json_data)   
            self.tweets=incontents
        
    def getSearchTerm(self,id):
        return self.tweets[id]['searchTerm']
    
    def getSearchTime(self,id):
        return self.tweets[id]['searchTime']
    
    def getText(self,id):
        tweet = self.getTweet(id)
        text=tweet['full_text']
        if 'retweeted_status'in tweet:
            original = tweet['retweeted_status']
            text=original['full_text']
        return text


# Put the values of your keys into these variables

# In[3]:


consumer_key = 'cwFAJagaDWthpmvfrZZOunUF1'
consumer_secret = 'q1EXSFavtHHK0Xq9h84gfZ9pLRmycGfrR5lLdbmfmtoSaeQ8Tb'
access_token = '1048305517230219266-NenBCf8nFUGopL3749VvnOC6C6UjST'
access_secret = '4Mc8wiV6jYRRp0Hn1DTiaRMXLMrxo6TNysIuSj3vZCZHe'


# In[4]:


from tweepy import OAuthHandler

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)


# ## 2.1 Annotating Tweets

# Now that we have a corpus of tweets, what do we want to do with them? Turning a relatively vague notion into a well-defined research question is often a significant challenge, as examination of the data often reveals both shortcomings and unforeseen opportunities.
# 
# In our case, we are interested in looking at tweets about smoking, but we're not quite sure exactly *what* we are looking for. We have a vague notion that we might learn something interesting, but understanding exactly what that is, and what sort of analyses we might need, will require a bit more work.
# 
# In situations such as this, we might look at some of the data to form some preliminary impressions of the content. Specifically, we can look at indidividual tweets, assigning them to one or more categories - known as *codes* - based on their content.  We can add categories as needed to capture important ideas that we might want to refer back to. This practice - known as *open coding* allows us to begin to make sense of unfamiliar data sets. 
# 
# This sounds much more complicated than it is. For now, let's start with some of the data that you used  in [Part 1](SocialMedia%20-%20Part%201.ipynb). If you completed this exercise, you should have 100 tweets associated with the search term 'smoking' in a file names `tweets.json`. If you didn't complete this exercise, please go back and do so.

# In[5]:


tweets =Tweets()
tweets.readTweets("tweets.json")


# We check the count, to verify the contents...

# In[6]:


print(tweets.countTweets())


# We will begin by taking a look at a subset of the first 20 tweets
# 
# To get this list, we'll sort the ids of the tweets and take the first 10 in the list, as ordered by ID

# In[7]:


ids=list(tweets.getIds())
ids.sort()
working=[]
for i in range(20):
    id = ids[i]
    working.append(id)


# *working* now has 20 tweets ids. Let's start with the first.

# In[8]:


td = working[0]
print(tweets.getSearchTerm(id))
print(tweets.getSearchTime(id))
print(tweets.getText(td))


# This tweet might have any of several interesting characteristics. It might be a retweet; it might specifically mention marijunana or tobacco; or it might not be related to either of these. 
# 
# We can model any and all of these points through relevant annotation. Specifically, we will a new array of codes to each tweet object. This array will contain a list of categorical annotations associated with the tweet.  We add routines to add a single code to a tweet (by ID), to add multiple codes, and to retrieve the list of codes associated with a tweet.
# 
# 
# See modifications to the  Tweets object in this new definition. 

# In[9]:


class Tweets:
    
    
    def __init__(self,term="",corpus_size=100):
        self.tweets={}
        if term !="":
            self.searchTwitter(term,corpus_size)
                
    def searchTwitter(self,term,corpus_size):
        searchTime=datetime.now()
        while (self.countTweets() < corpus_size):
            new_tweets = api.search(term,lang="en",tweet_mode='extended',count=corpus_size)
            for nt_json in new_tweets:
                nt = nt_json._json
                if self.getTweet(nt['id_str']) is None and self.countTweets() < corpus_size:
                    self.addTweet(nt,searchTime,term)
            time.sleep(120)
                
    def addTweet(self,tweet,searchTime,term="",count=0):
        id = tweet['id_str']
        if id not in self.tweets.keys():
            self.tweets[id]={}
            self.tweets[id]['tweet']=tweet
            self.tweets[id]['count']=0
            self.tweets[id]['searchTime']=searchTime
            self.tweets[id]['searchTerm']=term
        self.tweets[id]['count'] = self.tweets[id]['count'] +1

    def combineTweets(self,other):
        for otherid in other.getIds():
            tweet = other.getTweet(otherid)
            searchTerm = other.getSearchTerm(otherid)
            searchTime = other.getSearchTime(otherid)
            self.addTweet(tweet,searchTime,searchTerm)
        
    def getTweet(self,id):
        if id in self.tweets:
            return self.tweets[id]['tweet']
        else:
            return None
    
    def getTweetCount(self,id):
        return self.tweets[id]['count']
    
    def countTweets(self):
        return len(self.tweets)
    
    # return a sorted list of tupes of the form (id,count), with the occurrence counts sorted in decreasing order
    def mostFrequent(self):
        ps = []
        for t,entry in self.tweets.items():
            count = entry['count']
            ps.append((t,count))  
        ps.sort(key=lambda x: x[1],reverse=True)
        return ps
    
    # reeturns tweet IDs as a set
    def getIds(self):
        return set(self.tweets.keys())
    
    # save the tweets to a file
    def saveTweets(self,filename):
        json_data =jsonpickle.encode(self.tweets)
        with open(filename,'w') as f:
            json.dump(json_data,f)
    
    # read the tweets from a file 
    def readTweets(self,filename):
        with open(filename,'r') as f:
            json_data = json.load(f)
            incontents = jsonpickle.decode(json_data)   
            self.tweets=incontents
        
    def getSearchTerm(self,id):
        return self.tweets[id]['searchTerm']
    
    def getSearchTime(self,id):
        return self.tweets[id]['searchTime']
    
    def getText(self,id):
        tweet = self.getTweet(id)
        text=tweet['full_text']
        if 'retweeted_status'in tweet:
            original = tweet['retweeted_status']
            text=original['full_text']
        return text
                
    ### NEW ROUTINE - add a code to a tweet
    def addCode(self,id,code):
        tweet=self.getTweet(id)
        if 'codes' not in tweet:
            tweet['codes']=set()
        tweet['codes'].add(code)
        
    ### NEW ROUTINE  - add multiple  codes for a tweet
    def addCodes(self,id,codes):
        for code in codes:
            self.addCode(id,code)
        
    ### NEW ROUTINE get codes for a tweet
    def getCodes(self,id):
        tweet=self.getTweet(id)
        if 'codes' in tweet:
            return tweet['codes']
        else:
            return None


# Now that we have this set up, we can load tweets from a file and reload the subset.
# 
# Note that thsis file is not acutally real tweets - it's just fabricated tweets that have been put together for purposes of this lesson. The text, user name, and other details are all wrong or omitted.  Later, you'll try this with the tweets you loaded in your search.

# In[10]:


tweets =Tweets()
tweets.readTweets("tweets-fake.json")
ids=list(tweets.getIds())
ids.sort()
working=[]

for i in range(len(ids)):
    id = ids[i]
    working.append(id)

td = working[0]
t = tweets.getTweet(td)
tweets.getText(td)


# so, this tweet seems to be about marijuana. We can add a code to this effect.

# In[11]:


tweets.addCode(td,"MARIJUANA")


# We can also confirm that this tweet is associated with the desired codes:

# In[12]:


tweets.getCodes(td)


# Good. Let's look at the next tweet. 

# In[13]:


td = working[1]
tweets.getText(td)


# This tweet contains a generic mention of smoking, without much detail, but we might gues that it is also about marijuana

# In[14]:


tweets.addCodes(td,['MARIJUANA'])


# ok.. moving on to the third tweet..

# In[15]:


td = working[2]
tweets.getText(td)


# In[16]:


tweets.addCodes(td,['QUITTING','BENEFITS'])


# next...

# In[17]:


td = working[3]
tweets.getText(td)


# This note mentions government quitting efforts, and has both a link and a user mention.

# In[18]:


tweets.addCodes(td,['LINK','USERMENTION','GOVERNMENT','QUITTING'])


# In[19]:


td = working[4]
tweets.getText(td)


# This tweet mentions a negatigve attitude towards vaping.

# In[20]:


tweets.addCodes(td,['VAPING','NEGATIVE'])


# Now that we've gone through several tweets, we can review the codes used.

# In[21]:


for i in range(5):
    td=working[i]
    print(tweets.getCodes(td))


# Having annotated several tweets, we might want to save the annotations in a file for future use. Fortnuately, the approach that we've used in our save and reload code is flexible enough to handle this without any further changes to the implementation. 
# 
# How does this work? The `Tweets` class stores all of the information abou the tweets in a simple dictionary. Tweet counts and codes are then stored inside the tweet object. When we go to save the set of Tweets, we simply turn this dictionary into JSON and then write it to a file. To read things in, we just read the JSON from the file and convert the result back into a dictionary. Thus, anything that we add to the dictionary will automatically be writen out and read back in.  We still need additional routines to access this data (like `addCode`, `addCodes`, and `getCodes`), but we  don't need to change the save/load routines.  Let's try it out.
# 

# In[22]:


tweets.saveTweets("tweets-fake-annotated.json")


# In[23]:


tweets2=Tweets()
tweets2.readTweets("tweets-fake-annotated.json")


# In[24]:


print(tweets.getText(td))
print(tweets.getCodes(td))
print(tweets2.getText(td))
print(tweets2.getCodes(td))


# *END CUT HERE*
# ****

# # Exercise 2.2: Code the Next 10 tweets in the set. 
# Start with the tags used above, adding your own as needed.  Code up to and including the tweet  with index 15 in the `working` array. Examine the code profile and save your tweets  to a new file when you are done. 

# *ANSWER FOLLOWS - insert here*

# In[25]:


tweets = Tweets()
tweets.readTweets("tweets-fake-annotated.json")
ids=list(tweets.getIds())
ids.sort()
working=[]
for i in range(5, len(ids)):
    id = ids[i]
    working.append(id)

td = working[0]
tweets.getText(td)


# In[26]:


tweets.addCodes(td,['LINK','USERMENTION','GOVERNMENT','QUITTING'])


# In[27]:


td = working[1]
t = tweets.getTweet(td)
tweets.getText(td)


# In[28]:


tweets.addCodes(td,['LINK','USERMENTION','GOVERNMENT','QUITTING'])


# In[29]:


td = working[2]
t = tweets.getTweet(td)
tweets.getText(td)


# In[30]:


tweets.addCodes(td,['VAPING','TEENS','RESEARCH'])


# In[31]:


td = working[3]
t = tweets.getTweet(td)
tweets.getText(td)


# In[32]:


tweets.addCodes(td,['VAPING','TEENS','RESEARCH'])


# In[33]:


td = working[4]
t = tweets.getTweet(td)
tweets.getText(td)


# In[34]:


tweets.addCodes(td,['NEGATIVE'])


# In[35]:


td = working[5]
t = tweets.getTweet(td)
tweets.getText(td)


# In[36]:


tweets.addCodes(td,['VAPING','NEGATIVE'])


# In[37]:


td = working[6]
t = tweets.getTweet(td)
tweets.getText(td)


# In[38]:


tweets.addCodes(td,['MARIJUANA'])


# In[39]:


td = working[7]
t = tweets.getTweet(td)
tweets.getText(td)


# In[ ]:


tweets.addCodes(td,['PUBLIC','VOTE'])


# In[40]:


td = working[8]
t = tweets.getTweet(td)
tweets.getText(td)


# In[ ]:


tweets.addCodes(td,['QUITING','USERMENTION'])


# In[41]:


td = working[9]
t = tweets.getTweet(td)
tweets.getText(td)


# In[42]:


tweets.addCodes(td,['VAPING','NEGATIVE'])


# In[44]:


tweets.saveTweets("tweets-fake-annotated.json")


# *END CUT*
# 
# ---

# # EXERCISE 2.3: Reflection on coding
# 
# Open coding can often be an iterative process. When we first start out, we don't really know what we're looking for. As a result, the first few items annotated might only get a few codes, and we might miss ideas that we don't initially think are important. As we see more and more items, our ideas of what needs to be annotated will change, and we'll start adding in codes that might also apply to earlier messages. Thus, we often need to review and re-annotate earlier tweets to account for changes in our interpreations.
# 
# Review your annotations the tweets that you reviewed. Revise the codes associated with these tweets, adding items from the overall list of codes as appropriate. Describe the change that you have made.
# 

# ---
# *ANSWER FOLLOWS - insert here*

# In[46]:


td = working[1]
t = tweets.getTweet(td)
tweets.getText(td)
tweets.addCodes(td,['LINK','USERMENTION','GOVERNMENT','QUITTING','PUBLIC'])


# After we see all the tweets,  we can give code through different dimention. This one is about the health for 8 million people, so we should add 'PUBLIC'

# In[47]:


tweets.saveTweets("tweets-fake-annotated.json")


# 
# *END CUT*
# 
# ---

# # EXERCISE 2.4: Reflection on storage/serialization
# 
# In working with this small set of 100 tweets, we are taking a very simple approach to storage and management of the tweets and annotations. Storing everything in a nested Python dictionary and then dumping it to disk as JSON text can be very appealing. What are the strengths and weaknesses of this approach, and how might these strengths and weaknesses differ with larger datasets containing 100,000 or 100 million datasets? What alternative  strategies might you use for larger datasets?

# ---
# *ANSWER FOLLOWS - insert here*

# The strengths of this approach is easy to implement, and all information will be presented in the file. The weaknesses are this approach will use big storage, and we should modify the whole file and re-write it. Also for very huge dataset, this approach will consume too many resources to store the data, and take long time to modify or search.

# For large dataset such as 100,000 or 100 million. We can use database, or we could use many JSON file to partition the whole dataset. If the data is used frequently, we also can implements stretagy like LRU or LFU to make it is easier to find those data. 

# 
# *END CUT*
# 
# ---

# # 2.2 Final Notes
# 
# [Part 3](SocialMedia%20-%20Part%203.ipynb) will explore the application of Natural Language Processing  - NLP - techniques to Tweet data. 
