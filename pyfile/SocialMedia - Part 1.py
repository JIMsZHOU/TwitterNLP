#!/usr/bin/env python
# coding: utf-8

#  <table><tr><td><img src="images/dbmi_logo.png" width="75" height="73" alt="Pitt Biomedical Informatics logo"></td><td><img src="images/pitt_logo.png" width="75" height="75" alt="University of Pittsburgh logo"></td></tr></table>
#  
#  
#  # Social Media and Data Science - Part 1
#  
#  
# Data science modules developed by the University of Pittsburgh Biomedical Informatics Training Program with the support of the National Library of Medicine data science supplement to the University of Pittsburgh (Grant # T15LM007059-30S1). 
# 
# Developed by Harry Hochheiser, harryh@pitt.edu. All errors are my responsibility.
# 
# <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
# 

# ###  *Goal*: Learn how to retrieve, manage, and save social media posts.
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
import time
import tweepy
from datetime import datetime


# # 1.0 Introduction
# 
# Analysis of social-media discussions has grown to be an important tool for biomedical informatics researchers, particularly for addressing questions relevant to public perceptions of health and related matters. Studies have examination of a range of topics at the intersection of health and social media, including studies of how [Facebook might be used to commuication health information](http://www.jmir.org/2016/8/e218/), how Tweets might be used to understand how smokers perceive [e-cigarettes, hookahs and other emerging smoking products](https://www.jmir.org/2013/8/e174/), and many others.
# 
# Although each investigation has unique aspects, studies of social media generally share several common tasks. Data acquisition is often the first challenge: although some data may be freely available, there are often [limits](https://dev.twitter.com/rest/public/rate-limits) as to how much data can be queried easily. Researchers might look out for [opportunities for accessing larger amounts of data](https://www.wired.com/2014/02/twitter-promises-share-secrets-academia/). Some studies contract with [commercial services providing fee-based access](https://gnip.com). 
# 
# Once a data set is hand, the next step is often to identify key terms and phrases relating to the research question. Messages might be annotated to indicate specific categorizations of interest - indicating, for example, if a message referred to a certain aspect of a disease or symptom. Similarly, key words and phrases regularly occurring in the content might also be identified. Natural language and text processing techniques might be used to extract key words, phrases, and relationships, and machine learning tools might be used to build classifiers capable of distinguishing between types of tweets of interest. 
# 
# This module presents a preliminary overview of these techniques, using Python 3 and several auxiliary libraries to explore the application of these techniques to Twitter data. 
#   
#   1. Configuration of tools to access Twitter data
#   2. Twitter data retrieval
#   3. Searching for tweets
# 
# Our case study will apply these topics to Twitter discussions of smoking and tobacco. Although details of the tools used to access data and the format and content of the data may differ for various services, the strategies and procedures used to analyze the data will generalize to other tools.

# # 1.1 Configuration of tools to access Twitter data

# [Twitter](www.twitter.com) provides limited capabilities for searching tweets through an Application Programming Interface (API) based on Representational State Transfer (REST).  [REST](https://doi.org/10.1145/337180.337228) is an approach to using web-based Hypertext-Transfer Protocol (HTTP) requests as APIs. 
# 
# Essentially, a REST API specifies conventions for HTTP requests that might be used to retrieve specific data items from a remote server. Unlike traditional HTTP requests, which return HTML markup to be rendered in web browsers, REST APIs return data formatted in XML or JSON, suitable for interpretation by computer programs. REST APIs from familiar websites underlie frequently-seen functionality such as embedded twitter widgets and "like/share" links, among others.
# 
# Commercial REST applications often use "API-Keys" - unique identifiers used to associate requests with registered accounts. Here, we will walk through the process of registering for Twitter API keys and using a Python library to manage the details of making a Twitter API request and receiving a response.
# 
# ## 1.1.2 Register for a Twitter API key
# 
# ### 1.1.2.1 Signup for Twitter and a Developer Account
# 
# The first step in registering for a Twitter API key is to [signup](https://twitter.com/signup) for an account. If you don't want to post anything or to use the account in any way that might be linked to your regular email adddress, you might want to create a special-purpose account using a service such as gmail, and use this new email address for the twitter account.
# 
# Go to  [Twitter's developer site](https://dev.twitter.com) an click on the "apply" link to create a developer account (look on the upper-right cornder, near the magnifying glass search icon). Follow the listed steps, indicating that you are interested in access for your own personal use. You will need to answer a few questions about the assignment and your goals. Answer these questions as appropriate - simply indicate that you are working on a class assignment and that you won't be showing anything commecially. Submit your application and wait for it to be approved. This might take a bit of time - please start early.
# 
# 
# ### 1.1.2.2 Create a Twitter application: 
# 
# One you are approved, log in and go to the [developer site](https://dev.twitter.com). Select "Apps" to create new apps. Click on "Create New App" in the upper right and then fill out the form. The main thing that you need to focus on here is the application name, description, and website. The rest can be ignored.
# 
# Creating the application will lead to the display of some information with some URLs and a few tabs. Look under "Keys and Access Tokens" to see the Consumer API key and API Secret - these will come in handy later.
# 
# There will also be a button that says "Create my access token". Press this button and make a note of the Access Token and Access Token Secret values that are displayed. 
# 
# Although these tokens are always available on the application page, for the purpose of this exercise, it's best to store them in Python variables directly in this Jupyter notebook. Execute the following insstructions, substituting the keys for your application for the phrases "YOUR-CONSUMER-KEY", etc. 

# In[2]:


consumer_key = 'cwFAJagaDWthpmvfrZZOunUF1'
consumer_secret = 'q1EXSFavtHHK0Xq9h84gfZ9pLRmycGfrR5lLdbmfmtoSaeQ8Tb'
access_token = '1048305517230219266-NenBCf8nFUGopL3749VvnOC6C6UjST'
access_secret = '4Mc8wiV6jYRRp0Hn1DTiaRMXLMrxo6TNysIuSj3vZCZHe'


# In theory, you now have all that you need to start accessing Twitter. Using these keys and the information in the [Twitter Developer Documentation](https://dev.twitter.com/docs), you might conceivably create web requests to search for tweets, post, and read your timeline. In practice, it's a bit more complicated, so most folks use third-party tools that take care of the hard work. 
# 
# ## 1.1.3 Try the Tweepy library
# 
# [Tweepy](http://www.tweepy.org) is a Python 3 library for using the Twitter API. Like other similar libraries - there are many for Python and other languages - Tweepy takes care of the details of authorization and provides a few simple function calls for accessing the API.  
# 
# The first step in using Tweepy is *authorization* - establishing your credentials for using the Twitter API. Tweepy uses the [OAuth](http://www.oauth.net) authorization framework, which is widely used for both API and user access to services provided over HTTP. Fortunately Tweepy hides the oauth details. All you need to do is to make a few calls to the Tweepy library and you're all set to go. Run the following code, making sure that the four variables are set to the values you were given when you registered your Twitter application:

# In[3]:


from tweepy import OAuthHandler

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
api


# If this worked correctly, you should see something like this 
# ```
# <tweepy.api.API at 0x109da36d8>
# ``` 
# 
# If you get an error message, please check your keys and tokens to ensure that they are correct.

# # 1.2 Twitter data retrieval
# 
# Now that you have successfully accessed the Twitter API, it's time to access the data. The simplest thing to do is to grab some Tweets off of your timeline. Try the following code:

# In[30]:


top_ten = []
i =0
for tweet in tweepy.Cursor(api.home_timeline,tweet_mode='extended').items(10):
    top_ten.append(tweet._json)


# There are several key componnents to this block of code:
# * ```api.home_timeline``` is a component of the API object, referring to the user timeline - the tweets shown on your home page.
# * ```tweepy.Cursor``` is a construct in the Tweepy API that supports navigation through a large set of results.
# * ```tweepy.Cursor(api.home_timeline).items(10)``` essentially asks Tweepy to set up a cursor for the home timeline and then to get the first 10 items in that set. The result is a Python Iterator, which can be used to examine the items in the set in turn.
# * We will grab the JSON representation of each tweet (stored as "tweet.\_json") for maximum flexibility.
# * The loop takes each of those objects an adds them into a Python array.
# 
# Now, each of the items in ```top_ten``` is a Tweet object. Let's take a look inside. We'll start by grabbing the first text:

# In[31]:


tweet1=top_ten[0]


# and looking at its text:

# In[32]:


tweet1['full_text']


# we can check for the length of the tweet

# In[33]:


len(tweet1['full_text'])


# Note that the `full_text` of the tweet might  contain the length beyond the original 140 characers assoicated with tweets. This value is returned in response to the `tweet_mode='extended'` argument to `tweepy.Cursor()`. Without that argument, a `text` field might be  returned instead of `full_text`, containing only the first 140 characters.

# We can also examine when the tweet was created...

# In[34]:


tweet1['created_at']


# .. whether it has been favorited...

# In[35]:


tweet1['favorited']


# .. The unique ID String of the Tweet...

# In[36]:


tweet1['id_str']


# .. and the name of the Twitter user responsible for the post. 

# In[37]:


tweet1['user']['name']


# We can look at another tweet in the list..

# In[38]:


tweet1=top_ten[1]


# In[39]:


tweet1['id_str']


# In[40]:


tweet1['full_text']


# We can check to see if a tweet is a retweet by seeing if it has the 'retweeted_status' attribute.

# In[41]:


'retweeted_status' in tweet1


# If the tweet is a retweet, the `retweeted_status` field will hold the original tweet - all of the fields contained in the main tweet can be found in the tweet contained in `retweeted_status`.

# In[70]:


tweet3=top_ten[0]
'retweeted_status' in tweet3


# The twitter API supports many other details for users, tweets, and other entities. See [The Twitter API Overview](https://dev.twitter.com/overview/api) for general details and subpages about [Tweets](https://dev.twitter.com/overview/api/tweets), [Users](https://dev.twitter.com/overview/api/users) and related pages for specific details of other data types.

# # Exercise 1.1: Retweets
# 
# Here, we're going to look at the contents of a retweet as compared to an original tweet. 
# 
# ## 1.1.1 Finding retweets
# 
# Using the `retweeted_status` field, find a retweet.  You might have to run the tweepy Cursor search above more than once.  
# 
# ## 1.1.2 Examining rewteets
# 
# Compare the text of the retweet and the original tweet. How do they differ? Which one should we analyze and why?

# ---
# *ANSWER FOLLOWS - insert anwer here*
# 

# In[66]:


top_ten = []
i =0
for tweet in tweepy.Cursor(api.home_timeline,tweet_mode='extended').items(10):
    top_ten.append(tweet._json)


# In[73]:


for tweet1 in top_ten:
    if 'retweeted_status' in tweet1:
        print(tweet1['full_text'])
    else:
        print('Not retweeted')


# The retweeted test contains RT @ORIGIN AUTHOR in the beginning.
# 
# We should analyze the origin tweet because the information is provided by the origin tweet, and the retweet may contain other opinions or judgement in it. For retweet, we should use it to analyze the reaction of spcially information.

# *END CUT*
# 
# ----

# # 1.3 Searching for tweets
# 
# Our next major goal will be to search for Tweets. Effective searching requires both construction of useful queries (the hard part) and use of the Tweepy search API (the easy part).
# 
# ## 1.3.1 Formulating a query
# 
# Formulating an effective search query is often a challenging, iterative process. Trying some searches in the Twitter web page is a good way to see both how a query might be formulated and which queries might be most useful.
# 
# If you look carefully at the URL bar in your browser after running a search, you might notice that the search term is embedded in the URL. Thus, if you search for "depression", you might see a URL that looks like https://twitter.com/search?q=depression. You might also see "&src=typed" at the end of the URL, indicating that the search was typed by hand.
# 
# You can also use Tweepy to conduct a search, as follows:

# In[74]:


tlist = api.search("smoking",lang="en",count=10,tweet_mode='extended')
tweets = [t._json for t in tlist]


# This search will find the first 10 English tweets matching the term "depression".

# In[75]:


tweets[0]['full_text']


# We can then look at the text for these tweets. This is a good way to check to ensure that we're getting what we think we should be getting.

# In[76]:


texts = [c['full_text'] for c in tweets]


# In[77]:


texts


# You may see some tweets that don't match exactly - perhaps using 'smoke' instead of 'smokiing'. This suggests that Twitter uses <em>stemming</em> - removing suffixes and variations to get to the core of the word - to increase search accuracy.

# At this point, we should be able to evaluate the results to see if we are on the right track. If we aren't, we'd want to try some different queries. For now, it looks good, so let's move on.
# 
# ## 1.3.2 Collecting and characterizing a larger corpus
# 
# Our original query only retrieved 10 tweets. This is a good start, but probably not enough for anything serious. We can loop through several times to create a longer list, with a delay between searches to avoid overstaying our welcome with Twitter:

# In[79]:



for i in range(3):
    new_tweets = api.search("smoking",lang="en",tweet_mode='extended',count=100)
    nt = [t._json for t in new_tweets]
    tweets= tweets+nt
    time.sleep(1)
    


# In[80]:


len(tweets)


# At this point, we might want to know something about the tweets that we have retrieved. As our goal is to shoot for linguistic diversity, we want to make sure that we don't have too many retweets, that we have a wide range of authors, and that we have enough different tweets (not too many repeats).  Let's run through the tweets and count the number of authors, the number of  retweets, and the number of times each tweet is seen.

# In[81]:


def getAuthors(tweets):
    authors={}
    retweets=0
    uniqTweets={}
    for t in tweets:
        # is it a retweet? If so, increment
        if 'retweeted_status' in t:
            retweets = retweets+1
        # get tweet author name
        uname = t['user']['name']
        # if not in authors, put it in with zero articles
        if uname not in authors:
            authors[uname]=0
        authors[uname]=authors[uname]+1
        id=t['id_str']
        if id not in uniqTweets.keys():
            uniqTweets[id]=0
        uniqTweets[id]=uniqTweets[id]+1

    # sort uniq tweets
    uts=[]
    for t,entry in uniqTweets.items():
        uts.append((t,entry))  
        uts.sort(key=lambda x: x[1],reverse=True)

    return (retweets,authors,uts)


# In[82]:


(retweets,authors,uniq) = getAuthors(tweets)
retweets


# In[83]:


len(tweets)


# In[84]:


len(authors.keys())


# We might see a lot of retweets here - I saw at least 80% in one instance, with about 193 authors. This suggests that this corpus has a good many authors with multiple tweets. 
# 
# To explore this, let's look at the histogram of the number of tweets/author.

# To examine the distribution of authors, we can use the [NumPy](http://www.numpy.org) and [Matplotlib](http://matplotlib.org) libraries to extract the number of tweets from each user (given by authors.values()) and to plot a histogram...

# In[91]:


vals = np.array(list(authors.values()))
# plt.xticks(range(min(vals),max(vals)+1))
plt.hist(vals,np.arange(min(vals)-0.5,max(vals)+1.5));


# To look at how frequently each tweet is seen we can look at the first few elements in the third item returned by `getAuthors`.

# In[92]:


uniq[1:10]


# It looks like a broad range of the number of tweets/user,with many users having multiple tweets. This is an intersting pattern, with no immediately obvious interpretation. Understanding the usage patterns might be an intersting area for further work, although larger data sets might be necessary to see meaningful patterns.

# # 1.4. A class for storing Tweets

# Given the complexity of the data being discussed, we should create a class to store our tweets. 
# 
# This class will be based on a dictionary called `tweets`. This dictionary will be indexed by the ID string of the tweet. Each element of the dictionary will itself be a dictionary, with the following contents:
# * `tweet` will refer to the full tweet
# * `count` contains the number of times it occurs in the dataset. Generally, this will be zero, but we might have a need for allowing for a tweet to occur multiple times.
# * `searchTime` will contain the timestamp of when a tweet was searched or added to the set.
# * `seachTerm` will contain the term used to search for the tweet.
# 
# We will have methods for adding tweets, retrieving tweets by id, tracking the number of times we see each tweet, and other useful features. We'll also add a routine to find the most frequently-seen tweets by ID.
# 
# We also provide a `searchTweets` routine to search for a set of unique tweets of a given size matching a given query. This will be called by the default constructor if search terms and counts are provided.  `searchTweets` will query for the number of desired tweets, up to the size required, knowing that Twitter will only return a maximum of 100 /query.  Any returned tweets will be checked for duplications and unique tweets will be added to the set. This will be repeated until the desired corpus size is retrieved.
# 
# Note that searching for a large set of tweets might take some time. I have noticed that the same search run twice in succession might yield almost identical results, which is not what we want if we are looking for a large, diverse set. The two-minute delay in `searchTwitter` might help, but it might not. To get around this, we can conduct multiple searches and combine results using `combineTweets`. 
# 
# `getText` will retrieve the text that we're interested in for a given tweet. If the tweet is not a retwet, the `full_text` field will be returned. However, if it is a retweet, the `full_text` of the original tweet will be returne. Thus, we will ahve the maximal amount of text available to analyze.  
# 
# A few extra methods are included for convenience in accessing various elements of the structure. 
# 
# This structure will evolve as we go along.  

# In[93]:


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
            time.sleep(1)
                
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
            searchTerm = otherid.getSearchTerm(id)
            searchTime = otherid.getSearchTime(id)
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


# In[94]:


tweets2 = Tweets("smoking",100)


# In[95]:


tweets2.countTweets()


# Now, we've got a good solid set of tweets to work with. Note that we also route some routines above to save and load tweets from a file. Let's try them out.

# In[96]:


tweets2.saveTweets('tweets.json')


# Let's  do some quick checks to confirm that we've got the right data out. Note that in future runs, you can just start here to read in your tweets.

# In[97]:


tweets3=Tweets()
tweets3.readTweets('tweets.json')


# In[98]:


tweets3.countTweets()


# [According to Python documentation](https://docs.python.org/2/reference/expressions.html#id24) dictionaries are equal if the keys and values are equal, so this looks good. To check in more detail, we can look at the keys, using subtraction to indicate set difference:

# In[99]:


tweets2.getIds()-tweets3.getIds()


# In[100]:


tweets3.getIds()-tweets2.getIds()


# ok, so we've got the same set of tweets IDs.  Now let's pick some random tweets and see if the tesxt look the same.

# In[101]:


tweet_id=random.choice(list(tweets3.getIds()))
t2=tweets2.getText(tweet_id)
t3=tweets3.getText(tweet_id)
print(tweets2.getText(tweet_id))
print(tweets3.getText(tweet_id))


# In[103]:


t3==t2


# Spot checks like this give some confidence that the loaded tweets are identical to the saved tweets. We might also run a slightly more rigorous check by iterating through the list to look for similarities. Since we know that the two dictionaries have identical sets of keys, we can iterate through the keys of one to get entries and compare equalities.

# In[104]:


errs =[]
for id in tweets2.getIds():
    t2 =tweets2.getTweet(id)
    t3 =tweets3.getTweet(id)
    if t2 != t3:
        errs.append(id)


# In[105]:


errs


# Great. No errors...

# ***
# 
# # Exercise 1.2: Dataset diversity
# 
# Having collected a data set, we might want to characterize it in different ways. We saw above how to identify the number of authors represented in a set of tweets. Here, we examine a slightly different question - what is the elapsed time period covered by a set of tweets? In other words, what are the  times of the first and last tweets in the set?
# 
# To do this, we'll need some help from Python libraries. Before we get into that, let's get a couple of tweets from our set. 
# 
# ## 1.2.1 Introduction

# In[106]:


tid1=random.choice(list(tweets3.getIds()))
tid2=random.choice(list(tweets3.getIds()))
tweet1=tweets3.getTweet(tid1)
tweet2=tweets3.getTweet(tid2)


# We can now look at their creation times.

# In[107]:


tweet1['created_at']


# In[108]:


tweet2['created_at']


# We can use the python [datetime](https://docs.python.org/2/library/datetime.html) library, and the *strptime* function in particular to convert these strings to datetime objects capable of being compared and manipulated. 

# In[109]:


from datetime import datetime


# to do this, we call *strptime* with a string pattern matchings of the strings returned in the tweet object. Specifically, we can see that each timestamp has a 3 letter string indicating a day of the week, the month, the date, the time in hh:mm:ss format, a time-zone indicate ("+000") and the year. These items can be specified in a string argument as "%a" ," "%b", "%d", "%H", "%M", "%S", "%z" and "%Y", respectively, thus providing a pattern to be used to create the time object, as follows:

# In[115]:


t1 = datetime.strptime(tweet1['created_at'], "%a %b %d %H:%M:%S %z %Y")


# In[116]:


t1


# In[117]:


t2 = datetime.strptime(tweet2['created_at'], "%a %b %d %H:%M:%S %z %Y")


# we can then find the difference between the two:

# In[118]:


t2-t1


# You can also take these *datetime* objects and convert them into dates, which then might be compared.

# In[114]:


t2.date() == t1.date()


# ## 1.2.2. Min and Max
# 
# Find the times of the minimum (earliest) and maximum (latest) tweets in the collection.

# *ANSWER FOLLOWS - insert anwer here*

# In[149]:


from datetime import timedelta
tweets4=Tweets()
tweets4.readTweets('tweets.json')
mint = datetime.max.strptime(tweet['created_at'], "%a %b %d %H:%M:%S %z %Y")
maxt = datetime.min.strptime(tweet['created_at'], "%a %b %d %H:%M:%S %z %Y")
for tid in list(tweets3.getIds()):
    tweet = tweets4.getTweet(tid)
    t = datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S %z %Y")
    if (t-mint) > timedelta(0):
        mint = t
        mintweet = tweet
    if (t-maxt) < timedelta(0):
        maxt = t
        maxtweet = tweet
print(mintweet['full_text'])
print(maxtweet['full_text'])


# *END CUT*
# ****

# ## 1.2.3 Frequency
# 
# Write a routine to find the distribution of the teets in the set by date. You might do this by creating a dictionary that has the dates of tweets as keys.  This routine will be very similar to the routine written above to count the number of tweets by author.
# 
# Note that for this dataset you will probably have all tweets coming from the same date. However, your routine should be generally enough to find the number of tweets for any date represented in the dataset.

# *ANSWER FOLLOWS - cut below here*

# In[150]:


def getDateFrequency(tweets):
    dates={}
    for id in tweets.getIds():
        ctimestring = tweets.getTweet(id)['created_at']
        ctime = datetime.strptime(ctimestring,"%a %b %d %H:%M:%S %z %Y")
        cdate = ctime.date()
        if cdate not in dates:
            dates[cdate]=0
        dates[cdate]=dates[cdate]+1
    return dates


# In[151]:


dfreq = getDateFrequency(tweets3)
len(dfreq)


# In[152]:


keys=list(dfreq.keys())


# In[153]:


d1=keys[0]
d1


# In[154]:


dfreq[d1]


# *END CUT*

# # Exercise 1.3 Other forms of data diversity
# 
# The open-ended nature of social media makes true sampling almost impossible. Unlike sampling based on geographical constraints such as place of residence, data sampled from social media does not draw from any well-characterized population. More simply stated, we might know how many people live in a city or town, but we don't know how many people might have tweeted on a given topic at any given time. 
# 
# However, we can look at the data to ensure that has some diversity. Our exploration of authors provides one example:
# a data set with a range of authors may cover more topics than one with a much smaller number of authors. Diversity of dates and times might help in the same way.  
#                                                                                                                                                                                     The following questions will encourage you to think about other forms of data diversity. 
#                                                
#                                                
# 1.3.1  Why might diversity  of times be of interest?    
#                                                                                                                                                                                      1.3.2  How might you ensure a diversity of times?  
#                                                                                                                                                                                      1.3.3 What other forms of data diversity might be of interest, and how might they be achieved?                                                                                                                                                                           

# *ANSWER FOLLOWS - insert here*
# 

# 1.3.1 Because we can evaluate the data wheather is out of date, and we can use timeline to generate the trend of a thing or things. Besides, through analyze time, we can infer the exact time that thing happened, also we can evaluate the importance of this data through the lasting  period.

# 1.3.2 Lasting for a period, have pretty high frequency in a time, repeat many times.

# 1.3.3 Gender, ages, athnics etc.
# get the author's public profile, or analyze through the ketword in their tweets.

# 
# *END CUT*
# 
# ---

# # Exercise 1.4 Increasing Diversity in Datasets
# 
# The `searchTweets` routine defined in the `Tweets` class atempts to find diverse tweets by ensuring that no tweet is included twice. However, we might see multiple tweets retweeting the same original. How might you modify `searchTweets` to avoid this sort of duplication?

# ----
# *ANSWER FOLLOWS - insert here*
# 

# Because the retweet has the same content RT @ORINGIN AUTHOR. Then, before we do the add modification. We first see does it contains RT in the beginning, if not add it. Else we check did we have same author publish same tweet. 

# *END CUT*
# 
# ----

# # 1.5 Some final notes
# Note that we might find that we will want to add additional fields to this file. We can always rewreite the file as needed. Saving the file as is gives us a good record that we can work from, without having to recreate the dataset. 
# 
# Now that you've mastered the basics of retrieving Twitter data, you can move on to [Part 2](SocialMedia%20-%20Part%202.ipynb).
