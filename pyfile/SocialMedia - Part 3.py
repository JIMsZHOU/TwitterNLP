#!/usr/bin/env python
# coding: utf-8

#  <table><tr><td><img src="images/dbmi_logo.png" width="75" height="73" alt="Pitt Biomedical Informatics logo"></td><td><img src="images/pitt_logo.png" width="75" height="75" alt="University of Pittsburgh logo"></td></tr></table>
#  
# 
# # Social Media and Data Science - Part 3
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

import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import jsonpickle
import json
import random
import tweepy
import spacy
import time
from datetime import datetime


# # 3.0 Introduction
# 
# This module continues the Social Media Data Science module started in [Part 1](SocialMedia%20-%20Part%201.ipynb) and [Part 2](SocialMedia%20-%20Part%202.ipynb), covering the natural language processing analysis of our tweet corpus, providing an introduction to basic concepts of Natural Language Processing.
#   
# Our case study will apply these topics to Twitter discussions of smoking and vaping. Although details of the tools used to access data and the format and content of the data may differ for various services, the strategies and procedures used to analyze the data will generalize to other tools.

# ## 3.0.1 Setup
# 
# Before we dig in, we must grab a bit of code from [Part 1](SocialMedia%20-%20Part%201.ipynb)and [Part 2](SocialMedia%20-%20Part%202.ipynb):
# 
# 1. Our Tweets class
# 3. Our twitter API Keys - be sure to copy the keys that you generated when you completed [Part 1](SocialMedia%20-%20Part%201.ipynb).
# 4. Configuration of our Twitter connection

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
            time.sleep(30)
                
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
                
    def addCode(self,id,code):
        tweet=self.getTweet(id)
        if 'codes' not in tweet:
            tweet['codes']=set()
        tweet['codes'].add(code)
        
   
    def addCodes(self,id,codes):
        for code in codes:
            self.addCode(id,code)
        
 
    def getCodes(self,id):
        tweet=self.getTweet(id)
        if 'codes' in tweet:
            return tweet['codes']
        else:
            return None
        
    # NEW -ROUTINE TO GET PROFILE
    def getCodeProfile(self):
        summary={}
        for id in self.tweets.keys():
            tweet=self.getTweet(id)
            if 'codes' in tweet:
                for code in tweet['codes']:
                    if code not in summary:
                            summary[code] =0
                    summary[code]=summary[code]+1
        sortedsummary = sorted(summary.items(),key=operator.itemgetter(0),reverse=True)
        return sortedsummary


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


# # 3.1 Natural langauge processing

# Our goal is to build a classifier capable of distinguishing tweets related to tobacco smoing from other, unrelated tweets. To do this, we will use ome basic natural language processing to explore the types of words and language found in the tweets. 
# 
# To do this, we will use the [spaCy](https://spaCy.io/) Python NLP package. spaCy provides significant NLP power out-of-the box, with customization facilities offering greater flexibility at various stages of the pipeline. Details can be found at the  [spaCy web site](https://spaCy.io/), and in this [tutorial](https://nicschrading.com/project/Intro-to-NLP-with-spaCy/). spaCy is built on a neural network model based on recent developments in NLP research. See the [spaCy architecture](https://spaCy.io/api/) description for an overview.
# 
# Before we get into the deails, a bit of an introduction. 
# 
# Natural Language Processing involves a series of operations on an input text, each building off of the previous step to add additional insight and undertanding.  Thus, many NLP packages run as pipeline processors providing modular components at each stage of the process. Separating key steps into discrete packages provides needed modularity, as developers can modify and customize individual components as needed. spaCy, like other NLP tools including [GATE](https://gate.ac.uk/) and [cTAKES](https://ctaes.apache.org)  operate on such a model. Although the specific components of each pipeline vary from system to system (and from tasks to task, the key tasks are roughly similar:
# 
# 1. *Tokenizing*: splitting the text into words, punctuation, and other markers.
# 2. *Part of speech tagging*: Classifying terms as nouns, verbs, adjective, adverbs, ec.
# 3. *Dependency Parsing* or *Chunking*: Defining relationships between tokens (subject and object of sentence) and grouping into noun and veb phrases.
# 4. *Named Entity Recognition*: Mapping words or phrases to standard vocabularies or other common, known values. This step is often key for linking free text to accepted terms for diseases, symptoms, and/or anatomic locations.
# 
# Each of these steps might be accomplished through rules, machine learning models, or some combination of approaches. After these initial steps are complete, results might be used to identify relationships between items in the text, build classifiers, or otherwise conduct further analysis. We'll get into these topics later.
# 
# The [spaCy documentation](https://spaCy.io/usage/spaCy-101) and [cTAKES default pipeline description](https://cwiki.apache.org/confluence/display/CTAKES/Default+Clinical+Pipeline) provide two examples of how these components might be arranged in practice.  For more information on NLP theory and methods, see [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/), perhaps the leading NLP textbook.
# 
# Given this introduction, we can read in our tweets and get to work.

# # 3.1.1 Reading in data

# Let's start by reading in the 'smoking' tweets that you created in [Part 1](SocialMedia%20-%20Part%201.ipynb)

# In[5]:


smoking=Tweets()
smoking.readTweets("tweets.json")


# In[6]:


smoking.countTweets()


# To go along with these tweets, let's search for, and save, a set of tweets for the search term 'vaping':

# In[7]:


vaping = Tweets("vaping",100)
vaping.saveTweets("tweets-vaping.json")


# # 3.1.2 NLP Roadmap
# 
# 
# spaCy, like many other natural laguage processing tools, operates as a *pipeline* - a sequential series of operations, each of which conducts some analysis and passes results on to the next.  Each of the steps on the pipeline can operate both on the original text and on any of the results of the previous stages. The basic Spacy pipeline starts with the following steps:
# 
# 1. Tokenizing - splitting into individual elements.
# 2. Tagging - assigning part-of-speech tags
# 3. Parsing - identifying relaionships between elements of a sentence.
# 4. Named Entity Recogntion (NER) - identifying domain-specific nounds and concepts. In biomedical literature, this might mean diseases, symptoms, anatomic locations, etc. 
# 
# Tokenizing is the assumed first stage of every pipeline. To see the contents of a pipeline, we can create an NLP object for the English language and iterate over the components of the pipeline. Although we'll usually use all of the components of the pipeline, they can be [customized](https://spacy.io/usage/processing-pipelines).

# If we can't use nlp = spacy.load('en'). we need to do this in terminal: ```python -m spacy download en```

# In[8]:


import spacy
nlp = spacy.load('en')
for name,proc in nlp.pipeline:
    print(name,proc)


# # 3.1.3 Tokenizing

# Tokenizing is the process of splitting a text into individual components - words - for further processing. Although this might sound simple, the pecularities of the English language and how it is used often make tokenizing more complex than we might expect.
# 
# To see some of the challenges, we will grab a specifc pre-chosen tweet and process it. For demonstration purposes, we will just use the text of the tweet.  
# 
# This will give us a beginning feel for what [Spacy](https://spacy.io) can do, how we might use it, and how we might want to extend and revise the tokenizing process.

# In[9]:


sample='#Smoking affects multiple parts of our body. Know more: https://t.co/hwTeRdC9Hf \n#SwasthaBharat #NHPIndia #mCessation #QuitSmoking https://t.co/x7xHO9G2Cr'
sample


# Tweets have usage patterns that are non-standard English - URLs, hashtags, user references (this particularly tweet was not selected accidentally). These patterns create challenges for extracting content - we might want to know that "#QuitSmoking" is, in a tweet, a hashtag that should be considered as a complete unit.  
# 
# We'll see soon how we might do this, but first, to start the NLP process, we can import the spaCy components and create an NLP object:

# In[10]:


import spacy
nlp = spacy.load('en')


# we can then parse out the text from the first tweet.

# In[11]:


parsed = nlp(sample)


# The result is a list of tokens. We can print out each token to start:

# In[12]:


print([token.text for token in parsed])


# We can see right away that this parsing isn't quite what we would like. Default English parsing treats  `#Smoking`  as two separate tokens - `#` and `Smoking`. Similar problems happen for other hashtags.
# 
# To treat this as a hashtag, we will indeed need to revise the tokenizer. 
# 
# For another example of potential problems, consider this tweet text:

# In[13]:


smoketweet='E-cigarette use by teens linked to later tobacco smoking, study says https://t.co/AhTpFUw0TW'
parsed=nlp(smoketweet)
print( [tok.text for tok in parsed])


# Note that "E-cigarette" becomes three tokens. This is not what we want - we want it to be held together as one. 
# 
# We will revise the spaCy tokenizer to handle these two difficulties - hashtags and "E-cigarette" tokenizing. 

# ## 3.1.3.1 Exception rules
# 
# "E-cigarette" can be handled with some simple exception rules.
# 
# To do this, we can refer to the spaCy docuentation, which describes the process for adding a [special-case tokenizer rule](https://spacy.io/usage/linguistic-features#section-tokenization). Essentially, these rules allow for the possibility of adding new rules to customize parsing for specific domains:
# 
# Each new rule will be a dictionary with three fields:
#     * `ORTH` is the text that will be matched
#     * `LEMMA` is the lemma form
#     * `POS` is the part-of-speech
#     
# These can then be added to the tokenizer:

# In[14]:


from spacy.symbols import ORTH, LEMMA, POS
special_case = [{ORTH: u'e-cigarette', LEMMA: u'e-cigarette', POS: u'NOUN'}]
nlp.tokenizer.add_special_case(u'e-cigarette', special_case)
nlp.tokenizer.add_special_case(u'E-cigarette', special_case)


# These commands suggest the text "e-cigarette" should be handled by the special case rule saying that it is a single token. Now, let's take a look at the result:

# In[15]:


parsed=nlp(smoketweet)
print( [tok.text for tok in parsed])


# Now we capture "E-cigarette" as one token. Note the importance of including both capitalizations in revised rules.

# # 3.1.3.2 Tokenizing hashtags

# As indicators of the progress and content of Twitter conversations, hashtags are important in tweets. For example, some analyses might want to use trends in hashtags, and their mentions in tweets and retweets, to understand conversational dynamics and the spread of ideas. However, as we saw, they are not handled properly by the deafult tokenier. As a reminder: 

# In[16]:


parsed = nlp(sample)
print( [tok.text for tok in parsed])


# we can look specifically at "#Smoking", which becomes two tokens:

# In[17]:


print(parsed[0])
print(parsed[1])


# Note how "#Smoking" is split into "#" and "Smoking". To avoid this, we will can add a specialized processing component as a member of a [spaCy pipeline](https://spacy.io/usage/processing-pipelines).
# 
# To process hashtags, we will use code suggested by a [spaCy
# GitHub issue](https://github.com/explosion/spaCy/issues/503). To see how this should work, let's walkt through some steps:
# 
# First, let's look at the tokens in the tweet parsed above. We can iterate through with enumerate. We can also look at a few interesting elements:
# 
# * `nbor` gets the next token after a token.
# * `idx ` is the position of the token in the list of characters, starting at 0.

# In[18]:


print(str(parsed[0].idx)+" "+parsed[0].text)
print(str(parsed[0].nbor().idx)+" "+str(parsed[0].nbor().text))
print(str(parsed[1].nbor().idx)+" "+str(parsed[1].nbor().text))
print(str(parsed[2].nbor().idx)+" "+str(parsed[2].nbor().text))


# Thus, '#' starts of the string,  and 'Smoking' occupies characters 7 characters starting with character 1.  The 9th characer (index 8) is a space, so the next token ('affects') starts on the 10th character, which has index 9, etc.
# 
# We can use this information to find a hash tag. essentially, we can look for a tag that has the text '#'. If we find one, we can look at the next tag and merge all of the characters from the start of the first tag to the end of the second tag. 

# In[19]:


start=parsed[0].idx
length = len(parsed[1].text)
end = start+length+1
print(str(start))
print(str(end))
parsed.merge(start,end)


# This combines the character starting with 0 up until the character before the character at index 8 (which is a space) to form a new token.

# Now, if we look at the list of tokens, we see that the first two are merged:

# In[20]:


print( [tok.text for tok in parsed])


# To get this to work for all of the tokens in a tweet, we need a routine that will repeatedly iterate over the tokens until we can't find anymore hashtags:
# 

# In[21]:


nlp = spacy.load('en')
def hashtag_pipe(doc):
    merged_hashtag = True
    while merged_hashtag == True:
        merged_hashtag = False
        for token_index,token in enumerate(doc):
            if token.text == '#':
                try:
                    nbor = token.nbor()
                    start_index = token.idx
                    end_index = start_index + len(token.nbor().text) + 1
                    if doc.merge(start_index, end_index) is not None:
                        merged_hashtag = True
                        break
                except:
                    pass
    return doc


# This routine might require a bit of explanation. The main routine in lines 6-16 does the bulk of the work shown above - we find a token that contains only the single character '#', we find the end of the next token, and we merge the two.
# 
# There is one catch in that inner loop. If the last token in the string is a '#', the attempt to read the next token (on line 9) will cause an exception. If this happens, we're done anyway. So we `try` to get the next token. If it fails, we must be at the end of the document, so the `except`  clause does nothing, as indicated by the `pass`.
# 
# However, this is not the whole story. The merging of these two tokens removes one from the list of tokens returned by `enumerate(doc)`. If we continue on, the result of the enumeration will evenutally blow  up, as the code will try to access an element in the set of tokens that is no longer there (try it and see). 
# 
# To get around this, we change the inner loop to `break` out as soon as a pair of tokens are merged. This will start the process over with a new enumeration. This process will repeat until we make it all the way through lines 6-16 - in other words, all of the way through the tweet -  without finding a pair of tokens to merge. When this happens, `merged_hashtag` will stay False, and the outer loop will exit.

# Once we have this routine written, we can then add it to the first position in the pipeline, which will put it after the default tokenizer, but before the part of speech tagger and other components.

# In[22]:


nlp.add_pipe(hashtag_pipe,first=True)


# And then we can try it out...

# In[23]:


doc = nlp("twitter #hashtag")
print(doc[0].text)
print(doc[1].text)


# Returning to our first example...

# In[24]:


sample='#Smoking affects multiple parts of our body. Know more: https://t.co/hwTeRdC9Hf \n#SwasthaBharat #NHPIndia #mCessation #QuitSmoking https://t.co/x7xHO9G2Cr'
print(sample+"\n")
parsed = nlp(sample)
print( [tok.text for tok in parsed])


# We can try a tweet that ends with a '#':

# In[25]:


doc = nlp("twitter #hashtag #")
print([tok.text for tok in doc])


# Great! We can also try a pathological example.

# In[26]:


parsed = nlp("weird hashtag ###tag")
print( [tok.text for tok in parsed])


# Oops. That doesn't work. It's not even clear that this is a legal hashtag. 
# 
# **BONUS CHALLENGE**: Perhaps you can extend the routine to make it handle hashtags started by multiple '#' symbols?

# Summarizing, we can combine the changes to the tokenizer, wrapping them up in a subroutine as follows:

# In[27]:


from spacy.symbols import ORTH, LEMMA, POS

def getTwitterNLP():
    nlp = spacy.load('en')
    special_case = [{ORTH: u'e-cigarette', LEMMA: u'e-cigarette', POS: u'NOUN'}]
    nlp.tokenizer.add_special_case(u'e-cigarette', special_case)
    nlp.tokenizer.add_special_case(u'E-cigarette', special_case)
    def hashtag_pipe(doc):
        merged_hashtag = True
        while merged_hashtag == True:
            merged_hashtag = False
            for token_index,token in enumerate(doc):
                if token.text == '#':
                    try:
                        nbor = token.nbor()
                        start_index = token.idx
                        end_index = start_index + len(token.nbor().text) + 1
                        if doc.merge(start_index, end_index) is not None:
                            merged_hashtag = True
                            break
                    except:
                        pass
        return doc
    nlp.add_pipe(hashtag_pipe,first=True)
    return nlp


# In[28]:


nlp = getTwitterNLP()


# In[29]:


parsed = nlp("weird e-cigarette hashtag ###tag")
print( [tok.text for tok in parsed])


# Note that spaCy can also detect sentences. If you have multiple sentences, they will be found in the results of the parser as spans, each with a start and endpoint, given in terms of the positions of the tokens: 

# In[30]:


parsed= nlp("This is an example of parsing two sentences. Here is the second sentence.")


# In[31]:


for span in parsed.sents:
    print(str(span.start)+" "+str(span.end))


# Thus the first sentence includes token 0-8 and the second includes 9-14:

# It's also possible to access the text of the sentences directly:

# In[32]:


sents = list(parsed.sents)
sents[0].text


# In[33]:


print(parsed[0].text)
print(parsed[8].text)
print(parsed[9].text)
print(parsed[14].text)


# Tokenizers are traditionally built using optimized [regular expressions](https://www.regular-expressions.info/). For more information about tokenizing in paCy, see [spaCy 101](https://spacy.io/usage/spacy-101#section-features) and the [detailed discussion of the spaCy tokenizer](https://spacy.io/usage/linguistic-features#tokenization). For a more general introduction, see [Chapter 2 of Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/).

# ## 3.1.3.3 Lemmatization, stop words, and alpha characterization
# 
# The spaCy tokenizer proivdes a few other useful features along the way:
# 
# * Lemmatization: For each token, spaCy can find the *lemma*: the "standard" or "base" form, reducing verb forms to their base verb, plurals to appropriate singular nouns, etc.  
# * Stop word identification - labelling words as commonly-found words taht add little or no information.
# * Alphanumeric identification - identifying those tokens that contain only alphanumeric values.
# 
# To see these in action, let's review a few tokens:

# In[34]:


sample='#Smoking affects multiple parts of our body. Know more: https://t.co/hwTeRdC9Hf \n#SwasthaBharat #NHPIndia #mCessation #QuitSmoking https://t.co/x7xHO9G2Cr'
parsed=nlp(sample)
print(sample)
print(parsed[1].text)
print(parsed[1].lemma)
print(parsed[1].lemma_)
print(parsed[1].is_stop)
print(parsed[1].is_alpha)


# So, `affects` has the lemma `affect`.  Note that spaCy stores many fields as both hashes for efficiency and as text  for readability. You'll want to use the text form for interpreting results, but the hash for computing. They differ only in the use of the trailing underscore - thus `lemma` is the hash while `lemma_` is the human readable form.
# 
# We can also see that `affect` is not a stop word, and it is alphabetic.
# 
# Some NLP systems will go a bit further than spaCy's lemmatization, using a process called "stemming" to reduce words to base forms. With a stemming algorithm, "scared" might be reduced to "scare" - see this description of [Porter's stemming algorithm](https://tartarus.org/martin/PorterStemmer/) for more detail. 

# If you play around a bit, you might notice that even very common words don't get called stop words:

# In[35]:


text="This is a test of the stop word tool"
doc=nlp(text)
for d in doc:
    print(d.text+", "+str(d.is_stop))


# Clearly, 'This', 'a',' of', and 'the' should be considered stop words. This is a bit of a minor bug. This [Stack Overflow](https://stackoverflow.com/questions/41170726/add-remove-stop-words-with-spacy) post provides a workaround:

# In[36]:


for word in nlp.Defaults.stop_words:
    lex = nlp.vocab[word]
    lex.is_stop = True


# Now try it...

# In[37]:


text="This is a test of the stop word tool"
doc=nlp(text)
for d in doc:
    print(d.text+", "+str(d.is_stop))


# Looks much better. Tying this together with the hashtag pipe routine above:

# In[38]:


def getTwitterNLP():
    nlp = spacy.load('en')
    
    for word in nlp.Defaults.stop_words:
        lex = nlp.vocab[word]
        lex.is_stop = True
    
    special_case = [{ORTH: u'e-cigarette', LEMMA: u'e-cigarette', POS: u'NOUN'}]
    nlp.tokenizer.add_special_case(u'e-cigarette', special_case)
    nlp.tokenizer.add_special_case(u'E-cigarette', special_case)
    def hashtag_pipe(doc):
        merged_hashtag = True
        while merged_hashtag == True:
            merged_hashtag = False
            for token_index,token in enumerate(doc):
                if token.text == '#':
                    try:
                        nbor = token.nbor()
                        start_index = token.idx
                        end_index = start_index + len(token.nbor().text) + 1
                        if doc.merge(start_index, end_index) is not None:
                            merged_hashtag = True
                            break
                    except:
                        pass
        return doc
    nlp.add_pipe(hashtag_pipe,first=True)
    return nlp


# In[39]:


text="This is a test of the stop word tool"
nlp=getTwitterNLP()
doc=nlp(text)
for d in doc:
    print(d.text+", "+str(d.is_stop))


# # 3.1.4  Part-Of-Speech Tagging 

# The next step in NLP is *Part of speech tagging* - classifying each token as one of the parts of speech that we all learned in elementrary school. Parts of speech are assigned to attributes of each token:
# 

# In[40]:


sample='#Smoking affects multiple parts of our body. Know more: https://t.co/hwTeRdC9Hf \n#SwasthaBharat #NHPIndia #mCessation #QuitSmoking https://t.co/x7xHO9G2Cr'
parsed=nlp(sample)
print(parsed[1].text)
print(parsed[1].pos)
print(parsed[1].pos_)


# As discussed before, we have two attributes here - `pos` is the hash code for the part of speech, used for efficiency, while `pos_` is the human readable form. Other attributes derived by spaCy follow the same pattern.
# 
# A second attribute - `tag` - provide es additional information.
# 
# As described in the [spaCy documentation for part-of-speech tags](https://spacy.io/api/annotation#pos-tagging), the tags associated with these two fields come from different sources. 'tag_' uses parts-of-speech from a version of the [Penn Treebank](https://www.seas.upenn.edu/~pdtb/), a well-known corpus of annotated text. 'pos_' uses a simpler set of tags from [A Universal Part-of-Speech Tagset](https://arxiv.org/abs/1104.2086), published by researchers from Google.  
# 
# The tags for `affects` provide an example of the difference. According to the [spaCy documentation ](https://spacy.io/api/annotation#pos-tagging) `VBZ` from the Penn tag set indicates a 'verb, 3rd person singular present', while 'the 'VERB' result for 'pos_' is a more general tag from the Google set. There are many types of verbs in the Penn Treebank that correspond tot the 'VERB' tag from the Google set. 

# In[41]:


print(parsed[1].text)
print(parsed[1].tag_)
print(parsed[1].pos_)


# If you want to learn more about a part of spech tag, you can use `spacy.explain`

# In[42]:


print(spacy.explain(parsed[1].pos_))
print(spacy.explain(parsed[1].tag_))


# Let's look at token 0 ("#Smoking"), token 3 ("parts"), token 11 ("https://t.co/hwTeRdC9Hf'"),  and token 13("#SwasthaBharat") to see a few more tokens in action.

# In[43]:


t0 = parsed[0]
t3 = parsed[3]
t11= parsed[11]
t13 = parsed[13]
print (t0.text,t0.lemma_,t0.pos_,t0.tag_,t0.is_stop,t0.is_alpha)
print (t3.text,t3.lemma_,t3.pos_,t3.tag_,t3.is_stop,t3.is_alpha)
print (t11.text,t11.lemma_,t11.pos_,t11.tag_,t11.is_stop,t11.is_alpha)
print (t13.text,t13.lemma_,t13.pos_,t13.tag_,t13.is_stop,t13.is_alpha)


# Note that URLS are neither alphabetical  nor stop-words, but they are proper nouns

# 
# Let's turn the code that we used above into a routine, along with a routine to print out token details and try another tweet or two. To make things easy to read, we'll use some spaces to format things in columns. 

# In[44]:


def printTokDetails(parsed):
    print("{:25} {:25} {:7}{:7}{:7}{:7}".format("Token text","Lemma","POS","Tag","Stop?","Alpha?"))
    for tok in parsed:
        print("{:25} {:25} {:7}{:7}{:7}{:7}".format(str(tok.text),str(tok.lemma_),str(tok.pos_),str(tok.tag_),str(tok.is_stop),str(tok.is_alpha)))


# In[45]:


tweet_id=random.choice(list(smoking.getIds()))
sample2 = smoking.getText(tweet_id)


# In[46]:


sample2


# In[47]:


parsed2=nlp(sample2)


# In[48]:


printTokDetails(parsed2)


# You might see some interesting pattners arising here.  For example:
# 
# * We see many different type of speech. Initially, we might want to focus on the nouns alone, as they provide much of the content.  
# 
# * Look for words like "is" or "was" - these might all refer to a common lemma term - "be", corresponding to the generic form of he verb. Do you see any other incidents of lemma forms that differ from the parsed text?
# 
# * URLs and icons might be present in tweets. Are they classified as alphanumeric? Should we include them as part of the "useful" text from a tweet? 
# 
# How about another?

# In[49]:


tweet_id=random.choice(list(smoking.getIds()))
sample2 = smoking.getText(tweet_id)
sample2


# In[50]:


parsed2=nlp(sample2)
printTokDetails(parsed2)


# Try a few more of these to get a bit more of a feel for he distribution of lemmas and POS tags. The following shortcut routine will make this a bit easier. 

# In[51]:


def getRandomTweetText(tweets):
    tweet_id = random.choice(list(tweets.getIds()))
    return tweets.getText(tweet_id)


# ---
# ## EXERCISE 3.1: Filtering tokens
# 
# Although NLP parsing is often a good start, further filtering is often necessary to focus on data relevant for specific tasks. In this problem, we will review some additional tweets and develop a post-processing routine capable of filtering tweets as necessary for our needs. 
# 
# 3.1.1 Using the `getRandomTweetText`, and `printTokDetails` routines above, aong with the spaCy `parser` command, examine several tweets to decide which tokens should be included or not.  List criteria for keeeping/removing tokens. Remember to use `spacy.explain()` for any unfamiliar POS or tag entries. Note that your  criteria will not be perfect, and will likely need refinining. Examine enough tweets to feel confident in your criteria. Because we are parsing tweets, please don't forget hashtags and user mentions.
# 
# 3.1.2 Write a routine  `includeToken` that will return a token to be inclued if it matches the criteria that you identified in 3.11, and false otherwise.  Assume for now that we are only interested in nouns and verbs, as they might be a good starting point to find information about vaping or smoking. For any tokens that are included,`includeToken` should return the lemmatized-version of the token, converted to all lower-case and stripped of any whitespace, using `strip()`. Zero-length tokens should not be included.
# 
# 3.1.3 Write a routine `filterTweetTokens` that will filter the parsed tokens from a single tweet, returning a list of the tokens to be included, based on your criteria from `includeToken`. To standardize matters, `filterTweetTokens` should also return the lemmatized-version of the token, converted to all lower-case.
# 
# 3.1.4 Run `filterTweetTokens` on a few tweets. Identify any inaccuracies and explain them. When possible, identify an approach for improving performance, and implement it in a revision version of `filterTweetTokens`.

# ---
# *ANSWER FOLLOWS insert here*
# 

# ```tweets3 = Tweets()
# tweets3.readTweets("tweets.json")
# sample3 = getRandomTweetText(tweets3)
# parsed3 = nlp(sample3)
# printTokDetails(parsed3)```

# In[ ]:


special_case1 = [{ORTH: u'cost-effective', LEMMA: u'cost-effective', POS: u'NOUN'}]
nlp.tokenizer.add_special_case(u'cost-effective', special_case1)
nlp.tokenizer.add_special_case(u'Cost-effective', special_case1)
special_case2 = [{ORTH: u'vape', LEMMA: u'vape', POS: u'NOUN'}]
nlp.tokenizer.add_special_case(u'vape', special_case2)
nlp.tokenizer.add_special_case(u'vaping', special_case2)
nlp.tokenizer.add_special_case(u'Vape', special_case2)
nlp.tokenizer.add_special_case(u'vap', special_case2)
special_case3 = [{ORTH: u'smoke', LEMMA: u'smoke', POS: u'VERB'}]
nlp.tokenizer.add_special_case(u'smoke', special_case3)
nlp.tokenizer.add_special_case(u'Smoke', special_case3)
nlp.tokenizer.add_special_case(u'smoking', special_case3)
nlp.tokenizer.add_special_case(u'Smoking', special_case3)


# ```def includeToken(tok, target):
#     if strip(tok.text) == strip(target):
#         return True
#     else:
#         return False```
#     

# ```def filterTweetTokens(tweets, target):
#     working = []
#     for token in tweets:
#         if includeToken(token, target):
#             working.append(token)
#     return working```
#     

# ```filterTweetTokens(parsed3, smoking)```

# In[63]:


def includeToken(tok):
    val =False
    if tok.is_stop == False:
        if tok.is_alpha == True: 
            if tok.text =='RT':
                val = False
            elif tok.pos_=='NOUN' or tok.pos_=='PROPN' or tok.pos_=='VERB':
                val = True
        elif tok.text[0]=='#' or tok.text[0]=='@':
            val = True
    if val== True:
        stripped =tok.lemma_.lower().strip()
        if len(stripped) ==0:
            val = False
        else:
            val = stripped
    return val

def filterTweetTokens(tokens):
    filtered=[]
    for t in tokens:
        inc = includeToken(t)
        if inc != False:
            filtered.append(inc)
    return filtered


# In[65]:


sample = "You probably should stop smoking whatever it is you are smoking. #VoteRedToSaveAmerica #VoteDemsOut2018 #LiberalLies #LiberalHypocrisy #DefeatElizabethWarren #VoteDeihlForSenate https://t.co/wuwscYIksx"
persed = nlp(sample)
filtered = filterTweetTokens(persed)
filtered


# *END OF ANSWER*
# 
# ---

# We will come back to these routines in [Part 4](SocialMedia%20-%20Part%203.ipynb).

# # 3.1.6  Dependency parsing

# *Dependency parsing* is the process of identifying the syntactic linkages between elements in a sentence. Dependency parsers lin noun phrases and modifiers, subjects to objects, etc. The [spaCy description of dependency parsing](https://spacy.io/usage/linguistic-features#dependency-parse) provides a detailed introduction - here, we provide a brief summary.
# 
# To see the dependencies in action, we can iterate through the tokens, printing out the dependencies, and the head (ie, the token that a token depens upon, and 

# In[42]:


sample='#Smoking affects multiple parts of our body. Know more: https://t.co/hwTeRdC9Hf \n#SwasthaBharat #NHPIndia #mCessation #QuitSmoking https://t.co/x7xHO9G2Cr'
print(sample)
print("----")
parsed=nlp(sample)
for chunk in parsed.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)


# In[43]:


for token in parsed:
    print(token.text,token.dep_,token.head.text, token.pos_)


# We can see a few things from this example:
# 
# 1. `#Smoking` is a noun subject of the sentence, dependent on the verb `affects`.
# 2. `affects` is a verb at the ROOT level.
# 3. `multiple` is an adjective modifier that modifies `parts`.
# 4. `parts` is a noun that is the direct object of `affects`, etc..
# 
# We can look in more deail a the text, dependency,  head, and children, of each token.   

# In[44]:


def printParseTree(parsed):
    print("{:10} {:10} {:7} {:7} {:30}".format("Token text","dep","Head text","POS","Children"))
    for tok in parsed:
        children=[child.text for child in tok.children]
        children=",".join(children)
        print("{:10} {:10} {:7} {:7} {:30}".format(str(tok.text),str(tok.dep_),str(tok.head.text),str(tok.head.pos_),children))


# In[45]:


sents =list(parsed.sents)
print(sents[0])
printParseTree(sents[0])


# Here we can see that that 'affects' is the root verb, with `#Smoking` as a noun subjects and `parts` as the object. `Parts`  is modified by `muliple` and `of our body`'.  
# 
# We can use the [displacy](https://spacy.io/usage/visualizers#section-dep) renderer to show a graphical depiction of the dependencies. Since displacy seems to prefer showing thepare tree fror an entire document, we'll try it on a single sentence.
# 
# Note - the "%%capture" line below tells Jupyter to hide some very ugly errors, whie still displaying the nice graphical result. 

# In[46]:


get_ipython().run_cell_magic('capture', '--no-display', 'from spacy import displacy\n\ntext="#Smoking affects multiple parts of our body."\nparsed=nlp(text)\ndisplacy.render(docs=[parsed],jupyter=True, options={\'distance\': 90})')


# This diagram shows the structure given above in the printed version of the parse tree. 
# 
# These relationships might be useful for some NLP goals, particularly those involving relationships between concpets. 
# 
# A variety of approaches - including greedy algorithms, graph-based methods, and machine learning - can be used to extract dependencies. 

# # 3.1.7 Named Entity Recognition

# *Named entity recognition* is the process of extracting categories to known entities - places, people, things, ec. spaCy provides a statistical model capable of assigning an [entity type](https://spacy.io/api/annotation#named-entities) to many of the terms in a document. For an example, let's look at the entities found in a tweet:

# In[48]:


sample ='Scott Gottlieb of @FDA says 8 million lives could be saved by cutting nicotine levels #publichealth https://phony.url/123'
print(sample)
parsed=nlp(sample)
print("----")
for ent in parsed.ents:
    print(ent.text,ent.label_)


# Note that entities are not equivalent to tokens: `Scott Gottlieb` and `8 million` are entities, but not tokens. For comparison:

# In[49]:


print([tok.text for tok in parsed])


# Thus, two tokens - `Scott` and `Gottlieb` are combined to form a single entity - `Scott Gottlieb'.' We can modify the above to see where each entity starts and ends:

# In[50]:


print("{:15} {:5} {:5} {:5}".format("Text","Start","End","Type"))
for ent in parsed.ents:
    print("{:15} {:5} {:5} {:5}".format(ent.text,ent.start_char,ent.end_char,ent.label_))


# Thus, `Scott Gottlieb` starts at character 0 and goes up through (but not including) character 14.
# 
# We can also use the spaCy visualizer to look at the named entities in a sentence:

# In[51]:


get_ipython().run_cell_magic('capture', '--no-display', "displacy.render(docs=[parsed],jupyter=True, style='ent')")


# 
# 
# Let's try another.

# In[52]:


sample='How many people in New York are smokers?'
print(sample)
parsed=nlp(sample)
print("----")
print("{:15} {:5} {:5} {:5}".format("Text","Start","End","Type"))
for ent in parsed.ents:
    print("{:15} {:5} {:5} {:5}".format(ent.text,ent.start_char,ent.end_char,ent.label_))


# As a named entity type, `GPE` stands for geopolitical entity.
# 
# Here, we note that hashtags are not necessarily categorized as entities.  This might be a shortcoming if we were going to use named entities as part of our strategy for classiying tweets. The spaCy named entity recognizer is based on statistical models that can be extended given enough training data. See the discussion of [training the named entity recognizer](https://spacy.io/usage/training#section-ner) for details on how this might be done. 
# 
# *Challenge*: Collect some tweets with hashtags and train the spaCy named entity recognizer add a `HASHTAG` as a new entity type.

# # Exercise 3.2
# 
# The natural language processing pipeline consists of several processes that add substantial structure to our understanding of these Tweets. Tokenizing, part of speech tagging, lemmatiziation, dependency parsing, and named entity recognition each add different details that might be used to understand and classify documents, while also providing some hints as to interesting questions that we might ask.
# 
# Review some tweets and discuss any patterns or questions that arise. You might consider some of the following:
# 
# * Are there terms that show up more frequently in the vaping tweets as opposed to the smoking tweets?
# * Are the tokens that we filtered (in Exercise 3.1) useful, or do we need the whole set of tokens to inerpret
# * Are the named entities informative?
# 
# Describe any other interesting phenomena that you think you might see in the corpus. Note that this question is not asking for fully statistically supported models. Rather, we're just looking for things that might be interesting to pursue further: it may turn out that any "patterns" you identify here are just incidental.
# 

# ---
# *ANSWER FOLLOWS - insert answer here *
# 

# If we want to understand the meanning of the tweets, we should know the position of author. Therefore, we should revongnize the manner. We DONT need all tokens to analyze, we should put eyes on ADJ words, and maybe verb tense. Those information will be vital in analyze the meanning of author's words.

# *END OF ANSWER*
# 
# ---

# # 3.2 Comparing Vocabularies
# 
# Although the examination of a few selected tweets might help us understand some of the trends in terminology and how they differ between the `smoking` and the `vaping` sets, these spot checks may not give a balanced picture of text usage across both of the corpora.  Here we will try to more systematically address the questions that you considered in Exercise 3.2.
# 
# 
# A most systematic way to go about this would be to identify frequently-occurring tokens in  both corpora, using methods similar to those used in our examination of frequent authors from  [Part 1](SocialMedia%20-%20Part%201.ipynb) and in `getCodeProfile()` from [Part 2](SocialMedia%20-%20.ipynb). Specifically, we will write a routine that iterates through all of the tweets in a Tweets object and does the following:
# 
# 1. Parse the tweet
# 2. Filter tokens (using the routines developed above).
# 3. Adds each token to a hash assoicating each token with a count of the number of times it has appeared in the corpus
# 
# The result will be a hash with the number of times each term occurs in the corpus. We can then sort this hash by descending values of the count to find the most frequent terms, and we can comapare results for the two sets. We'll return this information in two forms - a hash (for quick access) and a list (for sorting):

# In[67]:


def getFrequentTerms(tweets,filtered=True):
    frequents={}
    for id in tweets.getIds():
        text = tweets.getText(id)
        parsed=nlp(text)
        if filtered ==True:
            toks = filterTweetTokens(parsed)
        else:
            toks = [tok for tok in parsed]
        
        for tok in toks:
            if tok not in frequents:
                frequents[tok]=0
            frequents[tok]=frequents[tok]+1
    sorts=sorted(frequents.items(),key=operator.itemgetter(1),reverse=True)
    return frequents,sorts


# In[68]:


smokFreqs,smokSorted=getFrequentTerms(smoking)


# In[69]:


len(smokFreqs)


# In[70]:


smokSorted[:20]


# In[71]:


vapFreqs,vapSorted = getFrequentTerms(vaping)


# In[72]:


vapSorted[:20]


# We can go through these lists to get an idea of some of the commonalities. One way to do this would be to create a new list containing all of the terms found in both lists, along with their counts for each list:

# In[73]:


merged=[]
for w,count in smokFreqs.items():
    if w in vapFreqs:
        vcount=vapFreqs[w]
        item= (w,count,vcount)
        merged.append(item)
merged


# From these lists, a few observations come to mind:
# 
# 1. There are significant repeats, particularly in the top 20 terms.
# 
# 2. Frequent occurrences of terms like 'cigarette' in both list suggest that distinguishing between the two sets of tweets might be difficult.
# 
# 3. The vaping datasset contains many similar frequent terms like 'vap','vaping', 'vapor'. These similarities are also seen in related hashtags - `#vape`, `#vaping`, `#vapelife`, `#vapor`, etc.

# # Exercise 3.3
# 
# Based on these observations of the frequent terms, we will consider some of the questions raised in exercise 3.2

# 3.3.1 Exercise 3.1 and the `getFrequentTerms` method developed above take the relevant tokens from each tweet to be only those that meet certain criteria. However, we do not separately include the named entities.  One possible improvement would be to add any named entities to the list of tokens to be included. Revise `filterTweetTokens` to add any named entities to the end of the token list.  
# 
# *Note* - be sure to add entities to the list only if they have a length of greater than zero! 
# 
# Try the result on a few tokens.  Do you see any potential problems with the simple approach to doing this? Does this strategy seem worth pursuing?
# 
# 3.3.2 We noticed that there are many repeated patterns in the tweets for vaping, including many terms and hashtags prefixed with 'vap'. One possible approach to this would be to further revise the lemmatizer to reduce these entries to common forms - perhaps 'vape' and '#vape'. Revise the `getTwitterNLP` routine above to include a lemmatizer that handles these cases.

# ---
# *ANSWER FOLLOWS - insert answer here*

# In[66]:


def filterTweetTokens(tokens):
    filtered=[]
    for t in tokens:
        inc = includeToken(t)
        if inc != False:
            filtered.append(inc)
    
    for ent in tokens.ents:
        filtered.append(ent)
    return filtered


# Because the entities also is a token, so when we add to the end, we will find many duplicates in the list. 

# In[ ]:


from spacy.symbols import ORTH, LEMMA, POS

def getTwitterNLP():
    nlp = spacy.load('en')
    special_case = [{ORTH: u'e-cigarette', LEMMA: u'e-cigarette', POS: u'NOUN'}]
    nlp.tokenizer.add_special_case(u'e-cigarette', special_case)
    nlp.tokenizer.add_special_case(u'E-cigarette', special_case)
    special_case1 = [{ORTH: u'cost-effective', LEMMA: u'cost-effective', POS: u'NOUN'}]
    nlp.tokenizer.add_special_case(u'cost-effective', special_case1)
    nlp.tokenizer.add_special_case(u'Cost-effective', special_case1)
    special_case2 = [{ORTH: u'vape', LEMMA: u'vape', POS: u'NOUN'}]
    nlp.tokenizer.add_special_case(u'vape', special_case2)
    nlp.tokenizer.add_special_case(u'vaping', special_case2)
    nlp.tokenizer.add_special_case(u'Vape', special_case2)
    nlp.tokenizer.add_special_case(u'vap', special_case2)
    special_case3 = [{ORTH: u'smoke', LEMMA: u'smoke', POS: u'VERB'}]
    nlp.tokenizer.add_special_case(u'smoke', special_case3)
    nlp.tokenizer.add_special_case(u'Smoke', special_case3)
    nlp.tokenizer.add_special_case(u'smoking', special_case3)
    nlp.tokenizer.add_special_case(u'Smoking', special_case3)
    def hashtag_pipe(doc):
        merged_hashtag = True
        while merged_hashtag == True:
            merged_hashtag = False
            for token_index,token in enumerate(doc):
                if token.text == '#':
                    try:
                        nbor = token.nbor()
                        start_index = token.idx
                        end_index = start_index + len(token.nbor().text) + 1
                        if doc.merge(start_index, end_index) is not None:
                            merged_hashtag = True
                            break
                    except:
                        pass
        return doc
    nlp.add_pipe(hashtag_pipe,first=True)
    return nlp

def includeToken(tok):
    val =False
    if tok.is_stop == False:
        if tok.is_alpha == True: 
            if tok.text =='RT':
                val = False
            elif tok.pos_=='NOUN' or tok.pos_=='PROPN' or tok.pos_=='VERB':
                val = True
        elif tok.text[0]=='#' or tok.text[0]=='@':
            val = True
    if val== True:
        stripped =tok.lemma_.lower().strip()
        if len(stripped) ==0:
            val = False
        else:
            val = stripped
    return val

def filterTweetTokens(tokens):
    filtered=[]
    for t in tokens:
        inc = includeToken(t)
        if inc != False:
            filtered.append(inc)
    
    for ent in tokens.ents:
        filtered.append(ent)
    return filtered


# *END OF ANSWER cut above here*
# 
# ---

# # 3.3 Final Notes
# 
# Now that you've seen the basics of using natural language processing to extract understanding from the tweets, you're ready to move on to the next step.  [Part 4](SocialMedia%20-%20Part%204.ipynb) will take the results of the NLP output and create basic classifier machine learning models. 
