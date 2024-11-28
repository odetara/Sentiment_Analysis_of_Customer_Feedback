#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis of Customer Feedback: Enhancing Products and Services with Precision

# ### Data Collection and Loading

# In[4]:


# Importing Libraries
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words("english"))


# In[6]:


# Data Loading
train_df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Amdari Project\Sentiment Analysis Amdari\datasets\e commerce reviews train.csv")


# In[8]:


test_df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Amdari Project\Sentiment Analysis Amdari\datasets\e commerce reviews test.csv")


# In[10]:


train_df.info()


# In[12]:


test_df.info()


# In[14]:


train_df.head(10)


# In[16]:


# get the row at index 6
print(train_df.iloc[6]['text'])


# ### Text Processing

# In[19]:


#First let change the label
###label 1: 1 and 2 stars ratings ==> negative
###label 2: 4 and 5 stars rating ==> positive


# In[21]:


train_df['labels'].unique()


# In[23]:


##lets map the labels to sentiment words, positive, negative

mapping_values = {
    '__label__1': "negative",
    "__label__2": "positive"
}


# In[25]:


train_df['labels'].map(mapping_values)


# In[27]:


#mapping labels columns
train_df['labels'] = train_df['labels'].map(mapping_values)


# In[29]:


test_df['labels'] = test_df['labels'].map(mapping_values)


# In[31]:


train_df.head(10)


# In[33]:


test_df.head(10)


# In[35]:


text = "I love this product, it is good"


# In[37]:


nltk.word_tokenize(text)


# In[39]:


# Tokenize the text (split it into words)
words = nltk.word_tokenize(text)


# In[41]:


# Remove stopwords from the text
filtered_words = [word for word in words if word.lower() not in stop_words]

# Reconstruct the text without stopwords
filtered_text = " ".join(filtered_words)

print(filtered_text)


# In[43]:


def remove_stopwords(text):
    """
    this function take a sentence
    tokenize.. the sentence
    filters out stopwords and return a more compactsentence
    """
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_words)
    return filtered_text


# In[45]:


# Remove stopwords from the text
remove_stopwords(text)


# In[47]:


train_df["text"]


# In[49]:


train_df["text"].head(10).apply(remove_stopwords)


# In[51]:


train_df["text"].head(10)


# In[57]:


##i would love to see a progress bar when we process for all the 3.6 million reviews
total_rows = len(train_df)
tqdm.pandas(total=total_rows)
train_df['stop words'] = train_df['text'].progress_apply(remove_stopwords)


# In[59]:


train_df


# In[63]:


total_rows = len(test_df)
tqdm.pandas(total=total_rows)
test_df['stop words'] = test_df['text'].progress_apply(remove_stopwords)


# In[65]:


test_df


# In[67]:


#Create a bag of words and TF-IDF


# A "Bag of Words" (BoW) is a simple and commonly used technique in natural language processing (NLP) and text analysis to represent text data as numerical features. It is used to transform a collection of text documents into a format that can be processed by machine learning algorithms. The idea behind the Bag of Words model is to disregard the order and structure of words in a text and focus only on the frequency of each word's occurrence.
# 
# The key idea is that the order of words and the grammatical structure of sentences are ignored, and the analysis is purely based on the presence or absence of specific words and their frequencies.

# TF-IDF, which stands for "Term Frequency-Inverse Document Frequency," is a numerical statistic used in information retrieval and natural language processing (NLP) to evaluate the importance of a word within a document relative to a collection of documents, typically a corpus.
# 
# The TF-IDF score provides a measure of how important a term is within a specific document and across a collection of documents. Terms that appear frequently in a document but rarely in other documents receive higher TF-IDF scores, making them indicative of the content of that document.

# In[70]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[72]:


vectorizer = CountVectorizer() # You can adjust max_features as needed
train_bow = vectorizer.fit_transform(train_df['stop words'])
test_bow = vectorizer.transform(test_df['stop words'])


# In[74]:


tfidf_vectorizer = TfidfVectorizer()  # You can adjust max_features as needed
train_tfidf = tfidf_vectorizer.fit_transform(train_df['stop words'])
test_tfidf = tfidf_vectorizer.transform(test_df['stop words'])


# **MODELLING AND EVALUATION**

# In[77]:


#Vader on normal Sentences
from sklearn.metrics import accuracy_score, classification_report


# In[79]:


import nltk
# download the VADER lexicon and model
nltk.download('vader_lexicon')


# In[81]:


# import the SentimentIntensityAnalyzer class from vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Vader: pretrain model for analyzing sentiment of sentence
analyzer = SentimentIntensityAnalyzer()


# In[91]:


## test out the sentiment analyzer with an example text

example_text  = "i love the orange flavor, good product"

sentiment_scores = analyzer.polarity_scores(example_text)

# The sentiment_scores dictionary will contain the scores.
print(sentiment_scores)


# In[93]:


# getting the sentiment scores
compound_score = sentiment_scores['compound']

#now lets make a decision for the cut off for a postitive or negative score
if compound_score > 0:
    sentiment = "Positive"
else:
  sentiment = "Negative"

print(f"The sentiment is {sentiment} (Compound Score: {compound_score})")


# In[95]:


## apply all the text in our dataset, so lets first
## create the function, then we apply the function

def analyze_sentence(sentence, threshold = 0):
  sentiment_scores = analyzer.polarity_scores(sentence)
  compound_score = sentiment_scores['compound']

  if compound_score > threshold:
    sentiment = "positive"
  else:
    sentiment = "negative"

  return sentiment


# In[97]:


inferences_0 = test_df['text'].progress_apply(analyze_sentence)


# **USING THE ACCURACY METRICS AND CLASSIFIACTION REPORT**

# In[100]:


accuracy_score(inferences_0, test_df['labels'])


# In[102]:


print(classification_report(test_df['labels'],inferences_0 ))


# In[104]:


# VADER ON STOP WORDS
# now lets repeat on stopwords, lets see if by removing context irrelvant words we can improve the scores of vader


# In[106]:


inferences_1 = test_df['stop words'].progress_apply(analyze_sentence)


# In[108]:


# get the accuracy scores, then the classifcation report
accuracy_score(inferences_1, test_df['labels'])


# In[110]:


print(classification_report(test_df['labels'],inferences_1 ))


# In[112]:


## TRAINING AND TESTING CUSTOM MODELS: Multinomial NB
## choosing it for its simplicity, speed and compatibility with bag of words and tfidf


# In[114]:


from sklearn.naive_bayes import MultinomialNB


# In[116]:


#create a classifier
classifier = MultinomialNB()


# In[120]:


#fit on bag_of_words
classifier.fit(train_bow, train_df['labels'])


# In[122]:


##lets make predictions and evaluate the model

y_pred = classifier.predict(test_bow)
accuracy = accuracy_score(test_df['labels'], y_pred)

#printing results
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(test_df['labels'], y_pred))


# In[123]:


#create and train a second classifier on tf-idf
classifier2 = MultinomialNB()


# In[126]:


classifier2.fit(train_tfidf, train_df["labels"])


# In[128]:


y_pred = classifier.predict(test_tfidf)
accuracy = accuracy_score(test_df['labels'], y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(test_df['labels'], y_pred))


# **DEPLOYMENT: INFERENCE SCRIPT AND FLASK APP**

# In[131]:


## create an inference function to receive a text, remove stopwords, convert to bow and pass to MUltinomialNB model

stop_words = set(stopwords.words("english"))
def remove_stopwords(text,stop_words = stop_words):
  words = nltk.word_tokenize(text)
  # Remove stopwords from the text
  filtered_words = [word for word in words if word.lower() not in stop_words]
  # Reconstruct the text without stopwords
  filtered_text = " ".join(filtered_words)
  #print(filtered_text)

  return filtered_text

def inference(text):
  filtered_text = remove_stopwords(text)
  bow = vectorizer.transform([filtered_text])
  sentiment = classifier.predict(bow)
  return sentiment


# In[133]:


example_text = "i hate this book."


# In[135]:


inference(example_text)


# In[137]:


## FLASK APP

get_ipython().system('pip install Flask')


# In[139]:


from flask import Flask

app = Flask(__name__)

@app.route('/')
def inference(text):
  filtered_text = remove_stopwords(text)
  bow = vectorizer.transform([filtered_text])
  sentiment = classifier.predict(bow)
  return sentiment

if __name__ == '__main__':
    app.run()


# In[ ]:


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)

