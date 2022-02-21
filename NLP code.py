

"""Text preprocessing and operations"""

'''
Process -
Tokenize
POS Tagging
Applying Naive Bayes Classifier

'''

import nltk
nltk.download()
data="In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans. Colloquially, the term 'artificial intelligence' is often used to describe machines (or computers) that mimic 'cognitive' functions that humans associate with other human minds, such as learning and problem solving. As machines become increasingly capable, tasks considered to require 'intelligence' are often removed from the definition of AI, a phenomenon known as the AI effect.[2] A quip in Tesler's Theorem says, 'AI is whatever hasn't been done yet'.[3] For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.[4] Modern machine capabilities generally classified as AI include successfully understanding human speech,[5] competing at the highest level in strategic game systems (such as chess and Go),[6] autonomously operating cars, intelligent routing in content delivery networks, and military simulations. Artificial intelligence can be classified into three different types of systems: analytical, human-inspired, and humanized artificial intelligence.[7] Analytical AI has only characteristics consistent with cognitive intelligence; generating cognitive representation of the world and using learning based on past experience to inform future decisions. Human-inspired AI has elements from cognitive and emotional intelligence; understanding human emotions, in addition to cognitive elements, and considering them in their decision making. Humanized AI shows characteristics of all types of competencies (i.e., cognitive, emotional, and social intelligence), is able to be self-conscious and is self-aware in interactions with others."
print(data)

#Tokenize
from nltk import sent_tokenize
sent_tokenize(data)                     #Check this
#Separates lines of the paragraph

from nltk import word_tokenize
word_tokenize(data)
#Make word tokens

#Stem
from nltk.stem import PorterStemmer
ps=PorterStemmer()
ps.stem('Cars')
ps.stem('boys')
ps.stem('goes')
#Finds the root words
#We cannot find the root word of goes, so we use lemmatizers.

#ls.stem('goes')
#ls.stem('does')

#Lemmatizers
from nltk import stem
wd=stem.WordNetLemmatizer()
wd.lemmatize('goes')

wd.lemmatize('cars','n')    #Lemmatize as a noun
wd.lemmatize('went','v')    #Lemmatize as a verb
wd.lemmatize('is','v')    #Lemmatize as a verb



"""Text Similarity (Cosine Similarity)"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = ["In a country like India, where a mass of population lives in the villages", "there are times when parents have work to take care of and cannot keep an eye on their children all day round", "In many cases, when even the females in houses have to go to work, but cannot always carry their children along with them", "they have no other choice but to leave them back at their homes. Due to these reasons and more, there has been a rapid increase in security risks of children recently"]
vector.fit(corpus)
print(vector.transform(["country population mass times parents parents parents hello"]).toarray())
print(vector.transform(["country population mass times parents parents parents hello"]))

vector = CountVectorizer(binary = False)
corpus = ["In a country like India, where a mass of population lives in the villages", "there are times when parents have work to take care of and cannot keep an eye on their children all day round", "In many cases, when even the females in houses have to go to work, but cannot always carry their children along with them", "they have no other choice but to leave them back at their homes. Due to these reasons and more, there has been a rapid increase in security risks of children recently"]
vector.fit(corpus)
v1 = vector.transform(["country population mass times parents parents parents hello"]).toarray()
print(v1)
v2 = vector.transform(["hello world country population so scared"]).toarray()
print(vector.transform(["country population mass times parents parents parents hello"]))

similarity = cosine_similarity(v1, v2)
print(similarity)



"""Translating sentences"""
import textblob
from textblob import TextBlob
data=TextBlob('Hello everyone! Hope you are enjoying machine learning.')
data.translate(to='hi')     #Translate to hindi language
data.translate(to='fr')     #Translate to french language
data.translate(to='ar')     #Translate to arabian language
data.translate(to='bn')    #Translate to bangla language


"""Sentiment Analysis"""

'''
Polarity -
tends towards:
emotional negative(-2)
Rational negative(-1)
neutral(0)
Rational positive(+1)
Emotional positive(+2)

Subjectivity -
tends towards:
Personal opinion(+1)
General fact(0)

'''
data=TextBlob('The movie was good. The cinematography was very good.')
data.sentiment

data=TextBlob('The auto was uncomfortable. The driver was rude.')
data.sentiment

data=TextBlob('The task was done in a right way. Hard work pays off.')
data.sentiment


"""Spelling correction"""

data=TextBlob('I havv two cars')
print(data.correct())

data=TextBlob('He seeme to be nic')
print(data.correct())

data=TextBlob('The sceneuri is really beutifu')
print(data.correct())




"""Categorizing the sentences into some category, based on the sentences"""

'Model training'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()
data.target_names


categories=['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
#training the data on these categories
train=fetch_20newsgroups(subset='train',categories=categories)
#testing the data on these categories
test=fetch_20newsgroups(subset='test',categories=categories)
print(train.data[5])

print(len(train.data))

print(len(test.data))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


from sklearn.feature_extraction.text import TfidfVectorizer
#Creating model
model=make_pipeline(TfidfVectorizer(),MultinomialNB())
#Training the models
model.fit(train.data,train.target)
#Creating label for test data
labels=model.predict(test.data)


#Predicting categories
def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]


predict_category('The US has thousands of people killed each year by the militants. There are so many people moving into the US as immigrants.')
predict_category('Audi has better technology than BMW')
predict_category('Hockey is the national sport of India')
predict_category('Windows operating system is widely used across the world')
predict_category('Scientists have made the discovery of a new planet in another galaxy')




""""Chatbot"""
"""
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

bot=ChatBot('friday')
ChatBot.train(['What is your name?','My name is Sophie'])
trainer=ChatterBotCorpusTrainer(bot)
trainer.train('chatterbot.corpus.english')

def chat(s):
  print(bot.get_response(s))
  return

chat('who are you?')

chat('who is spiderman')

chat('I am a failure')

"""


#importing libraries
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request  

#fetching the content from the URL
fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/20th_century')

article_read = fetched_data.read()

#parsing the URL content and storing in a variable
article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')

#returning <p> tags
paragraphs = article_parsed.find_all('p')

article_content = ''

#looping through the paragraphs and adding them to the variable
for p in paragraphs:  
    article_content += p.text


def _create_dictionary_table(text_string) -> dict:
   
    #removing stop words
    stop_words = set(stopwords.words("english"))
    
    words = word_tokenize(text_string)
    
    #reducing words to their root form
    stem = PorterStemmer()
    
    #creating dictionary for the word frequency table
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table


def _calculate_sentence_scores(sentences, frequency_table) -> dict:   

    #algorithm for scoring a sentence by its words
    sentence_weight = dict()

    for sentence in sentences:
        sentence_wordcount = (len(word_tokenize(sentence)))
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:7] in sentence_weight:
                    sentence_weight[sentence[:7]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:7]] = frequency_table[word_weight]

        sentence_weight[sentence[:7]] = sentence_weight[sentence[:7]] / sentence_wordcount_without_stop_words

       

    return sentence_weight

def _calculate_average_score(sentence_weight) -> int:
   
    #calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    #getting sentence average value from source text
    average_score = (sum_values / len(sentence_weight))

    return average_score

def _get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''

    for sentence in sentences:
        if sentence[:7] in sentence_weight and sentence_weight[sentence[:7]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary

def _run_article_summary(article):
    
    #creating a dictionary for the word frequency table
    frequency_table = _create_dictionary_table(article)

    #tokenizing the sentences
    sentences = sent_tokenize(article)

    #algorithm for scoring a sentence by its words
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)

    #getting the threshold
    threshold = _calculate_average_score(sentence_scores)

    #producing the summary
    article_summary = _get_article_summary(sentences, sentence_scores, 1.5 * threshold)

    return article_summary

if __name__ == '__main__':
    summary_results = _run_article_summary(article_content)
    print(summary_results)


























































































































































































































