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