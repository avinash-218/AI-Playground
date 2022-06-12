'''
Disclaimer:

DeepSphere.AI developed these materials based on its team’s expertise and technical infrastructure, and we are sharing these materials strictly for learning and research.
These learning resources may not work on other learning infrastructures and DeepSphere.AI advises the learners to use these materials at their own risk. As needed, we will
be changing these materials without any notification and we have full ownership and accountability to make any change to these materials.

Author :                          Chief Architect :       Reviewer :
____________________________________________________________________________
Avinash R & Jothi Periasamy       Jothi Periasamy         Jothi Periasamy
'''

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow import keras
import json
import random

def load_dependencies():#load model,intents,words,classes files
    model = keras.models.load_model('../Utility/DSAI_Tbot_Chatbot_Model.h5') #load model
    intents = json.loads(open('../Utility/DSAI_Tbot_Intents.json').read()) #load json file
    words = pickle.load(open('../Utility/DSAI_Tbot_Words.pkl','rb')) #load saved words object (pickle)
    classes = pickle.load(open('../Utility/DSAI_Tbot_Classes.pkl','rb')) #load saved classes object (pickle)
    return (model, intents, words, classes)

def tokenize(sentence):#tokenize sentence into words
    #sentence - input sentence
    return nltk.word_tokenize(sentence)

def lemmatize(words):#lemmatize words of input sentence
    #words - tokenized sentence words
    lemmatizer = WordNetLemmatizer() #lemmatizer
    sentence_words =[]
    for i in words:
        sentence_words.append(lemmatizer.lemmatize(i.lower()))
    return sentence_words

def bag_of_words(sentence_words, words):# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    #sentence_words - tokenized words
    #words - words.pkl extracted from JSON
    bag = [0]*len(words)# bag of words - array of N words, vocabulary matrix
    for s in sentence_words:#loop through input sentence
        for i,w in enumerate(words):#loop through saved words
            if(w == s): #if input word present in sentence set as 1
                bag[i] = 1
    return(np.array(bag))

def preprocess(sentence, words):#preprocessing of input sentence
    #sentence - input sentence
    #words - words.pkl extracted from JSON
    sentence_words = tokenize(sentence)#tokenize
    sentence_words = lemmatize(sentence_words)#lemmatize
    return bag_of_words(sentence_words, words) #extract bag of words

def predict_class(bag, classes, model):
    #filter out predictions below a threshold
    #classes - classes from JSON
    #model - saved model by which class is to be predicted
    #model outputs predicted class array and probability
    res = model.predict(np.array([bag]))[0] #predict class- predicted array of probability for classes
    print("res", res, end='\n\n')
    results=[]
    for i,r in enumerate(res):
        if(r > 0.25): #probability threshold
            results.append([i,r]) #filter out classes with probability <= 0.25 appending [index, probability]
    print("results", results,end='\n\n')

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)#contains filtered predicted class array and probability in sorted order by probability
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})#based on index, take the corresponding tag from classes.pkl
    print("return list", return_list,end='\n\n')
    return return_list#return [{'intent':tag, 'probability':prob}]

def getResponse(ints, intents_json):
    #ints - [{'intent':tag, 'probability':prob}]
    #intents_json - json file
    # get response for the input message after predicting class
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:#for each tags in json data
        if(i['tag']== tag):#if tag matches
            return random.choice(i['responses'])#return random response

'''
Copyright Notice:

Local and international copyright laws protect this material. Repurposing or reproducing this material without written approval from DeepSphere.AI violates the law.

(c) DeepSphere.AI
'''