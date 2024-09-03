'''This is model file for training Chatbot with the intents'''
#we imported some required packages
import nltk #it is a python package used for tokenize words
import tensorflow #Tensor package is used to create chatbot model
from nltk.stem import WordNetLemmatizer #this nltk module help us to lemmatize the words
lemmatizer=WordNetLemmatizer() #created object for WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,MaxPool2D,Activation,Flatten,Conv2D
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import json
import random
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

#this lists are used to store the data from json
words=[]
docs=[]
labels=[]
ignore=[]

#Intents File Declared
data = open('intents1.json').read()
intents = json.loads(data)

#breaking text into small forms
for intent in intents["intents"]:
  for pattern in intent["patterns"]:
    word_token=nltk.word_tokenize(pattern)
    words.extend(word_token)
    docs.append((word_token,intent["tag"]))
    if intent["tag"] not in labels:
      labels.append(intent['tag'])

#Lemmatization is another technique used to reduce inflected words to their root word. 
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
labels=sorted(list(set(labels)))
print(words)

#To Store Words and Labels inside pickle 
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(labels,open('labels.pkl','wb'))
print(words)

#Mapping Text Based on Occurance
training=[]
output=[0]*len(labels)
for doc in docs:
  bag_of_words=[]
  pattern_words=doc[0]
  pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]

  for w in words:
    if w in pattern_words:
      bag_of_words.append(1)
    else:
      bag_of_words.append(0)
    output_row=list(output)
    output_row[labels.index(doc[1])]=1

    training.append([bag_of_words,output_row])

# Based on your needs(shuffle the words)
random.shuffle(training)
training=np.array(training,dtype=object)

x_train=list(training[:,0])
y_train=list(training[:,1])

#Sequentail Network (Basic RNN)
model=Sequential()
model.add(Dense(128,input_shape=(len(x_train[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

model.summary()

#stochastic gradient descendent used to optimize the model 
sgd_optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy",optimizer=sgd_optimizer,metrics=['accuracy'])

history=model.fit(np.array(x_train),np.array(y_train),epochs=80,batch_size=15,verbose=1)

#saves the model file with Given Intends
model.save('model_for_chatbot1.h5', history)
