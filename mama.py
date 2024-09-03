import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import numpy as np
import pickle
import json
from keras.models import load_model
import random
# UnCommand This part while Running First Time.....
'''nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')'''

model=load_model('model_for_chatbot1.h5')

intents=json.loads(open('intents1.json').read())
words=pickle.load(open('words.pkl','rb'))
labels=pickle.load(open('labels.pkl','rb'))

#this function perform text operation over our intents
def word_bank(s,words,show_details=True):
  bag_of_words=[0 for i in range (len(words))]
  sent_words=nltk.word_tokenize(s)
  sent_words=[lemmatizer.lemmatize(word.lower()) for word in sent_words]
  for sent in sent_words:
    for i,w in enumerate(words):
      if w==sent:
        bag_of_words[i]=1
  return np.array(bag_of_words)

#this function help us to predict words
def predict_label(s,model):
  pred=word_bank(s,words,show_details=False) #we calling the word_bank function
  response=model.predict(np.array([pred]))[0] #creating array of response using numpy
  Error_Threshold=0.25
  final_results=[[i,r] for i,r in enumerate(response) if r>Error_Threshold]
  final_results.sort(key=lambda x:x[1],reverse=True)
  return_list=[]
  for r in final_results:
    return_list.append({"intent":labels[r[0]],"probability":str(r[1])})
  return return_list

def Response(ints,intent_json):
  tags=ints[0]['intent']
  list_intents=intent_json['intents']
  for i in list_intents:
    if(i['tag']==tags):
      response=random.choice(i['responses'])
      break
  return response

def chatbot_response(msg):
  ints=predict_label(msg,model)
  response=Response(ints,intents)
  return response

def chat():
  print("Hi I Am Mama")
  while True:
    inp=input("you:")
    if inp.lower==1:
      break
    response=chatbot_response(inp)
    print("\n Boo:"+ response +  '\n\n')

chat()