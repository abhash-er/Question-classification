import numpy as np
import tensorflow as tf
import pickle as p
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re


# change the root_directory to your folder path
root_directory = "D:/Project/Niki.ai/"
data_dir = "saved-data/LSTM/"

#load Label Encoder
print("Loading Label Encoder.............")
encoder_path = root_directory + data_dir + "encoder.pickle"
encoder = p.load(open(encoder_path,'rb'))


#Loading Tokenizer
print("Loading Tokenizer.............")
tokenizer_path = root_directory + data_dir + "tokenizer.pickle"
tokenizer = p.load(open(tokenizer_path,'rb'))

#Loading Model
model_path = root_directory + data_dir + "lstm_model.h5"
print("Loading Model.............")
model = tf.keras.models.load_model(model_path)

flag = True

print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("------------------------------------------------------------")
print("------------------------------------------------------------")

while(flag):
    print()
    print("**********************************************")
    print("Welcome to LSTM model implementation")
    print("**********************************************")
    print("\nWhat is your Question ?")
    question = input()
    question = re.sub('[^a-zA-z0-9\s]','',question.lower())
    print("Cleaned : ")
    print(question)
    question = tokenizer.texts_to_sequences([question])
    print("Tokenized :")
    print(question)
    print("Padded :")
    question = pad_sequences(question, maxlen=35)
    print(question)


    ques_type  =  encoder.inverse_transform([np.argmax(model.predict(question))])
    print("\nQuestion Type :",ques_type[0])

    while(True):
        a = input("\nPress 1 to quit and 0 to continue  ")

        if a == str(0) :
            flag = True
            break
        elif a == str(1) :
            flag = False
            break
        else:
            print("\nTry Again !")

    print()
print("\nMade by Abhash Kumar Jha \n Thank You")
