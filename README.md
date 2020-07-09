# Question-classification
This repo is a part of assignment given by niki.ai 

## Problem:

Given a question, the aim is to identify the category it belongs to. The four categories to handle for this assignment are : Who, What, When, Affirmation(yes/no).

```
A labeled dataset was given such that questions and their labels i.e. { Who, What, When, Affirmation},separated by ',,,' -- were given in a text file to be trained upon. 
The goal was to come up with data structures to encapsulate these information as well as the code that populates the relevant data.
Any machine learning technique for training the labelled text file given. The output should be driven keeping in mind the reply to the question.
 
Example:
1. What is your name? Type: What
2. When is the show happening? Type: When
3. Is there a cab available for the airport? Type: Affirmation
There are ambiguous cases to handle as well like:
What time does the train leave(this looks like a what question but is actually a When type)
```


## 1 .Major Files :
See the Analysis Part in the [Notebook](nlp_classification.ipynb) <br>

To run the Random-Forest version Web Api- run [Web API](apiFLASK.py)

To run the LSTM version CLI Api- run [LSTM](lstm.py)




## 2. What I did! 

<b>1. The first part was to analyze the dataset or rather preprocess it. </b>

```
This is the preprocessing for only Naive Bayes and Random Forest Clasifiers only.
```

The very first part was to clean the dataset and form a meaningful corpus or vocabulary out of it.<br>

Count Vectorization was used to form one-hot encoded sparse matrix.

![Attached Image](https://miro.medium.com/proxy/1*YEJf9BQQh0ma1ECs6x_7yQ.png)

Inside the count vectorizer class, I also passed a stemming method to concurrently stem the words using SnowBallStemmer.

![Attached Image](https://pythonspot.com/wp-content/uploads/2016/08/word-stem.png)

<b> 2. Here comes the interesting part - Classifiers </b>

I applied 3 algorithms, the first one was Naive Bayes, the second was Random Forest and the third one was LSTM Model.

Naive Bayes Performed very poorly and was not able to classify many questions.
([Notebook](nlp_classification.ipynb)- cell 20)
An accuracy of 91.86% was achieved but it still was not enough and looked like to be overfitted.


But Random Forest performed very well.(([Notebook](nlp_classification.ipynb)- cell 27))
An accuracy of 97.28% was achieved and model seemed to give correct answers.
I saved this model inside "saved-data/random-forest" folder to call the model into API.

<b> 3. For LSTM I used pretrained glove word vector embeddings and made a Sequential Model.</b>

To download it, go to [Glove](http://nlp.stanford.edu/data/glove.6B.zip)

My model's architecture is as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 35)]              0         
_________________________________________________________________
embedding (Embedding)        (None, 35, 300)           1105800   
_________________________________________________________________
lstm (LSTM)                  (None, 512)               1665024   
_________________________________________________________________
dense (Dense)                (None, 128)               65664     
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 325       
=================================================================
Total params: 2,845,069
Trainable params: 1,739,269
Non-trainable params: 1,105,800
```
<h6><i> INPUT &rarr; EMBEDDING &rarr; LSTM &rarr; RELU-128 &rarr; RELU-64 &rarr; SOFTMAX &rarr; OUTPUT </i> </h6>   

- Note :  The maximum length of input stream is 35 here. It can be changed to other number also depending upon the use case. 

This model was also a success.

Upon Training a validation loss of just 7.63% and an accuracy of 98.64% was achieved.

I also saved this model for calling it inside the API.

```
The Tensorflow LSTM Model was very heavy and ran on GPU, thus while using Flask the gpu was overloaded. Hence, for LSTM, a command line version is implemented.
```




## 3. Flask API (Random-Forest)

This API is developed in Flask for implementing the Random Forest Model
See [Web API](apiFLASK.py) for the code. 

> Run the apiFlask.py

![Attached Image](https://raw.githubusercontent.com/abhash-er/Question-classification/master/Markdown%20Images/flask_shell.PNG)

> Index Page

![Attached Image](https://raw.githubusercontent.com/abhash-er/Question-classification/master/Markdown%20Images/index.png)

> Result Page

![Attached Image](https://raw.githubusercontent.com/abhash-er/Question-classification/master/Markdown%20Images/result.png)





## 4. CLI - LSTM API

This API is developed in Flask for implementing the Random Forest Model
See [LSTM](lstm.py) for the code.

> Initial Load

![Attached Image](https://raw.githubusercontent.com/abhash-er/Question-classification/master/Markdown%20Images/sample_lstm.PNG)

> RUN 


![Attached Image](https://raw.githubusercontent.com/abhash-er/Question-classification/master/Markdown%20Images/sample_lstm1.png)


## 5. Dependencies

You can run pip install dependencies.txt to check that you satisfy all the requirements.

```
tensorflow-gpu==2.1.0
tqdm==4.47.0
keras-gpu==2.3.1
pandas==1.0.2
numpy==1.18.1
scipy==1.41.1
flask==1.1.2
matplotlib==3.13
seaborn==0.10.0
nltk==3.5
sklearn==0.0
json5==0.9.2
requests==2.23.0

```

