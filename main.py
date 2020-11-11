# -*- coding: utf-8 -*-
from flask import  Flask, render_template, request, jsonify

app = Flask(__name__)


from werkzeug.utils import redirect
import nltk

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
from tensorflow.contrib.rnn import static_bidirectional_rnn
import pickle
import json
import random

with open("intents_PE") as file:
    data = json.load(file)



try:
    # hier x activieren, wenn etwas an der Json file geändert wurde, sonst nutzt es das schon trainieret Model
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)

            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)  # numpy - takes lists and change them to arrays - format that is needed for model
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

########Setting up Model
tf.reset_default_graph()  # Resetting the underlying  data graph

net = tflearn.input_data(shape=[None, len(training[0])])  # Defines input shape we are expecting for the model
net = tflearn.fully_connected(net, 8)  #  8 neurons for that hidden layer
net = tflearn.fully_connected(net, 8)  # another hidden layer
net = tflearn.fully_connected(net, 8)  # another hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),
                              activation="softmax")  # allows  to get probabilities for each output
net = tflearn.regression(net)  #

model = tflearn.DNN(net)  # trains the model

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)  # pass all training data , epochs: amount of times it sees training data
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


# def chat():
#     print("Sie können nun mit dem Chatbot schreiben. Um den Chat zu unterbrechen schreiben Sie 'stop'!")
#     while True:
#         inp = input("Du: ")
#         if inp.lower() == "stop":
#             break
#
#         results = model.predict([bag_of_words(inp, words)])[0]
#         results_index = numpy.argmax(results)
#         tag = labels[results_index]
#
#         if results[results_index] > 0.8:
#             for tg in data["intents"]:
#                 if tg['tag'] == tag:
#                     responses = tg['responses']
#
#             print("Chatbot:", random.choice(responses))
#
#         else:
#             print("Es tut mir leid, ich habe Ihre Frage nicht verstanden. Versuchen Sie es bitte noch einmal.")

@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/process")
def chatbot():

    inp = request.args.get("msg")



        # if inp.lower() == "stop":
        #     break

    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.8:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                answer = random.choice(responses)
        return answer
        # print("Chatbot:", answer)
        # return jsonify({"inp": inp ,"answer" : answer})


    else:
        alt_answer=("Bitte versuchen Sie es noch einmal")
        return alt_answer




if __name__== "__main__":
    app.run(debug=True, port=5004)
