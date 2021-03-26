#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer    # use for word stemming
stemmer = LancasterStemmer()

from tensorflow.python.framework import ops
import numpy
import tflearn
import random
import json
import pickle
import string

# ---------------------- NLTK IMPORTS ----------------------------------------------------------------------------------
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# ----------------------------------------------------------------------------------------------------------------------

with open("intents_normal.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:                    # if we've already saved our data, it will be restored here
        words, labels, training, output = pickle.load(f)
except:
    words = []                                      # all the words from intents (json file)
    labels = []                                     # list of "tags" from json file
    docs_x = []                                     # list of all different patterns
    docs_y = []                                     # corresponding list of intents

    for intent in data["intents"]:                  # loop through intents
        for pattern in intent["patterns"]:          # loop through patterns
            wrds = nltk.word_tokenize(pattern)      # split words in json patterns using nltk tokenizer -> .split(" "), returns list
            words.extend(wrds)                      # instead of iterating through list using loop we can extend it
            docs_x.append(wrds)                     # important for training our model
            docs_y.append(intent["tag"])            # -||-

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]    # stemming each word and changing them to lower case,
    words = sorted(list(set(words)))                                # takes all the words, removes all duplicate elements
                                                                    # and sort them alphabetically
    labels = sorted(labels)                                         # sorting labels

    training = []       # list of bag of words
    output = []         # corresponding list of 0's and 1's

    out_empty = [0 for _ in range(len(labels))]     # Bag of words for "tags" from json

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)   # word exists -> append
            else:
                bag.append(0)   # word does not exist -> do not append

        output_row = out_empty[:]                   # copy of the list
        output_row[labels.index(docs_y[x])] = 1     # set value 1 for a particular tag

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)    # changing our lists into numpy arrays for tflearn
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)   # saving our data for future use

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])    # defines the input shape that we're expecting for our model
net = tflearn.fully_connected(net, 16)                       # adds fully connected layer to our network (8 neurons for that layer)
net = tflearn.fully_connected(net, 16)                       # adds another hidden layer with 8 neurons
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")    # allows us to get probability for each output (each neuron in this layer)
net = tflearn.regression(net)                               #

model = tflearn.DNN(net)                                    # a deep neural network model

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)   # training our data
    model.save("model.tflearn")


# -------------------------------- METHODS -----------------------------------------------------------------------------
# checking if 'https://' phrase in provided string
def contains_url(sentence):
    return "https://" in sentence


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    # ---------------------------- INITIALIZING TIME -------------------------------------------------------------------
    import time
    start_time = time.time()
    # ------------------------------------------------------------------------------------------------------------------

    # ---------------------------- VARIABLES TO STORE CHATBOT METRICS --------------------------------------------------
    user_input_no = 0   # int[0:n]
    chat_time = 0       # int[0:n] in seconds
    url_useful = 0      # float[0-1]
    sent_term = 0       # int[0-1]
    avg_inp_sent = 0    # float[-1 - 1] from -1 to 1
    rating = 0          # int[1-5]
    answers = []        # ['no_of_msgs', 'time', 'url_useful', 's_term', 'usr_rating']
    # ------------------------------------------------------------------------------------------------------------------
    url_no = 0          # stores number of url's provided by chatbot
    sum_inp_sent = 0    # stores sum of input sentiment
    # ---------------------------- START OF CHATBOT --------------------------------------------------------------------
    print("Start talking with the bot (type quit to stop)!")

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)]) # changes user's input into bag of word and feed our model, returns probabilities
        results_index = numpy.argmax(results)               # returns the highest probability from our results
        tag = labels[results_index]                         # returns particular label with the highest probability

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        s_response = random.choice(responses)               # randomly picks response from recognized pattern
        print(s_response)

        # Sentiment of termination (positive/negative)
        if tag == "goodbye":
            sent_term = 1
        else:
            sent_term = 0

        # URL's - determining if url provided by chatbot was useful for the user
        if contains_url(s_response):
            url_no += 1
            go_loop = True
            while go_loop:
                go_back = input("Was the URL useful (y/n)?")
                if go_back.lower() == "y":
                    url_useful = 1
                    go_loop = False
                elif go_back.lower() == "n":
                    # url_useful = 0
                    go_loop = False
                else:
                    print("I can't understand!")
                    go_loop = True

        user_input_no += 1                      # increments number of user's messages

        #--------------------- SENTIMENT ANALYSIS ----------------------------------------------------------------------
        text = inp

        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

        # Using word_tokenize because it's faster than split()
        tokenized_words = word_tokenize(cleaned_text, "english")

        # Removing Stop Words
        final_words = []
        for word in tokenized_words:
            if word not in stopwords.words('english'):
                final_words.append(word)

        # Lemmatization - From plural to single + Base form of a word (example better-> good)
        lemma_words = []
        for word in final_words:
            word = WordNetLemmatizer().lemmatize(word)
            lemma_words.append(word)

        emotion_list = []
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')

                if word in lemma_words:
                    emotion_list.append(emotion)

        user_sent = sentiment_analyse(cleaned_text)
        sum_inp_sent += user_sent
        print(user_sent)
        #--------------------- END OF SENTIMENT ANALYSIS ---------------------------------------------------------------

    end_time = time.time()                      # 'stop' the timer
    chat_time = end_time - start_time           # total interaction time

    rating = input("What is your overall experience with me (1-5)? : ")
    while rating not in ["1", "2", "3", "4", "5"]:
        rating = input("Please provide number between 1-5! : ")
    # ------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------- OUTPUT ------------------------------------------------------------------
    print("Thank you!")
    print("\nChat Summary:")
    print("-------------------------------------------")

    # Getting the metrics:
    # 1. No of inputs
    print(f"{'Number of user inputs: ':<32}", user_input_no)
    answers.append(user_input_no)

    # 2. Time of 'conversation'
    # print("Chat time: %s seconds" % int(chat_time))
    print(f"{'Chat time: ':<32} {int(chat_time)} seconds")
    answers.append(int(chat_time))

    # 3. Was URL useful
    if url_useful != 0:
        url_useful /= url_no

    print(f"{'Was the URL usefull: ':<32}", round(url_useful, 4))
    answers.append(round(url_useful, 4))

    # 4. Sentiment of termination
    print(f"{'Sentiment termination: ':<32}", sent_term)
    answers.append(sent_term)

    # 5. Average Input Sentiment
    if user_input_no != 0:
        avg_inp_sent = sum_inp_sent / user_input_no
    print(f"{'Average Input Sentiment: ':<32}", round(avg_inp_sent, 4))
    answers.append(round(float(avg_inp_sent), 4))

    # 6. Explicit User Experience - in future version will be deduced from metrics obtained above
    print("User's experience in scale 1-5: ", rating)
    answers.append(int(rating))
    print("-------------------------------------------")

    print("Collected metrics: ", answers)
    ans_str = str(answers[0]) + "," + str(answers[1]) + "," + str(answers[2]) + "," + str(answers[3]) + "," +\
              str(answers[4]) + "," + str(answers[5]) + "," + '\n'
    # ------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------- WRITING TO FILE----------------------------------------------------------
    f = open("data_chat.csv", "a")
    f.write(ans_str)
    f.close()
    # ------------------------------------------------------------------------------------------------------------------


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
        return -1
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
        return 1
    else:
        print("Neutral Sentiment")
        return 0


chat()
