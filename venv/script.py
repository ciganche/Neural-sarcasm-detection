import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install("tensorflow")
install("emoji")
install("keras")
install("nltk")
install("numpy")
install("gensim")
install("sklearn")
install("urlextract")


import keras
import emoji
import string
import nltk
import numpy as np
import re
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from keras import preprocessing as pre
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, Input
from nltk.corpus import words
from nltk.corpus import wordnet
all_word_list = words.words()
from urlextract import URLExtract
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
extractor = URLExtract()



# --- FUNKCIJE ---
# --- FUNKCIJE ---


def reformat_output(Y):
    ret_val = []
    for x in Y:
        current_array = []
        if x == 0:
            current_array.append(1)
            current_array.append(0)
        if x == 1:
            current_array.append(0)
            current_array.append(1)
        ret_val.append(current_array)
    return ret_val


def winner(output):
    return np.argmax(output)

def display_result(outputs):
    result = []
    for output in outputs:
        result.append(winner(output))

    return result

def loadLabeledData(location):
    list_of_labels = []
    list_of_tweets = []
    f = open(location, "r", encoding='utf-8', errors='ignore')
    for line in f:
        line = line.replace("\n", " ")
        split_line = line.split("\t");
        list_of_labels.append(split_line[1])
        list_of_tweets.append(split_line[2])

    list_of_labels = list_of_labels[1:]
    numeral_label_list = []
    for l in list_of_labels:
        numeral_label_list.append(int(l))


    return (numeral_label_list, list_of_tweets[1:])


def preprocess(input):

    input = input.lower()

    input = emoji.demojize(input)
    input = input.replace("::", " ")
    input = input.replace(":", " ")
    input = input.replace("_", " ")
    input = input.replace("."," ")
    exclude = set(string.punctuation)
    input = ''.join(ch for ch in input if ch not in exclude)

    input = input.lower()

    return input



def right_padd(sequences, MAXSIZE):

    new_sequence = []

    for s in sequences:
        if len(s) > MAXSIZE:
            s = s[:MAXSIZE]
        else:
            kolko_fali = MAXSIZE - len(s)
            for x in range(0,kolko_fali):
                s.append(0)

        new_sequence.append(s)


    return new_sequence

def make_word_matrix(tweet_list):
    cnt = 0
    total = []
    additional = [] #3d matrix
    regex = re.compile('[^a-zA-Z0-9#@>> ]')

    for tweet in tweet_list:

        tweet = mark_emojis(tweet)
        tweet = mark_hashtags(tweet)
        tweet = removeURL(tweet)
        tweet = regex.sub(' ', tweet) #sklonio sve znakove interpunkcije osim par

        tweet_additional = np.zeros(shape=(PADDING_INPUT_LENGTH,6)) #2d matrix [PADDING_INPUT_LENGTH, numer_of_flags]
        tweet_tokens = []

        words = tweet.split(" ")
        for i in range(0,len(words)):
            word = words[i]
            if(word == ""):
                continue

            temp = generateAdditionalVec(word)
            word_info = temp[0]
            word = temp[1]

            tweet_tokens.append(word)

            if(i<PADDING_INPUT_LENGTH):
                tweet_additional[i,:] = word_info

            cnt = cnt + 1

        total.append(tweet_tokens)
        additional.append(tweet_additional)


    return (total,additional,cnt)

def generateAdditionalVec(word):
    vector = np.zeros(shape=(1,6))

    if (word[0] == '@'):
        word = "person"
        vector[0,0] = 1
    elif (word == "URL"):
        vector[0,1] = 1
        word = "url"
    elif (word[0] == ">"):
        vector[0,2] = 1
        word = word.replace(">>","")
    else:
        if (word[0] == "#"):
            vector[0,3] = 1
            word = word.replace("#","")
        if (word == word.upper()):
            vector[0,4] = 1
        if (repeatingVowels(word)):
            vector[0,5] = 1
        word = word.lower()

    return (vector,word)


def repeatingVowels(input):
    retVal = False
    input = input.lower()
    repeating_o_words =  re.findall("[a-z]*o{3,}[a-z]*", input)
    repeating_a_words =  re.findall("[a-z]*a{3,}[a-z]*", input)
    repeating_e_words =  re.findall("[a-z]*e{3,}[a-z]*", input)
    repeating_i_words =  re.findall("[a-z]*i{3,}[a-z]*", input)
    repeating_u_words =  re.findall("[a-z]*u{3,}[a-z]*", input)
    if len(repeating_a_words) == 0 and len(repeating_e_words) == 0 and len(repeating_i_words) == 0 and len(repeating_o_words) == 0 and len(repeating_u_words) == 0:
        retVal = False
    else:
        retVal = True

    return retVal

def removeURL(tweet):

    tokens = tweet.split(" ")
    for word in tokens:
        if "http" in word or "www" in word:
            tokens[tokens.index(word)] = "URL"
    tweet= ""
    for x in tokens:
       tweet = tweet + x + " "

    return tweet

def mark_emojis(tweet): #makrs all emojis with >>. :) turns into ">>smily >>face"

    tokens = tweet.split(" ")
    retVal=""
    for word in tokens:
        new_word = emoji.demojize(word)
        if(new_word!=word):
            new_word = new_word.replace("_"," >>")
            new_word = new_word.replace("::"," >>")
            new_word = new_word.replace(":"," >>",1)
            new_word = new_word.replace(":", "")
        retVal = retVal + " " + new_word

    return retVal


def mark_hashtags(tweet):
    hashtaged_words_in_line = re.findall("#(\w+)", tweet)

    tweet, stuff_in_hashtags = FindWordsInHashtag(tweet, hashtaged_words_in_line)

    return tweet


def split_hashtag(tag):
    pattern = re.compile(r"[A-Z][a-z]+|\d+|[A-Z]+(?![a-z])")
    return pattern.findall(tag)

def FindWordsInHashtag(input, hashtaged_words_in_line):
    stuff_in_hashtags = []
    for x in hashtaged_words_in_line:
        if len(split_hashtag(x)) > 1:

            split_words = ""
            for word in split_hashtag(x):
                split_words = split_words + "#"+word + " "
            input = input.replace(x, split_words)
            stuff_in_hashtags.append(split_words.lower())

    input = input.replace("##","#")
    input = input.replace("#", " #")

    return input, stuff_in_hashtags




def words2nums(word_matrix, PADDING_INPUT_LENGTH, tokenizer):
    sequences = []
    max = 0
    for t in word_matrix:

        if(len(t) > max):
            max = len(t)

        s = tokenizer.texts_to_sequences(t);
        sequence_line = []
        for x in s:
            if(len(x) == 0 ):
                continue #tokenizer.text_to_sequence puts [] if it comes across an unknown word
            else:
                sequence_line.append(x[0])
        sequences.append(sequence_line)

    # right padd all the tweets to maximum of 50 words
    sequences = right_padd(sequences, PADDING_INPUT_LENGTH)
    print("Maximum word count: " + str(max))
    return sequences









# --- MAIN ---
location_train = "trainSet.txt"
location_test = "testSet.txt"

(label_list_train, tweet_list_train) = loadLabeledData(location_train)
(label_list_test, tweet_list_test) = loadLabeledData(location_test)

print("Number of training examples: " + str(len(label_list_train)))
print("Number of test examples: " + str(len(label_list_test)))

PADDING_INPUT_LENGTH = 75 #word length of every tweet in set, if shorter zeros are added, if longer, it's trimmed

model = KeyedVectors.load_word2vec_format('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)
tokenizer = pre.text.Tokenizer(num_words=None)
(total_text_train,additional_data_train,cnt_train) = make_word_matrix(tweet_list_train) #list of lists words in every train tweet
(total_text_test,additional_data_test,cnt_test) = make_word_matrix(tweet_list_test) #list of lists words in every test tweet

print("Total words found in train set: " + str(cnt_train))
print("Total words found in test: " + str(cnt_test))

# dictionary is made from training tweets
tokenizer.fit_on_texts(total_text_train)
word_index = tokenizer.word_index #dictionary of unique words
print("Unique words found: " + str(len(word_index)))



sequences_train = words2nums(total_text_train, PADDING_INPUT_LENGTH, tokenizer)
sequences_test = words2nums(total_text_test, PADDING_INPUT_LENGTH, tokenizer)

sequences_train = np.asmatrix(sequences_train)
sequences_test = np.asmatrix(sequences_test)
additional_data_train = np.asarray(additional_data_train)
additional_data_test = np.asarray(additional_data_test)

print("TRAIN SEQUENCE:")
print(sequences_train.shape)

print("TEST SEQUENCE:")
print(sequences_test.shape)

print("ADDITIONAL TRAIN INPUT:")
print(additional_data_train.shape)

print("ADDITIONAL TEST INPUT:")
print(additional_data_test.shape)


label_list_train = np.asarray(reformat_output(label_list_train))

print("TRAIN LABELS SHAPE:")
print(label_list_train)

print("TRAIN LABELS SHAPE:")
print(label_list_test)


#the order of the labeled tweets is pretty much random, so there's no need to shuffle
VALIDATION_SET_SIZE = 390
test_size = sequences_test.shape[0] - VALIDATION_SET_SIZE

X_test = sequences_test[0:test_size]
Y_test = label_list_test[0:test_size]
ADDITIONAL_TEST = additional_data_test[0:test_size]

X_VALIDATION = sequences_test[test_size:]
Y_VALIDATION = label_list_test[test_size:]
ADDITIONAL_VALIDATION = additional_data_test[test_size:]





print("TRAINING SET:")
print("X: " + str(X_VALIDATION.shape) + " Y: " + str(len(Y_VALIDATION)) + " ADDITIONAL: " + str(ADDITIONAL_VALIDATION.shape))



EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM)) #the 0th row remains with all zero values
for word, i in word_index.items():
    try:
        embedding_vector = model.word_vec(word)
        embedding_matrix[i] = embedding_vector
    except KeyError:
        #all zero row stays if no word embeddings are found
        pass

main_input = Input(shape=(PADDING_INPUT_LENGTH,), dtype='int32', name='main_input')

embd = Embedding(input_dim=len(word_index)+1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True, trainable=False)(main_input)


additional_input = Input(shape=(PADDING_INPUT_LENGTH,6), name='additional_input') #the aditional information about a word in a sentence
appended = keras.layers.concatenate([embd, additional_input])

lstm = LSTM(225)(appended)

dropout = Dropout(0.1)(lstm)
main_output = Dense(2, activation='sigmoid', name='main_output')(dropout)

rnnModel = Model(inputs=[main_input, additional_input], outputs=[main_output])

rnnModel.summary()
rnnModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

rnnModel.fit([sequences_train, additional_data_train], label_list_train, epochs=9, shuffle=True)
Y_predict_probs = rnnModel.predict([X_test,ADDITIONAL_TEST])
Y_predict = display_result(Y_predict_probs)

print("\n")
print("* * * Classification report:")
print(classification_report(Y_test,Y_predict))
print("* * * Accuracy achieved:")
print(accuracy_score(Y_test, Y_predict))
print("\n")


#--- NEURAL NETWORK PARAMETERS OPTIMIZATION OF DROPOUT LAYER VALUE AND LSTM UNIT NUMBER ON THE VALIDATION SET----
# best_layer_num = 0
# best_f1 = 0
# best_rnn_model = ""
# best_dropout = 0.05
#
# for layer_num in range(5,1000,5): #PROMENI 10 NA 1000
#
#
#
#     main_input = Input(shape=(PADDING_INPUT_LENGTH,), dtype='int32', name='main_input')
#
#     embd = Embedding(input_dim=len(word_index)+1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True, trainable=False)(main_input)
#
#
#     additional_input = Input(shape=(PADDING_INPUT_LENGTH,6), name='additional_input') #the aditional information about a word in a sentence
#     appended = keras.layers.concatenate([embd, additional_input])
#
#     lstm = LSTM(layer_num)(appended)
#
#     dropout = Dropout(0.3)(lstm)
#     main_output = Dense(2, activation='sigmoid', name='main_output')(dropout)
#
#     rnnModel = Model(inputs=[main_input, additional_input], outputs=[main_output])
#
#     rnnModel.summary()
#     rnnModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
#     rnnModel.fit([sequences_train, additional_data_train], label_list_train, epochs=9, shuffle=True)
#     Y_predict_probs = rnnModel.predict([X_VALIDATION,ADDITIONAL_VALIDATION])
#     Y_predict = display_result(Y_predict_probs)
#
#
#     current_f1 = f1_score(Y_VALIDATION,Y_predict)
#     current_acc = accuracy_score(Y_VALIDATION, Y_predict)
#     if current_f1 > best_f1:
#         best_f1 = current_f1
#         best_layer_num = layer_num
#         best_rnn_model = rnnModel.to_json()
#
#     print("current accuracy: " + str(current_acc*100))
#     print("current f1: " + str(current_f1*100))
#     print("The best f1: " + str(best_f1) + " is for layers: " + str(best_layer_num) + " for dropout value: ")
#     print("-----------------------------------------")
#
#
# print("FINAL: the best f1: " + str(best_f1) + " is for layers: " + str(best_layer_num))
#
# with open("rnn_model.json", "w") as json_file:
#     json_file.write(best_rnn_model)
# rnnModel.save_weights("rnn_model.h5")
# print("Model saved")
#
# Y_predict_probs = rnnModel.predict([X_test,ADDITIONAL_TEST])
# Y_predict = display_result(Y_predict_probs)
# print("\n")
# print("* * * Classification report:")
# print(classification_report(Y_test,Y_predict))
# print("* * * Accuracy achieved:")
# print(accuracy_score(Y_test, Y_predict))
# print("\n")
#


# dropout_value = 0.05
# best_dropout = 0
# best_f1 = 0
# while dropout_value <= 0.7:
#
#     main_input = Input(shape=(PADDING_INPUT_LENGTH,), dtype='int32', name='main_input')
#     embd = Embedding(input_dim=len(word_index)+1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True, trainable=False)(main_input)
#
#
#     additional_input = Input(shape=(PADDING_INPUT_LENGTH,6), name='additional_input') #the aditional information about a word in a sentence
#     appended = keras.layers.concatenate([embd, additional_input])
#
#     lstm = LSTM(225)(appended)
#
#     dropout = Dropout(dropout_value)(lstm)
#     main_output = Dense(2, activation='sigmoid', name='main_output')(dropout)
#
#     rnnModel = Model(inputs=[main_input, additional_input], outputs=[main_output])
#
#     rnnModel.summary()
#     rnnModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
#     rnnModel.fit([sequences_train, additional_data_train], label_list_train, epochs=9, shuffle=True)
#     Y_predict_probs = rnnModel.predict([X_VALIDATION,ADDITIONAL_VALIDATION])
#     Y_predict = display_result(Y_predict_probs)
#
#
#     current_f1 = f1_score(Y_VALIDATION,Y_predict)
#     current_acc = accuracy_score(Y_VALIDATION, Y_predict)
#     if current_f1 > best_f1:
#         best_f1 = current_f1
#         best_dropout = dropout_value
#         best_rnn_model = rnnModel.to_json()
#     dropout_value = dropout_value + 0.05
#
#     print("current accuracy: " + str(current_acc*100))
#     print("current f1: " + str(current_f1*100))
#     print("The best f1: " + str(best_f1) + " for dropout value: " + str(best_dropout))
#     print("-----------------------------------------")
