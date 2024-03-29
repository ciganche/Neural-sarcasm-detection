# NeuralSarcasmDetection

1. venv/script.py loads the labeled tweets from "trainSet.txt" and "testSet.txt".

   * internet connection is needed for downloading packages and GoogleNews-vectors.

2. Preprocesses the tweets.

3. Converts words into a word vector using "GoogleNews-vectors-negative300.bin.gz" word2vec traind model.

4. Trains the neural network and makes predictions for the test examples.


#### Data used in this experiment is taken from the SemEval-2018 competition in sarcasm detection on Twitter.

#### Achieved accuracy of 71.2% (f1-score: 0.7).


Dropout layer valus (0.1) and the number of LSTM cells ware gained in the commented optimization process over the validation set.
Train/validation/test split is 80/10/10 %.
__________________________________________________________________________________________________

There are two input layers in the neural network:

1. [1x300] size vector for every word in a tweet

2. [1x6] flag vector for every word in a tweet:
    * flag 0 - is a word an username (e.g. @Tjokarda123), if so, it is replaced with "person"
    * flag 1 - is a word an URL, if so, it is replaced with "URL"
    * flag 2 - is an word emoji, if so, it is demojized - converted into words (emojis are broken down in the preprocessing part)
    * flag 3 - is an part of a hashtag (hashtags are broken down in the preprocessing part)
    * flag 4 - is a word all uppercase
    * flag 5 - does the word have repeating vowels (e.g. loveeeee)
    
    
 The two inputs are combined after the word embedding layer.
    





#### Architecture of the network:




Layer (type)  |     Output shape     |  Param #      | Connected to  
------------- | ------------- | --------------|------------
main_input (InputLayer)  | 1,75  |       0        |
embedding_1 (Embedding)  | 75, 300  |      2947500         |main_input 
additional_input (InputLayer)  | 75, 6  |      0         |
concatenate_1 (Concatenate)  | 75, 306  |     0          |embedding_1, additional_input
lstm_1 (LSTM) | 225 | 6240|concatenate_1
dropout_1 (Dropout)   | 225 |0| lstm_1
main_output (Dense)| 2 |  12 |dropout_1


