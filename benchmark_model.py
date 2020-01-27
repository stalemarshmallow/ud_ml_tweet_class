"""This is the benchmark model to use as a baseline when developing our production model
It is meant to be easy to understand and fast for training, for grader evaluation"""
import pandas as pd
import numpy as np
import re
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from gensim.models import Word2Vec
from nltk.corpus import brown
#dimension of the word vector
USED_WORD_DIM = 30
#number of words to pad, so that all input sentences(tweets) have length 31
USED_PAD_LENGTH= 31
#space character
SPACE_CHARACTER = " "
def text_to_words(text):
    """
    Converts text that is represented as a string to an array of strings of words
    :param text: raw string representing a sentence, or multiple sentences from a tweet
    :return: an array of words that have been retrieved from the given text
    """
    #remove all numbers and special characters
    only_characters = re.sub(pattern = "[^A-Za-z ]",repl="",string=text)
    #remove http links
    no_links = re.sub(pattern = "[^ ]*http[^ ]*", repl = " ", string = only_characters)
    #split by white spaces
    split_by_whitespace = re.split(pattern = " +",string = no_links)
    #filter out empties
    non_empty_words = [word for word in split_by_whitespace if word != ""]
    return [string.lower() for string in non_empty_words]
def build_embeddings(texts):
    """
    Creates embeddings for texts based on the skip-gram model
    :param texts: array of sentences, where each sentence is an array of words in sequence
    :return: the model that contains the word vectors that are our embeddings
    """
    try:
        model = Word2Vec.load("tweet_trained.model")
    except:
        model =  Word2Vec(texts, size=USED_WORD_DIM, window=20, min_count=1, workers=4,sg=1,seed = 20)
        model.save("tweet_trained.model")
    return model
def pad_inputs(sentences, pad_length = 10, word_dim=USED_WORD_DIM):
    """
    Converts varying sentences containing word embeddings of some length to sentences of pad_length
    0 is used for empty inputs (when length < pad_length)
    :param sentences: array of sentences, where each word is represented by a numerical vector
    :param pad_length: length to pad the sentences to
    :param word_dim: dimension of the word vectors
    :return: array of sentences with length pad_length
    """
    result = list()
    total = 0
    maximum = 0
    for i, sentence in enumerate(sentences):
        vec_length = len(sentence)
        total = total + vec_length
        maximum = max(vec_length,maximum)
        new_vector = [np.zeros((word_dim,),dtype=float)]*pad_length
        rel_length = min(pad_length,vec_length)
        new_vector[:rel_length] = sentence[:rel_length]
        result.append(new_vector)
    return result
def map_to_embedding(texts,embeddings):
    """
    Map sentences that consist of individual words to numerical vectors
    :param texts: array of sentences
    :param embeddings: map from word to embedding, which is a numerical vector
    :return: array of sentences where each word in the sentence is represented by a vector of real numbers
    """
    result = list()
    for sentence in texts:
        mapped_sentence = list()
        for word in sentence:
            if word in embeddings:
             mapped_sentence.append(embeddings[word])
        result.append(mapped_sentence)
    return result
def split_data(X,y):
    """
    Splits training data into train and test sets in a stratified fashion preserving class distribution in the data
    :param X: the features
    :param y: the labels
    :return: a generator that returns exactly one train/test split in a stratified fashion based on labels
    """
    splitter = model_selection.StratifiedShuffleSplit(n_splits = 1, test_size=.05, random_state = 10)
    return splitter.split(X=X,y=y)
def get_model_inputs():
    """
    Reads in input data from the train.csv file, extract features from input data and return array of sentences
    where each sentence is an array of single words (without punctuation, special characters, or urls)
    Map each array of words to an array of vectors (word embeddings), then pad each sentence length
    Finally pack it in a tuple along with the labels
    :return: tuple of
    (input features in a completely numerical format: an array of an array of word embeddings of same length)
    (target labels)
    """
    df_input = pd.read_csv('train.csv')
    texts = df_input["text"].values
    #there are some empty keywords unfortunately
    keywords = df_input["keyword"].fillna(" ")
    keywords = keywords.values
    #added the keyword to the beginning of each sentence for easy processing
    augmented_texts = [keyword + SPACE_CHARACTER + text for keyword, text in zip(keywords, texts)]
    #convert the sentences of words with punctuation, special chars, etc. to arrays of single words
    processed = [ text_to_words(text) for text in augmented_texts]
    #construct embedding model from processed text using skip-gram model
    embeddings = build_embeddings(processed)
    #map words to embeddings
    mapped_inputs = map_to_embedding(processed, embeddings.wv)
    #pad the sentences with vectors of 0 for missing words
    padded_inputs = np.array(pad_inputs(sentences=mapped_inputs, pad_length=USED_PAD_LENGTH))
    #flattened inputs so that each sentence can be fed as a training instance
    flattened_inputs = padded_inputs.reshape((padded_inputs.shape[0], USED_PAD_LENGTH * USED_WORD_DIM))
    #labels
    target_values = np.array(df_input["target"].astype('float32').values)
    return (flattened_inputs,target_values)
"""
Fits a logistic 
"""
if __name__ == "__main__":
    input_features, target_values = get_model_inputs()
    #split into train and test data in a stratified fashion
    split_generator= split_data(input_features, target_values)
    train_indices, test_indices = next(split_generator)
    train_x = input_features[train_indices]
    train_y = target_values[train_indices]
    test_x = input_features[test_indices]
    test_y = target_values[test_indices]
    #please forgive the magic C constant number, this is obviously heavily regularized
    model= linear_model.LogisticRegression(solver="liblinear",penalty="l2", C=.00775,random_state=1,
                                           tol=1e-10,max_iter=1000)
    model.fit(X=train_x, y=train_y)
    y_predicted_test = model.predict(X=test_x)
    y_predicted_train = model.predict(X=train_x)
    print("F1 Score for the Test Data:", metrics.f1_score(y_true=test_y,y_pred=y_predicted_test))
    print("F1 Score for the Training Data:",  metrics.f1_score(y_true=train_y,y_pred=y_predicted_train))



