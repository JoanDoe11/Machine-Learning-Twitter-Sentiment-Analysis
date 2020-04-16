import csv
import nlp_util
from collections import Counter

def load_training_set(path):
    """
    Extracts data from a csv file, 
    separates features and labels,
    cleans the features with nlp methods.

    Parameters
    ----------
    path : string
        absolute path to the csv file containing training data
    
    Returns
    -------
    (labels, features) : (list, list)
        returns a tuple of two lists.
        labels[] is a list of integers 0,1 denoting the class
        features[] is a list of strings denoting tweets

    """
    labels = []     # 0=negative, 1=positive 
    features = []   # tweets

    # rt = read as text
    with open(path,'rt', encoding = "ISO-8859-1") as datafile:
        reader = csv.reader(datafile)
        next(reader)    # skipping the header

        for row in reader:
            labels.append(int(row[1]))
            tweet = row[2]
            tweet = nlp_util.clean(tweet)
            tweet = nlp_util.letters_only(tweet)
            features.append(tweet)

    return (labels, features)

def split_training_data(training_lables, training_features, scale):
    """
    splits available data into two chunks: 
    learning data and testing data.

    Parameters
    ----------
    training_labels : list
        list of integers 0,1 denoting the class
    
    training_features : list
        list of strings denoting tweets

    scale : float
        number denoting the proportion of
        |learn data| : |train data|
    
    Returns
    -------
    (learning_data, testing_data) : (tuple, tuple)
        returns a tuple of two tuples.
        (learning_data) is a tuple of two lists:
            learning_lables[] is a list of integers 0,1 denoting the classes of the learning features
            learning_features[] is a list of strings denoting tweets to be used for learning
        (testing_data) is a tuple of two lists:
            testing_lables[] is a list of integers 0,1 denoting the classes of the testing features
            testing_features[] is a list of strings denoting tweets to be used for testing

    """
    data_amount = len(training_labels)
    border = int(scale * data_amount)

    learning_labels = training_labels[:border]
    learning_features = training_features[:border]
    testing_labels = training_labels[border:]
    testing_features = training_features[border:]

    learning_data = (learning_labels, learning_features)
    testing_data = (testing_labels, testing_features)

    return (learning_data, testing_data)

def learn(learning_data):
    """
    creates a naive bayes model for tweet sentiment based on the learning data

    Parameters
    ----------
    learning_data : tuple
        tuple of two lists:
            learning_lables[] is a list of integers 0,1 denoting the classes of the learning features
            learning_features[] is a list of strings denoting tweets to be used for learning
    
    Returns
    -------
    (positive_model, negative_model) : (tuple, tuple)
        returns a tuple of two tuples denoting class models.
        (positive_model) is a tuple of three elements:
            positive_features is a list containing words that are in the positive class
            positive_freq is a counter object with (word:frequency) pairs, denoting how many occurencies of that word the class has
            p_positive is a float number denoting the probability of that class
        (negative_model) is a tuple of three elements:
            negative_features is a list containing words that are in the positive class
            negative_freq is a counter object with (word:frequency) pairs, denoting how many occurencies of that word the class has
            p_negative is a float number denoting the probability of that class

    """
    labels = learning_data[0]
    features = learning_data[1]
    

    # initialization of classes and their initial members
    (positive_features, negative_features) = init_class(labels, features)

    # calculation of word frequences in classes
    positive_freq = Counter(positive_features) #see if you can only use this, because the other one counts repetitions
    negative_freq = Counter(negative_features)

    # probability calculations
    numOfPos = len(positive_freq)
    numOfNeg = len(negative_freq)
    numTotal = numOfPos + numOfNeg
    # P(class) = |class words| / |all words|
    p_positive = numOfPos / numTotal
    p_negative = numOfNeg / numTotal
    #print("Verovatnoca p/n:"+str(p_positive)+"/"+str(p_negative))
    
    #assembling models
    positive_model = (positive_features, positive_freq, p_positive)
    negative_model = (negative_features, negative_freq, p_negative)

    return (positive_model, negative_model)


def init_class(labels, features):
    """
    separates positive features from negative features

    Parameters
    ----------
    labels : list
        list of integers 0 and 1, 0 = negative class, 1 = positive class
    features : list
        list of strings denoting tweets that are to be classified

    Returns
    -------
    (positive_features, negative_features) : tuple
        tuple of two lists
            positive_features - list of strings denoting words in positive tweets
            negative_features - list of strings denoting words in negative tweets

    """
    positive_features = []
    negative_features = []
    for i in range(len(labels)):
        words = features[i].split(' ')
        if(labels[i]==0):
            for word in words:
                negative_features.append(word)
        else:
            for word in words:
                positive_features.append(word)
    #print("duzina pozitivnih:"+str(len(positive_features))+"; duzina negativnih:"+str(len(negative_features)))
    return (positive_features, negative_features)

# Naive Bayes prediction
# P(tweet|C) = (freq(words in tweet, C)+1) / (|C| + |words in tweet|)
# P(tweet) = sum(P(tweet|C)*P(C))
def prediction(tweet, class_model):
    """
    calculates the probability that the given tweet is in the given class

    Parameters
    ----------
    tweet : string
        
    class_model : tuple of three elements
            features is a list containing words that are in the positive class
            freq is a counter object with (word:frequency) pairs, denoting how many occurencies of that word the class has
            p_class is a float number denoting the probability of that class

    Returns
    -------
    float
        probability

    """
    freq = class_model[1]
    p_class = class_model[2]
    class_size = len(class_model[0])

    tweetWords = Counter(tweet.split(' ')) # words and their occurencies
    result = 1
    for word in tweetWords:
        freqTweet = tweetWords.get(word) # number of occurancies in the tweet
        freqClass = freq.get(word,0) # number of occurancies in the class
        #print("freq in tweet:"+str(freqTweet))
        #print("freq in class:"+str(freqClass))
        result *= freqTweet * (freqClass + 1) / class_size
    return result * p_class


def classify(tweet, model):
    """
    calculates the most probable class for the given tweet

    Parameters
    ----------
    tweet : string
        
    model : tuple of two tuples
        positive_model : tuple of three elements
            positive_features is a list containing words that are in the positive class
            positive_freq is a counter object with (word:frequency) pairs, denoting how many occurencies of that word the class has
            p_positive is a float number denoting the probability of that class
        (negative_model) is a tuple of three elements:
            negative_features is a list containing words that are in the positive class
            negative_freq is a counter object with (word:frequency) pairs, denoting how many occurencies of that word the class has
            p_negative is a float number denoting the probability of that class

    Returns
    -------
    int
        0 - tweet is classified as negative
        1 - tweet is classified as positive

    """
    positive_class = model[0]
    negative_class = model[1]

    prediction_positive = prediction(tweet, positive_class)
    prediction_negative = prediction(tweet, negative_class)
    
    # p(tweet) = p(positive)*p(tweet|positive) + p(negative)*p(tweet|negative)
    p_tweet = prediction_positive + prediction_negative
    
    # p(class|tweet) = p(class) * p(tweet|class) / p(tweet)
    p_pos_tweet = prediction_positive / p_tweet
    p_neg_tweet = prediction_negative / p_tweet

    # calculate max(p(class|tweet)) for all classes and tweets
    if(p_pos_tweet>=p_neg_tweet):
        return 1
        
    return 0



if __name__ == '__main__':
    print("Enter path to csv file:")
    path = input()

    print("Loading training data...")
    (training_labels, training_features) = load_training_set(path)
    
    # 80% of the data will be used for learning, 20% for testing accuracy
    (learning_data, testing_data) = split_training_data(training_labels, training_features, 0.8)


    print("Learning...")
    model = learn(learning_data)

    print("Testing...")
    correct = 0
    for index, tweet in enumerate(testing_data[1]):
        tweet_class = classify(tweet, model)
        if(tweet_class == testing_data[0][index]):
            correct += 1
    test_results = correct/len(testing_data[0]) * 100

    print("Corectness: "+str(test_results))
    