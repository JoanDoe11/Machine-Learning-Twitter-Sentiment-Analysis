import csv
import nlp_util
from collections import Counter

def load_training_set(path):
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

def init_class(labels, features):
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

    return (positive_features, negative_features)

# Naive Bayes prediction
# P(tweet|C) = (freq(words in tweet, C)+1) / (|C| + |words in tweet|)
# P(tweet) = sum(P(tweet|C)*P(C))
def prediction(tweet, class_model):
    freq = class_model[1]
    p_class = class_model[2]
    class_size = len(class_model[0])

    tweetWords = Counter(tweet) # words and their occurencies
    result = 1
    for word in tweetWords:
        freqTweet = tweetWords.get(word) # number of occurancies in the tweet
        freqClass = freq.get(word,0) # number of occurancies in the class
        
        result *= freqTweet * (freqClass + 1) / (len(tweet) + class_size)
    return result * p_class

def learn(learning_data):
    labels = learning_data[0]
    features = learning_data[1]
    

    # initialization of classes and their initial members
    (positive_features, negative_features) = init_class(labels, features)

    # calculation of word frequences in classes
    positive_freq = Counter(positive_features)
    negative_freq = Counter(negative_features)

    # probability calculations
    numOfPos = len(positive_features)
    numOfNeg = len(negative_features)
    numTotal = numOfPos + numOfNeg
    # P(class)
    p_positive = numOfPos / numTotal
    p_negative = numOfNeg / numTotal
    
    #assembling models
    positive_model = (positive_features, positive_freq, p_positive)
    negative_model = (negative_features, negative_freq, p_negative)

    return (positive_model, negative_model)


# calculate max(p(class|tweet)) for all classes and tweets
# p(class|tweet) = p(class) * p(tweet|class) / p(tweet)
# p(tweet) = p(positive)*p(tweet|positive) + p(negative)*p(tweet|negative)
def classify(tweet, model):
    positive_class = model[0]
    negative_class = model[1]

    prediction_positive = prediction(tweet, positive_class)
    prediction_negative = prediction(tweet, negative_class)
    
    p_tweet = prediction_positive + prediction_negative
    if(p_tweet == 0):
        print(tweet)
    else:
        p_pos_tweet = prediction_positive / p_tweet
        p_neg_tweet = prediction_negative / p_tweet

        if(p_pos_tweet>=p_neg_tweet):
            return 1
        
    return 0

def split_training_data(scale):
    data_amount = len(training_labels)
    border = int(scale * data_amount)
    learning_labels = training_labels[:border]
    learning_features = training_features[:border]
    testing_labels = training_labels[border:]
    testing_features = training_features[border:]

    learning_data = (learning_labels, learning_features)
    testing_data = (testing_labels, testing_features)
    return (learning_data, testing_data)


if __name__ == '__main__':
    print("Enter path to csv file:")
    path = input()

    print("Loading training data...")
    (training_labels, training_features) = load_training_set(path)
    
    # 80% of the data will be used for learning, 20% for testing accuracy
    (learning_data, testing_data) = split_training_data(0.8)

    print("Learning...")
    model = learn(learning_data)

    print("Testing...")
    correct = 0
    for index, tweet in enumerate(testing_data[1]):
        tweet_class = classify(testing_data[1], model)
        if(tweet_class == testing_data[0][index]):
            correct += 1
    test_results = len(test_results[0])/correct * 100

    print("Corectness: "+str(test_results))
    