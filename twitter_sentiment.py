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
# P(tweet|C) = (freq(words in tweet, C)+1) / |C| + |words in tweet|
def prediction(tweet, freq, p_class, class_size):
    tweetWords = Counter(tweet) # words and their occurencies
    result = 1
    for word in tweetWords:
        freqTweet = tweetWords.get(word) # number of occurancies in the tweet
        freqClass = (freq.get(word) +1) / sum(freq.values()) + class_size
        
        result *= freqTweet 


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

def classify(testing_data):
    features = testing_data[1] 
    for tweet in features:
        continue
    # P(class|data)
    
    return null

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
    test_results = classify(testing_data)
    