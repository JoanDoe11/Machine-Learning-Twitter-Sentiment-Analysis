import csv
import nlp_util

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

if __name__ == '__main__':
    print("Enter path to csv file:")
    path = input()

    print("Loading training data...")
    (training_labels, training_features) = load_training_set(path)
    print(training_features[1])