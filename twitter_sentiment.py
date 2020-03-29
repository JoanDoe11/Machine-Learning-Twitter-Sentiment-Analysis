import csv

def load_training_set(path):

    # rt = read as text
    with open(path,'rt', encoding = "ISO-8859-1") as datafile:
        reader = csv.reader(datafile)
        header = next(reader)

    return ""

if __name__ == '__main__':
    print("Enter path to csv file:")
    path = input()

    print("Loading training data...")
    training_data = load_training_set(path)