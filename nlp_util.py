import re

# clean the given tweet from punctuation signs and special characters
def clean(tweet):
    tmp = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)

    specials = '@#$&%'
    for item in specials:
        tmp = re.sub(str(item) + '\S+', '', tmp)

    tmp = re.sub('\t', ' ', tmp)    
    tmp = re.sub(' +', ' ', tmp)
    tmp = tmp.strip()
    return tmp

# leave only lowercase letters in the tweets
def letters_only(tweet):
    tmp = tweet.lower()

    result = ''
    for ch in tmp:
        if((ch>='a' and ch<='z') or ch == ' '):
            result += str(ch)

    return result