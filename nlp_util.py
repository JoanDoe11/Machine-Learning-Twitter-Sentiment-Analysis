import re



def clean(tweet):
    """
    clean the given tweet from punctuation signs and special characters

    Parameters
    ----------
    tweet : string
        
    Returns
    -------
    string
    """
    tmp = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)

    specials = '@#$&%'
    for item in specials:
        tmp = re.sub(str(item) + '\S+', '', tmp)

    tmp = re.sub('\t', ' ', tmp)    
    tmp = re.sub(' +', ' ', tmp)
    tmp = tmp.strip()
    
    return tmp

# 
def letters_only(tweet):
    """
    leave only lowercase letters in the tweets

    Parameters
    ----------
    tweet : string
        
    Returns
    -------
    string
    
    """
    tmp = tweet.lower()

    result = ''
    for ch in tmp:
        if((ch>='a' and ch<='z') or ch == ' '):
            result += str(ch)

    return result