# from text_classifer_utils import preprocess, custom_standardization 
import warnings
import re
import shutil
import string
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.files.storage import default_storage

from keras.models import Sequential
from keras.losses import BinaryCrossentropy
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout,TextVectorization
from keras.models import model_from_json


import warnings
warnings.filterwarnings("ignore")

abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    'hwy': 'highway',
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    'w/e': 'whatever',
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

# Change an abbreviation to its true word
def fix_abbrev(text):
    return ' '.join([abbreviations[word.lower()] if (word.lower() in abbreviations.keys()) else word for word in text.split()])

# Replace some others smiley face with SADFACE
def transcription_sad(text):
    smiley = re.compile(r'[8:=;][\'\-]?[(\\/]')
    return smiley.sub(r'sad', text)

# Replace <3 with HEART
def transcription_heart(text):
    heart = re.compile(r'<3')
    return heart.sub(r'love', text)

# Replace URLs
def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

# Remove HTML
def remove_html(text):
    return re.sub(r'<.*?>', '', text)

# Converts text to lowercase
def to_lower(text):
    return text.lower()

# Remove words contaning numbers
def remove_numbers(text):
    return re.sub(r'\w*\d\w*', '', text)

# Remove text in brackets
def remove_brackets(text):
    return re.sub(r'\[.*?\]', '', text)  

# Replace mentions
def remove_mentions(text):
    return re.sub(r'@\w*', '', text)

# Remove hashtags
def remove_hashtags(text):
    return re.sub(r'#\w*', '', text)

# Remove emojis
def remove_emojis(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
    "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

# Remove non-ASCII characters
def remove_non_ascii(text):
#     return ''.join(filter(lambda x: x in string.printable, text))
    return text.encode("ascii",errors="ignore").decode()

# Remove stopwords
def remove_stopwords(text):
    return ' '.join([token.text for token in nlp(text) if not token.is_stop])

# Remove punctuation
def remove_punctuation(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Remove white space (Extra step, just in case)
def remove_whitespace(text):
    return ' '.join(text.split())

# Filter out words with too few characters (2 by default)
def filter_words(text):
    return ' '.join([word for word in text.split() if len(word) > 2])

def custom_standardization(input_data):
  return tf.strings.regex_replace(input_data,'[%s]' % re.escape(string.punctuation),'')

def perform_processing_on_training_set():
    # fill na values
    # csv_file = default_storage.open('Dataset/train.csv')
    # train = pd.read_csv(csv_file)
    train = pd.read_csv("static/Dataset/train.csv")
    for col in ["keyword", "location"]:
        train[col] = train[col].fillna("None")

    raw_loc = train.location.value_counts()
    # print(raw_loc)
    top_loc = list(raw_loc[raw_loc>=10].index)

    def clean_loc(x):
        if x == 'None':
            return 'None'
        elif x == 'Earth' or x =='Worldwide' or x == 'Everywhere':
            return 'World'
        elif 'New York' in x or 'NYC' in x:
            return 'New York'    
        elif 'London' in x:
            return 'London'
        elif 'Mumbai' in x:
            return 'Mumbai'
        elif 'Washington' in x and 'D' in x and 'C' in x:
            return 'Washington DC'
        elif 'San Francisco' in x:
            return 'San Francisco'
        elif 'Los Angeles' in x:
            return 'Los Angeles'
        elif 'Seattle' in x:
            return 'Seattle'
        elif 'Chicago' in x:
            return 'Chicago'
        elif 'Toronto' in x:
            return 'Toronto'
        elif 'Sacramento' in x:
            return 'Sacramento'
        elif 'Atlanta' in x:
            return 'Atlanta'
        elif 'California' in x:
            return 'California'
        elif 'Florida' in x:
            return 'Florida'
        elif 'Texas' in x:
            return 'Texas'
        elif 'United States' in x or 'USA' in x:
            return 'USA'
        elif 'United Kingdom' in x or 'UK' in x or 'Britain' in x:
            return 'UK'
        elif 'Canada' in x:
            return 'Canada'
        elif 'India' in x:
            return 'India'
        elif 'Kenya' in x:
            return 'Kenya'
        elif 'Nigeria' in x:
            return 'Nigeria'
        elif 'Australia' in x:
            return 'Australia'
        elif 'Indonesia' in x:
            return 'Indonesia'
        elif x in top_loc:
            return x
        else: return 'Others'


    train['location'] = train['location'].apply(lambda x: clean_loc(str(x)))

    for df in [train]:
        
        text = df['text']
        
        # Convert to lowercase
        text = text.apply(to_lower)

        # Replace symbols
        text = text.replace(r'&amp;?', r'and')
        text = text.replace(r'&lt;', r'<')
        text = text.replace(r'&gt;', r'>')
        text = text.replace('&amp;', " and ")
        
        # Manual Lemmatize
        text = text.str.replace('won\'t', 'will not')
        text = text.str.replace('can\'t', 'cannot')
        text = text.str.replace('i\'m', 'i am')
        text = text.replace('ain\'t', 'is not')
        
        # Remove mentions and links (hashtags too?)
    #     text = text.apply(remove_hashtags)
        text = text.apply(remove_mentions)
        text = text.apply(remove_urls)

        # Fix abbreviations
        text = text.apply(fix_abbrev)
        
        # Remove HTML tags
        text = text.apply(remove_html)
        

        # Fix emojies
        text = text.apply(transcription_sad)   # Sad emojies
        text = text.apply(transcription_heart) # Heart emoji
        text = text.apply(remove_emojis)       # General emojies

        # Remove non-ASCII characters
        text = text.apply(remove_non_ascii)
        
        # Remove words contaning numbers
        text = text.apply(remove_numbers)
        
        # Remove punctuations
        text = text.apply(remove_punctuation)

        
        # Fill entry if turns out empty
        text = text.apply(lambda x: x if x != '' else '?')

        df['text'] = text    

    train['text'] = custom_standardization(train['text'])

    return train

## function to classify and preprocess the data
def text_classify(data):
    data = {'text' : [data]}
    data = pd.DataFrame(data)
    print(type(data))
    print(data.head())

    train = perform_processing_on_training_set()
   
    max_features = 10000 # no of word in vocab
    sequence_length = 250   

    vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
    
    vectorize_layer.adapt(train['text'].values)
    # load json and create model

    json_file = open('static/text_classifier.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("static/text_classifier.h5")
    
    print("Loaded model from disk")


    data['text'] = custom_standardization(data['text'])
    print(type(data['text'][0]))
    print(data.head())
    loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    input_data = vectorize_layer(data['text'][0])
    
    predictions = loaded_model.predict(np.expand_dims(input_data,0))

    print(data)
    print(predictions)
    return predictions[0][0] >= 0.5

if __name__ == "__main__":
    data = str(input("Enter your text = "))
    print(text_classify(data))