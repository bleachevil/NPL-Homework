# NPL-Homework
## 1. Sentiment Analysis
### Initial imports
```
import os
import pandas as pd
from dotenv import load_dotenv
import nltk as nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

%matplotlib inline
```

### Read your api key environment variable
```
load_dotenv()
api_key = os.getenv("News_API_KEY")
```

### Create a newsapi client
```
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key=api_key)
```

### Fetch the Bitcoin news articles
```
btc_headlines = newsapi.get_everything(
    q="Bitcoin", language="en", sort_by="relevancy"
)

btc_headlines
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/BTCnews.png?raw=true)

### Fetch the Ethereum news articles
```
eth_headlines = newsapi.get_everything(
    q="Ethereum", language="en", sort_by="relevancy"
)

eth_headlines
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/ETHnews.png?raw=true)

### Create the Bitcoin sentiment scores DataFrame
```
btc_sentiments = []

for article in btc_headlines["articles"]:
    try:
        text = article["content"]
        date = article["publishedAt"][:10]
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neu = sentiment["neu"]
        neg = sentiment["neg"]
        
        btc_sentiments.append({
            "text": text,
            "date": date,
            "compound": compound,
            "positive": pos,
            "negative": neg,
            "neutral": neu
            
        })
        
    except AttributeError:
        pass
    
btc_df = pd.DataFrame(btc_sentiments)

cols = ["date", "text", "compound", "positive", "negative", "neutral"]
btc_df = btc_df[cols]

btc_df.head()
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/BTCscoredatabase.png?raw=true)

### Create the Ethereum sentiment scores DataFrame
```
eth_sentiments = []

for article in eth_headlines["articles"]:
    try:
        text = article["content"]
        date = article["publishedAt"][:10]
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neu = sentiment["neu"]
        neg = sentiment["neg"]
        
        eth_sentiments.append({
            "text": text,
            "date": date,
            "compound": compound,
            "positive": pos,
            "negative": neg,
            "neutral": neu
            
        })
        
    except AttributeError:
        pass
    
eth_df = pd.DataFrame(eth_sentiments)

cols = ["date", "text", "compound", "positive", "negative", "neutral"]
eth_df = eth_df[cols]

eth_df.head()
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/ETHscoredatabase.png?raw=true)

### Describe the Bitcoin & Ethereum Sentiment
```
btc_df.describe()
eth_df.describe()
```
pic btc and eth decr

### Question and Answer

Q: Which coin had the highest mean positive score?<br />

A: BTC has highest mean positive score.<br />

Q: Which coin had the highest compound score?<br />

A: ETH has hihgest compound score.<br />

Q. Which coin had the highest positive score?<br />

A: ETH has hihgest positive score.<br />


## 2. Natural Language Processing
### Initial imports
```
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
import re
```
### Instantiate the lemmatizer and Create a list of stopwords
```
lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words('english'))
```

### Complete the tokenizer function
```
def tokenizer(text):
    """Tokenizes text."""
    
    sw = set(stopwords.words('english'))
    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', text)
    words = word_tokenize(re_clean)
    lem = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word.lower() for word in lem if word.lower() not in sw]
    
    
    return tokens
```

### Create a new tokens column for Bitcoin and Ethereum
```
tokenizer(btc_headlines["articles"][0]['content'])
tokenizer(eth_headlines["articles"][0]['content'])
```
#### Bitcoin
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/btctoken.png?raw=true)

#### Ethereum
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/ethtoken.png?raw=true)

### NGrams and Frequency Analysis
```
from collections import Counter
from nltk import ngrams
```

### Generate the Bitcoin N-grams where N=2
```
processed_btc = tokenizer(btc_headlines["articles"][0]['content'])
btc_counts = Counter(ngrams(processed_btc, n=2))
btc_counts
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/btcngram.png?raw=true)

### Generate the Ethereum N-grams where N=2
```
processed_eth = tokenizer(eth_headlines["articles"][0]['content'])
eth_counts = Counter(ngrams(processed_eth, n=2))
eth_counts
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/ethngram.png?raw=true)

### Function token_count generates the top 10 words for a given coin
```
def token_count(tokens, N=3):
    """Returns the top N tokens from the frequency count"""
    return Counter(tokens).most_common(10)
token_count(processed_btc, N=3)
token_count(processed_eth, N=3)
```
#### Bitcoin
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/top10btc.png?raw=true)

#### Ethereum
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/top10eth.png?raw=true)

### Word Clouds
```
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [20.0, 10.0]
```

### Generate the Bitcoin word cloud
```
big_string = ' '.join(processed_btc)
wc_btc = WordCloud().generate(big_string)
plt.imshow(wc_btc)
plt.title("Bitcoin word cloud", fontsize=48)
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/btcwordcloud.png?raw=true)

### Generate the Ethereum word cloud
```
big_string = ' '.join(processed_eth)
wc_eth = WordCloud().generate(big_string)
plt.imshow(wc_eth)
plt.title("Ethereum word cloud", fontsize=48)
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/ethwordcloud.png?raw=true)


### Named Entity Recognition
```
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
```

### Bitcoin NER
```
total_btc =[]
for x in btc_headlines["articles"]:
    total_btc.append(x['content'])
right_btc = ''.join(total_btc)
doc_btc = nlp(right_btc)
doc_btc.user_data["title"] = "Bitcoin NER"
displacy.render(doc_btc, style='ent')
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/btcNER.png?raw=true)

```
print([ent.text for ent in doc_btc.ents])
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/btclistofentities.png?raw=true)

### Eehereum NER
```
total_eth =[]
for x in eth_headlines["articles"]:
    total_eth.append(x['content'])
right_eth = ''.join(total_btc)
doc_eth = nlp(right_eth)
doc_eth.user_data["title"] = "Ethereum NER"
displacy.render(doc_eth, style='ent')
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/ethNER.png?raw=true)
```
print([ent.text for ent in doc_eth.ents])
```
![](https://github.com/bleachevil/NPL-Homework/blob/main/pic/ethlistofentities.png?raw=true)
