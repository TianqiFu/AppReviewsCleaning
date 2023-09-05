import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

from textblob import TextBlob

import re #正则表达式
import langid

# %%
df = pd.read_csv('./data/Flipboard.csv')
# %%
df.head()
# %%
# the purpose of demo is practice review data cleaning.
# i will follew the paper that Semantic-aware and Fine-grained App Review Bug Mining Approach <sup><a href="#ref1">1</a></sup> step by step.
# >1.Non-English reviews removal <br>
# >2.Emoji icons filtering <br>
# >3.split sentence <br>
# >4.Lemmatization and Lowercasing <br>
# >5.Contractions expansion <br>
# >6.Misspelled words correction <br>
# >7.Normalization use &lt;number&gt; replace number
# 1. <p name = "ref1">C:\Users\futia\Zotero\storage\IAX69KZ9</p>

# %% 1. Non-English reviews removal
def detect_en(text):
    if not isinstance(text, str):  # filter non-str data
        return []
    result = langid.classify(text)
    return result[0] == 'en'

# Apply the function to filter rows
df['detect_en'] = df['content'].apply(detect_en)
# Filter rows where the "detect" column is True
df = df[df['detect_en'] == True]
df.to_csv('df1.csv', index=False)

# %% 2. Emoji icons filtering

# %% 3. split sentence
# Function to tokenize sentences using NLTK
def tokenize_sentences(text):
    if not isinstance(text, str): #filter non-str data
        return []
    sentences = sent_tokenize(text)
    return sentences

# demo
text="All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
text2=tokenize_sentences(text)
print(text2)

# %%
# Apply the sentence splitting function to the second column and store the results in the fourth column
df['splited'] = df['content'].apply(tokenize_sentences)
# Save the updated DataFrame back to the CSV file
df.to_csv('df.csv', index=False)
# %%
df.head()
# %% Emoji icons filtering
