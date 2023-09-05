# %%
import langid
import pandas as pd

def detect_en(text):
    if not isinstance(text, str):  # filter non-str data
        return []
    result = langid.classify(text)
    return result[0] == 'en'

# Apply the language detection function to filter out non-English comments
df = pd.read_csv('./data/Flipboard.csv')
text = "Constantly refreshes, unfortunately sometimes in the middle of reading an article 😑 when that happens it takes you back to the start page with all new articles and there's no way to get back to it, even by searching"
temp = detect_en(text)
print(temp)
# Apply the function to filter rows
df['detect_en'] = df['content'].apply(detect_en)
# %%
df.head()
# Filter rows based on the 'IsEnglish' column
# df1 = df[df['detcet_en']]
# df1.to_csv('english_comments5.csv', index=False)
# %%
# Filter rows where the "detect" column is True
df = df[df['detect_en'] == True]
# df = df.drop(df[df.detect_en == 'False'].index, inplace=True)
df.to_csv('df1.csv', index=False)
# %%
df.head()