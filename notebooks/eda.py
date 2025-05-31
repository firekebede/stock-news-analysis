# Descriptive Statistics
# Headline Length

# Import basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional for Jupyter notebooks (ignore if you're using .py files)
# %matplotlib inline  

# Load your dataset (adjust the filename)
df = pd.read_csv(r"C:\Users\Administrator\Downloads\raw_analyst_ratings.csv")
# Calculate headline length
df['headline_length'] = df['headline'].astype(str).str.len()

# Print basic statistics
print(df['headline_length'].describe())

# Visualize headline length distribution
sns.histplot(df['headline_length'], bins=30, kde=True)
plt.title("Distribution of Headline Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()

#Count Articles per Publisher
top_publishers = df['publisher'].value_counts().head(10)
print(top_publishers)

top_publishers.plot(kind='bar')
plt.title("Top 10 Publishers by Article Count")
plt.ylabel("Number of Articles")
plt.show()
#Trend in Publication Dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['pub_day'] = df['date'].dt.date

df['pub_day'].value_counts().sort_index().plot(figsize=(12, 4), title="Publication Trend Over Time")
plt.ylabel("Article Count")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Text Analysis (Topic Modeling)
#Preprocess Text

import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

df['clean_headline'] = df['headline'].apply(preprocess)


#Topic Modeling
vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
dtm = vectorizer.fit_transform(df['clean_headline'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

# Show top keywords for each topic
for idx, topic in enumerate(lda.components_):
    print(f"Topic #{idx+1}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print("\n")

    
#Time Series Analysis
#Publication Frequency Over Time

df.groupby(df['date'].dt.date).size().plot(title="Article Frequency Over Time", figsize=(12, 4))
plt.xlabel("Date")
plt.ylabel("Article Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze Publishing Time (if time exists)
if 'time' in df.columns:
    df['hour'] = pd.to_datetime(df['time'], errors='coerce').dt.hour
    df['hour'].value_counts().sort_index().plot(kind='bar')
    plt.title("Publishing Hours Distribution")
    plt.xlabel("Hour of Day")
    plt.ylabel("Article Count")
    plt.show()

    #. Publisher Analysis
    # Most Active Publishers
df['publisher'].value_counts().head(10).plot(kind='bar')
plt.title("Top Publishers")
plt.ylabel("Article Count")
plt.xticks(rotation=45)
plt.show()

# Domain Analysis (for email-like publishers)
if df['publisher'].str.contains('@').any():
    df['domain'] = df['publisher'].str.extract(r'@([A-Za-z0-9.-]+)')
    df['domain'].value_counts().head(10).plot(kind='bar')
    plt.title("Top Publisher Domains")
    plt.ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.show()

