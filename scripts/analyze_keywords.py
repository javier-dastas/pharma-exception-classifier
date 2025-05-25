import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load data
df = pd.read_csv('data/exceptions.csv')

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['exception_text'])

# Create DataFrame
words_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
words_df['label'] = df['label']

# Calculate mean frequency per label
mean_by_label = words_df.groupby('label').mean().T

# Get top 5 keywords per label
top_keywords = mean_by_label.apply(lambda x: x.sort_values(ascending=False).head(5))
print(top_keywords)
