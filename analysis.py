import spacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models


with open("transcription.txt", "r", encoding="utf-8") as file:
    interviews = file.readlines()

nlp = spacy.load("tr_core_news_lg")

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

preprocessed_interviews = [preprocess(interview) for interview in interviews]


all_tokens = [token for interview in preprocessed_interviews for token in interview]
token_counts = Counter(all_tokens)

# Get the most common tokens
common_tokens = token_counts.most_common(20)
# print(common_tokens)

with open("common_tokens.txt", "w", encoding="utf-8") as file:
    for common_token in common_tokens:
        file.write(f"{common_token}\n")

# wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(token_counts)

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Create a dictionary and corpus for topic modeling
dictionary = corpora.Dictionary(preprocessed_interviews)
corpus = [dictionary.doc2bow(interview) for interview in preprocessed_interviews]

# Train an LDA model
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Print the topics
with open("topics.txt", "w", encoding="utf-8") as file:
    for idx, topic in lda_model.print_topics(-1):
        file.write(f"Topic {idx}: {topic} \n")