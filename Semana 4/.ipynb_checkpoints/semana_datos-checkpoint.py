import requests
import pandas as pd
from bs4 import BeautifulSoup
import json

def extract_canonical_urls(data, urls=[]):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "canonical_url":
                urls.append(value)
            elif isinstance(value, (dict, list)):
                extract_canonical_urls(value, urls)
    elif isinstance(data, list):
        for item in data:
            extract_canonical_urls(item, urls)

    return urls

num_queries=100
for i in range(1, num_queries):
    if i%10 == 0:
        print(f"First {100*i} links...")
        
    url = "https://www.semana.com/pf/api/v3/content/fetch/content-apir?query={%22feedOffset%22:" + str(100*i) + ",%22feedSize%22:100,%22includeSections%22:%22politica%22,%22sourceInclude%22:%22canonical_url,_id,promo_items,headlines.basic,subheadlines.basic,description.basic,subtype,publish_date,taxonomy,copyright%22}&d=6860&_website=semana"
    response = requests.get(url)
    links = extract_canonical_urls(json.loads(response.text))
    
    
url = "https://www.semana.com/"
df = pd.DataFrame(columns=['content', 'date', 'headline', 'description'])

# Perform GET requests for each filtered hyperlink
for i, link in enumerate(links):
    if i%10 == 0:
        print(f"Extracted content of first {i} links...")
        
    full_url = url.rstrip('/') + link  # Construct the full URL by combining the base URL with the relative link
    link_response = requests.get(full_url)
#    print(f"Status code for {full_url}: {link_response.status_code}")
    link_soup = BeautifulSoup(link_response.text, 'html.parser')
    
    soup = BeautifulSoup(link_response.content, "html.parser")

    # Find all the <p> elements with the class "section sp-8"
    p_elements = soup.find_all("p", class_="section sp-8")

    # Extract the text from the elements and join them into a single string
    extracted_text = " ".join([element.get_text() for element in p_elements])
    
    # Find all the script tags with type="application/ld+json"
    script_tags = soup.find_all('script', {'type': 'application/ld+json'})

    for script_tag in script_tags:
        try:
            # Load the JSON data from the script tag
            json_data = json.loads(script_tag.string)

            # Check if the desired keys are present
            if 'datePublished' in json_data and 'headline' in json_data and 'description' in json_data:
                # Extract the 'datePublished', 'headline', and 'description' values
                date_published = json_data['datePublished']
                headline = json_data['headline']
                description = json_data['description']

                break
                
        except json.JSONDecodeError:
            # Skip this script tag if there's an error decoding JSON
            date_published = 'NA'
            headline = 'NA'
            description = 'NA'
            pass
    
    df = pd.concat([df, pd.DataFrame({'content': [extracted_text], 'date': [date_published], 'headline': [headline], 'description': [description]})])
    
    
top_two_feature_indices[-2:][::-1]


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

# Download the NLTK Spanish stopwords if you haven't already
spanish_stopwords = nltk.corpus.stopwords.words('spanish')

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, stop_words=spanish_stopwords, ngram_range=(1,3))

# Train the TfidfVectorizer and transform the texts
X = vectorizer.fit_transform(df['content']).todense()

# Get the feature names (tokens)
feature_names = vectorizer.get_feature_names()

# For each text, find the top two features and print the tokens separated by a comma
top_tokens = []
for i, text in enumerate(df['content']):
    row = X[i][0]
    top_two_feature_indices = np.array(row.argsort())[0][-5:][::-1]
    top_two_feature_tokens = ", ".join([feature_names[idx] for idx in top_two_feature_indices])
    top_tokens.append(top_two_feature_tokens)

df['top_tokens'] = top_tokens



import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Download the NLTK Spanish stopwords if you haven't already
nltk.download('stopwords')
spanish_stopwords = nltk.corpus.stopwords.words('spanish')

vectorizer = CountVectorizer(max_features=2000, stop_words=spanish_stopwords, ngram_range=(1,3))
articles_embeddings = vectorizer.fit_transform(articles).todense()

# Extract the vocabulary and the corresponding frequencies
vocabulary = vectorizer.get_feature_names_out()
frequencies = articles_embeddings.sum(axis=0).A1

# Sort the tokens by frequency in descending order
sorted_indices = frequencies.argsort()[::-1]

# Display the most frequent tokens along with their frequencies
for idx in sorted_indices:
    print(f"{vocabulary[idx]}: {frequencies[idx]}")