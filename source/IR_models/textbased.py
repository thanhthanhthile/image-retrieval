from fileinput import filename
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import nltk
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle

IMAGE_DIR = './source/data/img'
CAPTION_DIR = './source/data/txt'

vectorizer = TfidfVectorizer()

nltk.download('punkt')
nltk.download('stopwords')

'''---Remove punctuation---'''
def remove_punctuation(text):
    return text.translate(str.maketrans('','', string.punctuation))

'''
Remove all the duplicate whitespaces
and newline characters
'''
def remove_whitespace(text):
    return " ".join(text.split())


'''
Remove stopwords and noisy words
'''
def remove_stopwords(doc):
    # noisy_words = TfidfVectorizer.get_feature_names_out()
    # noisy_words = noisy_words[:350].tolist()

    words_to_remove = stopwords.words('english')
    cleaned_doc = []
    for word in doc:
        if word not in words_to_remove:
            cleaned_doc.append(word)
    return cleaned_doc

'''
Tokenization
'''
def get_tokenized_list(doc):
    tokens = nltk.word_tokenize(doc)
    return tokens

'''
Stemming
'''
def word_stemmer(token_list):
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

'''
Preprocessing captions
'''
def prepocessing_text(caption_path):
    """
    Argument:
    path -- path of the caption txt files
    
    Returns:
            id_corpus -- list of id corpus and
            corpus -- list of corpus
    """
    id_corpus = []
    corpus = []

    for file_name in sorted(os.listdir(caption_path)):
        lines = open(os.path.join(caption_path, file_name), 'r', encoding='cp1252')
        temp_str = ''
        try:
          for line in lines:
              # remove endline character
              if line[-1] == '\n':
                  temp_str += line[:-1] + ' '
              else:
                  temp_str += line
          temp_str = temp_str.lower()
          temp_str = remove_whitespace(temp_str)
          temp_str = remove_punctuation(temp_str)
          corpus.append(temp_str)
          id_corpus.append(str(file_name[:-4]))
          lines.close()
        except:
          pass
    cleaned_corpus = []
    for d in corpus:
        tokens = get_tokenized_list(d)
        doc = remove_stopwords(tokens)
        doc = word_stemmer(doc)
        doc = ' '.join(doc)
        cleaned_corpus.append(doc)
    
    return (id_corpus, cleaned_corpus)


'''
Preprocessing query
'''
def preprocessing_query(query):
    """
    Argument:
        query -- a query

    Returns:
        vector_query
    """
    query = query.lower()
    query = remove_punctuation(query)
    query = remove_whitespace(query)
    query = get_tokenized_list(query)
    query = remove_stopwords(query)
    q = []
    for w in word_stemmer(query):
        q.append(w)
    q = ' '.join(q)
    vector_query = vectorizer.transform([q])
    return vector_query

'''
Show Images retrived
'''
def show_img_retrieved(related_docs_indices, id_corpus):
    fig = plt.figure(figsize=(15,6))
    for idx, id in enumerate(related_docs_indices):
        img_name = str(id_corpus[id]) + '.png'
        img = mpimg.imread(os.path.join(IMAGE_DIR, img_name))
        fig.add_subplot(len(related_docs_indices)//5 + 1, 5, idx+1)
        plt.title('#{}'.format(idx+1))
        plt.axis('off')
        plt.imshow(img)
    plt.show()


# if __name__ == "__main__":

#     id_corpus, corpus = prepocessing_text(CAPTION_DIR)

#     filename = 'source/corpus'
#     outfile = open(filename, 'wb')
#     pickle.dump(prepocessing_text(CAPTION_DIR), outfile)
#     outfile.close()

#     vector_doc = vectorizer.fit_transform(corpus)

#     query = input('Your query: ')
#     number_of_img = int(input('Number of images retrieved: '))
#     start = time.time()
#     vector_query = preprocessing_query(query)
#     # Calculate cosine similarities
#     similar = cosine_similarity(vector_doc, vector_query).flatten()
#     related_docs_indices = similar.argsort()[:-(number_of_img+1):-1]
#     stop = time.time()
#     running_time = stop - start
#     print('{} images retrieved in {}s'.format(number_of_img, running_time))
#     show_img_retrieved(related_docs_indices, id_corpus)
