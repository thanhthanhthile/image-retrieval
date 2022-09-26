import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import time

from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.neighbors import NearestNeighbors

IMAGE_DIR = './source/data/img'

image_input = Input(shape=(224,224,3))
model = VGG16(input_tensor=image_input)
# model.summary()


def extract_feature(img_path):
    img = image.load_img(path=img_path, target_size=(224,224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    vgg16_feature = model.predict(img_arr)
    vgg16_feature_np = np.array(vgg16_feature).flatten()
    return vgg16_feature_np

def extract_feature_list():
    filename_list = []
    feature_list = []
    for file_name in os.listdir(IMAGE_DIR):
        filename_list.append(file_name)
        img_path = os.path.join(IMAGE_DIR, file_name)
        feature_list.append(extract_feature(img_path))
        
    feature_list = np.array(feature_list)
    return (filename_list, feature_list)

# def compute_distance_knn():
# def compute_distance_cos(vector_query, vector_corpus):
#     similar = cosine_similarity(vector_doc, vector_query).flatten()


def show_img_retrieved(related_img_indices, filename_list):
    fig = plt.figure(figsize=(15,6))
    i = 1
    for idx in related_img_indices:
        img_name = filename_list[idx]
        img = mpimg.imread(os.path.join(IMAGE_DIR, img_name))
        fig.add_subplot(len(related_img_indices)//5 + 1, 5, i)
        plt.axis('off')
        plt.title('#{}'.format(img_name))
        plt.imshow(img)
        i += 1
    plt.show()

if __name__=="__main__":

# --------------------------------------------------------------
    # filename_corpus, feature_corpus = extract_feature_list()
    # print(feature_corpus.shape)

    # filename = 'source/corpus_img'
    # outfile = open(filename, 'wb')
    # pickle.dump((filename_corpus, feature_corpus), outfile)
    # outfile.close()
# --------------------------------------------------------------
    start = time.time()
    FILENAME_LIST, VECTOR_CORPUS = pickle.load(open('source/corpus_img', 'rb'))

    knn = NearestNeighbors(n_neighbors=20, metric='euclidean')

    query_path = 'source/static/img/23760.png'
    feature_query = []
    feature_query.append(extract_feature(query_path))
    feature_query = np.array(feature_query)
    print(feature_query.shape)
    a = knn.fit(VECTOR_CORPUS)
    # filename = 'source/corpus_knn'
    # outfile = open(filename, 'wb')
    # pickle.dump(knn, outfile)
    # outfile.close()
    result = knn.kneighbors(feature_query, return_distance=False)
    stop = time.time()
    print(f'{round((stop - start) * 1000)} ms')
    # print(result[0][1])
    show_img_retrieved(result[0], FILENAME_LIST)
