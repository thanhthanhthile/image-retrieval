import pickle

filename = 'source/corpus_img'
infile = open(filename, 'rb')
id, corpus = pickle.load(infile)
infile.close()
print(corpus.shape)
print(id)