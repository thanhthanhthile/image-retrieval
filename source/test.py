import pickle

filename = 'source/corpus'
infile = open(filename, 'rb')
id, corpus = pickle.load(infile)
infile.close()
print(corpus)
