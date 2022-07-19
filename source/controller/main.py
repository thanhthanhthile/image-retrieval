import sys
from copyreg import pickle
from fastapi import FastAPI, File
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
import time
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

import pickle
sys.path.append('source')
from IR_models.textbased import *

app = FastAPI()

@app.get('/')
def hello_world(name: str):
    return f"Hello {name}!"

@app.post('/img-retrieval/text-based')
async def ir_textbased(query: str, number_of_img: int):
    # start time
    start = time.time()
    # preprocessing data text
    filename = 'source/corpus'
    infile = open(filename, 'rb')
    id_corpus, corpus = pickle.load(infile)
    infile.close()
    # id_corpus, corpus = prepocessing_text(CAPTION_DIR)
    vector_doc = vectorizer.fit_transform(corpus)
    # doing processing query
    vector_query = preprocessing_query(query)   
    # Calculate cosine similarities
    similar = cosine_similarity(vector_doc, vector_query).flatten()
    related_docs_indices = similar.argsort()[:-(number_of_img+1):-1]
    # stop time
    stop = time.time()
    running_time = stop - start
    # show result
    print('{} images retrieved in {}s'.format(number_of_img, running_time))
    show_img_retrieved(related_docs_indices, id_corpus)
    return running_time

if __name__ == "__main__":
    uvicorn.run(app, port=8000)