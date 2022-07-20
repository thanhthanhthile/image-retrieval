import sys
import os
from fastapi import FastAPI, Form, Request
from requests import request
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
import time
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle

sys.path.append('source')
from IR_models.textbased import *


app = FastAPI()

ID_CORPUS, CORPUS = pickle.load(open('source/corpus', 'rb'))
VECTOR_DOC = vectorizer.fit_transform(CORPUS)

app.mount('/static', StaticFiles(directory='source/static'), name='static')

templates = Jinja2Templates(directory='source/views')


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    context = {'request': request}
    return templates.TemplateResponse('index.html', context)


@app.post('/textbased')
async def IR_textbased(query: str = Form(...)):
    start = time.time()
    vector_query = preprocessing_query(query)
    similar = cosine_similarity(VECTOR_DOC, vector_query).flatten()
    related_docs_indices = similar.argsort()[:-(10+1):-1]
    
    related_img_path = []
    for idx, id in enumerate(related_docs_indices):
        img_name = str(ID_CORPUS[id]) + '.png'
        img_path = os.path.join(IMAGE_DIR, img_name)
        related_img_path.append(img_path)
    # show_img_retrieved(related_docs_indices, ID_CORPUS)
    stop = time.time()
    running_time = stop - start
    text_rs = '{} images retrieved in {}s'.format(10, running_time)
    return {'text': text_rs, 'img_path': related_img_path}
    # return templates.TemplateResponse('index.html', {'request': request, 'text_rs': text_rs, 'img_path': img_path}) 


# @app.post('/img-retrieval/text-based')
# async def ir_textbased(query: str, number_of_img: int):
#     # start time
#     start = time.time()
#     # preprocessing data text
#     filename = 'source/corpus'
#     infile = open(filename, 'rb')
#     id_corpus, corpus = pickle.load(infile)
#     infile.close()
#     # id_corpus, corpus = prepocessing_text(CAPTION_DIR)
#     vector_doc = vectorizer.fit_transform(corpus)
#     # doing processing query
#     vector_query = preprocessing_query(query)   
#     # Calculate cosine similarities
#     similar = cosine_similarity(vector_doc, vector_query).flatten()
#     related_docs_indices = similar.argsort()[:-(number_of_img+1):-1]
#     # stop time
#     stop = time.time()
#     running_time = stop - start
#     # show result
#     print('{} images retrieved in {}s'.format(number_of_img, running_time))
#     show_img_retrieved(related_docs_indices, id_corpus)
#     return running_time

if __name__ == "__main__":
    uvicorn.run(app, port=8000)