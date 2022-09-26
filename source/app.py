import sys
import shutil
from fastapi import FastAPI, Form, Request, File, UploadFile
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
import time
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
from sklearn.neighbors import NearestNeighbors


sys.path.append('source')
from IR_models.textbased import *
from IR_models.contentbased import *

app = FastAPI()
app.mount('/static', StaticFiles(directory='source/static'), name='static')
templates = Jinja2Templates(directory='source/views')

# text
ID_CORPUS, CORPUS = pickle.load(open('source/corpus', 'rb'))
VECTOR_DOC = vectorizer.fit_transform(CORPUS)

# content
FILENAME_LIST, IMG_CORPUS = pickle.load(open('source/corpus_img', 'rb'))
knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(IMG_CORPUS)

@app.get('/index.html', response_class=HTMLResponse)
def index(request: Request):
    context = {'request': request}
    return templates.TemplateResponse('index.html', context)

@app.get('/contentbased.html', response_class=HTMLResponse)
def index(request: Request):
    context = {'request': request}
    return templates.TemplateResponse('contentbased.html', context)


@app.post('/text-based_query')
async def IR_textbased(request: Request, query: str = Form(...)):
    start = time.time()
    vector_query = preprocessing_query(query)
    similar = cosine_similarity(VECTOR_DOC, vector_query).flatten()
    related_docs_indices = similar.argsort()[:-(10+1):-1]
    
    related_img_name = []
    for _, id in enumerate(related_docs_indices):
        img_name = str(ID_CORPUS[id]) + '.png'
        img_path = os.path.join('/img/', img_name)
        related_img_name.append(img_path)
    # show_img_retrieved(related_docs_indices, ID_CORPUS)
    stop = time.time()
    running_time = f'{round((stop - start) * 1000)} ms'

    output = {'query': query, 'running_time': running_time, 'image_name': related_img_name}
    return templates.TemplateResponse('index.html', {'request': request,'output': output})


@app.post('/content-based_query')
async def IR_contentbased(request: Request, file: UploadFile = File(...)):
    dir_query = 'source/static/img_query'
    for f in os.listdir(dir_query):
        os.remove(os.path.join(dir_query, f))
    if file:
        query_location = f'source/static/img_query/{file.filename}'
        with open(query_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        start = time.time()
        feature_query = []
        feature_query.append(extract_feature(query_location))
        feature_query = np.array(feature_query)
        knn.fit(IMG_CORPUS)
        indices = knn.kneighbors(feature_query, return_distance=False)
        related_img_name = []
        for i in indices[0]:
            img_path = os.path.join('/img/', FILENAME_LIST[i])
            related_img_name.append(img_path)
        stop = time.time()
        running_time = f'{round((stop - start) * 1000)} ms'
        query_path = query_location[13:]
        output = {'query_path': query_path, 'running_time': running_time, 'image_name': related_img_name}
        
    return templates.TemplateResponse('contentbased.html', {'request': request, 'output': output})

if __name__ == "__main__":
    uvicorn.run(app)