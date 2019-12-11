import uvicorn
from fastapi import FastAPI, Form
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import numpy as np

from model import preprocess_text, run_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def postprocess(val):
    if val < 0.5:
        return 0
    else:
        return 1

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    return loaded_model


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model")
def rerun_model():
    run_model()
    return "completed running model"


@app.post("/predict")
def run_prediction(request: Request, comment: str = Form(...)):
    model = load_model()
    comment = preprocess_text(comment)
    tk = Tokenizer()
    tk.fit_on_texts(comment)
    index_list = tk.texts_to_sequences(comment)
    x_train = pad_sequences(index_list, maxlen=200)
    output = model.predict(x_train)
    indexes = np.argmax(output, axis=0)
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    dict = {}
    for i in range(len(indexes)):
        prediction = output[indexes[i]][i]
        prediction = postprocess(prediction)
        dict[labels[i]] = prediction
    return templates.TemplateResponse("output.html", {"request": request, "output":dict})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
