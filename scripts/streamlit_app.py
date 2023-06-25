from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st
from transformers.pipelines import pipeline
import json
from predict import run_prediction

import os
import shutil
import zipfile
import requests

########
# Change to parent directory
#os.chdir("..")

# Clone the Git repository
#os.system("git clone https://github.com/TheAtticusProject/cuad.git")

# Rename the directory
#shutil.move("cuad", "cuad-training")

# Extract data.zip
#with zipfile.ZipFile("cuad-training/data.zip", "r") as zip_ref:
#    zip_ref.extractall("cuad-data")

# Create the cuad-models directory
#os.mkdir("cuad-models")

# Download roberta-base.zip
#url = "https://zenodo.org/record/4599830/files/roberta-base.zip?download=1"
#response = requests.get(url)
#with open("cuad-models/roberta-base.zip", "wb") as file:
#    file.write(response.content)

# Extract roberta-base.zip
#with zipfile.ZipFile("cuad-models/roberta-base.zip", "r") as zip_ref:
#    zip_ref.extractall("cuad-models")

########


st.set_page_config(layout="wide")

@st.cache_resource(show_spinner=False)
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', use_fast=False)
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_questions():
    with open('test.json') as json_file:
        data = json.load(json_file)
    questions = []
    for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
        question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
        questions.append(question)
    return questions

@st.cache_data(show_spinner=False, persist=True)
def load_contracts():
    with open('test.json') as json_file:
        data = json.load(json_file)
    contracts = []
    for i, q in enumerate(data['data']):
        contract = ' '.join(data['data'][i]['paragraphs'][0]['context'].split())
        contracts.append(contract)
    return contracts

model, tokenizer = load_model()
questions = load_questions()
contracts = load_contracts()

st.header("Contract Understanding Atticus Dataset (CUAD) Demo")
st.write("This demo uses a machine learning model for Contract Understanding.")

add_text_sidebar = st.sidebar.title("Sidebar Test Menu")
add_text_sidebar = st.sidebar.text("Hello, world!")

question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
paragraph = st.text_area(label="Contract")

if not (len(paragraph) == 0) and not (len(question) == 0):
    prediction = run_prediction(question, paragraph, 'Rakib/roberta-base-on-cuad')
    st.write("Answer: " + prediction.strip())
    st.write("Answer: ")
    st.write(prediction)

my_expander = st.expander("Sample Contract", expanded=False)
my_expander.write(contracts[1])
