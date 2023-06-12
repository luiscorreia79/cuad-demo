from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st
from transformers.pipelines import pipeline
import json
from predict import run_prediction

st.set_page_config(layout="wide")

@st.cache(show_spinner=False, persist=True)
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', use_fast=False)
    return model, tokenizer

@st.cache(show_spinner=False, persist=True)
def load_questions():
    with open('cuad-data/test.json') as json_file:
        data = json.load(json_file)
    questions = []
    for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
        question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
        questions.append(question)
    return questions

@st.cache(show_spinner=False, persist=True)
def load_contracts():
    with open('cuad-data/test.json') as json_file:
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

add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Hello, world!")

question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
paragraph = st.text_area(label="Contract")

if not (len(paragraph) == 0) and not (len(question) == 0):
    prediction = run_prediction(question, paragraph, 'cuad-models/roberta-base/')
    st.write("Answer: " + prediction.strip())
    st.write("Answer: ")
    st.write(prediction)

my_expander = st.beta_expander("Sample Contract", expanded=False)
my_expander.write(contracts[1])
