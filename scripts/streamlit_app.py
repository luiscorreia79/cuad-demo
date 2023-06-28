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
add_text_sidebar = st.sidebar.text("This Demo is available on:")
add_text_sidebar = st.sidebar.markdown("[https://huggingface.co/spaces/marshmellow77/contract-review](https://huggingface.co/spaces/marshmellow77/contract-review)")


question = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
paragraph = st.text_area(label="Contract")

if not (len(paragraph) == 0) and not (len(question) == 0):
    prediction = run_prediction(question, paragraph, 'Rakib/roberta-base-on-cuad')
    st.write("Answer: " + str(prediction).strip())
    st.write("Answer: ")
    st.write(prediction)

my_expander = st.expander("Sample Contract", expanded=False)
my_expander.write(contracts[1])
