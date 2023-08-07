import os

import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

OPEN_AI_KEYPATH = "/Users/timlee/Dropbox/keys/openai_key.txt"


def load_openai_key():
    with open(OPEN_AI_KEYPATH, "r") as f:
        return f.read().strip()


# App framework
st.title("Youtube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

# prompt templates
title_template = PromptTemplate(
    input_variables = ["topic"],
    template="Write me a youtube title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title"],
    template="Write me a youtube video script based on this TITLE: {title}"
)


# LLMs
llm = OpenAI(
    temperature=0.9,  # creative
    openai_api_key=load_openai_key()
)

title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key="title",
)

script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key="script",
)

seq_chain = SequentialChain(
    chains=[
        title_chain,
        script_chain,
    ],
    input_variables=["topic"],
    output_variables=["title", "script"],
    verbose=True
)


if prompt:
    response = seq_chain({"topic": prompt})
    st.write(response["title"])
    st.write(response["script"])
