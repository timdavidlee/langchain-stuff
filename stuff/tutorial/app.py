import os

import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

OPEN_AI_KEYPATH = "/Users/timlee/Dropbox/keys/openai_key.txt"


def load_openai_key():
    with open(OPEN_AI_KEYPATH, "r") as f:
        return f.read().strip()


# App framework
st.title("Youtube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

# prompt templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Write me a youtube title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="""
Write me a youtube video script for children based on this TITLE: {title}
Leverage this wikipedia research:
{wikipedia_research}
"""
)

title_memory = ConversationBufferMemory(
    input_key="topic",
    memory_key="chat_history"
)

script_memory = ConversationBufferMemory(
    input_key="title",
    memory_key="chat_history"
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
    memory=title_memory,
)

script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key="script",
    memory=script_memory
)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(
        title=title,
        wikipedia_research=wiki_research
    )

    st.write(title)
    st.write(script)

    with st.expander("Message History"):
        st.info(title_memory.buffer)

    with st.expander("Message History"):
        st.info(script_memory.buffer)

    with st.expander("Message History"):
        st.info(wiki_research)
