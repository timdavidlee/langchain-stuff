"""cd ~/myrepos/langchain-stuff/stuff/demo/guess_city/ && streamlit run app.py"""
import json
from typing import List
from random import choice

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import streamlit as st

from glob import glob

TEXT_JSON_FILES = "/Users/timlee/myrepos/langchain-stuff/stuff/demo/guess_city/descriptions/*"
OPENAI_KEY = "/Users/timlee/Dropbox/keys/openai_key.txt"


def load_city_wiki_obj(fl: str):
    with open(fl, "r") as f:
        return json.load(f)


def load_openai_key():
    with open(OPENAI_KEY, "r") as f:
        return f.read()


def load_city_objs():
    fls = glob(TEXT_JSON_FILES)
    city_obj_collector = []
    for fl in fls:
        try:
            city_obj = load_city_wiki_obj(fl)
            city_obj_collector.append(city_obj)
        except Exception as e:
            print(fl)
            raise e

    return city_obj_collector


def match_phrase(input_str: str, correct_matches: List[str]):
    for phr in correct_matches:
        if phr.lower() in input_str:
            return True

    return False


def catch_gen_ai_has_answer(input_str: str):
    if "the correct city is" in input_str:
        return True
    return False


def main():
    city_obj_collector = load_city_objs()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    single_selection = choice(city_obj_collector)
    correct_city = (single_selection["city"])
    correct_country = (single_selection["country"])
    single_selection_list = [single_selection["wiki_text"]]

    text_parts = text_splitter.create_documents(single_selection_list)
    encoder = OpenAIEmbeddings(openai_api_key=load_openai_key())
    db = Chroma.from_documents(text_parts, encoder)

    st.title("Guess the City")
    st_guess_prompt = st.text_input(
        "Ask some questions! When ready ask `is the correct answer` or `is the correct city`..."
    )

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=load_openai_key()),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "verbose": True,
        }
    )

    if st_guess_prompt:
        if match_phrase(st_guess_prompt, ["is the correct answer", "is the correct city"]):
            if match_phrase(st_guess_prompt, [correct_city,]):
                st.info("correct! You got it right, the answer was: {} {}".format(
                    correct_city,
                    correct_country,
                ))
            elif match_phrase(st_guess_prompt, [correct_country, ]):
                st.info("close! You got the country correct [{}], but what's the city?".format(correct_country))
            else:
                answer = qa.run(st_guess_prompt)
                st.info("Nope, Sorry!")
                if not catch_gen_ai_has_answer(answer):
                    st.info(answer)

        else:
            answer = qa.run(st_guess_prompt)
            st.info(answer)


if __name__ == "__main__":
    main()
