
import pandas as pd
import pandasql as ps

import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

OPEN_AI_KEYPATH = "/Users/timlee/Dropbox/keys/openai_key.txt"


def load_openai_key():
    with open(OPEN_AI_KEYPATH, "r") as f:
        return f.read().strip()


csv_file = "/Users/timlee/Documents/data/customer-shopping-dataset/customer_shopping_data.csv"
customer_shopping_data = pd.read_csv(csv_file, parse_dates=["invoice_date"])

# App framework
st.title("Data Navigator GPT")
st.text("Here is a few rows some sample sales data")
st.table(data=customer_shopping_data.head(3).transpose())
prompt = st.text_input("As a Question about the data: `customer shoppping`")

# prompt templates
query_template = PromptTemplate(
    input_variables=["question"],
    template="""
customer_shopping_data:
    invoice_no: text
    customer_id: text
    gender: text
    age: integer
    category: text
    quantity: integer
    price: float
    payment_method: str
    invoice_date: datetime
    shopping_mall: str

Generate a sql query for the question: {question}?
"""
)

type_of_chart_template = PromptTemplate(
    input_variables=["question"],
    template="""
Available charts:
    bar
    line

Given the following question, did the user ask for a chart, answer the chart type or No? {question}
"""
)

# LLMs
llm = OpenAI(
    temperature=0.5,  # creative
    openai_api_key=load_openai_key()
)

query_chain = LLMChain(
    llm=llm,
    prompt=query_template,
    verbose=True,
    output_key="query",
)

table_chain = LLMChain(
    llm=llm,
    prompt=type_of_chart_template,
    verbose=True,
    output_key="table_type",
)


if prompt:
    generated_query = query_chain.run(prompt)
    type_of_chart = table_chain.run(prompt)
    # st.write(generated_query)

    results = ps.sqldf(generated_query, locals())
    results = results.set_index(results.columns[0])
    st.table(data=results.head(10))
    # st.write(type_of_chart)

    if "Yes" in type_of_chart:
        if "bar" in type_of_chart:
            st.bar_chart(data=results)
        if "line" in type_of_chart:
            st.line_chart(data=results)
