"""python -m stuff.tutorial.test_openai"""
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from stuff.util import load_openai_key


def main():
    chat = ChatOpenAI(temperature=.7, openai_api_key=load_openai_key())
    print(
        chat(
            [
                SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
                HumanMessage(content="I like tomatoes, what should I eat?")
            ]
        )
    )


if __name__ == "__main__":
    main()
