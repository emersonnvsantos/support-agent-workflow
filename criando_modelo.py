import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.agents import AgentState
from typing_extensions import NotRequired
from typing import Literal

load_dotenv()

chave_api = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(model="gpt-5.5", temperature=0, api_key=chave_api, stream_usage=True)

EtapaSuporte = Literal[
    "coletor_garantia",
    "classificador_problema",
    "especialista_resolucao"
]