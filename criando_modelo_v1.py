import os
from dotenv import load_dotenv
from typing import Literal

from typing_extensions import NotRequired

from langchain_openai import ChatOpenAI
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage

from langgraph.types import Command


# ==========================================================
# CONFIGURAÇÃO DO MODELO
# ==========================================================

load_dotenv()

chave_api = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0,
    api_key=chave_api,
    stream_usage=True,
)


# ==========================================================
# ETAPAS DO FLUXO
# ==========================================================

EtapaOrcamento = Literal[
    "identificacao_cliente",
    "consulta_produtos",
    "coleta_informacoes",
    "validacao_informacoes",
    "geracao_orcamento",
    "confirmacao_cliente",
    "envio_orcamento",
]


# ==========================================================
# STATE
# ==========================================================

class EstadoSuporte(AgentState):
    """Estado do fluxo de geração de orçamento."""

    etapa_atual: NotRequired[EtapaOrcamento]

    cliente_localizado: NotRequired[
        Literal["localizado", "nao_localizado"]
    ]

    produto_localizado: NotRequired[
        Literal["localizado", "nao_localizado"]
    ]

    quantidade_pessoas: NotRequired[int]

    regiao: NotRequired[str]

    informacoes_validas: NotRequired[
        Literal["correto", "incorreto"]
    ]

    orcamento_gerado: NotRequired[
        Literal["sim", "nao"]
    ]

    cliente_confirmou: NotRequired[
        Literal["sim", "nao"]
    ]

    orcamento_enviado: NotRequired[
        Literal["sim", "nao"]
    ]


# ==========================================================
# TOOLS
# ==========================================================

@tool
def localizar_cadastro_cliente(
    status: Literal["localizado", "nao_localizado"],
    runtime: ToolRuntime[None, EstadoSuporte],
) -> Command:
    """Registra se o cadastro do cliente foi localizado no sistema."""

    proxima_etapa = (
        "consulta_produtos"
        if status == "localizado"
        else "identificacao_cliente"
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        f"Status da localização do cadastro "
                        f"do cliente: {status}"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "cliente_localizado": status,
            "etapa_atual": proxima_etapa,
        }
    )


@tool
def localizar_produto_cliente(
    status: Literal["localizado", "nao_localizado"],
    runtime: ToolRuntime[None, EstadoSuporte],
) -> Command:
    """Registra se o produto do cliente foi localizado no sistema."""

    proxima_etapa = (
        "coleta_informacoes"
        if status == "localizado"
        else "consulta_produtos"
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        f"Status da localização do produto "
                        f"do cliente: {status}"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "produto_localizado": status,
            "etapa_atual": proxima_etapa,
        }
    )


@tool
def registrar_informacoes_orcamento(
    quantidade_pessoas: int,
    regiao: str,
    runtime: ToolRuntime[None, EstadoSuporte],
) -> Command:
    """Registra as informações necessárias para gerar o orçamento."""

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        f"Informações registradas: "
                        f"{quantidade_pessoas} pessoas, "
                        f"região {regiao}"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "quantidade_pessoas": quantidade_pessoas,
            "regiao": regiao,
            "etapa_atual": "validacao_informacoes",
        }
    )


# ==========================================================
# PROMPTS
# ==========================================================

IDENTIFICACAO_CLIENTE_PROMPT = """
Você é um assistente responsável por auxiliar clientes
na geração de orçamentos.

ETAPA ATUAL: Identificação do cliente

CONTEXTO:
Este canal é exclusivo para clientes previamente cadastrados.

Nesta etapa:

1. Cumprimente o cliente de forma cordial.
2. Identifique o cliente no sistema.
3. Use localizar_cadastro_cliente para registrar o resultado.
4. Se o cliente for localizado, avance para consulta dos produtos.

Não solicite informações do orçamento nesta etapa.
"""


CONSULTA_PRODUTOS_PROMPT = """
Você é um assistente responsável por auxiliar clientes
na geração de orçamentos.

ETAPA ATUAL: Consulta dos produtos

Status do cliente:
{cliente_localizado}

CONTEXTO:
Os clientes possuem produtos previamente cadastrados.

Nesta etapa:

1. Consulte os produtos cadastrados do cliente.
2. Identifique o produto relacionado ao orçamento.
3. Use localizar_produto_cliente para registrar o resultado.
4. Se o produto for localizado, avance para coleta das informações.

Não invente produtos ou preços.
"""


COLETA_INFORMACOES_PROMPT = """
Você é um assistente responsável por auxiliar clientes
na geração de orçamentos.

ETAPA ATUAL: Coleta das informações

Cliente localizado:
{cliente_localizado}

Produto localizado:
{produto_localizado}

Nesta etapa, colete:

1. Quantidade de pessoas
2. Região onde o serviço será realizado

Regras:

- Faça perguntas simples.
- Não pergunte novamente informações já fornecidas.
- Se faltar apenas um dado, peça somente esse dado.
- Não gere o orçamento ainda.
- Quando quantidade e região estiverem disponíveis,
  use registrar_informacoes_orcamento.

Exemplo:

Cliente:
"Quero orçamento para 50 pessoas."

Resposta:
"Qual é a região onde o serviço será realizado?"
"""


# ==========================================================
# CONFIGURAÇÃO DAS ETAPAS
# ==========================================================

CONFIG_ETAPAS = {

    "identificacao_cliente": {
        "prompt": IDENTIFICACAO_CLIENTE_PROMPT,
        "tools": [
            localizar_cadastro_cliente
        ],
        "requires": [],
    },

    "consulta_produtos": {
        "prompt": CONSULTA_PRODUTOS_PROMPT,
        "tools": [
            localizar_produto_cliente
        ],
        "requires": [
            "cliente_localizado"
        ],
    },

    "coleta_informacoes": {
        "prompt": COLETA_INFORMACOES_PROMPT,
        "tools": [
            registrar_informacoes_orcamento
        ],
        "requires": [
            "cliente_localizado",
            "produto_localizado",
        ],
    },

}


# ==========================================================
# AGENTE
# ==========================================================

agente = create_agent(
    model=model,
    tools=[
        localizar_cadastro_cliente,
        localizar_produto_cliente,
        registrar_informacoes_orcamento,
    ],
    state_schema=EstadoSuporte,
)


# ==========================================================
# FUNÇÃO PARA EXIBIR STATE
# ==========================================================

def mostrar_estado(resultado):

    print("\n===================================")
    print("        ESTADO ATUAL DO CHAT")
    print("===================================")

    print(
        "Etapa atual:",
        resultado.get("etapa_atual")
    )

    print(
        "Cliente localizado:",
        resultado.get("cliente_localizado")
    )

    print(
        "Produto localizado:",
        resultado.get("produto_localizado")
    )

    print(
        "Quantidade de pessoas:",
        resultado.get("quantidade_pessoas")
    )

    print(
        "Região:",
        resultado.get("regiao")
    )

    print(
        "Informações válidas:",
        resultado.get("informacoes_validas")
    )

    print(
        "Orçamento gerado:",
        resultado.get("orcamento_gerado")
    )

    print(
        "Cliente confirmou:",
        resultado.get("cliente_confirmou")
    )

    print(
        "Orçamento enviado:",
        resultado.get("orcamento_enviado")
    )

    print("===================================")


# ==========================================================
# TESTE 1 - IDENTIFICAR CLIENTE
# ==========================================================

resultado = agente.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content":
                    "O cadastro do cliente foi localizado."
            }
        ],

        "etapa_atual":
            "identificacao_cliente",
    }
)

print("\n--- APÓS IDENTIFICAR CLIENTE ---")

mostrar_estado(resultado)


# ==========================================================
# TESTE 2 - LOCALIZAR PRODUTO
# ==========================================================

resultado = agente.invoke(
    {
        "messages":
            resultado["messages"]
            + [
                {
                    "role": "user",
                    "content":
                        "O produto do cliente foi localizado."
                }
            ],

        "etapa_atual":
            resultado.get("etapa_atual"),

        "cliente_localizado":
            resultado.get("cliente_localizado"),
    }
)

print("\n--- APÓS LOCALIZAR PRODUTO ---")

mostrar_estado(resultado)


# ==========================================================
# TESTE 3 - COLETAR DADOS DO ORÇAMENTO
# ==========================================================

resultado = agente.invoke(
    {
        "messages":
            resultado["messages"]
            + [
                {
                    "role": "user",
                    "content":
                        (
                            "Quero um orçamento para "
                            "50 pessoas na região "
                            "de Santo André."
                        )
                }
            ],

        "etapa_atual":
            resultado.get("etapa_atual"),

        "cliente_localizado":
            resultado.get("cliente_localizado"),

        "produto_localizado":
            resultado.get("produto_localizado"),
    }
)

print("\n--- APÓS COLETAR INFORMAÇÕES ---")

mostrar_estado(resultado)


# ==========================================================
# HISTÓRICO DAS MENSAGENS
# ==========================================================

print("\n===================================")
print("       HISTÓRICO DO AGENTE")
print("===================================")

for mensagem in resultado["messages"]:

    print(
        f"\nTipo: "
        f"{type(mensagem).__name__}"
    )

    print(
        "Conteúdo:",
        mensagem.content
    )

    if (
        hasattr(mensagem, "tool_calls")
        and mensagem.tool_calls
    ):

        print(
            "Tools chamadas:",
            mensagem.tool_calls
        )