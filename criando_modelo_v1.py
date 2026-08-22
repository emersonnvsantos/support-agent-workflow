import os
from dotenv import load_dotenv
from typing import Literal, Callable
from typing_extensions import NotRequired

from langchain_openai import ChatOpenAI
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)

from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.utils.uuid import uuid7


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


# ----------------------------------------------------------
# SIMULAÇÃO 1
# Cliente localizado
# ----------------------------------------------------------

cliente = "localizado"


@tool
def localizar_cadastro_cliente(
    status: Literal["localizado", "nao_localizado"],
    runtime: ToolRuntime[None, EstadoSuporte],
) -> Command:
    """Registra se o cadastro do cliente foi localizado no sistema."""

    # Sobrescreve o status conforme o resultado
    # simulado da consulta.
    if cliente == "localizado":

        status = "localizado"
        proxima_etapa = "consulta_produtos"

    else:

        status = "nao_localizado"
        proxima_etapa = "identificacao_cliente"

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


# ----------------------------------------------------------
# SIMULAÇÃO 2
# Produto NÃO localizado
# ----------------------------------------------------------

produto = "nao_localizado"


@tool
def localizar_produto_cliente(
    status: Literal["localizado", "nao_localizado"],
    runtime: ToolRuntime[None, EstadoSuporte],
) -> Command:
    """Registra se o produto do cliente foi localizado no sistema."""

    # Sobrescreve o status conforme o resultado
    # simulado da consulta.
    if produto == "localizado":

        status = "localizado"
        proxima_etapa = "coleta_informacoes"

    else:

        status = "nao_localizado"
        proxima_etapa = "consulta_produtos"

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

Se o cliente não for localizado:
- Informe que o cadastro não foi localizado.
- Não avance para a consulta dos produtos.
- Não solicite informações do orçamento.

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

Se o produto não for localizado:
- Informe ao cliente que o produto não foi localizado.
- Não avance para coleta das informações.

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
# MIDDLEWARE
# ==========================================================

@wrap_model_call
def aplicando_configuracao_etapa_atual(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configura o comportamento do agente com base na etapa atual."""

    # 1. Obtém a etapa atual.
    # Se não existir, inicia em identificacao_cliente.
    etapa_atual = request.state.get(
        "etapa_atual",
        "identificacao_cliente"
    )

    # 2. Consulta a configuração da etapa atual.
    estagio_configuracao = CONFIG_ETAPAS[etapa_atual]

    # 3. Valida os campos obrigatórios.
    for key in estagio_configuracao["requires"]:

        if request.state.get(key) is None:

            raise ValueError(
                f"{key} deve ser definido "
                f"antes de chegar a {etapa_atual}"
            )

    # 4. Monta o prompt usando os dados do State.
    system_prompt = estagio_configuracao["prompt"].format(
        **request.state
    )

    # 5. Define prompt e tools disponíveis
    # somente para a etapa atual.
    request = request.override(
        system_prompt=system_prompt,
        tools=estagio_configuracao["tools"],
    )

    # 6. Continua a execução.
    return handler(request)


# ==========================================================
# AGENTE
# ==========================================================

todas_tools = [
    localizar_cadastro_cliente,
    localizar_produto_cliente,
    registrar_informacoes_orcamento,
]


agente = create_agent(
    model,
    tools=todas_tools,
    state_schema=EstadoSuporte,
    middleware=[
        aplicando_configuracao_etapa_atual
    ],
    checkpointer=InMemorySaver(),
)


# ==========================================================
# CRIAR THREAD DA CONVERSA
# ==========================================================

thread_id = str(uuid7())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# ==========================================================
# TURNO 1
# IDENTIFICAÇÃO DO CLIENTE
# ==========================================================

print("\n========================================")
print("TURNO 1 - IDENTIFICAÇÃO DO CLIENTE")
print("========================================")


result = agente.invoke(
    {
        "messages": [
            HumanMessage(
                content="Olá, gostaria de fazer um orçamento!"
            )
        ]
    },
    config
)


print("\n--- ESTADO APÓS TURNO 1 ---")

print(
    "Cliente localizado:",
    result.get("cliente_localizado")
)

print(
    "Produto localizado:",
    result.get("produto_localizado")
)

print(
    "Etapa atual:",
    result.get("etapa_atual")
)


# ==========================================================
# TURNO 2
# CONSULTA DO PRODUTO
# ==========================================================

print("\n========================================")
print("TURNO 2 - CONSULTA DO PRODUTO")
print("========================================")


result = agente.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Quero fazer um orçamento "
                    "do meu produto cadastrado."
                )
            )
        ]
    },
    config
)


print("\n--- ESTADO APÓS TURNO 2 ---")

print(
    "Cliente localizado:",
    result.get("cliente_localizado")
)

print(
    "Produto localizado:",
    result.get("produto_localizado")
)

print(
    "Etapa atual:",
    result.get("etapa_atual")
)


# ==========================================================
# HISTÓRICO COMPLETO
# ==========================================================

print("\n========================================")
print("HISTÓRICO DO AGENTE")
print("========================================")


for mensagem in result["messages"]:

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