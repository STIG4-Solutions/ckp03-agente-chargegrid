"""
Agente ReAct (Aulas 09 a 11): AgentExecutor + 4 ferramentas + memória por sessão.

Fluxo de um turno:
    pergunta → guardrail de input → AgentExecutor (ReAct: Thought, Action,
    Action Input, Observation ... Final Answer) → resposta + passos + fontes
O histórico da sessão entra no prompt como texto (janela das últimas trocas),
e o scratchpad é comprimido quando cresce (`ckp03/compressao.py`).
"""

from __future__ import annotations

import re
import textwrap
import time
from dataclasses import dataclass, field

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

from ckp03.compressao import CompressorDeScratchpad
from ckp03.config import criar_llm
from ckp03.ferramentas import criar_ferramentas
from ckp03.guardrail import verificar_entrada

TURNOS_NA_MEMORIA = 6

PROMPT_REACT = PromptTemplate.from_template("""Você é o assistente do ChargeGrid Intelligence (GoodWe × FIAP) para operadores comerciais de eletropostos em shoppings, supermercados e estacionamentos. Responda sempre em português do Brasil, de forma direta e profissional.

Ferramentas disponíveis:
{tools}

Como decidir:
- Regra, norma, especificação de equipamento, protocolo (OCPP, MODBUS), preço do piloto ou procedimento → use buscar_base_chargegrid e cite as fontes no formato [documento, p. N].
- Pergunta com números para calcular folga de demanda, custo de uma recarga ou cobrança de ultrapassagem → use a calculadora correspondente. Não faça essas contas de cabeça.
- Cumprimento, agradecimento, pergunta sobre o que você faz ou dúvida conceitual simples que não exige fonte nem cálculo → responda direto, sem ferramenta.
- Se a base não tiver a informação, diga que ela não está na base. Nunca invente especificação, preço ou norma.
- Nunca oriente intervenção física em instalação elétrica: encaminhe a eletricista habilitado.

Use EXATAMENTE este formato (as palavras-chave em inglês são obrigatórias; o raciocínio pode ser em português):

Question: a pergunta do operador
Thought: o que preciso fazer e por quê
Action: o nome da ferramenta, um entre [{tool_names}]
Action Input: a entrada da ferramenta
Observation: o resultado da ferramenta
... (Thought/Action/Action Input/Observation podem se repetir)
Thought: já sei a resposta
Final Answer: a resposta final ao operador, com as fontes citadas quando houver

Quando não precisar de ferramenta, vá direto para:
Thought: não preciso de ferramenta porque ...
Final Answer: ...

Conversa anterior desta sessão:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}""")

class ParserReActTolerante(ReActSingleInputOutputParser):
    """Parser ReAct que aceita resposta final sem o rótulo "Final Answer:".

    O gemma4 às vezes escreve a resposta direto, depois de uma Observation,
    sem o prefixo. O parser padrão levanta OutputParserException e o loop
    gasta uma volta inteira só para o modelo repetir a resposta no formato.
    Se o texto não contém "Action:", ele é a resposta final.
    """

    def parse(self, text: str):
        try:
            return super().parse(text)
        except OutputParserException:
            if "Action:" not in text:
                resposta = text.split("Thought:")[-1].strip() if "Thought:" in text else text.strip()
                return AgentFinish({"output": resposta}, text)
            raise


MENSAGEM_ERRO_FORMATO = (
    "Formato inválido. Responda com 'Thought:' seguido de 'Action:' e 'Action Input:', "
    "ou com 'Thought:' seguido de 'Final Answer:'."
)


@dataclass
class Turno:
    pergunta: str
    resposta: str
    passos: list = field(default_factory=list)
    ferramentas_usadas: list = field(default_factory=list)
    fontes: list = field(default_factory=list)
    bloqueado: bool = False
    motivo_bloqueio: str = ""
    latencia_s: float = 0.0
    compressoes: int = 0

    def raciocinio(self) -> str:
        """Log legível do loop ReAct, para o notebook e para a interface."""
        if self.bloqueado:
            return f"Guardrail de input: pergunta bloqueada ({self.motivo_bloqueio}). O agente não foi chamado."
        if not self.passos:
            return "O agente respondeu sem usar ferramenta."
        def quebrar(rotulo: str, texto: str) -> str:
            corpo = " ".join(str(texto).split())
            return textwrap.fill(f"{rotulo}: {corpo}", width=78, subsequent_indent="    ")

        linhas = []
        for i, (acao, observacao) in enumerate(self.passos, start=1):
            pensamento = acao.log.split("Action:")[0].replace("Thought:", "").strip()
            observacao = str(observacao)
            observacao = observacao[:500] + ("…" if len(observacao) > 500 else "")
            linhas.append("\n".join([
                f"Passo {i}",
                quebrar("Thought", pensamento),
                quebrar("Action", acao.tool),
                quebrar("Action Input", acao.tool_input),
                quebrar("Observation", observacao),
            ]))
        return "\n\n".join(linhas)


class AgenteChargeGrid:
    def __init__(self, verbose: bool = False) -> None:
        self.llm = criar_llm()
        self.ferramentas = criar_ferramentas()
        self.compressor = CompressorDeScratchpad(criar_llm(num_predict=400))
        agente = create_react_agent(
            self.llm, self.ferramentas, PROMPT_REACT, output_parser=ParserReActTolerante()
        )
        self.executor = AgentExecutor(
            agent=agente,
            tools=self.ferramentas,
            verbose=verbose,
            max_iterations=6,
            handle_parsing_errors=MENSAGEM_ERRO_FORMATO,
            return_intermediate_steps=True,
            trim_intermediate_steps=self.compressor,
        )
        self.sessoes: dict[str, InMemoryChatMessageHistory] = {}

    def historico(self, sessao: str) -> InMemoryChatMessageHistory:
        return self.sessoes.setdefault(sessao, InMemoryChatMessageHistory())

    def limpar(self, sessao: str) -> None:
        self.sessoes.pop(sessao, None)

    def _historico_como_texto(self, sessao: str) -> str:
        mensagens = self.historico(sessao).messages[-2 * TURNOS_NA_MEMORIA :]
        if not mensagens:
            return "(início da conversa)"
        return "\n".join(
            f"{'Operador' if isinstance(m, HumanMessage) else 'Assistente'}: {m.content}" for m in mensagens
        )

    def responder(self, pergunta: str, sessao: str = "padrao") -> Turno:
        inicio = time.perf_counter()
        veredito = verificar_entrada(pergunta)
        if not veredito.permitido:
            turno = Turno(pergunta, veredito.mensagem, bloqueado=True, motivo_bloqueio=veredito.motivo)
        else:
            compressoes_antes = self.compressor.compressoes
            saida = self.executor.invoke({"input": pergunta, "chat_history": self._historico_como_texto(sessao)})
            passos = [(a, o) for a, o in saida.get("intermediate_steps", []) if a.tool != "_Exception"]
            fontes = []
            for acao, observacao in passos:
                if acao.tool == "buscar_base_chargegrid":
                    fontes += re.findall(r"^\[([^\]]+)\]", str(observacao), flags=re.MULTILINE)
            turno = Turno(
                pergunta=pergunta,
                resposta=saida["output"],
                passos=passos,
                ferramentas_usadas=[a.tool for a, _ in passos],
                fontes=list(dict.fromkeys(fontes)),
                compressoes=self.compressor.compressoes - compressoes_antes,
            )
        turno.latencia_s = round(time.perf_counter() - inicio, 2)
        self.historico(sessao).add_messages([HumanMessage(content=pergunta), AIMessage(content=turno.resposta)])
        return turno
