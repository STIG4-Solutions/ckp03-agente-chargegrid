"""
Compressão de contexto do scratchpad (diferencial, técnica da Aula 10).

A cada passo do loop ReAct o AgentExecutor reenvia ao modelo todas as
Observations anteriores. Uma busca no RAG devolve 5 a 10 trechos de até 1.000
caracteres, e duas ou três buscas seguidas fazem o prompt crescer rápido.

Política aplicada antes de cada novo passo (`trim_intermediate_steps`):
- janela deslizante: os 2 passos mais recentes ficam intactos;
- resumo: Observations mais antigas, quando o total passa do limite, são
  substituídas por um resumo curto feito pelo próprio LLM, preservando números
  e a referência [documento, página]. O resumo é guardado em cache, para não
  resumir a mesma Observation duas vezes.
Se o LLM falhar, cai num corte extrativo (início da Observation e as fontes).
"""

from __future__ import annotations

import hashlib
import re

LIMITE_CARACTERES = 4000
JANELA = 2

PROMPT_RESUMO = (
    "Resuma o resultado de ferramenta abaixo em no máximo 3 frases, em português, mantendo números, "
    "valores em R$ e as referências entre colchetes [documento, p. N]. Não acrescente nada.\n\n{texto}"
)


class CompressorDeScratchpad:
    def __init__(self, llm, limite: int = LIMITE_CARACTERES, janela: int = JANELA) -> None:
        self.llm = llm
        self.limite = limite
        self.janela = janela
        self._cache: dict[str, str] = {}
        self.compressoes = 0

    def _resumir(self, texto: str) -> str:
        chave = hashlib.sha1(texto.encode()).hexdigest()
        if chave not in self._cache:
            try:
                resumo = self.llm.invoke(PROMPT_RESUMO.format(texto=texto[:6000])).content.strip()
            except Exception:
                resumo = ""
            if not resumo:
                fontes = " ".join(dict.fromkeys(re.findall(r"\[[^\]]{5,120}\]", texto)))
                resumo = texto[:400] + " … " + fontes
            self._cache[chave] = "[resumo de observação anterior] " + resumo
            self.compressoes += 1
        return self._cache[chave]

    def __call__(self, passos: list) -> list:
        total = sum(len(str(obs)) for _, obs in passos)
        if total <= self.limite or len(passos) <= self.janela:
            return passos
        antigos, recentes = passos[: -self.janela], passos[-self.janela :]
        return [(acao, self._resumir(str(obs))) for acao, obs in antigos] + recentes
