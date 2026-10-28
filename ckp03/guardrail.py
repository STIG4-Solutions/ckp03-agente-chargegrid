"""
Guardrail de input (diferencial): verificação antes de a pergunta chegar ao agente.

Reaproveita a camada determinística do Challenge (Sprint 3): tentativa de
subverter as instruções (jailbreak e injeção) e assunto claramente fora do
domínio são recusados com mensagem educada, sem chamar o modelo. Perguntas do
domínio e conversas simples (cumprimento, "o que você faz?") passam: o agente
decide se precisa de ferramenta.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto.lower()).strip()


SUBVERSAO = [re.compile(p) for p in (
    r"\b(ignore|ignora|esqueca|desconsidere)\b.{0,40}\b(instrucoes|regras|diretrizes|prompt)\b",
    r"\b(a partir de agora voce e|voce agora e|finja (ser|que))\b",
    r"\bmodo\b.{0,15}\b(desenvolvedor|developer|irrestrito|sem filtro|dan)\b",
    r"\b(revele|mostre|imprima|repita)\b.{0,40}\b(system prompt|prompt de sistema|suas instrucoes)\b",
    r"^\s*(system|sistema)\s*:",
)]

FORA_DO_DOMINIO = [re.compile(p) for p in (
    r"\b(receita|bolo|culinaria|cozinhar)\b",
    r"\b(futebol|campeonato|brasileirao|jogo do|placar)\b",
    r"\b(horoscopo|signo|piada|poema|letra de musica)\b",
    r"\b(presidente|eleicao|partido politico|candidato)\b",
    r"\b(namoro|relacionamento amoroso|dieta|emagrecer)\b",
    r"\b(dever de casa|redacao do enem|trabalho escolar)\b",
)]

MENSAGEM_SUBVERSAO = (
    "Não consigo atender a esse pedido: minhas instruções não podem ser alteradas nem reveladas. "
    "Posso ajudar com a operação dos eletropostos: demanda, tarifas, regras da ANEEL, OCPP, MODBUS "
    "ou equipamentos GoodWe."
)
MENSAGEM_FORA = (
    "Esse assunto está fora do que eu atendo. Sou o assistente do ChargeGrid para operadores de "
    "eletropostos comerciais: posso calcular folga de demanda, custo de recarga e cobrança de "
    "ultrapassagem, e consultar regras e especificações com a fonte citada."
)


@dataclass(frozen=True)
class Veredito:
    permitido: bool
    motivo: str = ""
    mensagem: str = ""


def verificar_entrada(pergunta: str) -> Veredito:
    texto = normalizar(pergunta)
    if not texto:
        return Veredito(False, "vazia", "Digite uma pergunta sobre a operação dos eletropostos.")
    if len(texto) > 2000:
        return Veredito(False, "longa_demais", "A pergunta é longa demais. Resuma em até 2.000 caracteres.")
    for padrao in SUBVERSAO:
        if padrao.search(texto):
            return Veredito(False, "subversao_de_instrucoes", MENSAGEM_SUBVERSAO)
    for padrao in FORA_DO_DOMINIO:
        if padrao.search(texto):
            return Veredito(False, "fora_do_dominio", MENSAGEM_FORA)
    return Veredito(True)
