"""
As 4 ferramentas do agente.

1. `buscar_base_chargegrid` (RAG do grupo via `create_retriever_tool`)
2. `calcular_folga_demanda`
3. `calcular_custo_recarga`
4. `calcular_cobranca_ultrapassagem`

O agente é ReAct (Thought → Action → Action Input → Observation), e no ReAct
clássico cada ferramenta recebe UM texto como entrada. As calculadoras aceitam
esse texto em JSON ou no formato "chave=valor, chave=valor", e a description de
cada uma diz exatamente quais chaves usar. A description é o que o LLM lê para
decidir QUANDO usar a ferramenta, por isso todas têm "use quando" e "não use".

As regras numéricas vêm dos documentos da base, citados na saída de cada
ferramenta: Manual de Operação ChargeGrid (limiares de demanda), Política
Tarifária ChargeGrid (preços do piloto) e REN ANEEL nº 1.000/2021, Art. 301
(cobrança de ultrapassagem).
"""

from __future__ import annotations

import json
import re

from langchain.retrievers import EnsembleRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from ckp03.base_conhecimento import abrir_indice, todos_os_fragmentos


# ---------------------------------------------------------------------------
# Entrada das calculadoras
# ---------------------------------------------------------------------------

def _ler_parametros(entrada: str) -> dict:
    """Aceita JSON ou "chave=valor, chave=valor" (também "chave: valor")."""
    texto = (entrada or "").strip().strip("`").strip()
    if texto.startswith("{"):
        try:
            return {k.strip().lower(): v for k, v in json.loads(texto).items()}
        except json.JSONDecodeError:
            pass
    # Separa os pares só na vírgula que antecede outra "chave=", para que a
    # vírgula decimal ("vrdult=45,50") não corte o valor.
    pares = re.split(r"[;\n]|,\s*(?=[a-zA-Z_]+\s*[=:])", texto)
    resultado = {}
    for par in pares:
        m = re.match(r"\s*([a-zA-Z_]+)\s*[=:]\s*(.+?)\s*$", par)
        if m:
            resultado[m.group(1).lower()] = m.group(2).strip().strip("\"'")
    return resultado


def _numero(valor, nome: str) -> float:
    if isinstance(valor, (int, float)):
        return float(valor)
    limpo = re.sub(r"[^\d,.\-]", "", str(valor))
    if "," in limpo and "." in limpo:
        limpo = limpo.replace(".", "").replace(",", ".")
    else:
        limpo = limpo.replace(",", ".")
    try:
        return float(limpo)
    except ValueError as erro:
        raise ValueError(f"valor inválido para '{nome}': {valor}") from erro


def _br(valor: float, casas: int = 1) -> str:
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _reais(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# Ferramenta 1: RAG
# ---------------------------------------------------------------------------

DESCRICAO_RAG = (
    "Busca trechos nos documentos oficiais e do projeto ChargeGrid: Resolução ANEEL nº 1.000/2021 "
    "(regras de recarga de veículos elétricos, demanda contratada, ultrapassagem), especificação OCPP 1.6 "
    "(mensagens como SetChargingProfile, MeterValues, StartTransaction), especificação MODBUS, datasheets e "
    "manual dos carregadores GoodWe HCA e HCA G2, norma Celesc de estações de recarga, reajuste tarifário da "
    "Enel SP, e os documentos do ChargeGrid (manual de operação, política tarifária, FAQ). "
    "Use quando a pergunta pedir uma regra, norma, especificação de equipamento, definição de protocolo, "
    "preço ou procedimento que precisa de fonte. A entrada é a pergunta reescrita como consulta de busca. "
    "Não use para fazer contas (use as calculadoras) nem para cumprimentos ou perguntas sobre o próprio assistente. "
    "Cada trecho volta com [documento, página], que deve ser citado na resposta."
)

FORMATO_TRECHO = PromptTemplate.from_template(
    "[{titulo}, p. {pagina}{artigos_rotulo}]\n{page_content}"
)


def criar_ferramenta_rag(k: int = 5):
    """`create_retriever_tool` sobre busca híbrida: vetorial (ChromaDB) + BM25.

    A busca híbrida foi a decisão de recuperação com maior ganho medido no
    Challenge (MRR da evidência 0,335 → 0,466): o BM25 resgata termos exatos
    como número de artigo e nome de mensagem OCPP, que o embedding aproxima mal.
    """
    loja = abrir_indice()
    vetorial = loja.as_retriever(search_kwargs={"k": k})
    lexical = BM25Retriever.from_documents(todos_os_fragmentos(loja))
    lexical.k = k
    hibrido = EnsembleRetriever(retrievers=[vetorial, lexical], weights=[0.5, 0.5])
    return create_retriever_tool(
        hibrido,
        name="buscar_base_chargegrid",
        description=DESCRICAO_RAG,
        document_prompt=FORMATO_TRECHO,
        document_separator="\n\n---\n\n",
    )


# ---------------------------------------------------------------------------
# Ferramenta 2: folga de demanda
# ---------------------------------------------------------------------------

@tool
def calcular_folga_demanda(entrada: str) -> str:
    """Calcula quanta potência sobra para os carregadores e em que faixa de alerta o estabelecimento está.
    Use quando o operador informar a demanda contratada e o consumo da loja (em kW) e quiser saber a folga,
    quantos carros cabem ou se há risco de ultrapassagem. Não use para dúvidas conceituais sem números.
    Entrada: demanda_contratada_kw=<número>, consumo_loja_kw=<número>, sessoes_ativas=<número, opcional, padrão 1>.
    Regra (Manual de Operação ChargeGrid, seção 3): reserva de 5% da demanda contratada; a partir de 85% de consumo
    a potência dos carregadores é reduzida; a 95% vai à corrente mínima; a 100% novas sessões são suspensas."""
    p = _ler_parametros(entrada)
    contratada = _numero(p.get("demanda_contratada_kw"), "demanda_contratada_kw")
    consumo = _numero(p.get("consumo_loja_kw"), "consumo_loja_kw")
    sessoes = max(1, int(_numero(p.get("sessoes_ativas", 1), "sessoes_ativas")))
    if contratada <= 0 or consumo < 0:
        return "Entrada inválida: a demanda contratada deve ser positiva e o consumo não pode ser negativo."
    folga_bruta = max(0.0, contratada - consumo)
    margem = 0.05 * contratada
    distribuivel = max(0.0, folga_bruta - margem)
    ocupacao = consumo / contratada
    if ocupacao >= 1.0:
        faixa = "LIMITE (≥ 100%): novas sessões suspensas"
    elif ocupacao >= 0.95:
        faixa = "CRÍTICA (≥ 95%): todos os conectores na corrente mínima"
    elif ocupacao >= 0.85:
        faixa = "ATENÇÃO (≥ 85%): redução escalonada da potência dos carregadores"
    else:
        faixa = "NORMAL (< 85%): operação sem restrição"
    return (
        f"Ocupação da demanda: {_br(ocupacao * 100)}% ({_br(consumo)} de {_br(contratada)} kW). Faixa {faixa}.\n"
        f"Folga bruta: {_br(folga_bruta)} kW. Margem de proteção (5%): {_br(margem)} kW. "
        f"Potência distribuível aos carregadores: {_br(distribuivel)} kW, "
        f"ou {_br(distribuivel / sessoes)} kW por sessão com {sessoes} sessão(ões) ativa(s).\n"
        "Fonte das regras: ChargeGrid · Manual de Operação, seção 3 (Controle de demanda)."
    )


# ---------------------------------------------------------------------------
# Ferramenta 3: custo da recarga
# ---------------------------------------------------------------------------

TARIFAS = [(7.4, "AC lenta", 1.90), (22.0, "AC semirrápida", 2.20), (float("inf"), "DC rápida (DC Fast)", 3.30)]


@tool
def calcular_custo_recarga(entrada: str) -> str:
    """Calcula o valor cobrado de uma sessão de recarga no piloto ChargeGrid.
    Use quando o operador quiser saber quanto custa ou quanto foi cobrada uma recarga a partir de energia (kWh),
    potência do carregador, horário e minutos parado após carregar. Não use para preços de outros
    estabelecimentos nem para tarifas da distribuidora.
    Entrada: energia_kwh=<número>, potencia_kw=<potência do carregador>, horario=<HH:MM>,
    dia_util=<sim|nao, padrão sim>, minutos_ociosos=<número, padrão 0>.
    Regra (Política Tarifária ChargeGrid): R$ 1,90/kWh até 7,4 kW; R$ 2,20/kWh até 22 kW; R$ 3,30/kWh acima de 22 kW;
    multiplicador 1,30 das 18h às 21h em dias úteis; ociosidade de R$ 1,00/min após 10 min de tolerância, máximo R$ 60,00."""
    p = _ler_parametros(entrada)
    energia = _numero(p.get("energia_kwh"), "energia_kwh")
    potencia = _numero(p.get("potencia_kw"), "potencia_kw")
    horario = str(p.get("horario", "12:00"))
    dia_util = str(p.get("dia_util", "sim")).lower() not in ("nao", "não", "false", "0", "n")
    ociosos = _numero(p.get("minutos_ociosos", 0), "minutos_ociosos")
    if energia < 0 or potencia <= 0:
        return "Entrada inválida: energia não pode ser negativa e a potência do carregador deve ser positiva."
    hora = int(re.match(r"\s*(\d{1,2})", horario).group(1)) if re.match(r"\s*\d", horario) else 12
    tipo, base = next((t, v) for limite, t, v in TARIFAS if potencia <= limite)
    ponta = dia_util and 18 <= hora < 21
    multiplicador = 1.30 if ponta else 1.00
    tarifa = round(base * multiplicador, 2)
    valor_energia = energia * tarifa
    ociosidade = min(60.0, max(0.0, ociosos - 10) * 1.00)
    total = round(valor_energia + ociosidade, 2)
    return (
        f"Tipo de recarga: {tipo} ({_br(potencia)} kW). Tarifa base {_reais(base)}/kWh × multiplicador "
        f"{_br(multiplicador, 2)} ({'ponta, 18h às 21h em dia útil' if ponta else 'fora de ponta'}) = {_reais(tarifa)}/kWh.\n"
        f"Energia: {_br(energia, 2)} kWh × {_reais(tarifa)} = {_reais(valor_energia)}. "
        f"Ociosidade: {_br(ociosos, 0)} min ({_reais(ociosidade)}, tolerância de 10 min, teto de R$ 60,00).\n"
        f"Total da sessão: {_reais(total)}.\n"
        "Fonte das regras: ChargeGrid · Política Tarifária do Piloto, seções 1 a 3."
    )


# ---------------------------------------------------------------------------
# Ferramenta 4: cobrança de ultrapassagem
# ---------------------------------------------------------------------------

@tool
def calcular_cobranca_ultrapassagem(entrada: str) -> str:
    """Calcula a cobrança de ultrapassagem de demanda que a distribuidora soma à fatura.
    Use quando o operador informar a demanda medida, a demanda contratada e o valor da tarifa de demanda
    (VRDULT, em R$/kW) e quiser saber se haverá cobrança e de quanto. Não use para explicar a regra sem números
    (para isso, busque na base).
    Entrada: demanda_medida_kw=<número>, demanda_contratada_kw=<número>, vrdult=<R$ por kW>.
    Regra (REN ANEEL nº 1.000/2021, Art. 301): para consumidor, só há cobrança se a demanda medida exceder em mais
    de 5% a contratada; a cobrança é (demanda medida − demanda contratada) × 2 × VRDULT."""
    p = _ler_parametros(entrada)
    medida = _numero(p.get("demanda_medida_kw"), "demanda_medida_kw")
    contratada = _numero(p.get("demanda_contratada_kw"), "demanda_contratada_kw")
    vrdult = _numero(p.get("vrdult", p.get("vrdult_reais_kw")), "vrdult")
    if contratada <= 0 or medida < 0 or vrdult < 0:
        return "Entrada inválida: demanda contratada positiva, demanda medida e VRDULT não negativos."
    limite = contratada * 1.05
    if medida <= limite:
        return (
            f"Sem cobrança de ultrapassagem: a demanda medida ({_br(medida)} kW) não excede a tolerância de 5% "
            f"sobre a contratada ({_br(limite)} kW).\nFonte: REN ANEEL nº 1.000/2021, Art. 301, inciso II."
        )
    excedente = medida - contratada
    valor = excedente * 2 * vrdult
    return (
        f"Há cobrança: a demanda medida ({_br(medida)} kW) passou da tolerância de 5% ({_br(limite)} kW).\n"
        f"Cobrança = (demanda medida − contratada) × 2 × VRDULT = {_br(excedente)} kW × 2 × {_reais(vrdult)}/kW "
        f"= {_reais(valor)}, somada ao faturamento regular.\n"
        "Fonte: REN ANEEL nº 1.000/2021, Art. 301, § 1º."
    )


def criar_ferramentas() -> list:
    return [criar_ferramenta_rag(), calcular_folga_demanda, calcular_custo_recarga, calcular_cobranca_ultrapassagem]
