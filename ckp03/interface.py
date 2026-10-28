"""
Interface Gradio do agente: chat com histórico, botão de limpar sessão e
painéis com o raciocínio do agente (loop ReAct), as ferramentas usadas e as
fontes citadas. `share=True` gera a URL pública exigida pelo CKP03.
"""

from __future__ import annotations

import html
import uuid

import gradio as gr

from ckp03.agente import AgenteChargeGrid

EXEMPLOS = [
    "Olá! O que você consegue fazer?",
    "Tenho 200 kW contratados e a loja consome 170 kW. Quanto sobra para 4 carros?",
    "Quanto custa uma recarga de 30 kWh num carregador DC de 60 kW às 19h de uma terça, com 25 minutos parado depois?",
    "A ANEEL permite cobrar preço livre pela recarga no meu estacionamento?",
    "Qual mensagem OCPP limita a potência de um carregador?",
    "Medi 230 kW com 200 kW contratados e tarifa de demanda de R$ 45,50/kW. Vou pagar ultrapassagem?",
]

NOMES_FERRAMENTAS = {
    "buscar_base_chargegrid": "Busca na base de documentos (RAG)",
    "calcular_folga_demanda": "Calculadora de folga de demanda",
    "calcular_custo_recarga": "Calculadora de custo de recarga",
    "calcular_cobranca_ultrapassagem": "Calculadora de cobrança de ultrapassagem",
}


def construir_interface(agente: AgenteChargeGrid) -> gr.Blocks:
    def responder(pergunta, historico, sessao):
        pergunta = (pergunta or "").strip()
        if not pergunta:
            return "", historico, gr.update(), gr.update(), gr.update()
        turno = agente.responder(pergunta, sessao)
        historico = (historico or []) + [
            {"role": "user", "content": pergunta},
            {"role": "assistant", "content": turno.resposta},
        ]
        if turno.bloqueado:
            ferramentas = "**Guardrail de input** bloqueou a pergunta. O agente não foi chamado."
        elif turno.ferramentas_usadas:
            ferramentas = "\n".join(f"- {NOMES_FERRAMENTAS.get(f, f)}" for f in turno.ferramentas_usadas)
        else:
            ferramentas = "Nenhuma: o agente decidiu responder direto."
        ferramentas += f"\n\nTempo de resposta: {turno.latencia_s:.1f} s"
        if turno.compressoes:
            ferramentas += f" · {turno.compressoes} observação(ões) comprimida(s)"
        fontes = "\n".join(f"- {html.escape(f)}" for f in turno.fontes) or "Nenhum documento consultado neste turno."
        return "", historico, ferramentas, fontes, f"```text\n{turno.raciocinio()}\n```"

    def limpar(sessao):
        agente.limpar(sessao)
        return [], str(uuid.uuid4()), "", "", ""

    with gr.Blocks(title="ChargeGrid · Agente do operador") as demo:
        sessao = gr.State(lambda: str(uuid.uuid4()))
        gr.Markdown(
            "# ChargeGrid · Agente do operador de eletropostos\n"
            "Pergunte sobre demanda contratada, custo de recarga, cobrança de ultrapassagem, regras da ANEEL, "
            "OCPP, MODBUS ou carregadores GoodWe. O agente decide sozinho quando consultar os documentos e "
            "quando calcular, e mostra as fontes. *GoodWe × FIAP · EV Challenge 2026 · CKP03*"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversa", height=480)
                with gr.Row():
                    entrada = gr.Textbox(placeholder="Escreva sua pergunta e aperte Enter", show_label=False, scale=5)
                    enviar = gr.Button("Enviar", variant="primary", scale=1)
                gr.Examples(EXEMPLOS, inputs=entrada, label="Exemplos para começar")
                botao_limpar = gr.Button("Limpar conversa (nova sessão)")
            with gr.Column(scale=2):
                gr.Markdown("### Ferramentas usadas")
                ferramentas = gr.Markdown()
                gr.Markdown("### Documentos consultados na base (RAG)")
                fontes = gr.Markdown()
                with gr.Accordion("Raciocínio do agente (loop ReAct)", open=False):
                    raciocinio = gr.Markdown()
        saidas = [entrada, chat, ferramentas, fontes, raciocinio]
        entrada.submit(responder, [entrada, chat, sessao], saidas)
        enviar.click(responder, [entrada, chat, sessao], saidas)
        botao_limpar.click(limpar, [sessao], [chat, sessao, ferramentas, fontes, raciocinio])
    return demo
