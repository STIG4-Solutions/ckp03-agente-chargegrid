"""
Configuração do agente: credencial, modelo e caminhos.

A chave da Ollama Cloud nunca aparece no código: no Colab ela vem do Secret
`OLLAMA_API_KEY` (ícone de chave na barra lateral); fora do Colab, da variável
de ambiente ou do arquivo `.env`, que está no `.gitignore`.
"""

from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Modelo exigido pelo enunciado do CKP03.
MODELO_CHAT = "gemma4:cloud"
URL_OLLAMA_CLOUD = "https://ollama.com"

# Embeddings: o mesmo nomic-embed-text do RAG do Challenge (Sprint 4), servido
# pelo Ollama local. A Ollama Cloud não expõe o endpoint de embeddings.
MODELO_EMBEDDINGS = "nomic-embed-text"
URL_OLLAMA_LOCAL = os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434")

# Base de conhecimento do Challenge: 16 PDFs com manifesto de procedência.
REPO_BASE = "https://github.com/STIG4-Solutions/chatbot_sprint4_challenge2026.git"
DIRETORIO_BASE = Path(os.getenv("CKP03_BASE_DIR", RAIZ / "base_challenge" / "data" / "knowledge_base"))
DIRETORIO_CHROMA = Path(os.getenv("CKP03_CHROMA_DIR", RAIZ / ".chroma_ckp03"))
COLECAO = "chargegrid_ckp03"


def obter_chave_api() -> str:
    """Lê a OLLAMA_API_KEY do Colab Secrets, do ambiente ou do `.env`."""
    try:
        from google.colab import userdata  # type: ignore

        chave = userdata.get("OLLAMA_API_KEY")
        if chave:
            return chave.strip()
    except Exception:
        pass
    chave = os.getenv("OLLAMA_API_KEY", "").strip()
    if not chave and (RAIZ / ".env").exists():
        for linha in (RAIZ / ".env").read_text().splitlines():
            if linha.startswith("OLLAMA_API_KEY="):
                chave = linha.split("=", 1)[1].strip()
    if not chave:
        raise EnvironmentError(
            "OLLAMA_API_KEY não encontrada. No Colab: Secrets (ícone de chave) > "
            "adicionar OLLAMA_API_KEY e liberar acesso ao notebook. Fora do Colab: "
            "variável de ambiente ou arquivo .env."
        )
    return chave


def criar_llm(temperature: float = 0.0, num_predict: int = 1500):
    """ChatOllama apontado para a Ollama Cloud com o gemma4:cloud."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=MODELO_CHAT,
        base_url=URL_OLLAMA_CLOUD,
        temperature=temperature,
        num_predict=num_predict,
        client_kwargs={"headers": {"Authorization": f"Bearer {obter_chave_api()}"}, "timeout": 180},
    )
