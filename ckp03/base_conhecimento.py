"""
RAG do grupo (base do CKP02): ChromaDB persistente com nomic-embed-text.

O grupo não entregou o CKP02 individual de RAG. O RAG encapsulado aqui é o do
EV Challenge (Sprint 4), feito pelo mesmo grupo, no mesmo semestre: 16 PDFs
com procedência declarada, PyMuPDFLoader, RecursiveCharacterTextSplitter,
nomic-embed-text com prefixos de tarefa e ChromaDB persistente. A avaliação
RAGAS completa desse pipeline está no repositório do Challenge; este módulo
o reconstrói do zero para que o notebook rode sozinho no Colab.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ckp03.config import (
    COLECAO,
    DIRETORIO_BASE,
    DIRETORIO_CHROMA,
    MODELO_EMBEDDINGS,
    REPO_BASE,
    URL_OLLAMA_LOCAL,
)

TAMANHO_CHUNK = 1000
SOBREPOSICAO = 150

# Frases de documento dirigidas a "sistemas de IA" são removidas na ingestão:
# um PDF de terceiros não pode dar ordens ao agente (injeção via documento).
PADRAO_INJECAO = re.compile(
    r"(ignore (todas )?as instru|a partir de agora voc[eê] [eé]|revele .{0,30}prompt|"
    r"aten[cç][aã]o,? sistema|ignore (all )?previous instructions)",
    re.IGNORECASE,
)
MARCADOR_REMOCAO = "[trecho removido: instrução embutida em documento]"


class NomicEmbeddings(Embeddings):
    """nomic-embed-text com os prefixos de tarefa do modelo (documento × consulta)."""

    def __init__(self) -> None:
        self.base = OllamaEmbeddings(model=MODELO_EMBEDDINGS, base_url=URL_OLLAMA_LOCAL)

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return self.base.embed_documents(["search_document: " + t for t in textos])

    def embed_query(self, texto: str) -> list[float]:
        return self.base.embed_query("search_query: " + texto)


def baixar_base() -> Path:
    """Clona (uma vez) o repositório do Challenge para obter os PDFs da base."""
    if not DIRETORIO_BASE.exists():
        destino = DIRETORIO_BASE.parent.parent
        subprocess.run(["git", "clone", "--depth", "1", REPO_BASE, str(destino)], check=True)
    return DIRETORIO_BASE


def _limpar(texto: str) -> str:
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    texto = re.sub(r"(?:[ \t ]*\.){4,}[ \t ]*", " … ", texto)
    return re.sub(r"[ \t]+", " ", texto).strip()


def _sanitizar(texto: str) -> tuple[str, bool]:
    frases = re.split(r"(?<=[.!?:])\s+|\n+", texto)
    limpas, removeu = [], False
    for frase in frases:
        if PADRAO_INJECAO.search(frase):
            removeu = True
            if not limpas or limpas[-1] != MARCADOR_REMOCAO:
                limpas.append(MARCADOR_REMOCAO)
        elif frase.strip():
            limpas.append(frase.strip())
    return (" ".join(limpas), True) if removeu else (texto, False)


def carregar_fragmentos(diretorio: Path = DIRETORIO_BASE) -> list[Document]:
    """PDFs do manifesto → páginas (PyMuPDFLoader) → fragmentos com metadados de citação."""
    manifesto = {d["arquivo"]: d for d in json.loads((diretorio / "manifesto.json").read_text())["documentos"]}
    divisor = RecursiveCharacterTextSplitter(chunk_size=TAMANHO_CHUNK, chunk_overlap=SOBREPOSICAO)
    fragmentos: list[Document] = []
    for arquivo, info in sorted(manifesto.items()):
        for pagina in PyMuPDFLoader(str(diretorio / arquivo)).load():
            texto = _limpar(pagina.page_content)
            if not texto:
                continue
            numero = int(pagina.metadata.get("page", 0)) + 1
            for i, pedaco in enumerate(divisor.split_text(texto)):
                if len(pedaco) < 60 or pedaco.count("…") >= 4:
                    continue  # rodapé, número de página ou sumário
                pedaco, suspeito = _sanitizar(pedaco)
                artigos = ", ".join(dict.fromkeys(re.findall(r"\bArt\.\s*(\d{1,3})", pedaco)))
                fragmentos.append(Document(
                    page_content=pedaco,
                    metadata={
                        "titulo": info["titulo"], "arquivo": arquivo, "tipo_fonte": info["tipo"],
                        "pagina": numero, "artigos": artigos, "suspeito_injecao": suspeito,
                        # Rótulo pronto para a citação "[documento, p. N, Art. X]".
                        "artigos_rotulo": f", Art. {artigos}" if artigos else "",
                        "id": f"{arquivo}:{numero}:{i}",
                    },
                ))
    return fragmentos


def abrir_indice(reconstruir: bool = False) -> Chroma:
    """Abre o ChromaDB persistente, indexando a base na primeira execução."""
    loja = Chroma(
        collection_name=COLECAO,
        embedding_function=NomicEmbeddings(),
        persist_directory=str(DIRETORIO_CHROMA),
        collection_metadata={"hnsw:space": "cosine"},
    )
    if reconstruir:
        loja.reset_collection()
    if loja._collection.count() == 0:
        fragmentos = carregar_fragmentos(baixar_base())
        for inicio in range(0, len(fragmentos), 64):
            lote = fragmentos[inicio : inicio + 64]
            loja.add_documents(lote, ids=[f.metadata["id"] for f in lote])
    return loja


def todos_os_fragmentos(loja: Chroma) -> list[Document]:
    dados = loja.get(include=["documents", "metadatas"])
    return [Document(page_content=t, metadata=m or {}) for t, m in zip(dados["documents"], dados["metadatas"])]
