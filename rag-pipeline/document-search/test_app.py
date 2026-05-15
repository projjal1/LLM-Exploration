import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.dirname(__file__))

import app


def test_text_splitter_respects_chunk_size_and_overlap():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    text = "sentence one. " * 500
    splits = splitter.split_documents([Document(page_content=text)])

    assert len(splits) > 1
    assert all(len(s.page_content) <= 1000 for s in splits)
    assert all("start_index" in s.metadata for s in splits)


@patch.object(app, "Chroma")
@patch.object(app, "PyPDFLoader")
def test_process_index_docs_loads_pdf_and_indexes_splits(mock_loader_cls, mock_chroma_cls):
    mock_loader_cls.return_value.load.return_value = [
        Document(page_content="financial filing content. " * 300)
    ]
    mock_vector_store = MagicMock()
    mock_chroma_cls.return_value = mock_vector_store

    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.0] * 8

    app.process_index_docs(embeddings)

    mock_loader_cls.assert_called_once_with("nke-10k-2023.pdf")
    mock_chroma_cls.assert_called_once()
    assert mock_chroma_cls.call_args.kwargs["collection_name"] == "financial-document-rag"

    mock_vector_store.add_documents.assert_called_once()
    indexed_docs = mock_vector_store.add_documents.call_args.kwargs["documents"]
    assert len(indexed_docs) > 1
    assert all(isinstance(d, Document) for d in indexed_docs)


@patch.object(app, "process_index_docs")
@patch.object(app, "RetrievalQA")
@patch.object(app, "ChatOllama")
@patch.object(app, "Chroma")
@patch.object(app, "OllamaEmbeddings")
def test_main_skips_indexing_when_chroma_dir_exists(
    mock_embeddings_cls,
    mock_chroma_cls,
    mock_chat_cls,
    mock_qa_cls,
    mock_process,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chroma_langchain_db").mkdir()

    mock_qa_cls.from_chain_type.return_value.run.return_value = "answer"

    app.main()

    mock_process.assert_not_called()
    mock_qa_cls.from_chain_type.return_value.run.assert_called()
