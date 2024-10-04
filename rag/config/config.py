import os
from typing import Optional

from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from langchain_core.retrievers import BaseRetriever

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
import re
from typing import List, Union

import requests
from bs4 import BeautifulSoup


def __html_document_loader__(url: Union[str, bytes]) -> str:
    """
    Loads the HTML content of a document from a given URL and return it's content.

    Args:
        url: The URL of the document.

    Returns:
        The content of the document.

    Raises:
        Exception: If there is an error while making the HTTP request.

    """
    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # Create a Beautiful Soup object to parse html
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.extract()

        # Get the plain text from the HTML document
        text = soup.get_text()

        # Remove excess whitespace and newlines
        text = re.sub("\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""


def create_embeddings(
    embedding_model: NVIDIAEmbeddings,
    embedding_path: str,
) -> BaseRetriever:
    print(f"Storing embeddings to {embedding_path}")

    # List of web pages containing NVIDIA Triton technical documentation
    urls = [
        "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html",
        "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html",
        "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html",
        "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html",
        "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html",
    ]

    documents = []
    for url in urls:
        document = __html_document_loader__(url)
        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
    retriever = __index_docs__(
        embedding_model=embedding_model,
        url=url,
        splitter=text_splitter,
        documents=texts,
        dest_embed_dir=embedding_path,
    )
    print("Generated embedding successfully")
    return retriever


def __index_docs__(
    embedding_model: NVIDIAEmbeddings,
    url: str,
    splitter: TextSplitter,
    documents: List[str],
    dest_embed_dir,
) -> BaseRetriever:
    """
    Split the document into chunks and create embeddings for the document

    Args:
        url: Source url for the document.
        splitter: Splitter used to split the document
        documents: list of documents whose embeddings needs to be created
        dest_embed_dir: destination directory for embeddings

    Returns:
        None
    """

    for document in documents:
        texts = splitter.split_text(document.page_content)

        # metadata to attach to document
        metadatas = [{"source": url}]

        # create embeddings and add to vector store
        if os.path.exists(dest_embed_dir):
            docsearch = FAISS.load_local(
                folder_path=dest_embed_dir,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True,
            )
            docsearch.add_texts(texts, metadatas=metadatas)
            docsearch.save_local(folder_path=dest_embed_dir)
        else:
            docsearch = FAISS.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas,
            )
            docsearch.save_local(folder_path=dest_embed_dir)

    return docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


def init_vectorstore_retriever(config: RailsConfig):
    config.knowledge_base.embedding_search_provider
    emd_config = config.knowledge_base.embedding_search_provider

    embedding_model = NVIDIAEmbeddings(
        model=emd_config.parameters.get("embedding_model"),
        base_url=emd_config.parameters.get("base_url"),
        truncate=emd_config.parameters.get("truncate"),
    )
    embedding_path = os.path.join(
        config.config_path,
        os.path.pardir,
        "data",
        "nv_embedding",
    )
    retriever = create_embeddings(
        embedding_model=embedding_model,
        embedding_path=embedding_path,
    )
    return retriever


@action(is_system_action=True)
async def retrieve_relevant_chunks(
    retriever: BaseRetriever,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
):
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    user_message = context.get("last_user_message")
    # only call the retriever as the message will be generated by `generate_bot_message` action
    documents = await retriever.ainvoke(input=user_message)
    citing_text = "\n".join([doc.page_content for doc in documents])
    source_ref = "\n".join([doc.metadata["source"] for doc in documents])

    context_updates = {
        "relevant_chunks": f"""
            Question: {user_message}
            Citing : {citing_text},
            Source : {source_ref}
    """
    }

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


def init(llm_rails: LLMRails):
    if isinstance(llm_rails, LLMRails):
        # check that `llm_rails` is an instance of `LLMRails` as multiple libraries uses the same
        # config.py and `init` methods, e.g. FastAPI
        config = llm_rails.config

        # Initialize the various models
        vectorstore_retriever = init_vectorstore_retriever(config)

        # Register the custom `retrieve_relevant_chunks` for custom retrieval
        llm_rails.register_action(
            action=retrieve_relevant_chunks,
            name="retrieve_relevant_chunks",
        )
        llm_rails.register_action_param(
            name="retriever",
            value=vectorstore_retriever,
        )
