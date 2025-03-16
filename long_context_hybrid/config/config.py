import httpx
import pandas as pd

from typing import Optional

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult

from nemoguardrails.server.api import register_datastore
from nemoguardrails.server.datastore.memory_store import MemoryStore

@action(is_system_action=True)
async def retrieve_relevant_chunks(
    context: Optional[dict] = None
):
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    user_message = context.get("last_user_message")

    vectordb_url="http://vectordb.runai-vast.svc.cluster.local/retrieve-chunks/"

    query_string = {'query':user_message, 'limit':100}

    async with httpx.AsyncClient() as client:
        response = await client.post(vectordb_url,json=query_string)

    try:
        results = pd.DataFrame(response.json()['results'])
    except:
        pass

    context_updates = {
        "relevant_chunks": f"""
            Question: {user_message},
            Citing : {results=}
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
        # config = llm_rails.config

        register_datastore(MemoryStore())

        # Register the custom `retrieve_relevant_chunks` for custom retrieval
        llm_rails.register_action(
            action=retrieve_relevant_chunks,
            name="retrieve_relevant_chunks",
        )
