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

    query_string = {'query':user_message}

    async with httpx.AsyncClient() as client:
        response = await client.post(vectordb_url,json=query_string)

    try:
        results = pd.DataFrame(response.json()['results'])
    except:
        pass

    citing_text = ""
    source_ref = ""
        
    if not results.empty:
        rerank_url="http://rerankqa-mistral-4b.runai-genai.svc.cluster.local/v1/ranking"
        start_string = '{"model": "nvidia/nv-rerankqa-mistral-4b-v3","query": {"text":"' + user_message + '"},"passages": [{"text": "'
        middle_string = '"},{"text": "'.join(results['cleaned_text'])
        middle_string = middle_string.encode("ascii","ignore")
        middle_string = middle_string.decode()
        middle_string = middle_string.replace('"','')
        end_string='"}],"truncate": "END"}'
        rerank_string = start_string + middle_string + end_string
        headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(rerank_url, headers=headers, data=rerank_string)
        try:
            ranking = pd.DataFrame(response.json()['rankings'])
            if ranking['logit'].loc[0] > 10:
                citing_text = results['cleaned_text'].loc[ranking['index'].loc[0]]
                source_ref = results['filename'].loc[ranking['index'].loc[0]]   
        except:
            pass

    context_updates = {
        "relevant_chunks": f"""
            Question: {user_message},
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
        # config = llm_rails.config

        register_datastore(MemoryStore())

        # Register the custom `retrieve_relevant_chunks` for custom retrieval
        llm_rails.register_action(
            action=retrieve_relevant_chunks,
            name="retrieve_relevant_chunks",
        )
