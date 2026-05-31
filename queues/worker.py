from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

_openai_client = None
_vector_db = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _get_vector_db():
    global _vector_db
    if _vector_db is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        _vector_db = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="learning_rag",
            embedding=embedding_model,
        )
    return _vector_db


def process_query(query: str):
    print(f"Searching chunks for query: {query}")
    search_results = _get_vector_db().similarity_search(
    query,
    k=5
)

    context = "\n\n\n".join(
        [
            f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
            for result in search_results
        ]
    )

    system_prompt = f"""
    You are a helpfull AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

    If the answer is not present in the context,
respond with:
"I could not find this information in the uploaded documents."

    Context:
    {context}
"""

    response = _get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message.content
