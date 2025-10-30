from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document


from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# RETRIEVAL_K = 3

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# retriever = vectorstore.as_retriever()

# Stage 1 - Bi-Encoder
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# Stage 2 - Cross-Encoder
model_name = "BAAI/bge-reranker-base"
model_kwargs = {'device': 'cpu'}  
model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
compressor = CrossEncoderReranker(model=model, top_n=3)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

llm = ChatOpenAI(temperature=0, model_name=MODEL)

def fetch_context(question: str) -> list:
    """
    Retrieve relevant context documents for a question.
    (This now uses the two-stage re-ranking retriever)
    """
    # return retriever.invoke(question, k=RETRIEVAL_K)
    return retriever.invoke(question)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs