from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document as LangChainDocument
from langchain.text_splitter import (
    CharacterTextSplitter
)
import openai
from app.api.config import (
    logger
)
from langchain import OpenAI

def get_embedding(text, model="bge-large-zh"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(base_url="http://36.212.226.3:43101",
                                   input = [text],
                                   model=model).data[0].embedding

def embed_documents(texts):
    embeddings = []
    for i in texts:
        embedding = get_embedding(i)
        embeddings.append(embedding)
    return embeddings

def test_llm(question):
    llm = OpenAI(
        temperature="1.0",
        model_name="Qwen1.5-72B-Chat",
        max_tokens="1000",
    )
    try:
        result = llm(prompt=question)
    except openai.error.InvalidRequestError as e:
        logger.error(f"🚨 LLM error: {e}")
    logger.debug(f"💬 LLM result: {str(result)}")
    return result

if __name__ == '__main__':
    # test_llm("我是一个")

    print(openai.__version__)
    LLM_CHUNK_SIZE = 512
    LLM_CHUNK_OVERLAP = 20
    document_data = "今天天气怎么样"

    documents = [LangChainDocument(page_content=document_data)]
    logger.debug(documents)

    embed_func = OpenAIEmbeddings(openai_api_base="http://36.212.226.3:43101/v1", model="bge-large-zh")
    # Returns an array of Documents
    doc_splitter = CharacterTextSplitter(
        chunk_size=LLM_CHUNK_SIZE, chunk_overlap=LLM_CHUNK_OVERLAP
    )
    split_documents = doc_splitter.split_documents(documents)
    # Lets convert them into an array of strings for OpenAI
    arr_documents = [doc.page_content for doc in split_documents]
    embeddings = embed_documents(arr_documents)
    logger.info(embeddings)
