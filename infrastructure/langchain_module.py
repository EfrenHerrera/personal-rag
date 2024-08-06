# from langchain_ollama.llms import OllamaLLM
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from os import environ
import argparse

from common.utils.constants import CHROMA_SETTINGS
from common.chromadb.db_settings import Chroma
from common.utils.assistant_prompt import assistant_prompt


model = environ.get('MODEL')
embeddings_model_name = environ.get('EMBEDDINGS_MODEL_NAME')
target_source_chunks = int(environ.get('TARGET_SOURCE_CHUNKS', 5))


def parse_args():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, ' 
        'using the power of LLMs.'
    )
    
    parser.add_argument(
        "--hide-source", 
        "-S", 
        action='store_true',
        help='Use this flag to disable printing of source documents used for answers.'
    )

    parser.add_argument(
        "--mute-stream", 
        "-M",
        action='store_true',
        help='Use this flag to disable the streaming StdOut callback for LLMs.'
    )

    return parser.parse_args()


def response(query: str) -> str:
    args = parse_args()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    db = Chroma(client=CHROMA_SETTINGS, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={'k': target_source_chunks})
    
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    llm = Ollama(model=model, callbacks=callbacks, temperature=0, base_url=environ.get('OLLAMA_URL'))
    # llm = OllamaLLM(model=model, callbacks=callbacks, temperature=0,
    
    prompt = assistant_prompt()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(query)