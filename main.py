import pinecone
import os
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from decouple import config

class ChainHandler:
    def __init__(
        self,
        openai_api_key,
        pinecone_api_key,
        pinecone_env,
        pinecone_index,
    ):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.pinecone_index = pinecone_index

    def create_chain(self, temperature, vector_k, model, namespace):
        """
        Langchain flow for creating a ConversationalRetrievalChain that fetches data from a vectordatabase and passes it as context to the AI model.
        Flow Diagram: https://github.com/mayooear/gpt4-pdf-chatbot-langchain/blob/main/visual-guide/gpt-langchain-pdf.png
        We are in a pre-ingestion state, we are not ingesting in real time we are doing so before queries and fetching from the existing namespaces.
        """

        # Init OpenAI Embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=self.openai_api_key
        )

        # Init Pinecone Environment
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

        """
        This is the main piece that needs to be reworked and integrated with the langchain library accordingly.

        Desired rework example:
        vectorstore = Pinecone.from_existing_index(
            index_name=self.pinecone_index,
            embedding=embeddings,
            text_key="text",
            namespace=['test-123', 'test-456', 'test-789'], # This is the part that needs to be edited, we need a list input of namespaces.
        )
        """
        vectorstore = Pinecone.from_existing_index(
            index_name=self.pinecone_index,
            embedding=embeddings,
            text_key="text",
            namespace=namespace,
        )

        # Init AI model
        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=self.openai_api_key,
            verbose=True,
        )

        # Prompts for the retriever and the question generator
        QA_PROMPT = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.\nIf you don't know the answer, just say you don't know. DO NOT try to make up an answer.\nIf the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.\nUse as much detail when as possible when responding.\n\n{context}\n\nQuestion: {question}"""
        CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"""

        # The vectorstore integrates with the retriever to fetch the top k most similar documents to the query
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": vector_k},
            qa_template=QA_PROMPT,
            question_generator_template=CONDENSE_PROMPT,
        )

        # The chain integrates the retriever and the llm to create a conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            retriever=retriever,
            llm=llm,
            return_source_documents=True,
            verbose=True,
        )

        return chain


if __name__ == "__main__":
    chain_handler = ChainHandler(
        openai_api_key=config('OPENAI_API_KEY'),
        pinecone_api_key=config('PINECONE_API_KEY'),
        pinecone_env=config('PINECONE_ENV'),
        pinecone_index=config('PINECONE_INDEX'),
    )
    chain = chain_handler.create_chain(
        temperature=0.9,
        vector_k=5,
        model="gpt-3.5-turbo",
        namespace="test-123",
    )

    # Query the chain
    query = "What is the meaning of life?"
    query_dict = {"question": query, "chat_history": []}
    
    answer = chain(query_dict)
    print(answer)
