import pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import OpenAIEmbeddings
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

    def create_chain(self, temperature, vector_k, model, namespaces):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=self.openai_api_key
        )

        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

        vectorstore = self.create_multi_namespace_vectorstore(
            index_name=self.pinecone_index,
            embedding=embeddings,
            text_key="text",
            namespaces=namespaces,
        )

        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=self.openai_api_key,
            verbose=True,
        )
        QA_PROMPT = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.\nIf you don't know the answer, just say you don't know. DO NOT try to make up an answer.\nIf the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.\nUse as much detail when as possible when responding.\n\n{context}\n\nQuestion: {question}"""
        CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"""


        retriever = vectorstore.as_retriever(
            search_kwargs={"k": vector_k},
            qa_template=QA_PROMPT,
            question_generator_template=CONDENSE_PROMPT,
        )

        chain = ConversationalRetrievalChain.from_llm(
            retriever=retriever,
            llm=llm,
            return_source_documents=True,
            verbose=True,
        )

        return chain

    def create_multi_namespace_vectorstore(self, index_name, embedding, text_key, namespaces):
        vector_store = LangchainPinecone.from_existing_index(index_name, embedding, text_key)

        def multi_namespace_search(query, **kwargs):
            all_results = []
            for namespace in namespaces:
                kwargs['namespace'] = namespace
                results = vector_store.search(query, **kwargs)
                all_results.extend(results)
            return all_results

        vector_store.search = multi_namespace_search
        return vector_store


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
        namespaces=["namespace-a", "namespace-b", "namespace-c"],
    )

    query = "What is the meaning of life?"
    query_dict = {"question": query, "chat_history": []}
    
    answer = chain(query_dict)
    print(answer)