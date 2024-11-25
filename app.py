from langchain_community.llms import Ollama
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from typing import List, Dict

app = Flask(__name__)

# Configuração melhorada do text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Aumentado para capturar mais contexto
    chunk_overlap=200,  # Aumentado para melhor continuidade
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
    is_separator_regex=False,
)

# Template de prompt melhorado
custom_prompt = PromptTemplate(
    template="""Você é um assistente especialista que tem acesso a documentos específicos.
    Use as seguintes partes de contexto para responder à pergunta do usuário.
    Se você não souber a resposta, diga simplesmente que não sabe - não tente inventar uma resposta.
    Se a pergunta não estiver relacionada ao contexto, tente dar uma resposta geral útil.
    Use o histórico da conversa quando relevante.
    
    Contexto: {context}
    Histórico: {chat_history}
    Pergunta: {question}
    
    Resposta útil:""",
    input_variables=["context", "chat_history", "question"]
)

class EnhancedQASystem:
    def __init__(self, data_path: str = "./data/"):
        self.setup_document_processing(data_path)
        self.setup_model()
        
    def setup_document_processing(self, data_path: str):
        # Carregamento e processamento de documentos
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        self.text_chunks = text_splitter.split_documents(documents)
        
        # Configuração de embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Criação do vector store com configurações otimizadas
        self.vector_store = FAISS.from_documents(
            self.text_chunks,
            self.embeddings
        )
        
    def setup_model(self):
        # Configuração do modelo com parâmetros otimizados
        self.llm = Ollama(
            model="llama3.2",
            temperature=0.7,  # Ajuste conforme necessário
            top_p=0.9,
        )
        
        # Memória com configuração melhorada
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
        
        # Configuração da chain com retrieval melhorado
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance para diversidade
                search_kwargs={
                    "k": 4,  # Aumentado para mais contexto
                    "fetch_k": 8,  # Busca inicial maior
                    "lambda_mult": 0.7  # Balanceamento entre relevância e diversidade
                }
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={'prompt': custom_prompt},
            return_source_documents=True,
            verbose=True
        )
    
    def get_relevant_chunks(self, query: str) -> List[str]:
        """Retorna os chunks mais relevantes para debug"""
        docs = self.vector_store.similarity_search(query, k=3)
        return [doc.page_content for doc in docs]
    
    def answer_question(self, question: str) -> Dict:
        """Processa a pergunta e retorna resposta com metadados"""
        try:
            result = self.chain({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": [
                    {"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
                    for doc in result.get("source_documents", [])
                ],
                "success": True
            }
        except Exception as e:
            return {
                "answer": "Desculpe, ocorreu um erro ao processar sua pergunta.",
                "error": str(e),
                "success": False
            }

# Instância global do sistema
qa_system = EnhancedQASystem()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form["user_input"]
        print(f"Received input: {user_input}")  # Log de entrada
        
        result = qa_system.answer_question(user_input)
        print(f"QA Result: {result}")  # Log do resultado
        
        serialized_result = {
            "answer": result.get("answer", "Não foi possível gerar uma resposta."),
            "source_documents": result.get("source_documents", []),
            "success": result.get("success", False)
        }
        
        return jsonify(serialized_result)
    except Exception as e:
        print(f"Error in chat route: {e}")  # Log de erro
        return jsonify({
            "answer": f"Erro interno: {str(e)}",
            "source_documents": [],
            "success": False
        })

# No método answer_question da classe, ajuste para garantir serialização:
def answer_question(self, question: str) -> Dict:
    try:
        result = self.chain({"question": question})
        
        return {
            "answer": result.get("answer", "Não foi possível gerar uma resposta."),
            "source_documents": [
                {"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
                for doc in result.get("source_documents", [])
            ],
            "success": True
        }
    except Exception as e:
        return {
            "answer": f"Erro ao processar: {str(e)}",
            "source_documents": [],
            "success": False
        }

if __name__ == "__main__":
    app.run(debug=True)