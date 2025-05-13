from langchain_ollama import ChatOllama
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid
import os, time 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory

app = Flask(__name__)
app.secret_key = "MY_SECRET_KEY"
 
# Configuring Session
app.config['PERMANENT_SESSION_LIFETIME'] = 60  # Session Lifetime
app.config['SESSION_TYPE'] = "filesystem"  # Session Storage Type
 
# Path to Storing Session
app.config['SESSION_FILE_DIR'] = "session_data"

Session(app)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL)

class ConversationContextManager:
    def __init__(self):
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        
        # Initialize the language model
        self.llm = ChatOllama(
            model="qwen2.5:latest",
            temperature=0.8,
            top_p=0.9,
            stream=True
        )
        
        # Create the conversational chain
        self.chain = self.prompt | self.llm
        
        # Create a runnable with message history
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            get_redis_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def process_message(self, input_message: str, session_id: str):
        """Process a message with conversation history"""
        response = self.chain_with_history.invoke(
            {"input": input_message},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

class EnhancedQASystem:
    def __init__(self, data_path, faiss_index_path):
        self.setup_model()
        self.conversation_manager = ConversationContextManager()

        print("🔄 Inicializando sistema de QA...")
        self.faiss_index_path = faiss_index_path
        print("🔄 Baixando modelo...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Modelo carregado com sucesso!")
        
        # Tenta carregar o índice FAISS salvo, senão recria do zero
        if os.path.exists(f"{self.faiss_index_path}.pkl"):
            self.load_faiss_index(self.faiss_index_path)
        else:
            self.setup_document_processing(data_path)
            self.save_faiss_index(self.faiss_index_path)

    def setup_document_processing(self, data_path: str):
        print("📂 Carregando documentos...")

        # Verifica se o índice FAISS já está salvo
        if self.load_faiss_index(self.faiss_index_path):
            return  # Já carregou o índice, então não precisa refazer tudo

        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        self.text_chunks = text_splitter.split_documents(documents)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        self.vector_store = FAISS.from_documents(
            self.text_chunks,
            self.embeddings
        )

        self.save_faiss_index(self.faiss_index_path)  # Salva o índice FAISS para reutilizar depois

    def save_faiss_index(self, index_path):
        """Salva o índice FAISS para evitar reconstrução demorada."""
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        self.vector_store.save_local(index_path)
        print("💾 Índice FAISS salvo em disco!")
        
    def load_faiss_index(self, index_path):
        """Carrega o índice FAISS se já existir em disco."""
        if os.path.exists(index_path):
            print("🔄 Carregando índice FAISS salvo...")
            self.vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Índice FAISS carregado!")
            return True
        return False

    def setup_model(self):
        self.llm = ChatOllama(
            model="llama3.2:1b", 
            temperature=0.8, 
            top_p=0.9,
            stream=True
        )
    
    def find_relevant_documents(self, query: str, top_k: int = 5):
        """Encontra documentos relevantes para a consulta"""
        return self.vector_store.similarity_search(query, k=top_k)
    
    def process_query(self, session_id, query: str) -> Dict:
        try:
            # Busca documentos relevantes
            relevant_docs = self.find_relevant_documents(query)
            
            # Prepara contexto com documentos relevantes
            context = f"""Documentos Relevantes:
            {chr(10).join(doc.page_content for doc in relevant_docs)}

            Pergunta: {query}"""
            
            # Gera resposta usando a cadeia com histórico
            response = self.conversation_manager.process_message(query, session_id)
            print(session_id)
            
            return {
                "answer": response,
                "source_documents": [
                    {"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
                    for doc in relevant_docs
                ],
                "success": True
            }
        
        except Exception as e:
            return {
                "answer": f"Erro ao processar: {str(e)}",
                "source_documents": [],
                "success": False
            }

# Instância global
qa_system_registro = EnhancedQASystem("./data_registro", "faiss_index_registro")
qa_system_juridico = EnhancedQASystem("./data_juridico", "faiss_index_juridico")

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4().hex)   
    return render_template("index.html")

@app.route("/chat_registro", methods=["POST"])
def chat_registro():
    try:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4().hex)   
        start_time = time.time()
        user_input = request.form["user_input"]
        session_id = session.get('session_id')
        print("aaaaa", session_id)
        print(user_input)
        
        result = qa_system_registro.process_query(session_id, user_input)

        end_time = time.time()
        execution_time = end_time - start_time
        
        return jsonify({
            "answer": result.get("answer", "Não foi possível gerar resposta"),
            "source_documents": result.get("source_documents", []),
            "execution_time": execution_time,
            "success": result.get("success", False)
        })
    
    except Exception as e:
        return jsonify({
            "answer": f"Erro interno: {str(e)}",
            "source_documents": [],
            "execution_time": 0,
            "success": False
        })

@app.route("/chat_juridico", methods=["POST"])
def chat_juridico():
    try:
        start_time = time.time()
        user_input = request.form["user_input"]
        session_id = session.get('session_id')
        print(session_id)
        result = qa_system_juridico.process_query(session_id, user_input)

        end_time = time.time()
        execution_time = end_time - start_time
        
        return jsonify({
            "answer": result.get("answer", "Não foi possível gerar resposta"),
            "source_documents": result.get("source_documents", []),
            "execution_time": execution_time,
            "success": result.get("success", False)
        })
    
    except Exception as e:
        return jsonify({
            "answer": f"Erro interno: {str(e)}",
            "source_documents": [],
            "execution_time": 0,
            "success": False
        })

if __name__ == "__main__":
    app.run(debug=True)