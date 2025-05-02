from langchain_ollama import ChatOllama
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from typing import List, Dict
import os, time 

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

class ConversationContextManager:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.conversation_context = []
    
    def add_interaction(self, user_query: str, ai_response: str):
        """Adiciona uma interação ao contexto da conversa"""
        self.conversation_context.append({
            "user_query": user_query,
            "ai_response": ai_response
        })
        
        # Limita o histórico ao tamanho máximo
        if len(self.conversation_context) > self.max_history:
            self.conversation_context.pop(0)
    
    def get_context_summary(self) -> str:
        """Gera um sumário do contexto da conversa"""
        if not self.conversation_context:
            return "Sem histórico de conversa."
        
        context_summary = "Histórico recente da conversa:\n"
        for interaction in self.conversation_context:
            context_summary += f"- Pergunta: {interaction['user_query']}\n"
        
        return context_summary
    
    def enhance_query(self, current_query: str) -> str:
        """Tenta adicionar contexto à pergunta atual"""
        if not self.conversation_context:
            return current_query
        
        # Palavras que indicam continuidade
        context_indicators = [
            'isso', 'aquilo', 'ele', 'ela', 'este', 'esta', 
            'esse', 'essa', 'aí', 'ali', 'então'
        ]
        
        # Verifica se a pergunta atual usa palavras de contexto
        query_lower = current_query.lower()
        uses_context_indicator = any(
            indicator in query_lower for indicator in context_indicators
        )
        
        if uses_context_indicator:
            # Usa a última interação para contextualizar
            last_interaction = self.conversation_context[-1]
            enhanced_query = (
                f"Considerando a pergunta anterior '{last_interaction['user_query']}' "
                f"e o contexto: {last_interaction['ai_response']}, "
                f"responda: {current_query}"
            )
            return enhanced_query
        
        return current_query

class EnhancedQASystem:
    def __init__(self, data_path: str = "./data/", faiss_index_path: str = "faiss_index"):
        self.setup_model()
        self.conversation_manager = ConversationContextManager()

        print("🔄 Inicializando sistema de QA...")
        self.faiss_index_path = faiss_index_path
        print("🔄 Baixando modelo...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )
        print("✅ Modelo carregado com sucesso!")
        
        # Tenta carregar o índice FAISS salvo, senão recria do zero
        if os.path.exists(f"{self.faiss_index_path}.pkl"):
            self.load_faiss_index()
        else:
            self.setup_document_processing(data_path)
            self.save_faiss_index()

    def setup_document_processing(self, data_path: str):
        print("📂 Carregando documentos...")

        # Verifica se o índice FAISS já está salvo
        if self.load_faiss_index():
            return  # Já carregou o índice, então não precisa refazer tudo

        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        self.text_chunks = text_splitter.split_documents(documents)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )

        self.vector_store = FAISS.from_documents(
            self.text_chunks,
            self.embeddings
        )

        self.save_faiss_index()  # Salva o índice FAISS para reutilizar depois


    def save_faiss_index(self, index_path="faiss_index"):
        """Salva o índice FAISS para evitar reconstrução demorada."""
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        self.vector_store.save_local(index_path)
        print("💾 Índice FAISS salvo em disco!")
        

    def load_faiss_index(self, index_path="faiss_index"):
        """Carrega o índice FAISS se já existir em disco."""
        if os.path.exists(index_path):
            print("🔄 Carregando índice FAISS salvo...")
            self.vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # 🚨 Adicionando essa flag
            )
            print("✅ Índice FAISS carregado!")
            return True
        return False


    def setup_model(self):
        self.llm = ChatOllama(
            model="qwen2.5:14b", 
            temperature=0.8, 
            top_p=0.9,
            stream=True,
            base_url="http://localhost:11434"
        )
    
    def find_relevant_documents(self, query: str, top_k: int = 5):
        """Encontra documentos relevantes para a consulta"""
        return self.vector_store.similarity_search(query, k=top_k)
    
    def process_query(self, query: str) -> Dict:
        try:
            # Melhora a consulta com contexto
            enhanced_query = self.conversation_manager.enhance_query(query)
            
            # Busca documentos relevantes
            relevant_docs = self.find_relevant_documents(enhanced_query)
            
            # Prepara contexto da conversa
            conversation_context = self.conversation_manager.get_context_summary()
            
            # Prompt personalizado com contexto
            prompt = f"""Sistema de Resposta Contextual:

    Contexto da Conversa:
    {conversation_context}

    Documentos Relevantes:
    {chr(10).join(doc.page_content for doc in relevant_docs)}

    Pergunta: {enhanced_query}

    Responda de forma precisa e concisa, baseando-se nos documentos disponíveis."""
            
            # Gera resposta usando Ollama
            response = self.llm.invoke(prompt)

            # Extrai o conteúdo da resposta (AIMessage)
            response_content = response.content  # Isso retorna uma string

            # Adiciona interação ao contexto da conversa
            self.conversation_manager.add_interaction(query, response_content)
            
            return {
                "answer": response_content,  # Retorna o conteúdo da resposta
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
qa_system = EnhancedQASystem()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        start_time = time.time()  # Captura o tempo inicial
        user_input = request.form["user_input"]
        
        # Processa a pergunta
        result = qa_system.process_query(user_input)

        end_time = time.time()  # Captura o tempo final
        execution_time = end_time - start_time  # Calcula o tempo de execução
        
        return jsonify({
            "answer": result.get("answer", "Não foi possível gerar resposta"),
            "source_documents": result.get("source_documents", []),
            "execution_time": execution_time,  # Adiciona o tempo de execução à resposta
            "success": result.get("success", False)
        })
    
    except Exception as e:
        return jsonify({
            "answer": f"Erro interno: {str(e)}",
            "source_documents": [],
            "execution_time": 0,  # Tempo de execução em caso de erro
            "success": False
        })

if __name__ == "__main__":
    app.run(debug=True)
