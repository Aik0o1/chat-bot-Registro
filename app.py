from langchain_community.llms import Ollama
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from typing import List, Dict

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

class EnhancedQASystem:
    def __init__(self, data_path: str = "./data/"):
        self.setup_document_processing(data_path)
        self.setup_model()
        
    def setup_document_processing(self, data_path: str):
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
        
    def setup_model(self):
        # Prompt mais simples e direto
        self.prompt = PromptTemplate(
            template="""Use o contexto fornecido para responder a pergunta. 
            Se a resposta não estiver no contexto, diga "Não encontrei informações nos documentos".

            Contexto: {context}
            
            Pergunta: {question}
            
            Resposta:""",
            input_variables=["context", "question"]
        )
        
        self.llm = Ollama(
            model="llama3.2", 
            temperature=0.2, 
            top_p=0.9
        )
        
        # Cria chain de QA com base no prompt personalizado
        self.qa_chain = load_qa_chain(
            llm=self.llm, 
            chain_type="stuff", 
            prompt=self.prompt
        )
        
        # Configuração do retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )
    
    def process_query(self, question: str) -> Dict:
        try:
            # Recupera documentos relevantes
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            # Processa a pergunta com os documentos relevantes
            result = self.qa_chain(
                {"input_documents": relevant_docs, "question": question},
                return_only_outputs=True
            )
            
            return {
                "answer": result.get('output_text', "Não foi possível gerar resposta."),
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
        user_input = request.form["user_input"]
        
        # Processa a pergunta
        result = qa_system.process_query(user_input)
        
        return jsonify({
            "answer": result.get("answer", "Não foi possível gerar resposta"),
            "source_documents": result.get("source_documents", []),
            "success": result.get("success", False)
        })
    
    except Exception as e:
        return jsonify({
            "answer": f"Erro interno: {str(e)}",
            "source_documents": [],
            "success": False
        })

if __name__ == "__main__":
    app.run(debug=True)