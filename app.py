from langchain_ollama import ChatOllama
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid
import os
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cache import RedisCache
from langchain.globals import set_llm_cache
from langchain_core.messages import HumanMessage, AIMessage
import redis
import traceback
from functools import wraps
from typing import Callable
import re
from pathlib import Path
import markdown
# Configuração de logging avançada
def setup_logger():
    """Configura um sistema de logging avançado com rotação de arquivos."""
    # Cria o diretório de logs se não existir
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configura o logger principal
    logger = logging.getLogger("chatbot_registro")
    logger.setLevel(logging.DEBUG)
    
    # Handler para arquivo com rotação (10 arquivos de 10MB cada)
    file_handler = RotatingFileHandler(
        "logs/chatbot.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatação rica para ambos os handlers com defaults para os campos extras
    class SafeLogFormatter(logging.Formatter):
        def format(self, record):
            # Adiciona 'session_id' ao record se não existir
            if not hasattr(record, 'session_id'):
                record.session_id = 'NO_SESSION'
            return super().format(record)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(session_id)s] - %(message)s'
    formatter = SafeLogFormatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Adiciona os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
# Cria logger global
logger = setup_logger()

# Adiciona contexto de sessão aos logs
class SessionAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        if 'session_id' not in kwargs['extra']:
            kwargs['extra']['session_id'] = getattr(self, 'session_id', 'NO_SESSION')
        return msg, kwargs

# Decorador para medir tempo de execução e logar
def log_execution_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_id = kwargs.get('session_id', 'unknown')
        start_time = time.time()
        
        logger_adapter = SessionAdapter(logger, {'session_id': session_id})
        logger_adapter.info(f"Iniciando {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger_adapter.info(f"Concluído {func.__name__} em {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger_adapter.error(f"Erro em {func.__name__} após {execution_time:.2f}s: {str(e)}")
            logger_adapter.debug(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

# Inicializa app Flask
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "MY_SECRET_KEY")

# Configura sessão avançada
app.config['PERMANENT_SESSION_LIFETIME'] = int(os.getenv("SESSION_LIFETIME", "3600"))  # em segundos
app.config['SESSION_TYPE'] = "filesystem"
app.config['SESSION_FILE_DIR'] = "session_data"
app.config['SESSION_USE_SIGNER'] = True  # Assina cookies para segurança adicional

Session(app)

# Configuração Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_DB_HISTORY = int(os.getenv("REDIS_DB_HISTORY", "0"))
REDIS_DB_CACHE = int(os.getenv("REDIS_DB_CACHE", "1"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))  # 24 horas

# Conecta ao Redis
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()  # Testa conexão
    logger.info(f"Conectado ao Redis em: {REDIS_URL}")
    
    # Configura cache LLM via Redis
    set_llm_cache(RedisCache(redis_=redis_client, ttl=REDIS_TTL))
    logger.info("Cache LLM configurado via Redis")
except Exception as e:
    logger.error(f"Erro ao conectar ao Redis: {str(e)}")
    logger.warning("Funcionalidades dependentes de Redis serão limitadas")
    redis_client = None

# Configuração de chunking mais inteligente para documentos jurídicos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,               # Chunks menores para precisão
    chunk_overlap=200,            # Sobreposição significativa para manter contexto
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Prioriza quebras naturais de texto
)

# Configurações gerais carregadas de arquivo ou variáveis de ambiente
CONFIG = {
    "max_history_messages": int(os.getenv("MAX_HISTORY_MESSAGES", "10")),
    "default_model": os.getenv("DEFAULT_MODEL", "llama3.2:latest"),
    "fallback_model": os.getenv("FALLBACK_MODEL", "qwen2.5:latest"),
    "temperature": float(os.getenv("TEMPERATURE", "0.9")),
    "top_p": float(os.getenv("TOP_P", "0.9")),
    "top_k": int(os.getenv("RETRIEVAL_TOP_K", "5")),
    "embedding_model": os.getenv(
        "EMBEDDING_MODEL", 
        "stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0"
    ),
    "reindex_interval_days": int(os.getenv("REINDEX_INTERVAL_DAYS", "30")),
}

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    """Obtém histórico de mensagens do Redis com limite configurável."""
    history = RedisChatMessageHistory(
        session_id, 
        redis_client=redis_client,
        ttl=REDIS_TTL
    )
    
    # Limita o histórico ao número máximo configurado
    if len(history.messages) > CONFIG["max_history_messages"]:
        # Mantém apenas as mensagens mais recentes
        history.messages = history.messages[-CONFIG["max_history_messages"]:]
    
    return history

def summarize_history(messages: List[Any], max_length: int = 1000) -> str:
    """Resumir histórico longo para context window menor."""
    if not messages:
        return ""
    
    # Se for curto o suficiente, retorna sem resumir
    total_chars = sum(len(m.content) for m in messages)
    if total_chars <= max_length:
        return ""
    
    # Implementação básica: pega pontos-chave das últimas interações
    summary_messages = []
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            summary_messages.append(f"Pergunta {i//2+1}: {msg.content[:100]}...")
        elif isinstance(msg, AIMessage):
            key_points = re.findall(r'(?:^|\. )([^.]{20,100}\.)', msg.content)
            if key_points:
                summary_messages.append(f"Resposta {i//2+1} incluiu: {' '.join(key_points[:2])}")
    
    return "Resumo da conversa anterior: " + " ".join(summary_messages)

class ConversationContextManager:
    def __init__(self):
        # Sistema de prompts melhorado com instruções mais claras
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente especialista em registro empresarial. 
            Baseie-se **exclusivamente** no conteúdo dos documentos fornecidos:
            
            Instruções:
                - Fundamente cada resposta com artigos ou leis, ex: (Art. 33 da Lei 8.934/94)
                - Liste documentos obrigatórios e prazos em formato claro e estruturado
                - Caso a resposta não esteja nos documentos, diga exatamente: "Esta informação específica não consta nos documentos fornecidos."
                - Nunca invente informações ou cite leis que não estejam nos documentos
                - Organize respostas longas em tópicos para facilitar a leitura
                - Prefira respostas diretas e objetivas
                
            {context}
            
            {history_summary}

            Regras:
            - Responda SOMENTE com base nos documentos acima.
            - Seja preciso e conciso. Evite textos desnecessários.
            - Priorize a precisão legal sobre a generalidade.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        # Inicializa modelo primário e modelo de fallback
        self.setup_models()
        
        # Cria o fluxo conversacional
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Cria um runnable com histórico de mensagens
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            get_redis_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def setup_models(self):
        """Configura modelo principal e fallback."""
        try:
            # Modelo principal
            self.llm = ChatOllama(
                model=CONFIG["default_model"],
                temperature=CONFIG["temperature"],
                top_p=CONFIG["top_p"],
                num_ctx=4096,  # Contexto mais amplo
                repeat_penalty=1.1,  # Evita repetições
                system_prompt="""
                Você é um especialista em registro empresarial. Responda com:
                1. Fundamentação legal precisa (ex: Art. 33 da Lei 8.934/94)
                2. Lista de documentos necessários em formato claro
                3. Prazos legais quando aplicáveis
                """
            )
            logger.info(f"Modelo principal configurado: {CONFIG['default_model']}")
            
            # Configura modelo de fallback (será usado apenas se o principal falhar)
            self.fallback_llm = ChatOllama(
                model=CONFIG["fallback_model"],
                temperature=CONFIG["temperature"] + 0.1,  # Um pouco mais criativo
                system_prompt="""
                Você é um especialista em registro empresarial. Responda com:
                1. Fundamentação legal precisa
                2. Lista de documentos necessários
                3. Prazos legais quando aplicáveis
                """
            )
            logger.info(f"Modelo de fallback configurado: {CONFIG['fallback_model']}")
        except Exception as e:
            logger.error(f"Erro ao configurar modelos: {str(e)}")
            # Fallback para modelo mais simples em caso de erro
            self.llm = ChatOllama(model="llama3:8b", temperature=0.1)
            self.fallback_llm = self.llm

    @log_execution_time
    def process_message(self, input_message: str, context: str, session_id: str):
        """Processa uma mensagem com histórico de conversação e contexto documental."""
        logger_adapter = SessionAdapter(logger, {'session_id': session_id})
        
        try:
            # Obtém o histórico para potencial resumo
            history = get_redis_history(session_id)
            history_summary = summarize_history(history.messages) if len(history.messages) > 6 else ""
            
            # Tentativa com modelo principal
            logger_adapter.debug(f"Processando mensagem: '{input_message[:100]}...'")
            
            response = self.chain_with_history.invoke(
                {
                    "context": context,
                    "question": input_message,
                    "history_summary": history_summary
                },
                config={"configurable": {"session_id": session_id}}
            )
            
            logger_adapter.info(f"Resposta gerada com sucesso: {len(response)} caracteres")
            return {
                "content": response,
                "model_used": CONFIG["default_model"],
                "success": True
            }
            
        except Exception as e:
            logger_adapter.error(f"Erro ao gerar resposta com modelo principal: {str(e)}")
            logger_adapter.info("Tentando modelo de fallback...")
            
            try:
                # Tenta com modelo de fallback em caso de erro
                fallback_chain = (self.prompt | self.fallback_llm | StrOutputParser())
                response = fallback_chain.invoke({
                    "context": context,
                    "question": input_message,
                    "history": history.messages,
                    "history_summary": history_summary
                })
                
                logger_adapter.info("Resposta gerada com modelo de fallback")
                return {
                    "content": response,
                    "model_used": CONFIG["fallback_model"],
                    "fallback_used": True,
                    "success": True
                }
            except Exception as fallback_error:
                logger_adapter.error(f"Erro também no modelo de fallback: {str(fallback_error)}")
                return {
                    "content": "Não foi possível processar sua pergunta no momento. Por favor, tente novamente.",
                    "error": str(e),
                    "success": False
                }


class EnhancedQASystem:
    def __init__(self, data_path, faiss_index_path, domain_name="registro"):
        """
        Inicializa o sistema com indexação inteligente e cache
        
        Args:
            data_path: Caminho para os documentos
            faiss_index_path: Caminho para salvar/carregar índice FAISS
            domain_name: Nome do domínio para logging e diferenciação
        """
        self.domain_name = domain_name
        self.logger = SessionAdapter(logger, {'session_id': f'system_{domain_name}'})
        self.data_path = data_path
        self.faiss_index_path = faiss_index_path
        
        self.logger.info(f"Inicializando sistema de QA para domínio: {domain_name}")
        
        # Inicializa componentes
        self.setup_embeddings()
        self.conversation_manager = ConversationContextManager()
        
        # Carrega ou cria índice FAISS
        self.load_or_create_index()
        
        # Verifica se é necessário reindexar
        self.check_reindex_needed()
        
        self.logger.info(f"Sistema QA para {domain_name} inicializado com sucesso!")

    @log_execution_time
    def setup_embeddings(self):
        """Configura modelo de embeddings com tratamento de erro."""
        try:
            self.logger.info(f"Carregando modelo de embeddings: {CONFIG['embedding_model']}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=CONFIG['embedding_model'],
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}  # Normalização para melhor performance
            )
            self.logger.info("Modelo de embeddings carregado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo de embeddings: {str(e)}")
            self.logger.info("Usando modelo de embeddings fallback")
            # Fallback para modelo mais leve
            self.embeddings = HuggingFaceEmbeddings(
                model_name="stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0",
                model_kwargs={'device': 'cpu'}
            )

    def check_reindex_needed(self):
        """Verifica se é necessário reindexar baseado na data do último índice."""
        index_metadata_path = Path(f"{self.faiss_index_path}_metadata.json")
        
        # Verifica se arquivo de metadata existe
        if not index_metadata_path.exists():
            self.logger.info("Metadata de índice não encontrado. Criando novo.")
            self.save_index_metadata()
            return
            
        # Carrega metadata
        try:
            with open(index_metadata_path, 'r') as f:
                metadata = json.load(f)
                
            last_indexed = datetime.fromisoformat(metadata['last_indexed'])
            reindex_interval = timedelta(days=CONFIG['reindex_interval_days'])
            
            # Verifica se passou o tempo de reindexação
            if datetime.now() - last_indexed > reindex_interval:
                self.logger.info(f"Índice com mais de {CONFIG['reindex_interval_days']} dias. Reindexando...")
                self.setup_document_processing(self.data_path)
                self.save_faiss_index(self.faiss_index_path)
            else:
                self.logger.info("Índice ainda dentro do prazo de validade.")
                
        except Exception as e:
            self.logger.error(f"Erro ao verificar necessidade de reindexação: {str(e)}")
            
    def save_index_metadata(self):
        """Salva metadata sobre o índice."""
        metadata = {
            'last_indexed': datetime.now().isoformat(),
            'document_count': len(getattr(self, 'text_chunks', [])),
            'embedding_model': CONFIG['embedding_model'],
            'domain': self.domain_name
        }
        
        try:
            with open(f"{self.faiss_index_path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
            self.logger.info("Metadata de índice salvo com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao salvar metadata de índice: {str(e)}")

    def load_or_create_index(self):
        """Carrega índice existente ou cria novo se necessário."""
        if self.load_faiss_index(self.faiss_index_path):
            return
            
        self.logger.info("Índice FAISS não encontrado. Criando novo...")
        self.setup_document_processing(self.data_path)
        self.save_faiss_index(self.faiss_index_path)

    @log_execution_time
    def setup_document_processing(self, data_path: str):
        """Carrega e processa documentos para criar índice vetorial."""
        self.logger.info(f"Carregando documentos de: {data_path}")
        
        try:
            loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            self.logger.info(f"Carregados {len(documents)} documentos")
            
            # Pré-processamento dos documentos
            self.text_chunks = text_splitter.split_documents(documents)
            self.logger.info(f"Gerados {len(self.text_chunks)} chunks de texto")
            
            # Cria store de vetores
            self.vector_store = FAISS.from_documents(
                self.text_chunks,
                self.embeddings
            )
            
            self.logger.info("Índice vetorial criado com sucesso")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao processar documentos: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False

    def save_faiss_index(self, index_path):
        """Salva o índice FAISS com tratamento de erro."""
        try:
            if not os.path.exists(index_path):
                os.makedirs(index_path)
                
            self.vector_store.save_local(index_path)
            self.save_index_metadata()
            self.logger.info(f"Índice FAISS salvo em: {index_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar índice FAISS: {str(e)}")
            return False
        
    def load_faiss_index(self, index_path):
        print(f"{index_path}faiss.pkl")
        """Carrega o índice FAISS se já existir em disco."""
        if os.path.exists(f"{index_path}/index.pkl"):
            try:
                self.logger.info(f"Carregando índice FAISS de: {index_path}")
                self.vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Índice FAISS carregado com sucesso")
                return True
            except Exception as e:
                self.logger.error(f"Erro ao carregar índice FAISS: {str(e)}")
                return False
        return False

    @log_execution_time
    def find_relevant_documents(self, query: str, top_k: int = None):
        """Encontra documentos relevantes para a consulta com busca híbrida."""
        if top_k is None:
            top_k = CONFIG["top_k"]
            
        try:
            # Busca vetorial semântica
            semantic_results = self.vector_store.similarity_search(query, k=top_k)
            
            # Implementação básica de híbrido: enriquece os resultados com metadados
            for doc in semantic_results:
                if 'score' not in doc.metadata:
                    doc.metadata['score'] = 0.0
                    
                # Aumenta pontuação se houver keywords importantes
                keywords = self.extract_keywords(query)
                for keyword in keywords:
                    if keyword.lower() in doc.page_content.lower():
                        doc.metadata['score'] += 0.1
                        
            # Ordena por pontuação (básico, poderia ser mais sofisticado)
            semantic_results.sort(key=lambda doc: doc.metadata.get('score', 0.0), reverse=True)
            
            return semantic_results
        except Exception as e:
            self.logger.error(f"Erro na busca de documentos: {str(e)}")
            return []
            
    def extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave de uma query. Poderia ser mais sofisticado."""
        # Versão básica - extrair palavras importantes
        stopwords = ["o", "a", "os", "as", "um", "uma", "uns", "umas", "e", "de", "para", "com", "em"]
        words = text.lower().split()
        return [w for w in words if len(w) > 3 and w not in stopwords]
    
    @log_execution_time
    def process_query(self, session_id, query: str) -> Dict:
        """Processa uma consulta completa - busca documentos e gera resposta."""
        logger_adapter = SessionAdapter(logger, {'session_id': session_id})
        logger_adapter.info(f"Processando consulta: '{query[:50]}...'")
        
        try:
            # Busca documentos relevantes
            relevant_docs = self.find_relevant_documents(query)
            
            if not relevant_docs:
                logger_adapter.warning("Nenhum documento relevante encontrado")
                return {
                    "answer": "Não encontrei informações relevantes sobre essa questão nos documentos disponíveis.",
                    "source_documents": [],
                    "success": True
                }
            
            # Formata contexto com os documentos encontrados
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = doc.metadata.get("source", "desconhecido").split("/")[-1]
                page = doc.metadata.get("page", "?")
                
                context_parts.append(f"--- Documento {i}: {source} (Página {page}) ---\n{doc.page_content}\n")
            
            context = "\n".join(context_parts)
            
            # Gera resposta usando o gerenciador de conversa
            response_data = self.conversation_manager.process_message(query, context, session_id)
            
            if response_data.get("success", False):
                return {
                    "answer": response_data["content"],
                    "model_used": response_data.get("model_used", "unknown"),
                    "source_documents": [
                        {
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page", "?")
                        }
                        for doc in relevant_docs
                    ],
                    "success": True
                }
            else:
                # Em caso de falha
                logger_adapter.error(f"Falha ao gerar resposta: {response_data.get('error', 'Erro desconhecido')}")
                return {
                    "answer": response_data.get("content", "Não foi possível processar sua consulta."),
                    "source_documents": [],
                    "success": False
                }
        
        except Exception as e:
            logger_adapter.error(f"Erro ao processar consulta: {str(e)}")
            logger_adapter.debug(traceback.format_exc())
            return {
                "answer": f"Ocorreu um erro ao processar sua consulta. Por favor, tente novamente.",
                "source_documents": [],
                "success": False
            }

# Instâncias globais com melhor estruturação
try:
    # Assegura que diretórios existam
    for path in ["./data/data_registro", "./data/data_juridico", 
                "./faiss/faiss_index_registro", "./faiss/faiss_index_juridico"]:
        os.makedirs(path, exist_ok=True)
        
    # Inicializa sistemas QA
    qa_system_registro = EnhancedQASystem(
        "./data/data_registro", 
        "./faiss/faiss_index_registro",
        "registro"
    )
    qa_system_juridico = EnhancedQASystem(
        "./data/data_juridico", 
        "./faiss/faiss_index_juridico",
        "juridico"
    )
    logger.info("Sistemas QA inicializados com sucesso")
except Exception as e:
    logger.critical(f"Erro ao inicializar sistemas QA: {str(e)}")
    logger.debug(traceback.format_exc())
    # Fallback para versões simplificadas em caso de erro crítico
    qa_system_registro = None
    qa_system_juridico = None

# Middleware para gerenciar sessões
@app.before_request
def session_management():
    """Gerencia criação e renovação de sessões antes de cada requisição."""
    if "session_id" not in session:
        session_id = str(uuid.uuid4().hex)
        session["session_id"] = session_id
        session["created_at"] = datetime.now().isoformat()
        logger.info(f"Nova sessão criada: {session_id}")
    
    # Registra atividade para manter sessão ativa
    session["last_active"] = datetime.now().isoformat()
    
    # Registra estatísticas básicas de uso
    metrics_path = Path("logs/metrics")
    metrics_path.mkdir(exist_ok=True, parents=True)
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        with open(f"logs/metrics/usage_{today}.txt", "a") as f:
            f.write(f"{datetime.now().isoformat()},{session.get('session_id', 'unknown')},{request.endpoint or 'unknown'}\n")
    except Exception as e:
        logger.warning(f"Não foi possível registrar métricas: {str(e)}")

def format_response(text):
    # Converte markdown para HTML
    html = markdown.markdown(text)
    return html

# Limpeza de sessões expiradas
@app.cli.command("clean-sessions")
def clean_sessions():
    """Comando para limpar sessões expiradas via Flask CLI."""
    session_dir = Path(app.config['SESSION_FILE_DIR'])
    count = 0
    
    if session_dir.exists():
        for session_file in session_dir.glob("*"):
            if session_file.is_file() and (datetime.now() - datetime.fromtimestamp(session_file.stat().st_mtime)).days > 7:
                session_file.unlink()
                count += 1
    
    print(f"Removidas {count} sessões expiradas")

# Rotas com limites de taxa e melhor tratamento de erros
@app.route("/")
def index():
    """Página inicial redirecionando para os chatbots disponíveis."""
    return render_template("index.html", 
                          sistemas=["registro", "juridico"],
                          session_id=session.get("session_id", ""))


@app.route("/chat_registro")
def chat_registro():
    """Rota para o chatbot de registro empresarial."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4().hex)   
    
    # Verifica disponibilidade do sistema
    if qa_system_registro is None:
        return render_template("error.html", 
                               error="Sistema de registro temporariamente indisponível",
                               retry_in="5 minutos")
    
    return render_template("registro.html", 
                          session_id=session.get("session_id"),
                          max_history=CONFIG["max_history_messages"])


@app.route("/pergunta_chat_registro", methods=["POST"])
def pergunta_chat_registro():
    """Endpoint para processar perguntas do chatbot de registro."""
    try:
        # Verifica se o sistema está disponível
        if qa_system_registro is None:
            return jsonify({
                "answer": "Sistema temporariamente indisponível. Tente novamente em instantes.",
                "source_documents": [],
                "success": False
            }), 503  # Service Unavailable
            
        # Obtém dados da requisição
        start_time = time.time()
        user_input = request.form.get("user_input", "")
        session_id = session.get('session_id', str(uuid.uuid4().hex))
        
        # Valida entrada
        if not user_input or len(user_input.strip()) < 3:
            return jsonify({
                "answer": "Por favor, faça uma pergunta mais completa.",
                "source_documents": [],
                "success": False
            }), 400  # Bad Request
        
        # Limite de tamanho para prevenir abuso
        if len(user_input) > 1000:
            user_input = user_input[:1000] + "..."
        
        # Registra a pergunta no log com session_id
        logger_adapter = SessionAdapter(logger, {'session_id': session_id})
        logger_adapter.info(f"Pergunta registro: {user_input[:100]}...")
        
        # Processa a consulta
        result = qa_system_registro.process_query(session_id, user_input)
        
        # Calcula tempo de execução
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Adiciona informação de tempo à resposta
        result["execution_time"] = execution_time
        
        logger_adapter.info(f"Resposta gerada em {execution_time:.2f}s")
        result["answer"] = format_response(result["answer"])
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Erro interno: {str(e)}")
        logger.debug(traceback.format_exc())
        
        return jsonify({
            "answer": "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente.",
            "source_documents": [],
            "execution_time": 0,
            "success": False
        }), 500  # Internal Server Error


@app.route("/chat_juridico")
def chat_juridico():
    """Rota para o chatbot jurídico."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4().hex)   
    
    # Verifica disponibilidade do sistema
    if qa_system_juridico is None:
        return render_template("error.html", 
                               error="Sistema jurídico temporariamente indisponível",
                               retry_in="5 minutos")
    
    return render_template("juridico.html", 
                          session_id=session.get("session_id"),
                          max_history=CONFIG["max_history_messages"])


@app.route("/pergunta_chat_juridico", methods=["POST"])
def pergunta_chat_juridico():
    """Endpoint para processar perguntas do chatbot jurídico."""
    try:
        # Verifica se o sistema está disponível
        if qa_system_juridico is None:
            return jsonify({
                "answer": "Sistema temporariamente indisponível. Tente novamente em instantes.",
                "source_documents": [],
                "success": False
            }), 503  # Service Unavailable
            
        # Obtém dados da requisição
        start_time = time.time()
        user_input = request.form.get("user_input", "")
        session_id = session.get('session_id', str(uuid.uuid4().hex))
        
        # Valida entrada
        if not user_input or len(user_input.strip()) < 3:
            return jsonify({
                "answer": "Por favor, faça uma pergunta mais completa.",
                "source_documents": [],
                "success": False
            }), 400  # Bad Request
        
        # Limite de tamanho para prevenir abuso
        if len(user_input) > 1000:
            user_input = user_input[:1000] + "..."
        
        # Registra a pergunta no log
        logger_adapter = SessionAdapter(logger, {'session_id': session_id})
        logger_adapter.info(f"Pergunta jurídico: {user_input[:100]}...")
        
        # Processa a consulta
        result = qa_system_juridico.process_query(session_id, user_input)
        
        # Calcula tempo de execução
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Adiciona informação de tempo à resposta
        result["execution_time"] = execution_time
        
        logger_adapter.info(f"Resposta gerada em {execution_time:.2f}s")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Erro interno: {str(e)}")
        logger.debug(traceback.format_exc())
        
        return jsonify({
            "answer": "Ocorreu um erro ao processar sua pergunta. Por favor, tente novamente.",
            "source_documents": [],
            "execution_time": 0,
            "success": False
        }), 500  # Internal Server Error

# Endpoint para limpar histórico de conversa
@app.route("/limpar_historico", methods=["POST"])
def limpar_historico():
    """Endpoint para limpar o histórico de conversa da sessão atual."""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"success": False, "message": "Sessão inválida"}), 400
        
        # Tenta remover o histórico do Redis
        if redis_client:
            try:
                history_key = f"message_store:{session_id}"
                redis_client.delete(history_key)
            except Exception as e:
                logger.warning(f"Erro ao limpar histórico no Redis: {str(e)}")
        
        logger.info(f"Histórico de conversa limpo para sessão: {session_id}")
        
        return jsonify({
            "success": True,
            "message": "Histórico de conversa limpo com sucesso"
        })
    
    except Exception as e:
        logger.error(f"Erro ao limpar histórico: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Erro ao limpar histórico de conversa"
        }), 500

# Endpoint para salvar feedback do usuário
@app.route("/feedback", methods=["POST"])
def feedback():
    """Endpoint para salvar feedback do usuário sobre respostas."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Dados inválidos"}), 400
        
        session_id = session.get('session_id', 'unknown')
        rating = data.get('rating')
        message_id = data.get('message_id')
        comment = data.get('comment', '')
        
        # Valida os dados
        if rating is None or message_id is None:
            return jsonify({"success": False, "message": "Dados incompletos"}), 400
        
        # Salva o feedback em arquivo
        feedback_dir = Path("logs/feedback")
        feedback_dir.mkdir(exist_ok=True, parents=True)
        
        with open(f"logs/feedback/feedback_{datetime.now().strftime('%Y-%m')}.csv", "a") as f:
            f.write(f"{datetime.now().isoformat()},{session_id},{message_id},{rating},{comment.replace(',', ' ')}\n")
        
        logger.info(f"Feedback recebido: sessão={session_id}, rating={rating}")
        
        return jsonify({
            "success": True,
            "message": "Feedback recebido. Obrigado!"
        })
    
    except Exception as e:
        logger.error(f"Erro ao salvar feedback: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Erro ao processar feedback"
        }), 500

# Endpoint de status do sistema
@app.route("/status")
def status():
    """Endpoint para verificar status dos sistemas."""
    try:
        sistemas = {
            "registro": {
                "status": "online" if qa_system_registro is not None else "offline",
                "documentos": len(getattr(qa_system_registro, 'text_chunks', [])) if qa_system_registro else 0
            },
            "juridico": {
                "status": "online" if qa_system_juridico is not None else "offline",
                "documentos": len(getattr(qa_system_juridico, 'text_chunks', [])) if qa_system_juridico else 0
            },
            "redis": "conectado" if redis_client else "desconectado",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(sistemas)
    except Exception as e:
        logger.error(f"Erro ao verificar status: {str(e)}")
        return jsonify({"error": "Erro ao verificar status"}), 500

# Tratamento de erro 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', 
                          error="Página não encontrada",
                          message="A página que você está procurando não existe."), 404

# Tratamento de erro 500
@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Erro interno do servidor: {str(e)}")
    return render_template('error.html',
                          error="Erro interno do servidor",
                          message="Ocorreu um erro interno. Por favor, tente novamente mais tarde."), 500

# Ponto de entrada principal
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    
    logger.info(f"Iniciando aplicação em {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)