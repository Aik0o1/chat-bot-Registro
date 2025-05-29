from langchain_ollama import ChatOllama
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
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
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

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

# MELHORIA 1: Chunking semântico para documentos jurídicos
# Configuração de chunking mais inteligente com base em estrutura semântica
# Primeiro dividimos por cabeçalhos (títulos, artigos, etc.) e depois por conteúdo
markdown_headers_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Título"),
        ("##", "Capítulo"),
        ("###", "Seção"),
        ("####", "Artigo"),
    ]
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,               
    chunk_overlap=150,           
    length_function=len,
    separators=[
        "\nArtigo ", "\nArt. ", "\nParágrafo ", "\n§ ", 
        "\nCAPÍTULO ", "\nSEÇÃO ", "\nTÍTULO ",
        "\n\n", "\n", ". ", " ", ""
    ]  # Separadores específicos para documentos jurídicos brasileiros
)

# Configurações gerais
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
    # reraking
    "rerank_top_n": int(os.getenv("RERANK_TOP_N", "20")),  # Recupera mais documentos para reranking
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.75")),  # Limiar para filtragem
    "use_hybrid_search": os.getenv("USE_HYBRID_SEARCH", "True").lower() in ("true", "1", "t"),
    "use_context_compression": os.getenv("USE_CONTEXT_COMPRESSION", "True").lower() in ("true", "1", "t"),
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
    """Resumir histórico longo para context window menor com foco em termos jurídicos."""
    if not messages:
        return ""
    
    # Se for curto o suficiente, retorna sem resumir
    total_chars = sum(len(m.content) for m in messages)
    if total_chars <= max_length:
        return ""
    
    summary_messages = []
    
    # Padrões para identificar termos jurídicos importantes
    legal_patterns = [
        r'(?:Lei|Decreto|Portaria|Instrução Normativa|Resolução)\s+[nN]º\s*[\d\.]+\/?\d*',
        r'[Aa]rt(?:igo)?\.\s*\d+',
        r'[Pp]arágrafo\s+(?:único|[\d]+º)',
        r'[Ii]nciso\s+[IVXLCDMivxlcdm]+',
        r'[Aa]línea\s+[a-z]',
    ]
    
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            # Extrai termos jurídicos da pergunta
            legal_terms = []
            for pattern in legal_patterns:
                found = re.findall(pattern, msg.content)
                legal_terms.extend(found)
            
            if legal_terms:
                legal_terms_str = ", ".join(legal_terms)
                summary_messages.append(f"Pergunta {i//2+1} sobre: {legal_terms_str}")
            else:
                summary_messages.append(f"Pergunta {i//2+1}: {msg.content[:80]}...")
                
        elif isinstance(msg, AIMessage):
            # Extrai citações legais e pontos-chave das respostas
            legal_refs = []
            for pattern in legal_patterns:
                found = re.findall(pattern, msg.content)
                legal_refs.extend(found)
            
            if legal_refs:
                legal_refs_str = ", ".join(legal_refs[:3])  # Limita a 3 referências
                summary_messages.append(f"Resposta {i//2+1} citou: {legal_refs_str}")
            else:
                # Extrai frases importantes
                key_points = re.findall(r'(?:^|\. )([^.]{20,100}\.)', msg.content)
                if key_points:
                    summary_messages.append(f"Resposta {i//2+1} incluiu: {' '.join(key_points[:2])}")
    
    return "Resumo da conversa anterior: " + " ".join(summary_messages)

class ConversationContextManager:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente especialista em registro empresarial e direito comercial. 
            Baseie-se **exclusivamente** no conteúdo dos documentos fornecidos:
            
            Instruções:
                - Fundamente cada resposta com artigos ou leis específicos, ex: (Art. 33 da Lei 8.934/94)
                - Liste documentos obrigatórios e prazos em formato claro e estruturado
                - Caso a resposta não esteja nos documentos, diga exatamente: "Esta informação específica não consta nos documentos fornecidos."
                - Nunca invente informações ou cite leis que não estejam nos documentos
                - Organize respostas longas em tópicos para facilitar a leitura
                - Prefira respostas diretas e objetivas
                - Ao citar artigos de lei, transcreva o texto exato quando disponível
                - Priorize informações mais recentes em caso de conflito entre normas
                
            {context}
            
            {history_summary}

            Regras:
            - Responda SOMENTE com base nos documentos acima.
            - Seja preciso e conciso. Evite textos desnecessários.
            - Priorize a precisão legal sobre a generalidade.
            - Quando houver múltiplas interpretações possíveis, indique isso claramente.
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
            )
            logger.info(f"Modelo principal configurado: {CONFIG['default_model']}")
            
            # Configura modelo de fallback 
            self.fallback_llm = ChatOllama(
                model=CONFIG["fallback_model"],
                temperature=CONFIG["temperature"],  
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
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}  # Normalização 
            )
            self.logger.info("Modelo de embeddings carregado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo de embeddings: {str(e)}")
            self.logger.info("Usando modelo de embeddings fallback")
            # Fallback para modelo mais leve
            self.embeddings = HuggingFaceEmbeddings(
                model_name="stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0",
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}
            )

    def check_reindex_needed(self):
        """Verifica se é necessário reindexar os documentos."""
        try:
            # Verifica se existe arquivo de metadados
            metadata_path = Path(f"{self.faiss_index_path}_metadata.json")
            if not metadata_path.exists():
                self.logger.info("Arquivo de metadados não encontrado. Reindexação necessária.")
                self.reindex_documents()
                return
            
            # Carrega metadados
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verifica data da última indexação
            last_indexed = datetime.fromisoformat(metadata.get('last_indexed', '2000-01-01'))
            days_since_indexed = (datetime.now() - last_indexed).days
            
            if days_since_indexed > CONFIG['reindex_interval_days']:
                self.logger.info(f"Índice desatualizado ({days_since_indexed} dias). Reindexando...")
                self.reindex_documents()
            else:
                self.logger.info(f"Índice atualizado (última indexação: {last_indexed.isoformat()})")
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar necessidade de reindexação: {str(e)}")
            # Em caso de erro, força reindexação
            self.reindex_documents()

    @log_execution_time
    def load_or_create_index(self):
        """Carrega índice FAISS existente ou cria um novo."""
        try:
            # Verifica se o índice existe
            if os.path.exists(f"{self.faiss_index_path}.faiss"):
                self.logger.info(f"Carregando índice FAISS de {self.faiss_index_path}")
                
                self.vectorstore = FAISS.load_local(
                    self.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Carrega metadados adicionais
                metadata_path = Path(f"{self.faiss_index_path}_metadata.json")
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.index_metadata = json.load(f)
                        
                    # Carrega chunks
                    if CONFIG["use_hybrid_search"]:
                        chunks_path = Path(f"{self.faiss_index_path}_chunks.json")
                        if chunks_path.exists():
                            with open(chunks_path, 'r') as f:
                                self.text_chunks = json.load(f)
                                self.setup_hybrid_retriever()
                
                self.logger.info(f"Índice FAISS carregado com sucesso: {len(self.vectorstore.index_to_docstore_id)} documentos")
            else:
                self.logger.info(f"Índice FAISS não encontrado. Criando novo índice...")
                self.reindex_documents()
        
        except Exception as e:
            self.logger.error(f"Erro ao carregar índice FAISS: {str(e)}")
            self.logger.info("Criando novo índice...")
            self.reindex_documents()

    @log_execution_time
    def reindex_documents(self):
        """Reindexar todos os documentos."""
        try:
            self.logger.info(f"Iniciando indexação de documentos em: {self.data_path}")
            
            # Carrega documentos PDF
            loader = DirectoryLoader(
                self.data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            self.logger.info(f"Carregados {len(documents)} documentos PDF")
            
            # Primeiro, converte para markdown para identificar cabeçalhos
            markdown_docs = []
            for doc in documents:
                # Extrai metadados do documento
                source = doc.metadata.get('source', 'unknown')
                filename = os.path.basename(source)
                
                # Tenta identificar tipo de documento e número da lei/normativa
                doc_type = "Documento"
                doc_number = ""
                
                # Padrões para identificar tipos de documentos e números
                patterns = [
                    (r'lei[_\s-]*(\d+[\.\d]*)', 'Lei'),
                    (r'decreto[_\s-]*(\d+[\.\d]*)', 'Decreto'),
                    (r'instrucao[_\s-]*normativa[_\s-]*(\d+[\.\d]*)', 'Instrução Normativa'),
                    (r'resolucao[_\s-]*(\d+[\.\d]*)', 'Resolução'),
                    (r'portaria[_\s-]*(\d+[\.\d]*)', 'Portaria'),
                ]
                
                for pattern, doc_type_name in patterns:
                    match = re.search(pattern, filename.lower())
                    if match:
                        doc_type = doc_type_name
                        doc_number = match.group(1)
                        break
                
                # Converte para formato markdown com cabeçalhos
                content = doc.page_content
                
          
                content = re.sub(r'(?m)^(TÍTULO|CAPÍTULO)\s+([IVXLCDM]+)', r'# \1 \2', content)
                content = re.sub(r'(?m)^(Seção|SEÇÃO)\s+([IVXLCDM]+)', r'## \1 \2', content)
                content = re.sub(r'(?m)^Art\.\s*(\d+)', r'### Artigo \1', content)
                content = re.sub(r'(?m)^Parágrafo\s+(único|[0-9]+º)', r'#### Parágrafo \1', content)
                content = re.sub(r'(?m)^§\s*(\d+º)', r'#### Parágrafo \1', content)
                
                # Adiciona metadados enriquecidos
                enhanced_metadata = {
                    'source': source,
                    'filename': filename,
                    'doc_type': doc_type,
                    'doc_number': doc_number,
                    'page': doc.metadata.get('page', 0)
                }
                
                markdown_docs.append({
                    'content': content,
                    'metadata': enhanced_metadata
                })
            
            # Aplica o splitter de cabeçalhos markdown
            chunked_docs = []
            for doc in markdown_docs:
                try:
                    # Tenta dividir por cabeçalhos
                    header_splits = markdown_headers_splitter.split_text(doc['content'])
                    
                    # Adiciona metadados originais + metadados de cabeçalho
                    for split in header_splits:
                        split.metadata.update(doc['metadata'])
                        
                        # Adiciona informações de hierarquia como metadados
                        if 'Título' in split.metadata:
                            split.metadata['section_title'] = split.metadata['Título']
                        if 'Capítulo' in split.metadata:
                            split.metadata['section_chapter'] = split.metadata['Capítulo']
                        if 'Artigo' in split.metadata:
                            split.metadata['section_article'] = split.metadata['Artigo']
                            
                        chunked_docs.append(split)
                except Exception as e:
                    # Se falhar, usa o documento original
                    self.logger.warning(f"Erro ao dividir por cabeçalhos: {str(e)}")
                    # Aplica divisão recursiva diretamente
                    splits = text_splitter.split_text(doc['content'])
                    for i, split in enumerate(splits):
                        chunked_docs.append({
                            'page_content': split,
                            'metadata': {**doc['metadata'], 'chunk_id': i}
                        })
            
            # Se não conseguiu dividir por cabeçalhos, aplica o splitter recursivo
            if not chunked_docs:
                self.logger.info("Aplicando chunking recursivo padrão")
                chunked_docs = text_splitter.split_documents(documents)
            
            self.logger.info(f"Documentos divididos em {len(chunked_docs)} chunks")
            
            # Salva chunks para uso com BM25
            self.text_chunks = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in chunked_docs
            ]
            
            # Cria índice FAISS
            self.vectorstore = FAISS.from_documents(
                chunked_docs, 
                self.embeddings
            )
            
            # Salva índice
            self.vectorstore.save_local(self.faiss_index_path)
            
            # Salva metadados
            self.index_metadata = {
                'last_indexed': datetime.now().isoformat(),
                'document_count': len(documents),
                'chunk_count': len(chunked_docs),
                'embedding_model': CONFIG['embedding_model']
            }
            
            with open(f"{self.faiss_index_path}_metadata.json", 'w') as f:
                json.dump(self.index_metadata, f)
                
            # Salva chunks para uso com BM25
            with open(f"{self.faiss_index_path}_chunks.json", 'w') as f:
                json.dump(self.text_chunks, f)
                
            # Configura retriever híbrido se habilitado
            if CONFIG["use_hybrid_search"]:
                self.setup_hybrid_retriever()
                
            self.logger.info(f"Indexação concluída: {len(chunked_docs)} chunks indexados")
            
        except Exception as e:
            self.logger.error(f"Erro durante indexação: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

    # MELHORIA 9: Busca híbrida (dense + sparse) para melhor recall
    def setup_hybrid_retriever(self):
        """Configura um retriever híbrido combinando embeddings e BM25."""
        try:
            # Cria documentos para BM25
            bm25_docs = []
            for chunk in self.text_chunks:
                from langchain_core.documents import Document
                bm25_docs.append(Document(
                    page_content=chunk['content'],
                    metadata=chunk['metadata']
                ))
            
            # Configura retriever FAISS (dense)
            faiss_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": CONFIG["top_k"]}
            )
            
            # Configura retriever BM25 (sparse)
            bm25_retriever = BM25Retriever.from_documents(
                bm25_docs,
                preprocess_func=lambda text: text.lower()  # Normalização simples
            )
            bm25_retriever.k = CONFIG["top_k"]
            
            # Configura retriever híbrido
            self.retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[0.7, 0.3] 
            )
            
            if CONFIG["use_context_compression"]:
                # Filtro de redundância baseado em embeddings
                redundant_filter = EmbeddingsRedundantFilter(
                    embeddings=self.embeddings,
                    similarity_threshold=CONFIG["similarity_threshold"]
                )
                
                # Filtro de relevância baseado em embeddings
                relevance_filter = EmbeddingsFilter(
                    embeddings=self.embeddings,
                    similarity_threshold=CONFIG["similarity_threshold"] * 0.9  # Limiar um pouco menor
                )
                
                # Pipeline de compressão
                compressor = DocumentCompressorPipeline(
                    transformers=[redundant_filter, relevance_filter]
                )
                
                # Retriever com compressão
                self.retriever = ContextualCompressionRetriever(
                    base_retriever=self.retriever,
                    base_compressor=compressor
                )
                
            self.logger.info("Retriever híbrido configurado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar retriever híbrido: {str(e)}")
            # Fallback para retriever padrão
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": CONFIG["top_k"]}
            )

    @log_execution_time
    def process_query(self, session_id: str, query: str):
        """
        Processa uma consulta do usuário.
        
        Args:
            session_id: ID da sessão do usuário
            query: Consulta do usuário
            
        Returns:
            Dict com resposta e documentos fonte
        """
        logger_adapter = SessionAdapter(logger, {'session_id': session_id})
        logger_adapter.info(f"Processando consulta: '{query[:100]}...'")
        
        try:
            if CONFIG["use_hybrid_search"]:
                retrieved_docs = self.retriever.get_relevant_documents(query)
            else:
                retrieved_docs = self.vectorstore.similarity_search(
                    query, 
                    k=CONFIG["rerank_top_n"]
                )
            
            logger_adapter.info(f"Recuperados {len(retrieved_docs)} documentos")
            
            # Reranking manual baseado em similaridade semântica com a query
            if len(retrieved_docs) > CONFIG["top_k"]:
                query_embedding = self.embeddings.embed_query(query)
                
                query_embedding_np = np.array(query_embedding)
                
                # Calcula similaridade para cada documento
                doc_scores = []
                for i, doc in enumerate(retrieved_docs):
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    doc_embedding_np = np.array(doc_embedding)
                    
                    similarity = np.dot(query_embedding_np, doc_embedding_np) / (
                        np.linalg.norm(query_embedding_np) * np.linalg.norm(doc_embedding_np)
                    )
                    
                    boost = 0.0
                    
                    if 'section_article' in doc.metadata:
                        article_pattern = r'(?:art(?:igo)?\.?\s*(\d+))'
                        article_matches = re.findall(article_pattern, query.lower())
                        if article_matches and any(match == doc.metadata['section_article'].lower().replace('artigo ', '') for match in article_matches):
                            boost += 0.2
                    
                    if 'doc_type' in doc.metadata and doc.metadata['doc_type'].lower() in query.lower():
                        boost += 0.1
                        
                    if 'doc_number' in doc.metadata and doc.metadata['doc_number'] in query:
                        boost += 0.15
                    
                    final_score = similarity + boost
                    
                    doc_scores.append((i, final_score))
                
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in doc_scores[:CONFIG["top_k"]]]
                
                # Filtra documentos
                retrieved_docs = [retrieved_docs[i] for i in top_indices]
                
                logger_adapter.info(f"Documentos reranqueados para os {CONFIG['top_k']} mais relevantes")
            
            # Formata documentos para contexto
            context_docs = []
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'Desconhecido')
                filename = os.path.basename(source) if 'source' in doc.metadata else 'Documento'
                page = doc.metadata.get('page', '')
                doc_type = doc.metadata.get('doc_type', 'Documento')
                doc_number = doc.metadata.get('doc_number', '')
                
                # Formata referência
                if doc_number:
                    reference = f"{doc_type} {doc_number}"
                else:
                    reference = filename
                
                if page:
                    reference += f", Página {page}"
                
                # Adiciona informações de seção se disponíveis
                section_info = []
                if 'section_title' in doc.metadata:
                    section_info.append(f"Título {doc.metadata['section_title']}")
                if 'section_chapter' in doc.metadata:
                    section_info.append(f"Capítulo {doc.metadata['section_chapter']}")
                if 'section_article' in doc.metadata:
                    section_info.append(f"Artigo {doc.metadata['section_article']}")
                
                section_text = ", ".join(section_info)
                if section_text:
                    reference += f" ({section_text})"
                
                # Formata documento para contexto
                context_docs.append(f"[Documento {i+1}: {reference}]\n{doc.page_content}\n")
            
            # Junta documentos em um único contexto
            context = "\n".join(context_docs)
            
            # Processa mensagem com o contexto
            result = self.conversation_manager.process_message(query, context, session_id)
            
            # Adiciona documentos fonte à resposta
            result["source_documents"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get('source', 'Desconhecido'),
                    "page": doc.metadata.get('page', ''),
                    "doc_type": doc.metadata.get('doc_type', 'Documento'),
                    "doc_number": doc.metadata.get('doc_number', '')
                }
                for doc in retrieved_docs
            ]
            
            return result
            
        except Exception as e:
            logger_adapter.error(f"Erro ao processar consulta: {str(e)}")
            logger_adapter.debug(traceback.format_exc())
            
            return {
                "answer": f"Erro ao processar sua consulta: {str(e)}",
                "source_documents": [],
                "success": False
            }

# Inicializa sistemas de QA
qa_system_registro = None
qa_system_juridico = None

try:
    # Sistema para registro empresarial
    qa_system_registro = EnhancedQASystem(
        data_path="data/registro",
        faiss_index_path="indexes/registro_index",
        domain_name="registro"
    )
    
    # Sistema para documentos jurídicos gerais
    qa_system_juridico = EnhancedQASystem(
        data_path="data/juridico",
        faiss_index_path="indexes/juridico_index",
        domain_name="juridico"
    )
    
except Exception as e:
    logger.error(f"Erro ao inicializar sistemas de QA: {str(e)}")

def format_response(text):
    # Converte markdown para HTML
    html = markdown.markdown(text)
    return html


# Rota principal
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
        
        # Limite de tamanho
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
