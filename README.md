# Chatbot de Direito Empresarial - Langchain, llama3.2, e Flask

Este projeto implementa um chatbot de IA capaz de responder perguntas com base em arquivos PDF de um conjunto de dados. O chatbot utiliza as frameworks Langchain, Ollama e Flask, além da técnica RAG (Retrieval Augmentation Generation) para gerar respostas precisas.

Projeto base: [PetCare-AI-Chatbot](https://github.com/rajveersinghcse/PetCare-AI-Chatbot)

## Conceito Principal

O conceito central deste chatbot de IA está no uso de técnicas avançadas de processamento de linguagem natural (NLP) para fornecer respostas precisas e contextuais às perguntas dos usuários. A seguir, um breve resumo dos principais componentes:

- **Langchain**: Framework que integra várias ferramentas e bibliotecas de NLP, facilitando a construção de sistemas de conversação complexos. Neste projeto, é usada para carregamento de documentos, divisão de textos, embeddings e mais.

- **Ollama**: Modelo de linguagem avançado (LLM) que permite ao chatbot compreender e gerar respostas semelhantes às humanas. Ao incorporar o Ollama na cadeia de conversação, o chatbot ganha uma compreensão mais profunda das perguntas dos usuários e do contexto.

- **Flask**: Framework web leve usado para construir aplicações web, incluindo APIs e interfaces baseadas na web. No projeto, o Flask serve como servidor backend, hospedando o chatbot e lidando com as interações dos usuários.

- **RAG (Retrieval Augmentation Generation)**: Metodologia que combina a recuperação de informações (retrieval) com a geração de linguagem (generation) para produzir respostas de alta qualidade. Ao recuperar informações relevantes dos PDFs usando as capacidades de recuperação do Langchain e complementá-las com a geração do Ollama, o chatbot pode fornecer respostas precisas e informativas.

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/Aik0o1/chat-bot-Registro.git
   cd chat-bot-Registro
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Confirme que você tem os arquivos PDF necessários na pasta `data/`. Incluímos os seguintes documentos de Direito Empresarial:
   - [Legislações Federais](https://www.gov.br/empresas-e-negocios/pt-br/drei/legislacao/legislacoes-federais)
   - [Instruções Normativas](https://www.gov.br/empresas-e-negocios/pt-br/drei/legislacao/instrucoes-normativas)
   - [Leis Estaduais e complementares](https://www.piauidigital.pi.gov.br/home/legislacao/)
   - Você pode adicionar mais arquivos, mas isso exigirá mais recursos computacionais.

2. Você deve instalar o servidor Ollama e configurar o model llama3.2. 
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2
   ollama serve
   ```

   ```bash
   # Opção com Docker:
   docker run -p 11434:11434 ollama/ollama
   docker exec -it <container_id> bash
   ollama pull llama3.2
   ```


4. Rode a aplicação Flask:
   ```bash
   python app.py
   ```

5. Acesse a interface do chatbot no navegador, digitando `http://localhost:5000`.

## Estrutura do projeto Structure

- `app.py`: Aplicação Flask que executa o servidor do chatbot.
- `templates/index.html`: Template HTML para a interface do chatbot.
- `data/`: Diretório contendo os arquivos PDF usados para responder perguntas.
- `requirements.txt`: Lista de dependências do Python.

## Configurações

1. `text_splitter`: Divide os documentos em blocos menores para processamento.
- `chunk_size`: Delimita o máximo de caracteres que um pedaço pode ter
- `chunk_overlap`: Número de caracteres que devem se sobrepor entre dois pedaços adjacentes.
2. `class ConversationContextManager`: Gerencia o contexto da conversa, armazenando as interações e fornecendo um resumo contextual para perguntas futuras.
3. `class EnhancedQASystem`: Sistema principal que integra todas as partes e processa as consultas dos usuários.
- `loader`: Carrega documentos PDF da pasta data/ usando o DirectoryLoader do Langchain, que utiliza o PyPDFLoader para leitura dos arquivos.
- `text_chunks`: Blocos de texto gerados a partir dos documentos carregados, usando o text_splitter.
- `embeddings`: : Utiliza o modelo sentence-transformers/all-MiniLM-L6-v2 da Hugging Face para gerar representações vetoriais dos blocos de texto.
- `vector_store`: Armazena os blocos de texto usando FAISS, permitindo a recuperação eficiente baseada em similaridade.
- `llm`: Implementa o modelo Ollama (Llama3.2), configurado com parâmetros como temperature e top_p, para controlar a geração de texto.
