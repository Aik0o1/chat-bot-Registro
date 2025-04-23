FROM python:3.12-slim

# Instala dependências do sistema e limpa cache para reduzir tamanho da imagem
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clona o repositório
RUN git clone https://github.com/Aik0o1/chat-bot-Registro.git . && \
    # Instala dependências Python (incluindo gunicorn explicitamente)
    pip install --upgrade pip && \
    pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Define variáveis de ambiente (opcional)
ENV PYTHONUNBUFFERED=1 \
    PORT=5000

# Recomendado: Use gunicorn para produção
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]