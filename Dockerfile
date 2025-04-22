FROM python:3.12-slim

RUN apt update && apt install -y git && apt-get clean 
WORKDIR /app
RUN git clone https://github.com/Aik0o1/chat-bot-Registro.git .
RUN pip install --upgrade pip && pip install --default-timeout=1000 -r requirements.txt

CMD ["python3", "app.py"]