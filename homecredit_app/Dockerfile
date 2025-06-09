FROM python:3.11

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./

ENV PORT=8080
EXPOSE $PORT

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app