FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY flask_app/requirements.txt .

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

COPY flask_app/ .

COPY models/vectorizer.pkl ./models/vectorizer.pkl

EXPOSE 5000

# local
CMD ["python", "app.py"]

# production
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
