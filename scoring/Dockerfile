FROM python:3.9.3-slim

WORKDIR /app

COPY ["dict_vectorizer.pkl", "imputer.pkl", "XGBoost_2023-11-05_14-54-16.bin", "requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
