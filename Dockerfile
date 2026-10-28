FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.3.5 pandas==2.3.3 scikit-learn==1.7.2 catboost==1.2.10 statsmodels==0.14.5 shap==0.53.0 ngboost==0.5.11 interpret-core==0.7.8 fastapi==0.143.0 uvicorn==0.54.0
COPY paddy_helpers.py api.py paddy_model.pkl ./
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
