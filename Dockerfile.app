# The Streamlit front end (app.py) as its own container, next to the API in Dockerfile
FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.3.5 pandas==2.3.3 scikit-learn==1.7.2 catboost==1.2.10 statsmodels==0.14.5 shap==0.53.0 ngboost==0.5.11 interpret-core==0.7.8 streamlit==1.51.0 plotly==6.3.0 requests==2.32.5
COPY paddy_helpers.py app.py paddy_model.pkl holdout_sample.csv model_results.csv ./
COPY ["Raw Data/district_map.geojson", "./Raw Data/"]
COPY .streamlit/config.toml ./.streamlit/
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
