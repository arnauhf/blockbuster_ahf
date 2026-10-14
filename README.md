# Detector de Blockbusters

Aplicación educativa de Streamlit que usa un Random Forest para estimar si una película será blockbuster. El entrenamiento descarga el dataset público desde Kaggle; también se puede indicar una copia local.

## Preparar y ejecutar

```powershell
uv sync
uv run python train.py
uv run streamlit run app.py
```

Sin argumentos, `train.py` descarga automáticamente el dataset público `vedauv/what-makes-a-movie-a-blockbuster-5000-films` desde Kaggle mediante KaggleHub, evalúa el modelo con una partición de prueba y genera `modelo.joblib`. También acepta una ruta local:

```powershell
uv run python train.py --dataset-root "dataset"
```

La app carga el artefacto y no vuelve a entrenar durante el arranque.

## Modelo

Las variables de entrada son género, presupuesto, duración y año. Se excluyen recaudación, beneficio, ROI, valoración, votos y popularidad para evitar filtrar la respuesta o usar información posterior al estreno. El objetivo `blockbuster` es 1 si la recaudación supera 3 veces el presupuesto y los 100 millones de dólares.

## Render

Para desplegar la app solo necesitas incluir `app.py`, `modelo.joblib` y los archivos de dependencias/configuración (`pyproject.toml`, `uv.lock` y `render.yaml`). El CSV no hace falta para servir predicciones: la app carga el modelo ya entrenado. Configura Render para usar `render.yaml`; el build instala el entorno bloqueado por uv y el start ejecuta `app.py`. `requirements.txt` incluye las dependencias necesarias para instalaciones con pip.

Incluye `model.py` si quieres poder ejecutar `train.py`; el CSV se descarga de Kaggle automáticamente o puedes pasar uno local con `--dataset-root`. No hace falta subir la carpeta local `dataset/` a GitHub. No ignores `modelo.joblib`: sí es necesario para desplegar. El modelo se guarda con versiones fijadas de scikit-learn y joblib para mantener compatibilidad.

## Limitación

Los datos y títulos son sintéticos; el modelo sirve para practicar y no predice de forma fiable la taquilla real.
