# Detector de Blockbusters

Aplicación educativa de Streamlit que usa una red neuronal multicapa (MLP) para estimar si una película será blockbuster. El entrenamiento descarga el dataset público desde Kaggle; también se puede indicar una copia local.

## Preparar y ejecutar

```powershell
uv sync --extra training
uv run python train.py
uv run streamlit run app.py
```

Sin argumentos, `train.py` descarga automáticamente el dataset público `vedauv/what-makes-a-movie-a-blockbuster-5000-films` desde Kaggle mediante KaggleHub, evalúa el modelo con una partición de prueba y genera `modelo.joblib`. También acepta una ruta local:

```powershell
uv run python train.py --dataset-root "dataset"
```

La app carga el artefacto y no vuelve a entrenar durante el arranque.

## Modelo

Las variables de entrada son género, presupuesto, duración y año (este dataset no incluye sinopsis). El género se codifica con one-hot encoding y las variables numéricas se escalan antes de entrar a una MLP con capas ocultas de 64 y 32 neuronas. `train.py` compara en la misma partición estratificada cuatro variantes: MLP base con Cross-Entropy; Focal Loss (`gamma=2`) con pesos balanceados y AdamW; Dropout (`p=0.35`) con Label Smoothing (`0.1`) y AdamW; y la MLP base entrenada con augmentation tabular. Esta última añade dos copias de cada fila de entrenamiento con ruido gaussiano moderado en duración (σ=2 minutos) y año (σ=1 año), sin alterar género ni presupuesto, que es una variable especialmente relacionada con la etiqueta. Se evalúa tanto en la partición limpia como en cinco versiones perturbadas de esa partición. La app despliega la variante con augmentation e informa Macro F1 limpio y perturbado. La importancia de variables se estima por permutación sobre macro F1.

Se excluyen recaudación, beneficio, ROI, valoración, votos y popularidad para evitar filtrar la respuesta o usar información posterior al estreno. El objetivo `blockbuster` es 1 si la recaudación supera 3 veces el presupuesto y los 100 millones de dólares. Como no hay sinopsis ni estructura secuencial de texto, embeddings de lenguaje y Global Max Pooling semántico no se pueden evaluar con este dataset. Las etiquetas son binarias y sintéticas, así que Label Smoothing se compara como regularización, no como una solución demostrada para películas multi-género. El ruido en duración/año es un análogo para robustez tabular; no es una transformación espacial ni una representación de que estos metadatos suelan venir mal medidos en películas reales.

PyTorch es una dependencia opcional usada al entrenar las dos variantes adaptadas. Para ejecutar la app o desplegar el artefacto basta el entorno base; para volver a entrenar, instala el extra `training` como en los comandos anteriores.

## Render

Para desplegar la app solo necesitas incluir `app.py`, `modelo.joblib` y los archivos de dependencias/configuración (`pyproject.toml`, `uv.lock` y `render.yaml`). El CSV no hace falta para servir predicciones: la app carga el modelo ya entrenado. Configura Render para usar `render.yaml`; el build instala el entorno bloqueado por uv y el start ejecuta `app.py`. `requirements.txt` incluye las dependencias necesarias para instalaciones con pip.

Incluye `model.py` si quieres poder ejecutar `train.py`; el CSV se descarga de Kaggle automáticamente o puedes pasar uno local con `--dataset-root`. No hace falta subir la carpeta local `dataset/` a GitHub. No ignores `modelo.joblib`: sí es necesario para desplegar. El modelo se guarda con versiones fijadas de scikit-learn y joblib para mantener compatibilidad.

## Limitación

Los datos y títulos son sintéticos; el modelo sirve para practicar y no predice de forma fiable la taquilla real.
