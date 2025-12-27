# 🎬 IMDb Sentiment Analysis

Análisis de sentimiento de reseñas de películas del dataset **IMDb**, utilizando técnicas de **Procesamiento de Lenguaje Natural (NLP)**, **Machine Learning** y **Deep Learning**, con un enfoque comparativo sobre distintas representaciones del texto.

---

## 📌 Descripción del proyecto

Este proyecto tiene como objetivo desarrollar y evaluar distintos modelos de **clasificación binaria de sentimiento** (positivo / negativo) aplicados a reseñas de películas del dataset IMDb.

Se analizan y comparan enfoques basados en:
- Representaciones clásicas del texto (Bag of Words, TF-IDF)
- Embeddings distribuidos a nivel documento
- Modelos de Machine Learning tradicionales
- Modelos de Deep Learning implementados con **Keras**

El foco principal del trabajo es **comparar el impacto de la representación del texto y la complejidad del modelo sobre el desempeño final**, priorizando un análisis riguroso y controlado.

---

## 🎯 Objetivo general

Desarrollar y evaluar modelos de Machine Learning y Deep Learning capaces de predecir el sentimiento (positivo o negativo) de reseñas de películas del dataset IMDb, utilizando técnicas de procesamiento de texto.

---

## 🎯 Objetivos específicos

- Analizar exploratoriamente el dataset de reseñas de IMDb.
- Aplicar técnicas de preprocesamiento de texto mediante NLP.
- Comparar distintos métodos de vectorización del texto.
- Entrenar modelos clásicos de Machine Learning para clasificación de sentimiento.
- Implementar modelos de Deep Learning con Keras (ANN / MLP) utilizando embeddings.
- Evaluar y comparar el desempeño de los distintos enfoques.

---

## 🧪 Enfoque metodológico

El proyecto sigue un pipeline típico de **Ciencia de Datos aplicada a texto**, que incluye:

1. **Análisis exploratorio de los datos** (EDA)
   - Distribución de clases
   - Longitud de reseñas (caracteres y palabras)
   - Nubes de palabras
   - Análisis de tokens más frecuentes

2. **Preprocesamiento lingüístico del texto**
   - Limpieza y normalización
   - Tokenización
   - Eliminación de stopwords
   - Lematización con spaCy
   - Manejo de negaciones y expresiones emocionales

3. **Vectorización del contenido textual**
   - Bag of Words (BoW) con n-gramas
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Word2Vec embeddings (entrenados sobre el corpus)
   - Representaciones a nivel documento (promedio de embeddings)

4. **Entrenamiento de modelos de Machine Learning**
   - Regresión Logística
   - Random Forest
   - MLPClassifier

5. **Implementación de modelos de Deep Learning**
   - Red Neuronal Artificial (ANN) con Keras/TensorFlow
   - Entrenamiento sobre embeddings de documentos

6. **Evaluación y análisis comparativo de resultados**
   - Métricas de clasificación (Accuracy, Precision, Recall, F1-score)
   - Comparación entre distintos enfoques
   - Análisis de curvas de entrenamiento

El análisis avanza desde métodos basados en conteo hasta embeddings entrenados y redes neuronales, evaluando el equilibrio entre complejidad del modelo y ganancia real en performance.

---

## 🤖 Modelos implementados

### 🔹 Machine Learning
- **Regresión Logística** sobre TF-IDF y Bag of Words
- **Random Forest** sobre TF-IDF
- **MLPClassifier** (scikit-learn) sobre embeddings de documentos
- Comparación de n-gramas (unigrams, bigrams)

### 🔹 Deep Learning (Keras)
- Red Neuronal Artificial Multicapa (ANN)
- Arquitectura deliberadamente simple:
  - Una única capa densa oculta
  - Regularización mediante Dropout
  - Función de activación ReLU
  - Capa de salida sigmoide para clasificación binaria

---

## 🧠 Justificación del diseño de la ANN

La arquitectura utilizada corresponde a una **red neuronal artificial multicapa (ANN) con una única capa densa oculta**.

Esta elección responde a un diseño deliberadamente simple, con el objetivo de:
- Evaluar el aporte del enfoque de Deep Learning sobre embeddings.
- Evitar introducir complejidad innecesaria en la arquitectura.
- Facilitar la comparación directa con modelos más simples.

No se utilizaron arquitecturas más profundas debido a que:
- El tamaño y la naturaleza del dataset no justifican redes profundas.
- No se observaron mejoras sustanciales frente a modelos más simples.
- Arquitecturas más complejas incrementan el riesgo de sobreajuste y el costo computacional.

---

## 📊 Métricas de evaluación

Los modelos fueron evaluados utilizando:
- Accuracy
- Precision
- Recall
- F1-score
- Análisis de curvas de entrenamiento (loss y accuracy)
- Comparación entre métricas de entrenamiento y validación

---

## 📁 Estructura del repositorio

```
IMDb_Sentiment_Analysis/
│
├── IMDb_Sentiment_Analysis_VanesaMizrahi.ipynb  # Notebook principal con el análisis completo
├── imdb_sentiment_analysis_vanesamizrahi.py      # Script Python exportado del notebook
├── README.md                                      # Este archivo
└── .gitignore                                     # Archivos ignorados por Git
```


---

## 🛠️ Tecnologías utilizadas

### Lenguaje y Librerías Principales
- **Python 3.x**
- **pandas** - Manipulación y análisis de datos
- **numpy** - Cálculos numéricos
- **scikit-learn** - Machine Learning (Regresión Logística, Random Forest, MLPClassifier)
- **TensorFlow / Keras** - Deep Learning
- **gensim** - Word2Vec para embeddings
- **spaCy** - Procesamiento avanzado de lenguaje natural (lematización, tokenización)

### Visualización y Utilidades
- **matplotlib** - Visualización de datos
- **seaborn** - Visualización estadística
- **wordcloud** - Nubes de palabras
- **tqdm** - Barras de progreso

### Otras herramientas
- **kagglehub** - Descarga del dataset desde Kaggle
- **Jupyter Notebook** / **Google Colab** - Entorno de desarrollo

---

## 📦 Dataset

El proyecto utiliza el **IMDb Dataset de 50K Movie Reviews** disponible en Kaggle:

- **Fuente**: [Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)
- **Tamaño**: 50,000 reseñas (25,000 positivas y 25,000 negativas)
- **Formato**: CSV con columnas `review` y `sentiment`
- **Descarga**: El código incluye la descarga automática mediante `kagglehub`

---

## 🚀 Instalación y Uso

### Requisitos previos

- Python 3.7 o superior
- Acceso a Kaggle (para descargar el dataset)

### Instalación de dependencias

```bash
pip install pandas numpy scikit-learn tensorflow gensim spacy matplotlib seaborn wordcloud tqdm kagglehub
```

### Descarga del modelo de spaCy

Para el preprocesamiento de texto, se requiere el modelo de inglés de spaCy:

```bash
python -m spacy download en_core_web_sm
```

### Ejecución

El proyecto está disponible en dos formatos:

1. **Notebook Jupyter**: `IMDb_Sentiment_Analysis_VanesaMizrahi.ipynb`
   - Abrir en Jupyter Notebook o Google Colab
   - Ejecutar las celdas en orden

2. **Script Python**: `imdb_sentiment_analysis_vanesamizrahi.py`
   - Ejecutar directamente: `python imdb_sentiment_analysis_vanesamizrahi.py`

### Nota sobre Google Colab

El proyecto incluye código específico para Google Colab. Si ejecutas localmente, puedes comentar o adaptar las secciones relacionadas con `google.colab`.

---

## 📊 Resultados destacados

Los modelos fueron evaluados utilizando múltiples métricas (Accuracy, Precision, Recall, F1-score). Los principales hallazgos incluyen:

- **Regresión Logística + TF-IDF/Bag of Words**: Desempeño sólido con F1-score cercano a 0.90
- **Modelos basados en embeddings**: Representaciones más compactas y semánticamente ricas
- **Deep Learning (Keras)**: Flexibilidad adicional con mejoras moderadas en métricas globales

---

## 📌 Conclusiones generales

Los resultados muestran que, para este dataset en particular, los modelos clásicos bien ajustados sobre representaciones simples del texto alcanzan desempeños competitivos frente a enfoques de Deep Learning.

La implementación de Deep Learning con Keras aporta mayor flexibilidad y control sobre el proceso de entrenamiento, aunque las mejoras en métricas globales resultan moderadas. Esto refuerza la importancia de evaluar cuidadosamente el trade-off entre complejidad del modelo y ganancia real en performance.

---

## 📄 Licencia
El proyecto está disponible bajo la licencia MIT, permitiendo su uso libre para fines personales, académicos o experimentales.
Para más detalles, consulta el archivo LICENSE.

---

## ✋ About Me

Soy **Vanesa Mizrahi**, desarrolladora de software iOS y **Data Scientist en formación**, con foco en el análisis y modelado de datos aplicados a problemas reales.

Durante la Diplomatura en Data Science profundicé en el uso de **Python, SQL, Machine Learning y técnicas de Deep Learning**, abordando proyectos que integran análisis exploratorio, procesamiento de datos, modelado predictivo y evaluación comparativa de enfoques.

Mis principales áreas de interés incluyen:
- Análisis exploratorio de datos y visualización
- Procesamiento de Lenguaje Natural (NLP)
- Modelos de Machine Learning supervisados
- Introducción a Deep Learning aplicado a datos reales
- Desarrollo de soluciones analíticas con criterio metodológico y enfoque práctico

---

## 🎓 Propósito Educativo

Este repositorio forma parte del trabajo desarrollado en el marco del curso **Data Science III: NLP & Deep Learning aplicado a Ciencia de Datos**, donde se aplican técnicas de NLP, modelos clásicos y redes neuronales para el análisis de sentimiento sobre texto, el cual forma parte de mi especialización a través de la Diplomatura en Data Science [CoderHouse](https://www.coderhouse.com/ar/diplomaturas/data/?pipe_source=google&pipe_medium=cpc&pipe_campaign=1&gad_source=1&gad_campaignid=13952864596&gbraid=0AAAAACoxfTL7S4LjLGDCtBrigIZUvaOtI&gclid=CjwKCAiAxc_JBhA2EiwAFVs7XJlquLs6YOrHV_5FBSUgw11RG-8BGH6YVHXJN2QfehgVqOBGVghiqxoCOQsQAvD_BwE).



- 🌐 **GitHub**: [@vanerm](https://github.com/vanerm)  
- 💼 **LinkedIn**: [vanesamizrahi](https://www.linkedin.com/in/vanesamizrahi)  
- 📓 **Notebook en Google Colab**: [Ver notebook](https://colab.research.google.com/drive/1G_0RDVRkqttwNkXLlOUIeJdVHQyq25_w?usp=sharing)

---

## 🙏 Agradecimientos

- **Kaggle** por proporcionar el dataset de reseñas de IMDb
- Comunidad de código abierto por las librerías utilizadas (scikit-learn, TensorFlow, gensim, spaCy, entre otras)

---

## 📝 Notas adicionales

- El código está optimizado para ejecutarse en **Google Colab**, pero puede adaptarse fácilmente para ejecución local
- Se recomienda tener al menos 8GB de RAM para procesar el dataset completo
- El entrenamiento de modelos puede tomar varios minutos dependiendo del hardware disponible
- Para una experiencia interactiva completa, se recomienda usar el notebook en lugar del script Python

---

