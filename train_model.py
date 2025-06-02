import pandas as pd
import nltk
import re
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SpanishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Para evaluación
from sklearn.metrics import classification_report, accuracy_score # Para evaluación

print("DEBUG (train_model): Iniciando train_model.py...", flush=True)


# Configuración 

DATASET_FILE = 'Reddit_Data.csv' 
MODEL_DIR = 'model'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')

print(f"DEBUG (train_model): Configuración de archivos y rutas establecida.", flush=True)


#  Configuración y Definición de Preprocesamiento de Texto 

stop_words_lang = 'english' 

try:
    stop_words = set(stopwords.words(stop_words_lang))
    print(f"DEBUG (train_model): Stopwords cargadas para el idioma '{stop_words_lang}'.", flush=True)
except LookupError:
    print(f"ERROR (train_model): No se encontraron datos de stopwords para '{stop_words_lang}'. Asegúrate de ejecutar `import nltk; nltk.download('stopwords')`.", flush=True)
    stop_words = set() 
stemmer_or_lemmatizer = None 
stemming_or_lematizing_method = lambda word: word 
print("DEBUG (train_model): Configuracion inicial de Lemmatizer/Stemmer.", flush=True)


if stop_words_lang == 'english':
    try:
        stemmer_or_lemmatizer = WordNetLemmatizer()
        stemming_or_lematizing_method = stemmer_or_lemmatizer.lemmatize 
        print("DEBUG (train_model): Lemmatizer (WordNet) configurado para preprocesamiento.", flush=True)
    except LookupError:
         print(f"ERROR (train_model): Datos para Lemmatizer ('wordnet'/'omw-1.4') no encontrados para '{stop_words_lang}'. Usando fallback simple.", flush=True)
elif stop_words_lang == 'spanish':
    try:
        stemmer_or_lemmatizer = SpanishStemmer()
        stemming_or_lematizing_method = stemmer_or_lemmatizer.stem 
        print("DEBUG (train_model): Stemmer (SpanishStemmer) configurado para preprocesamiento.", flush=True)
    except LookupError:
         print(f"ERROR (train_model): Datos o Stemmer ('spanish') no encontrado para '{stop_words_lang}'. Usando fallback simple.", flush=True)
else:
    print(f"WARNING (train_model): Idioma '{stop_words_lang}' no soporta lematización/stemming configurado. Usando fallback simple.", flush=True)
def clean_text(text):
    """
    Limpia y procesa una cadena de texto para análisis de sentimiento.
    Incluye minúsculas, eliminación de caracteres, tokenización,
    eliminación de stopwords y stemming/lematización.
    """
    if not isinstance(text, str): # Asegura que trabajamos con strings
        return ""

    text = text.lower() # Convertir todo a minúsculas
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        print("ERROR (train_model: clean_text): NLTK tokenization data ('punkt') not found. Cannot tokenize. Make sure to download NLTK data.", flush=True)
        return "" 

    processed_tokens = []
    for word in tokens:
        if word and word not in stop_words:
            if stemming_or_lematizing_method:
                processed_tokens.append(stemming_or_lematizing_method(word))
            else:
                 if re.fullmatch(r'[a-z]+', word):
                      processed_tokens.append(word)
    return ' '.join(processed_tokens) # Rejoin tokens into a single string
print("DEBUG (train_model): Configuracion y definicion de funciones de preprocesamiento completada.", flush=True)

# Cargar, Limpiar y Preparar Dataset 

print(f"DEBUG (train_model): Cargando dataset desde '{DATASET_FILE}'...", flush=True)
try:

    df = pd.read_csv(DATASET_FILE, header=None, names=['text', 'sentiment'])
    print(f"DEBUG (train_model): Dataset '{DATASET_FILE}' cargado exitosamente. Forma: {df.shape}", flush=True)
    print("DEBUG (train_model): Primeras 5 filas del dataset crudo:\n", df.head(), flush=True) 
    print("DEBUG (train_model): Limpiando y convirtiendo columna 'sentiment' a numérica...", flush=True)
    df['sentiment_cleaned_str'] = df['sentiment'].astype(str).str.strip()
    df['sentiment_numeric'] = pd.to_numeric(df['sentiment_cleaned_str'], errors='coerce')
    non_numeric_labels = df[df['sentiment_numeric'].isna()]['sentiment_cleaned_str'].unique()
    if len(non_numeric_labels) > 0:
         print(f"WARNING (train_model): Valores no numéricos encontrados en la columna 'sentiment' (se convertirán a NaN y se eliminarán): {non_numeric_labels}", flush=True)
    else:
         print("DEBUG (train_model): No se encontraron valores no numéricos inesperados en la columna 'sentiment'.", flush=True)
    df_cleaned_labels = df.dropna(subset=['sentiment_numeric']).copy()
    df_cleaned_labels['sentiment_numeric'] = df_cleaned_labels['sentiment_numeric'].astype(int)

    print(f"DEBUG (train_model): DataFrame después de limpiar etiquetas no numéricas. Forma: {df_cleaned_labels.shape}", flush=True)
    print("DEBUG (train_model): Distribución de sentimientos (numéricos INT) después de limpieza:\n", df_cleaned_labels['sentiment_numeric'].value_counts(), flush=True) 
    print("DEBUG (train_model): Tipos de datos en DataFrame después de limpieza de etiquetas:\n", df_cleaned_labels.dtypes, flush=True) 


except FileNotFoundError:
    print(f"ERROR (train_model): No se encontró el archivo del dataset en '{DATASET_FILE}'. Por favor, asegúrate de que '{DATASET_FILE}' esté en el directorio raíz del proyecto.", flush=True)
    exit()
except Exception as e:
    print(f"ERROR (train_model): Error durante la carga, limpieza o conversión inicial del dataset: {type(e).__name__}: {e}", flush=True)
    exit()

# Aplicar preprocesamiento de texto 

print("DEBUG (train_model): Aplicando preprocesamiento de texto a la columna 'text'...", flush=True)
try:
    df_cleaned_labels['cleaned_text'] = df_cleaned_labels['text'].astype(str).apply(clean_text)
    print("DEBUG (train_model): Preprocesamiento de texto completado.", flush=True)
    print("DEBUG (train_model): Primeras 5 filas con 'cleaned_text':\n", df_cleaned_labels[['text', 'cleaned_text', 'sentiment_numeric']].head(), flush=True)

except Exception as e:
    print(f"ERROR (train_model): Error durante el preprocesamiento de texto: {type(e).__name__}: {e}", flush=True)
    df_cleaned_labels['cleaned_text'] = '' 
#  Preparación Final de Datos para Entrenamiento 
print("DEBUG (train_model): Preparando datos finales para Vectorizer y Modelo...", flush=True)

df_final_train = df_cleaned_labels[df_cleaned_labels['cleaned_text'].str.strip() != ''].copy()

if df_final_train.empty:
     print("ERROR (train_model): El DataFrame final quedó vacío después de la limpieza. No hay datos suficientes con texto y etiquetas válidas para entrenar el modelo.", flush=True)
     print("Asegúrate de que el dataset tiene filas con texto y etiquetas -1, 0, 1, y que el preprocesamiento no elimina todo.", flush=True)
     exit()

if 'cleaned_text' not in df_final_train.columns or 'sentiment_numeric' not in df_final_train.columns:
    print("ERROR (train_model): Columnas 'cleaned_text' o 'sentiment_numeric' no encontradas después de la preparación final del DataFrame.", flush=True)
    exit()

print(f"DEBUG (train_model): Filas finales con datos limpios y válidos para entrenamiento: {df_final_train.shape}", flush=True)


X = df_final_train['cleaned_text']
y = df_final_train['sentiment_numeric']

print(f"DEBUG (train_model): Datos de entrenamiento (X e y) definidos. X shape: {X.shape}, y shape: {y.shape}", flush=True)
print(f"DEBUG (train_model): Distribución de etiquetas finales en 'y':\n", y.value_counts(), flush=True)


#  Feature Engineering (TF-IDF Vectorization) 
print("DEBUG (train_model): Entrenando TfidfVectorizer...", flush=True)

vectorizer = TfidfVectorizer(
    max_features=10000, 
    ngram_range=(1, 2) )


X_vectorized = vectorizer.fit_transform(X)

print("DEBUG (train_model): TfidfVectorizer entrenado y datos vectorizados.", flush=True)
print(f"DEBUG (train_model): X_vectorized shape: {X_vectorized.shape}", flush=True) 


# Entrenamiento del Modelo ML
print("DEBUG (train_model): Entrenando modelo LogisticRegression...", flush=True)

model = LogisticRegression(
    max_iter=2000,      
    solver='liblinear' )


model.fit(X_vectorized, y)

print("DEBUG (train_model): Entrenamiento del modelo principal completado.", flush=True)



print(f"DEBUG (train_model): model.classes_ después del entrenamiento: {model.classes_}", flush=True)
print(f"DEBUG (train_model): Tipo de model.classes_: {type(model.classes_)}", flush=True)

if hasattr(model.classes_, 'dtype'):
     print(f"DEBUG (train_model): Dtype de model.classes_: {model.classes_.dtype}", flush=True)


print("DEBUG (train_model): Realizando evaluación del modelo (división train/test)...", flush=True)

try:
    X_eval_train, X_eval_test, y_eval_train, y_eval_test = train_test_split(
        df_final_train['cleaned_text'],
        df_final_train['sentiment_numeric'], 
        test_size=0.2, 
        random_state=42, 
        stratify=y )
    print(f"DEBUG (train_model): Dataset dividido para evaluacion. Train: {X_eval_train.shape[0]} samples, Test: {X_eval_test.shape[0]} samples.", flush=True)
    if X_eval_train.empty or X_eval_test.empty:
        print("WARNING (train_model): Los conjuntos de train/test para evaluación están vacíos después de train_test_split. Saltando evaluación.", flush=True)
    else:

        eval_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_eval_train_vec = eval_vectorizer.fit_transform(X_eval_train)
        X_eval_test_vec = eval_vectorizer.transform(X_eval_test)
        eval_model = LogisticRegression(max_iter=2000, solver='liblinear')
        eval_model.fit(X_eval_train_vec, y_eval_train)
        y_pred = eval_model.predict(X_eval_test_vec)


        print("\n--- Resultados de Evaluación del Modelo ---", flush=True)
        print(classification_report(y_eval_test, y_pred), flush=True)
        print(f"Precisión General (Accuracy): {accuracy_score(y_eval_test, y_pred):.4f}", flush=True)
        print("------------------------------------------", flush=True)
        print(f"DEBUG (train_model): model.classes_ del modelo de evaluación: {eval_model.classes_}", flush=True)


except ValueError as ve:
     print(f"WARNING (train_model): Error al realizar train_test_split o evaluación: {type(ve).__name__}: {ve}. Saltando evaluación.", flush=True)
except Exception as e:
     print(f"ERROR (train_model): Error inesperado durante la evaluación del modelo: {type(e).__name__}: {e}. Saltando evaluación.", flush=True)


print("DEBUG (train_model): Evaluación completada (o saltada debido a errores/falta de datos).", flush=True)


# Persistir Modelo y Vectorizador Entrenados

print(f"DEBUG (train_model): Guardando vectorizador en '{VECTORIZER_PATH}'...", flush=True)
try:
    # Asegurar que el directorio 'model' existe
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"DEBUG (train_model): Directorio '{MODEL_DIR}' creado.", flush=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("DEBUG (train_model): Vectorizador guardado.", flush=True)

    print(f"DEBUG (train_model): Guardando modelo en '{MODEL_PATH}'...", flush=True)
    joblib.dump(model, MODEL_PATH)
    print("DEBUG (train_model): Modelo guardado.", flush=True)

    print("DEBUG (train_model): Persistencia completada con éxito.", flush=True)

except Exception as e:
    print(f"ERROR (train_model): Error durante la persistencia de archivos: {type(e).__name__}: {e}. Los archivos .pkl podrían no haberse guardado.", flush=True)


print("DEBUG (train_model): Fin del script train_model.py", flush=True)
