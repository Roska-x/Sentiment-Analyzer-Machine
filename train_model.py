# sentiment_analyzer/train_model.py

# Importar las librerías necesarias
import pandas as pd
import nltk
import re
import joblib
import os

# Importaciones específicas de NLTK para preprocesamiento
from nltk.corpus import stopwords
# Lemmatizer común para INGLÉS (puede necesitar datos 'wordnet')
from nltk.stem import WordNetLemmatizer
# Stemmer para ESPAÑOL (puede necesitar datos 'spanish')
from nltk.stem.snowball import SpanishStemmer

# Librerías para ML (scikit-learn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Para evaluación
from sklearn.metrics import classification_report, accuracy_score # Para evaluación


# Añadir mensajes de depuración al inicio para rastrear la ejecución del script de entrenamiento
print("DEBUG (train_model): Iniciando train_model.py...", flush=True)


# --- Configuración ---
# Nombre del archivo del dataset CSV. Debe estar en el mismo directorio que este script.
DATASET_FILE = 'Reddit_Data.csv' # <--- Asegura que este nombre coincide exactamente con tu archivo
# Directorio donde se guardarán el modelo y el vectorizador entrenados
MODEL_DIR = 'model'
# Rutas completas a los archivos serializados (donde se guardarán)
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')

print(f"DEBUG (train_model): Configuración de archivos y rutas establecida.", flush=True)


# --- Configuración y Definición de Preprocesamiento de Texto ---
# Esta sección se mueve AQUÍ, ANTES de la carga y limpieza del dataset,
# para que la función `clean_text` y sus dependencias estén definidas
# cuando se utilicen más adelante en el script.

# Configura el idioma para stopwords, Stemmer/Lemmatizer.
# DEBE COINCIDIR CON EL IDIOMA PREDOMINANTE EN TU DATASET.
# Si el texto del dataset es principalmente en inglés, usa 'english'.
# Si es español, usa 'spanish'.
stop_words_lang = 'english' # <--- **AJUSTA ESTO AL IDIOMA DE TU DATASET**

# Obtener las stopwords para el idioma configurado
try:
    # nltk.download('stopwords') si no tienes los datos
    stop_words = set(stopwords.words(stop_words_lang))
    print(f"DEBUG (train_model): Stopwords cargadas para el idioma '{stop_words_lang}'.", flush=True)
except LookupError:
    print(f"ERROR (train_model): No se encontraron datos de stopwords para '{stop_words_lang}'. Asegúrate de ejecutar `import nltk; nltk.download('stopwords')`.", flush=True)
    stop_words = set() # Usar conjunto vacío como fallback


# Configurar el Lemmatizer o Stemmer basado en el idioma.
# Si stop_words_lang es 'english': usar WordNetLemmatizer (requiere datos 'wordnet').
# Si stop_words_lang es 'spanish': usar SpanishStemmer (más común y simple para español en NLTK).
# nltk.download('wordnet') o nltk.download('spanish') o nltk.download('omw-1.4') si es necesario.
stemmer_or_lemmatizer = None # Inicializamos
stemming_or_lematizing_method = lambda word: word # Fallback simple (función identidad)
print("DEBUG (train_model): Configuracion inicial de Lemmatizer/Stemmer.", flush=True)


if stop_words_lang == 'english':
    try:
        # nltk.download('wordnet') y nltk.download('omw-1.4') si no tienes los datos
        stemmer_or_lemmatizer = WordNetLemmatizer()
        stemming_or_lematizing_method = stemmer_or_lemmatizer.lemmatize # Método a usar
        print("DEBUG (train_model): Lemmatizer (WordNet) configurado para preprocesamiento.", flush=True)
    except LookupError:
         print(f"ERROR (train_model): Datos para Lemmatizer ('wordnet'/'omw-1.4') no encontrados para '{stop_words_lang}'. Usando fallback simple.", flush=True)
elif stop_words_lang == 'spanish':
    try:
        # nltk.download('spanish') si no tienes los datos necesarios para SpanishStemmer
        stemmer_or_lemmatizer = SpanishStemmer()
        stemming_or_lematizing_method = stemmer_or_lemmatizer.stem # Método a usar
        print("DEBUG (train_model): Stemmer (SpanishStemmer) configurado para preprocesamiento.", flush=True)
    except LookupError:
         print(f"ERROR (train_model): Datos o Stemmer ('spanish') no encontrado para '{stop_words_lang}'. Usando fallback simple.", flush=True)
else:
    print(f"WARNING (train_model): Idioma '{stop_words_lang}' no soporta lematización/stemming configurado. Usando fallback simple.", flush=True)


# Definición de la función de preprocesamiento del texto.
# Esta función se llama más adelante. Ahora está definida antes de su uso.
# ¡EL CONTENIDO DE ESTA FUNCIÓN DEBE COINCIDIR EXACTAMENTE con la función clean_text en app.py!
def clean_text(text):
    """
    Limpia y procesa una cadena de texto para análisis de sentimiento.
    Incluye minúsculas, eliminación de caracteres, tokenización,
    eliminación de stopwords y stemming/lematización.
    """
    if not isinstance(text, str): # Asegura que trabajamos con strings
        return ""

    text = text.lower() # Convertir todo a minúsculas

    # Eliminar puntuación y números.
    # AJUSTA ESTA EXPRESION REGULAR si quieres mantener números u otros caracteres.
    # DEBE COINCIDIR CON LA REGEX EN LA FUNCIÓN clean_text EN app.py.
    text = re.sub(r'[^a-z\s]', '', text) # Mantiene solo letras a-z y espacios

    # Tokenización (dividir el texto en palabras)
    try:
        # nltk.download('punkt') si no tienes los datos
        tokens = nltk.word_tokenize(text)
    except LookupError:
        print("ERROR (train_model: clean_text): NLTK tokenization data ('punkt') not found. Cannot tokenize. Make sure to download NLTK data.", flush=True)
        return "" # Si no se puede tokenizar, devuelve cadena vacía

    # Eliminar stopwords y aplicar stemming/lematización
    processed_tokens = []
    for word in tokens:
        # Asegura que la palabra no esté vacía y que no sea una stopword del idioma configurado
        if word and word not in stop_words:
            # Aplica el método de stemming/lematización si está configurado
            if stemming_or_lematizing_method:
                # Llama al método configurado (lemmatize o stem)
                processed_tokens.append(stemming_or_lematizing_method(word))
            else:
                 # Fallback simple si el stemmer/lemmatizer no se configuró correctamente
                 # Mantiene la palabra original si solo contiene caracteres alfabéticos a-z
                 if re.fullmatch(r'[a-z]+', word):
                      processed_tokens.append(word)

    return ' '.join(processed_tokens) # Rejoin tokens into a single string


print("DEBUG (train_model): Configuracion y definicion de funciones de preprocesamiento completada.", flush=True)


# --- Cargar, Limpiar y Preparar Dataset para Entrenamiento ---
# Este bloque ahora está DEPUÉS de la configuración y definicion de preprocesamiento.

print(f"DEBUG (train_model): Cargando dataset desde '{DATASET_FILE}'...", flush=True)
try:
    # Cargamos el CSV. Asumimos 2 columnas SIN encabezado (common en datasets raw).
    # Columna 0 -> texto, Columna 1 -> sentimiento (-1, 0, 1)
    # Ajusta 'header=None, names=['text', 'sentiment']' si tu CSV tiene un formato diferente.
    df = pd.read_csv(DATASET_FILE, header=None, names=['text', 'sentiment'])
    print(f"DEBUG (train_model): Dataset '{DATASET_FILE}' cargado exitosamente. Forma: {df.shape}", flush=True)
    print("DEBUG (train_model): Primeras 5 filas del dataset crudo:\n", df.head(), flush=True) # Mostrar head del crudo

    # --- Limpieza y Conversión de ETIQUETAS DE SENTIMIENTO ---
    print("DEBUG (train_model): Limpiando y convirtiendo columna 'sentiment' a numérica...", flush=True)

    # 1. Convertir la columna 'sentiment' a string y quitar espacios.
    df['sentiment_cleaned_str'] = df['sentiment'].astype(str).str.strip()

    # 2. Convertir la columna string limpia a numérico. Los errores se convierten a NaN.
    # Esto convierte '1', '0', '-1' a números y 'category', u otra basura, a NaN.
    df['sentiment_numeric'] = pd.to_numeric(df['sentiment_cleaned_str'], errors='coerce')

    # 3. Identificar y mostrar valores que no se pudieron convertir.
    non_numeric_labels = df[df['sentiment_numeric'].isna()]['sentiment_cleaned_str'].unique()
    if len(non_numeric_labels) > 0:
         print(f"WARNING (train_model): Valores no numéricos encontrados en la columna 'sentiment' (se convertirán a NaN y se eliminarán): {non_numeric_labels}", flush=True)
    else:
         print("DEBUG (train_model): No se encontraron valores no numéricos inesperados en la columna 'sentiment'.", flush=True)


    # 4. Eliminar filas donde la columna 'sentiment_numeric' es NaN (porque la conversión falló).
    # Creamos una copia del DataFrame resultante para evitar SettingWithCopyWarning.
    df_cleaned_labels = df.dropna(subset=['sentiment_numeric']).copy()

    # 5. Convertir la columna numérica a tipo entero explícitamente (-1, 0, 1 son enteros).
    # Esto es crucial para asegurar que model.classes_ contenga INTs puros.
    df_cleaned_labels['sentiment_numeric'] = df_cleaned_labels['sentiment_numeric'].astype(int)

    print(f"DEBUG (train_model): DataFrame después de limpiar etiquetas no numéricas. Forma: {df_cleaned_labels.shape}", flush=True)
    print("DEBUG (train_model): Distribución de sentimientos (numéricos INT) después de limpieza:\n", df_cleaned_labels['sentiment_numeric'].value_counts(), flush=True) # Debería mostrar solo -1, 0, 1 INT
    print("DEBUG (train_model): Tipos de datos en DataFrame después de limpieza de etiquetas:\n", df_cleaned_labels.dtypes, flush=True) # Debería mostrar 'int64' para sentiment_numeric


except FileNotFoundError:
    print(f"ERROR (train_model): No se encontró el archivo del dataset en '{DATASET_FILE}'. Por favor, asegúrate de que '{DATASET_FILE}' esté en el directorio raíz del proyecto.", flush=True)
    exit()
except Exception as e:
    print(f"ERROR (train_model): Error durante la carga, limpieza o conversión inicial del dataset: {type(e).__name__}: {e}", flush=True)
    exit()

# --- Aplicar preprocesamiento de texto (AHORA ESTA LLAMADA FUNCIONARÁ) ---
# Aplicamos la función clean_text a la columna 'text' del DataFrame con etiquetas limpias.
# Aseguramos que la columna 'text' es string antes de aplicar, para evitar errores.
print("DEBUG (train_model): Aplicando preprocesamiento de texto a la columna 'text'...", flush=True)
try:
    df_cleaned_labels['cleaned_text'] = df_cleaned_labels['text'].astype(str).apply(clean_text)
    print("DEBUG (train_model): Preprocesamiento de texto completado.", flush=True)

    # Mostrar algunas filas con texto limpio para verificar
    print("DEBUG (train_model): Primeras 5 filas con 'cleaned_text':\n", df_cleaned_labels[['text', 'cleaned_text', 'sentiment_numeric']].head(), flush=True)

except Exception as e:
    print(f"ERROR (train_model): Error durante el preprocesamiento de texto: {type(e).__name__}: {e}", flush=True)
    # Este error puede ser por NLTK data faltante, o un problema en clean_text.
    # Continuaremos para intentar dar un mensaje más específico en el siguiente paso si no hay cleaned_text.
    df_cleaned_labels['cleaned_text'] = '' # Añadir columna vacía para evitar errores posteriores si clean_text falló catastróficamente

# --- Preparación Final de Datos para Entrenamiento (Filtrar texto vacío y validar) ---
print("DEBUG (train_model): Preparando datos finales para Vectorizer y Modelo...", flush=True)

# Filtrar filas donde el texto limpio quedó vacío después del preprocesamiento
df_final_train = df_cleaned_labels[df_cleaned_labels['cleaned_text'].str.strip() != ''].copy()

# Validar que todavía tenemos datos para entrenar después de toda la limpieza
if df_final_train.empty:
     print("ERROR (train_model): El DataFrame final quedó vacío después de la limpieza. No hay datos suficientes con texto y etiquetas válidas para entrenar el modelo.", flush=True)
     print("Asegúrate de que el dataset tiene filas con texto y etiquetas -1, 0, 1, y que el preprocesamiento no elimina todo.", flush=True)
     exit()

# Asegurar que las columnas necesarias existen antes de asignarlas a X e y
if 'cleaned_text' not in df_final_train.columns or 'sentiment_numeric' not in df_final_train.columns:
    print("ERROR (train_model): Columnas 'cleaned_text' o 'sentiment_numeric' no encontradas después de la preparación final del DataFrame.", flush=True)
    exit()

print(f"DEBUG (train_model): Filas finales con datos limpios y válidos para entrenamiento: {df_final_train.shape}", flush=True)


# Definir X (features) como la columna de texto limpio y preprocesado
X = df_final_train['cleaned_text']
# Definir y (target/etiquetas) como la columna de sentimiento numérico INT limpia
y = df_final_train['sentiment_numeric']

print(f"DEBUG (train_model): Datos de entrenamiento (X e y) definidos. X shape: {X.shape}, y shape: {y.shape}", flush=True)
# Reconfirmar la distribución de etiquetas en y antes de entrenar
print(f"DEBUG (train_model): Distribución de etiquetas finales en 'y':\n", y.value_counts(), flush=True)


# --- Feature Engineering (TF-IDF Vectorization) ---
print("DEBUG (train_model): Entrenando TfidfVectorizer...", flush=True)
# Entrenar el vectorizador TF-IDF en los datos de texto limpios.
# Configura max_features para limitar el tamaño del vocabulario (importante para rendimiento y RAM).
# Configura ngram_range para incluir unigramas (palabras individuales) y bigramas (pares de palabras).
vectorizer = TfidfVectorizer(
    max_features=10000, # Limita el número de palabras/bigramas considerados
    ngram_range=(1, 2) # Incluye palabras sueltas y pares de palabras
)

# fit_transform aprende el vocabulario y las frecuencias Y transforma el texto a vectores
X_vectorized = vectorizer.fit_transform(X)

print("DEBUG (train_model): TfidfVectorizer entrenado y datos vectorizados.", flush=True)
print(f"DEBUG (train_model): X_vectorized shape: {X_vectorized.shape}", flush=True) # (num_samples, max_features)


# --- Entrenamiento del Modelo ML ---
print("DEBUG (train_model): Entrenando modelo LogisticRegression...", flush=True)
# Inicializar y entrenar el modelo. LogisticRegression maneja múltiples clases (-1, 0, 1) por defecto.
model = LogisticRegression(
    max_iter=2000,      # Aumenta si obtienes warnings sobre no convergencia
    solver='liblinear'  # Solver eficiente para datasets de tamaño medio/grande
)

# Entrenar el modelo con los datos vectorizados y las etiquetas numéricas
model.fit(X_vectorized, y)

print("DEBUG (train_model): Entrenamiento del modelo principal completado.", flush=True)

# IMPRIMIR model.classes_ DESPUÉS DEL ENTRENAMIENTO
# Esto te mostrará los valores EXACTOS que el modelo aprendió como clases.
print(f"DEBUG (train_model): model.classes_ después del entrenamiento: {model.classes_}", flush=True)
print(f"DEBUG (train_model): Tipo de model.classes_: {type(model.classes_)}", flush=True)
# Si model.classes_ es un array NumPy, mostrar su dtype (debería ser int64)
if hasattr(model.classes_, 'dtype'):
     print(f"DEBUG (train_model): Dtype de model.classes_: {model.classes_.dtype}", flush=True)


# --- Evaluar el Modelo (Opcional pero RECOMENDADO para saber qué tan bueno es) ---
print("DEBUG (train_model): Realizando evaluación del modelo (división train/test)...", flush=True)

# Divide el dataset final (texto limpio + etiquetas numéricas) para train/test split de EVALUACIÓN.
# OJO: Esto NO afecta al modelo que se guarda, ese se entrena con TODOS los datos.
# stratify=y ayuda a asegurar que la proporción de -1, 0, 1 sea similar en train y test.
# Si value_counts() mostró alguna clase con solo 1 ejemplo, quita `stratify=y` aquí.
try:
    X_eval_train, X_eval_test, y_eval_train, y_eval_test = train_test_split(
        df_final_train['cleaned_text'],
        df_final_train['sentiment_numeric'], # Usar la columna numérica limpia
        test_size=0.2, # 20% de los datos para prueba
        random_state=42, # Para resultados reproducibles
        stratify=y # Usa la 'y' numérica final para estratificar
        # Si el stratify=y de arriba causa un ValueError, cambialo a:
        # stratify=None # No estratificar
    )
    print(f"DEBUG (train_model): Dataset dividido para evaluacion. Train: {X_eval_train.shape[0]} samples, Test: {X_eval_test.shape[0]} samples.", flush=True)

    # Asegúrate de que los conjuntos de división no están vacíos (especialmente si el dataset es muy pequeño)
    if X_eval_train.empty or X_eval_test.empty:
        print("WARNING (train_model): Los conjuntos de train/test para evaluación están vacíos después de train_test_split. Saltando evaluación.", flush=True)
    else:
        # Entrena UN NUEVO vectorizador SOLO con los datos de entrenamiento del split.
        eval_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_eval_train_vec = eval_vectorizer.fit_transform(X_eval_train)
        # Transforma (no fit_transform) los datos de prueba usando el vectorizador entrenado solo en train.
        X_eval_test_vec = eval_vectorizer.transform(X_eval_test)

        # Entrena UN NUEVO modelo SOLO con los datos de entrenamiento del split.
        eval_model = LogisticRegression(max_iter=2000, solver='liblinear')
        eval_model.fit(X_eval_train_vec, y_eval_train)

        # Hacer predicciones en el conjunto de prueba
        y_pred = eval_model.predict(X_eval_test_vec)

        # Imprimir reporte de clasificación (precisión, recall, f1-score por clase) y precisión general.
        print("\n--- Resultados de Evaluación del Modelo ---", flush=True)
        print(classification_report(y_eval_test, y_pred), flush=True)
        print(f"Precisión General (Accuracy): {accuracy_score(y_eval_test, y_pred):.4f}", flush=True)
        print("------------------------------------------", flush=True)
        # Imprimir las clases del modelo de evaluación (deberían ser las mismas que el modelo principal)
        print(f"DEBUG (train_model): model.classes_ del modelo de evaluación: {eval_model.classes_}", flush=True)


except ValueError as ve:
    # Capturar específicamente errores de train_test_split (como el stratify)
     print(f"WARNING (train_model): Error al realizar train_test_split o evaluación: {type(ve).__name__}: {ve}. Saltando evaluación.", flush=True)
except Exception as e:
     # Capturar cualquier otro error durante la evaluación
     print(f"ERROR (train_model): Error inesperado durante la evaluación del modelo: {type(e).__name__}: {e}. Saltando evaluación.", flush=True)


print("DEBUG (train_model): Evaluación completada (o saltada debido a errores/falta de datos).", flush=True)


# --- Persistir Modelo y Vectorizador Entrenados ---
# Guardar el vectorizador y el modelo a archivos .pkl para usarlos en la aplicación Flask.
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