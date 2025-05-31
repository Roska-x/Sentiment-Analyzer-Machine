# sentiment_analyzer/app.py

# Importar las librerías necesarias
from flask import Flask, render_template, request
import joblib
import os
import nltk
import re

# Importaciones específicas de NLTK para preprocesamiento
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer       # Lemmatizer común para INGLÉS
from nltk.stem.snowball import SpanishStemmer # Stemmer para ESPAÑOL

# Añadir mensajes de depuración al inicio para ver si el script arranca
print("DEBUG: Iniciando script app.py...", flush=True)

# --- Configuración ---
# Directorio donde se guardan/cargan el modelo y el vectorizador
MODEL_DIR = 'model'
# Rutas completas a los archivos serializados
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')

print(f"DEBUG: Paths del modelo configurados: {VECTORIZER_PATH}, {MODEL_PATH}", flush=True)

# --- Cargar Modelo y Vectorizador Globalmente al iniciar la aplicación ---
# Intentamos cargar estos objetos una vez cuando el servidor Flask inicia.
# Deben estar fuera de cualquier función de request para ser accesibles globalmente.
# Inicializamos como None por si falla la carga.
vectorizer = None
model = None

print(f"DEBUG: Intentando cargar modelo y vectorizador desde '{MODEL_DIR}/'", flush=True)
try:
    # Cargar el vectorizador TF-IDF serializado
    vectorizer = joblib.load(VECTORIZER_PATH)
    # Cargar el modelo de sentimiento serializado
    model = joblib.load(MODEL_PATH)
    print("DEBUG: Modelo y vectorizador cargados exitosamente. Variables 'model' y 'vectorizer' definidas globalmente.", flush=True)
except FileNotFoundError:
    # Manejar el caso en que los archivos del modelo no existan
    print("-" * 50, flush=True)
    print(f"ERROR: Archivos del modelo no encontrados en '{MODEL_DIR}'.", flush=True)
    print("Por favor, ejecuta 'python train_model.py' primero para entrenar y guardar el modelo.", flush=True)
    print("-" * 50, flush=True)
    # Las variables ya están en None, no es necesario re-asignar, pero lo dejamos claro:
    # vectorizer = None
    # model = None
except Exception as e:
    # Capturar cualquier otro error durante la carga (permisos, archivo corrupto, etc.)
    print(f"ERROR inesperado al cargar modelo desde '{MODEL_DIR}/'. Tipo: {type(e).__name__}, Mensaje: {e}", flush=True)
    # vectorizer = None # Ya están en None
    # model = None


print("DEBUG: Bloque de carga de modelo/vectorizador finalizado.", flush=True)

# --- Inicializar la Aplicación Flask ---
app = Flask(__name__)
print("DEBUG: Aplicación Flask creada.", flush=True)


# --- Preprocesamiento del Texto (DEBE SER IDÉNTICO al utilizado en train_model.py) ---

# Configura el idioma para stopwords, Stemmer/Lemmatizer.
# Usa 'english' si tu dataset (Reddit_Data.csv) es en inglés.
# Usa 'spanish' si tu dataset (o el nuevo que uses) es en español.
# Asegúrate de haber descargado los datos de NLTK para el idioma seleccionado (ej: nltk.download('spanish')).
stop_words_lang = 'english' # <-- CAMBIA a 'spanish' si tu dataset y texto de entrada son en ESPAÑOL

# Obtener las stopwords para el idioma configurado
try:
    stop_words = set(stopwords.words(stop_words_lang))
    print(f"DEBUG: Stopwords cargadas para el idioma '{stop_words_lang}'.", flush=True)
except LookupError:
    print(f"ERROR: No se encontraron datos de stopwords para '{stop_words_lang}'. Asegúrate de descargar NLTK data.", flush=True)
    stop_words = set() # Usar conjunto vacío como fallback para evitar errores


# Configurar el Lemmatizer o Stemmer.
# Si stop_words_lang es 'english': usar WordNetLemmatizer.
# Si stop_words_lang es 'spanish': usar SpanishStemmer (NLTK no tiene un Lemmatizer robusto para español por defecto).
if stop_words_lang == 'english':
    stemmer_or_lemmatizer = WordNetLemmatizer()
    stemming_or_lematizing_method = stemmer_or_lemmatizer.lemmatize # Usar el método .lemmatize
    print("DEBUG: Lemmatizer (WordNet) configurado para preprocesamiento.", flush=True)
elif stop_words_lang == 'spanish':
    try:
        stemmer_or_lemmatizer = SpanishStemmer() # Asegúrate de haber importado SpanishStemmer
        stemming_or_lematizing_method = stemmer_or_lemmatizer.stem # Usar el método .stem para stemmers
        print("DEBUG: Stemmer (SpanishStemmer) configurado para preprocesamiento.", flush=True)
    except LookupError:
         print("ERROR: No se encontró SpanishStemmer. Asegúrate de tenerlo instalado (pip install nltk) y sus datos si son necesarios. Usando fallback simple.", flush=True)
         # Fallback: si el stemmer no carga, usa una función simple que no haga nada
         stemmer_or_lematizer = None
         stemming_or_lematizing_method = lambda word: word # Función identidad
else: # Fallback para idioma no reconocido o sin recursos
    stemmer_or_lemmatizer = None
    stemming_or_lematizing_method = lambda word: word # Función identidad
    print(f"WARNING: Idioma '{stop_words_lang}' no soportado para lematización/stemming. Usando fallback simple.", flush=True)


def clean_text(text):
    """
    Aplica el preprocesamiento al texto de entrada.
    DEBE COINCIDIR LO MÁS POSIBLE con la función usada en train_model.py.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower() # Convertir a minúsculas

    # Ajusta la regex si necesitas mantener números, puntuación específica, etc.
    # DEBE COINCIDIR CON LA DE TRAIN_MODEL.PY. Si allí usaste r'[^a-z0-9\s]', úsala aquí.
    text = re.sub(r'[^a-z\s]', '', text) # Eliminar todo excepto letras a-z y espacios

    # Tokenización
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        print("ERROR: NLTK tokenization data ('punkt') not found. Cannot tokenize.", flush=True)
        return "" # No se puede procesar si no se puede tokenizar


    # Eliminar stopwords y aplicar Stemming/Lemmatization
    processed_tokens = []
    for word in tokens:
        if word and word not in stop_words: # Asegura que la palabra no esté vacía y no sea una stopword
            if stemming_or_lematizing_method: # Si configuramos correctamente el stemmer/lemmatizer
                processed_tokens.append(stemming_or_lematizing_method(word))
            else: # Usar el fallback simple si stemmer/lemmatizer falló al configurar
                 if re.fullmatch(r'[a-z]+', word): # Solo palabras alfabéticas en el fallback
                      processed_tokens.append(word)


    return ' '.join(processed_tokens)


print("DEBUG: Funciones de preprocesamiento definidas.", flush=True)


# --- Rutas de Flask ---

@app.route('/')
def index():
    """
    Ruta principal que sirve la página HTML con el formulario.
    """
    print("DEBUG: Request received for '/'", flush=True)
    return render_template('index.html')

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    print("DEBUG: Request received for '/analyze-sentiment' (POST)", flush=True)

    # 1. Verificar modelo/vectorizador cargados (este bloque está bien)
    if 'model' not in globals() or model is None or 'vectorizer' not in globals() or vectorizer is None:
        print("DEBUG: ERROR: Variables 'model' o 'vectorizer' no definidas o son None. No se puede analizar.", flush=True)
        return render_template('_sentiment_result.html', sentiment="Error", details="El modelo ML no está disponible. Contacta al administrador.")

    # 2. Obtener texto de entrada (este bloque está bien)
    comment_text = request.form.get('comment_text')
    print(f"DEBUG: Texto de entrada recibido: '{comment_text}'", flush=True)

    # 3. Validar texto no vacío (este bloque está bien)
    if not comment_text or not comment_text.strip():
        print("DEBUG: Texto de entrada vacío o solo espacios. Devolviendo Info.", flush=True)
        return render_template('_sentiment_result.html', sentiment="Info", details="Por favor, ingresa texto para analizar.")

    # --- Bloque try-except principal APLICADO AL PROCESAMIENTO SENSIBLE ---
    # El try empieza aquí, después de obtener y validar el texto de entrada.
    # Esto significa que comment_text ya EXISTE si llegamos al try.
    try:
        # 4. Preprocesar el texto
        cleaned_text = clean_text(comment_text)
        print(f"DEBUG: Texto preprocesado: '{cleaned_text}'", flush=True)

        # 5. Verificar texto preprocesado NO vacío
        if not cleaned_text:
             print("DEBUG: Texto preprocesado resultó vacío. Devolviendo Info.", flush=True)
             # Este return Sale de la función si está vacío.
             return render_template('_sentiment_result.html', sentiment="Info", details="El texto ingresado no contenía palabras relevantes para analizar.")

        # 6. Vectorizar el texto (solo si cleaned_text NO está vacío)
        # AHORA text_vectorized SIEMPRE se definirá aquí ANTES de cualquier otro paso que lo use
        text_vectorized = vectorizer.transform([cleaned_text])
        print(f"DEBUG: Texto vectorizado. Dimensiones: {text_vectorized.shape}", flush=True)


        # 7. Predecir el sentimiento y obtener probabilidades
        prediction = model.predict(text_vectorized)[0] # Esto te da el valor numérico (-1, 0, 1)
        probability_array = model.predict_proba(text_vectorized)[0] # Probabilidades para cada clase

        print(f"DEBUG: Predicción numérica cruda: {prediction}", flush=True)


        # --- 8. Mapear la predicción numérica a etiqueta de texto Y calcular confianza ---
        # Esta sección combina y estructura correctamente la lógica de mapeo y confianza

        # A) Mapear predicción numérica a etiqueta de texto legible
        # Define el mapeo de clases numéricas a etiquetas de texto
        sentiment_mapping = {
            1: "Positivo",
            -1: "Negativo",
            0: "Neutral"
        }

        # Asegúrate de que la predicción numérica sea un entero para el mapeo.
        try:
            predicted_value_int = int(prediction) # Convierte a INT
        except (ValueError, TypeError):
             predicted_value_int = None # Si no es convertible, asigna None


        # Usa el diccionario .get() para obtener la etiqueta STRING.
        # El fallback sólo ocurrirá si predicted_value_int es None O si el valor no está en el diccionario (raro).
        if predicted_value_int is not None:
            sentiment_label = sentiment_mapping.get(predicted_value_int, f"Resultado Desconocido: Clase desconocida: {predicted_value_int}")
        else:
             sentiment_label = f"Resultado Desconocido: Predicción no numérica ({prediction})"

        print(f"DEBUG: Predicción como INT: {predicted_value_int}, Etiqueta de texto generada: '{sentiment_label}'", flush=True)


        # B) Calcular y formatear la confianza
        # Este bloque se ejecuta *después* de tener predicted_value_int y sentiment_label
        confidence_str = "Confianza: N/A" # Etiqueta de confianza por defecto
        try:
            # OBTENER LAS CLASES DEL MODELO SIN INTENTAR CONVERTIRLAS TODAS A INT AHORA
            # Usamos list() para manejar arrays NumPy u otros tipos
            model_classes = list(model.classes_)

            print(f"DEBUG (Confianza): Clases del modelo cargado: {model_classes} (Types: {[type(c) for c in model_classes]})", flush=True)
            print(f"DEBUG (Confianza): Predicted value INT para buscar índice: {predicted_value_int} (Type: {type(predicted_value_int)})", flush=True)

            # Busca el índice de predicted_value_int (el ENTERO) dentro de la lista original de clases
            if predicted_value_int is not None and predicted_value_int in model_classes:
                 class_index_in_model_classes = model_classes.index(predicted_value_int)
                 # Acceder a la probabilidad usando el índice encontrado
                 confidence_score = probability_array[class_index_in_model_classes]
                 confidence_str = f"Confianza: {confidence_score:.2f}" # Formatear si tuvimos éxito

                 print(f"DEBUG: Cálculo de confianza exitoso. Índice: {class_index_in_model_classes}, Score: {confidence_score:.4f}", flush=True)

            else:
                 # Este else se ejecuta si predicted_value_int es None O NO está en model_classes
                 print(f"DEBUG (Confianza): Predicción INT {predicted_value_int} NO encontrada en model.classes_ ({model_classes}) para índice de probabilidad.", flush=True)
                 confidence_str = f"Confianza: Error al encontrar índice de clase." # Mensaje de error limpio

        except Exception as e: # Captura *otros* errores durante el cálculo de índice/acceso a probability_array
             print(f"DEBUG: ERROR inesperado CÁLCULO DE CONFIANZA: {type(e).__name__}: {e}", flush=True)
             confidence_str = "Confianza: Error de cálculo inesperado" # Mensaje de error limpio para la interfaz


        # --- 9. Renderizar el partial HTML ---
        # Este paso final se ejecuta *después* de que sentiment_label y confidence_str se hayan determinado
        print(f"DEBUG: Renderizando template '_sentiment_result.html' con sentiment='{sentiment_label}', details='{confidence_str}'", flush=True)
        return render_template('_sentiment_result.html',
                               sentiment=sentiment_label, # Pasa el STRING
                               details=confidence_str) # Pasa el string de confianza/error


    except Exception as e: # ESTE ES EL EXCEPT GENERAL PRINCIPAL - CAPTURA ERRORES ANTES DEL PASO 8/9 O EN CUALQUIER LUGAR DEL TRY PRINCIPAL
         print(f"ERROR General durante el análisis del sentimiento: {type(e).__name__}: {e}", flush=True)
         # Asegurar que no se usan variables que pudieron no haberse definido
         return render_template('_sentiment_result.html', sentiment="Error", details=f"Error interno inesperado durante el análisis: {type(e).__name__}")



# Puedes añadir otras rutas @app.route(...) aquí si tu app tiene más páginas


# --- Bloque de ejecución principal para desarrollo ---
# Esto solo se ejecuta si corres el script directamente con `python app.py`
# Usar un servidor WSGI (Gunicorn, uWSGI) es recomendado para producción.
if __name__ == '__main__':
    print("DEBUG: Entrando al bloque __main__. Iniciando servidor Flask.", flush=True)
    # debug=True habilita el reloader automático y muestra más información de errores
    app.run(debug=True)
    # El print de abajo rara vez se ve en modo debug=True porque el reloader toma el control
    print("DEBUG: app.run() ha finalizado.", flush=True)