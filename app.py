from flask import Flask, render_template, request, render_template_string 
import joblib
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer     
from nltk.stem.snowball import SpanishStemmer 
print("DEBUG: Iniciando script app.py...", flush=True)

# --- Configuración ---
MODEL_DIR = 'model'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
print(f"DEBUG: Paths del modelo configurados: {VECTORIZER_PATH}, {MODEL_PATH}", flush=True)

# --- Cargar Modelo y Vectorizador Globalmente al iniciar la aplicación ---
vectorizer = None
model = None
print(f"DEBUG: Intentando cargar modelo y vectorizador desde '{MODEL_DIR}/'", flush=True)
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    print("DEBUG: Modelo y vectorizador cargados exitosamente. Variables 'model' y 'vectorizer' definidas globalmente.", flush=True)
except FileNotFoundError:
    print("-" * 50, flush=True)
    print(f"ERROR: Archivos del modelo no encontrados en '{MODEL_DIR}'.", flush=True)
    print("Por favor, ejecuta 'python train_model.py' primero para entrenar y guardar el modelo.", flush=True)
    print("-" * 50, flush=True)
except Exception as e:
    print(f"ERROR inesperado al cargar modelo desde '{MODEL_DIR}/'. Tipo: {type(e).__name__}, Mensaje: {e}", flush=True)
print("DEBUG: Bloque de carga de modelo/vectorizador finalizado.", flush=True)
# --- Inicializar la Aplicación Flask ---
app = Flask(__name__)
print("DEBUG: Aplicación Flask creada.", flush=True)

# --- Preprocesamiento del Texto ---
stop_words_lang = 'english' 
try:
    stop_words = set(stopwords.words(stop_words_lang))
    print(f"DEBUG: Stopwords cargadas para el idioma '{stop_words_lang}'.", flush=True)
except LookupError:
    print(f"ERROR: No se encontraron datos de stopwords para '{stop_words_lang}'. Asegúrate de descargar NLTK data.", flush=True)
    stop_words = set() # Usar conjunto vacío como fallback para evitar errores
if stop_words_lang == 'english':
    stemmer_or_lemmatizer = WordNetLemmatizer()
    stemming_or_lematizing_method = stemmer_or_lemmatizer.lemmatize 
    print("DEBUG: Lemmatizer (WordNet) configurado para preprocesamiento.", flush=True)
elif stop_words_lang == 'spanish':
    try:
        stemmer_or_lemmatizer = SpanishStemmer() 
        stemming_or_lematizing_method = stemmer_or_lemmatizer.stem 
        print("DEBUG: Stemmer (SpanishStemmer) configurado para preprocesamiento.", flush=True)
    except LookupError:
         print("ERROR: No se encontró SpanishStemmer. Asegúrate de tenerlo instalado (pip install nltk) y sus datos si son necesarios. Usando fallback simple.", flush=True)
         stemmer_or_lematizer = None
         stemming_or_lematizing_method = lambda word: word 
else: 
    stemmer_or_lemmatizer = None
    stemming_or_lematizing_method = lambda word: word 
    print(f"WARNING: Idioma '{stop_words_lang}' no soportado para lematización/stemming. Usando fallback simple.", flush=True)
def clean_text(text):
    """
    Aplica el preprocesamiento al texto de entrada.
    DEBE COINCIDIR LO MÁS POSIBLE con la función usada en train_model.py.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text) 
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        print("ERROR: NLTK tokenization data ('punkt') not found. Cannot tokenize.", flush=True)
        return "" 
    processed_tokens = []
    for word in tokens:
        if word and word not in stop_words: 
            if stemming_or_lematizing_method: 
                processed_tokens.append(stemming_or_lematizing_method(word))
            else: 
                 if re.fullmatch(r'[a-z]+', word): 
                      processed_tokens.append(word)
    return ' '.join(processed_tokens)
print("DEBUG: Funciones de preprocesamiento definidas.", flush=True)

# --- Rutas de Flask ---

@app.route('/')
def index():
    """
    Ruta principal que sirve la página HTML con el formulario.
    Renderiza el estado inicial del resultado del sentimiento usando el mismo fragmento que HTMX.
    """
    print("DEBUG: Request received for '/'", flush=True)
    
    initial_sentiment_data = {
        "sentiment": "i wait for your text...", 
    }
    initial_html_content = render_template_string(
        "{% include '_sentiment_result.html' %}", 
        **initial_sentiment_data
    )
    
    return render_template('index.html', initial_sentiment_html_content=initial_html_content)

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    print("DEBUG: Request received for '/analyze-sentiment' (POST)", flush=True)
    if 'model' not in globals() or model is None or 'vectorizer' not in globals() or vectorizer is None:
        print("DEBUG: ERROR: Variables 'model' o 'vectorizer' no definidas o son None. No se puede analizar.", flush=True)
        return render_template('_sentiment_result.html', sentiment="Error", details="El modelo ML no está disponible. Contacta al administrador.")
    comment_text = request.form.get('comment_text')
    print(f"DEBUG: Texto de entrada recibido: '{comment_text}'", flush=True)
    if not comment_text or not comment_text.strip():
        print("DEBUG: Texto de entrada vacío o solo espacios. Devolviendo Info.", flush=True)
        return render_template('_sentiment_result.html', sentiment="Info", details="Por favor, ingresa texto para analizar.")

    try:
        cleaned_text = clean_text(comment_text)
        print(f"DEBUG: Texto preprocesado: '{cleaned_text}'", flush=True)
        if not cleaned_text:
             print("DEBUG: Texto preprocesado resultó vacío. Devolviendo Info.", flush=True)
             return render_template('_sentiment_result.html', sentiment="Info", details="The entered text did not contain relevant words to analyze.")
        text_vectorized = vectorizer.transform([cleaned_text])
        print(f"DEBUG: Texto vectorizado. Dimensiones: {text_vectorized.shape}", flush=True)
        prediction = model.predict(text_vectorized)[0] 
        probability_array = model.predict_proba(text_vectorized)[0] 
        print(f"DEBUG: Predicción numérica cruda: {prediction}", flush=True)
        sentiment_mapping = {
            1: "Positive",
            -1: "Negative",
            0: "Neutral"
        }
        try:
            predicted_value_int = int(prediction) 
        except (ValueError, TypeError):
             predicted_value_int = None 

        if predicted_value_int is not None:
            sentiment_label = sentiment_mapping.get(predicted_value_int, f"Resultado Desconocido: Clase desconocida: {predicted_value_int}")
        else:
             sentiment_label = f"Resultado Desconocido: Predicción no numérica ({prediction})"

        print(f"DEBUG: Predicción como INT: {predicted_value_int}, Etiqueta de texto generada: '{sentiment_label}'", flush=True)

        confidence_str = "Confidence: N/A" 
        try:
            model_classes = list(model.classes_)

            print(f"DEBUG (Confidence): Clases del modelo cargado: {model_classes} (Types: {[type(c) for c in model_classes]})", flush=True)
            print(f"DEBUG (Confidence): Predicted value INT para buscar índice: {predicted_value_int} (Type: {type(predicted_value_int)})", flush=True)

            if predicted_value_int is not None and predicted_value_int in model_classes:
                 class_index_in_model_classes = model_classes.index(predicted_value_int)
                 confidence_score = probability_array[class_index_in_model_classes]
                 confidence_str = f"Confidence: {confidence_score:.2f}" 
                 print(f"DEBUG: Cálculo de confianza exitoso. Índice: {class_index_in_model_classes}, Score: {confidence_score:.4f}", flush=True)
            else:
                 print(f"DEBUG (Confidence): Predicción INT {predicted_value_int} NO encontrada en model.classes_ ({model_classes}) para índice de probabilidad.", flush=True)
                 confidence_str = f"Confianza: Error al encontrar índice de clase." 
        except Exception as e: 
             print(f"DEBUG: ERROR inesperado CÁLCULO DE CONFIANZA: {type(e).__name__}: {e}", flush=True)
             confidence_str = "Confianza: Error de cálculo inesperado" 
        print(f"DEBUG: Renderizando template '_sentiment_result.html' con sentiment='{sentiment_label}', details='{confidence_str}'", flush=True)
        return render_template('_sentiment_result.html',
                               sentiment=sentiment_label,
                               details=confidence_str) 
    except Exception as e: 
         print(f"ERROR General durante el análisis del sentimiento: {type(e).__name__}: {e}", flush=True)
         return render_template('_sentiment_result.html', sentiment="Error", details=f"Error interno inesperado durante el análisis: {type(e).__name__}")

if __name__ == '__main__':
    print("DEBUG: Entrando al bloque __main__. Iniciando servidor Flask.", flush=True)
    app.run(debug=True)
    print("DEBUG: app.run() ha finalizado.", flush=True)
