# Sentiment Analyzer Machine

![Screenshot de la máquina Sentiment Analyzer](assets/sentiment-machine-screenshot.png)


## Licencia

Este proyecto (todo el código: Python, HTML, CSS, JavaScript) está bajo la **Licencia MIT**. Puedes encontrar el texto completo de la licencia en el archivo [LICENSE](LICENSE) en la raíz del repositorio.

**Importante:** La **Licencia MIT aplica únicamente al CÓDIGO**. El **dataset** utilizado (`Reddit_Data.csv`) y el **modelo entrenado** derivado de él (`model/` archivos) están bajo la licencia **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)**. Esto significa que:
*   Debes acreditar al autor original del dataset (cosmos98, como se detalla en la sección de Instalación).
*   No puedes usar el dataset original ni el modelo entrenado (los archivos .pkl producidos por este código con ese dataset) con fines **comerciales**.
*   Si adaptas el dataset o el modelo entrenado y los distribuyes, debes hacerlo bajo la misma licencia CC BY-NC-SA 4.0.

Si planeas usar este código con un propósito comercial, deberás utilizar un dataset diferente cuya licencia lo permita y re-entrenar el modelo con ese nuevo dataset.

## Descripción del Proyecto

Esta es una aplicación web interactiva que simula una "Máquina de Análisis de Sentimiento". El backend, desarrollado con Python y Flask, implementa un **pipeline básico de Procesamiento de Lenguaje Natural (PLN) y Machine Learning** para **clasificación de texto**. El usuario ingresa texto a través de una interfaz web (construida con HTML/CSS moderno), y una petición asíncrona (manejada por HTMX) envía el texto al backend. El sistema predice el sentimiento predominante del texto (Positivo, Negativo o Neutral) utilizando un **modelo de clasificación supervisada (Regresión Logística)**. El resultado y un indicador de confianza son mostrados dinámicamente en la "pantalla digital" de la interfaz con animaciones (GSAP).

Este proyecto sirve como una demostración práctica de la integración de modelos predictivos en aplicaciones web funcionales, mostrando cómo un resultado de ML puede ser presentado de forma interactiva en el frontend.

## Características Clave

*   **Clasificación de Sentimiento:** Predice el sentimiento de un texto (`Positivo`, `Negativo`, `Neutral`) utilizando un modelo de **Regresión Logística (Logistic Regression)**.
*   **Pipeline de PLN:** Incluye etapas de preprocesamiento de texto (tokenización, eliminación de stopwords, stemming/lematización).
*   **Vectorización TF-IDF:** Transforma el texto preprocesado en características numéricas utilizando la técnica **TF-IDF (Term Frequency-Inverse Document Frequency)**.
*   Interfaz de usuario distintiva y animada, inspirada en una máquina digital.
*   Comunicación asíncrona frontend-backend con **HTMX**.
*   Animaciones fluidas en la interfaz (`GSAP`).
*   Visualización dinámica del resultado y confianza en pantalla simulada.
*   Serialización y carga eficiente del modelo y vectorizador (`joblib`).

## Tecnologías Utilizadas

**Backend (Python):**

*   **Flask:** Microframework web.
*   **scikit-learn:** Implementación del modelo de **Regresión Logística** y el **TfidfVectorizer**.
*   **pandas:** Carga y manipulación del dataset.
*   **nltk:** Herramientas para **Procesamiento de Lenguaje Natural (PLN)**.
*   **joblib:** Serialización de objetos Python.

**Frontend (Web):**

*   **HTML5, CSS3:** Estructura y estilizado (layout con Flexbox/Grid, diseño neumórfico/biselado).
*   **HTMX:** Interacción asíncrona sin JavaScript complejo.
*   **GSAP:** Animaciones web.
*   **Bootstrap (parcial):** Íconos (`Bootstrap Icons`), utilidades de layout.



## Cómo Funciona (Diagrama Simplificado del Pipeline)

```mermaid
%%{init: {
    'theme': 'dark',
    'themeVariables': {
        'lineColor': '#A9A9A9', // Un gris un poco más oscuro para las líneas
        'textColor': '#FFFFFF'
    },
    'flowchart': {
        'htmlLabels': true, // Permite el uso de <br> y otro HTML simple en etiquetas
        'nodeSpacing': 50,
        'rankSpacing': 60
    }
}}%%
graph TD
    A["Texto Crudo<br>(Input Usuario)"] --> B["Preprocesamiento<br>(NLTK)"];
    B -- "Texto Limpio" --> C["Vectorización TF-IDF<br>(scikit-learn)"];
    C -- "Vector Numérico" --> D["Modelo Regresión<br>Logística (scikit-learn)"];
    D -- "Predicción (-1, 0, 1)<br>+ Probabilidades" --> E["Mapeo a Etiqueta<br>Texto (Flask/Python)"];
    E -- "Resultado<br>Formateado" --> F["Backend Flask<br>genera HTML"];
    F -- "HTML Parcial" --> G["HTMX<br>actualiza Pantalla"];
    G --> H["GSAP<br>anima Aparición"];

    classDef grey fill:#001f3f,stroke:#E0E0E0,color:#FFFFFF,font-family:Arial,font-size:12px;
    classDef primary fill:#0074D9,stroke:#E0E0E0,color:#FFFFFF,font-family:Arial,font-size:12px;

    class A,G,H primary;
    class B,C,D,E,F grey;
```



## Habilidades 



    Machine Learning Aplicado: Implementación de un pipeline de clasificación de texto supervisada.

    Modelos Predictivos: Uso específico de la Regresión Logística para una tarea de clasificación categórica (-1, 0, 1).

    Procesamiento de Lenguaje Natural (PLN): Aplicación de técnicas estándar como tokenización, manejo de stopwords y stemming/lematización con nltk.

    Ingeniería de Características Textuales: Dominio de la técnica de vectorización TF-IDF (TfidfVectorizer) para representar datos textuales de forma cuantitativa.

    Desarrollo Full-Stack: Conexión y comunicación fluida entre un backend Python/Flask y un frontend web dinámico.

    Integración Web Asíncrona: Uso estratégico de HTMX para crear una experiencia de usuario responsiva sin la complejidad de un framework JS tradicional.

    Serialización y Persistencia de Modelos ML: Gestión del ciclo de vida del modelo y vectorizador mediante joblib.

    Diseño de Interfaz (UI) Avanzado con CSS: Creación de layouts complejos (Flexbox/Grid) y aplicación de efectos visuales detallados como el diseño neumórfico/biselado utilizando box-shadow.

    Animación Web Performante: Integración de GSAP para efectos visuales suaves.

    Manejo de Datos: Carga, limpieza y transformación de datos tabulares (pandas).

    Depuración Sistemática y Resolución de Problemas: Identificación y solución de errores en múltiples capas del stack (backend, frontend, pipeline de datos ML, serialización)

## Instalación y Ejecución Local

Sigue estos pasos para poner la aplicación en marcha en tu máquina local:

1. Clonar el Repositorio:

          
    git clone ... https://github.com/Roska-x/Sentymental-Analysis-Machine/ ...
    cd sentiment_analyzer

        


2. Crear y Activar Entorno Virtual: (Altamente recomendado para aislar las dependencias)

      
python3 -m venv venv
source venv/bin/activate


3. Instalar Dependencias de Python:

      
pip install -r requirements.txt

    


4. Descargar Datos de NLTK: Necesario para el preprocesamiento de texto.

      
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
    


(Si el descargador de NLTK interactivo se abre en tu terminal, confirma las descargas).

5. Obtener el Dataset de Entrenamiento:
El dataset utilizado es "Sentiment Analysis on Multi-Source Social Media Text" de cosmos98 en Kaggle. Contiene datos de Reddit y Twitter. Para este proyecto, se utiliza específicamente el archivo Reddit_Data.csv.
Enlace al Dataset Original: https://www.kaggle.com/datasets/cosmos98/sentiment-analysis-on-multi-source-social-media

Licencia: Este dataset está bajo la licencia Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). Se agradece y respeta la labor del autor original. Más detalles: https://creativecommons.org/licenses/by-nc-sa/4.0/

Nota sobre Modificaciones: El archivo Reddit_Data.csv descargado directamente de Kaggle no fue modificado. Las operaciones de limpieza de etiquetas (para convertir '1', '0', '-1' a valores numéricos enteros y eliminar filas con etiquetas no numéricas como 'category') y el preprocesamiento de texto se realizan dinámicamente durante la ejecución del script de entrenamiento (train_model.py) utilizando pandas y NLTK.

Acción: Descarga el archivo Reddit_Data.csv desde el enlace de Kaggle y colócalo en el directorio raíz de este proyecto (sentiment_analyzer/).


6. Entrenar y Guardar el Modelo:
Ejecuta el script de entrenamiento una sola vez. Esto procesará el dataset y generará los archivos vectorizer.pkl y sentiment_model.pkl en la carpeta model/. Asegúrate de haber ajustado la configuración de idioma (stop_words_lang) y la regex en train_model.py (¡y en app.py!) si tu dataset usa español o tiene un formato de texto particular.

      
python train_model.py

    

(Este script puede tardar unos minutos dependiendo de tu dataset y CPU. Debería imprimir mensajes de DEBUG sobre el proceso y la evaluación.)

7. Ejecutar la Aplicación Flask:

      
python app.py



(Verás mensajes DEBUG de Flask en tu terminal.)

8. Acceder a la Aplicación: Abre tu navegador web y visita http://127.0.0.1:5000/.

## Estructura del Proyecto

```text
sentiment_analyzer/
├── app.py                 # Aplicación Flask principal
├── train_model.py         # Script para entrenar y guardar el modelo ML
├── requirements.txt       # Dependencias de Python
├── Reddit_Data.csv        # Archivo del dataset de entrenamiento
├── model/                 # Carpeta para guardar archivos del modelo
│   ├── vectorizer.pkl
│   └── sentiment_model.pkl
├── static/                # Archivos estáticos (CSS, JS, fuentes, imágenes)
│   ├── css/
│   │   └── style.css      # Estilos de la UI y tema original
│   ├── fonts/             # (Ej: Orbitron-Regular.ttf si usas fuente digital)
│   │   └── ...
│   ├── img/               # (Ej: favicon.ico, etc.)
│   │   └── ...
│   └── js/
│       └── main.js        # Lógica JavaScript del frontend (GSAP, HTMX)
└── templates/             # Plantillas HTML (Jinja2)
    ├── index.html         # Página principal con la UI
    └── _sentiment_result.html # Fragmento HTML para la respuesta de HTMX
```


## Autor


Franco Donati

https://linkedin.com/in/franco-donati/

https://github.com/Roska-x/Sentymental-Analysis-Machine/


## ======================================================================

## Sentiment Analyzer Machine


## License

This project (all code: Python, HTML, CSS, JavaScript) is under the MIT License. You can find the full license text in the LICENSE file in the root of the repository.

Important: The MIT License applies only to the CODE. The dataset used (Reddit_Data.csv) and the trained model derived from it (model/ files) are under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) license. This means that:

You must give appropriate credit to the original author of the dataset (cosmos98, as detailed in the Installation section).

You may not use the original dataset or the trained model (the .pkl files produced by this code with that dataset) for commercial purposes.

If you remix, transform, or build upon the dataset or trained model and distribute your contributions, you must distribute them under the same CC BY-NC-SA 4.0 license.

If you plan to use this code for a commercial purpose, you will need to use a different dataset whose license permits it and retrain the model with that new dataset.

## Project Description

This is an interactive web application that simulates a "Sentiment Analysis Machine". The backend, developed with Python and Flask, implements a basic Natural Language Processing (NLP) and Machine Learning pipeline for text classification. The user enters text through a web interface (built with modern HTML/CSS), and an asynchronous request (handled by HTMX) sends the text to the backend. The system predicts the predominant sentiment of the text (Positive, Negative, or Neutral) using a supervised classification model (Logistic Regression). The result and a confidence indicator are dynamically displayed on the interface's "digital screen" with animations (GSAP).

This project serves as a practical demonstration of integrating predictive models into functional web applications, showing how an ML result can be interactively presented on the frontend.

## Key Features

Sentiment Classification: Predicts the sentiment of a text (Positive, Negative, Neutral) using a Logistic Regression model.

NLP Pipeline: Includes text preprocessing stages (tokenization, stopword removal, stemming/lemmatization).

TF-IDF Vectorization: Transforms preprocessed text into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.

Distinctive and animated user interface, inspired by a digital machine.

Asynchronous frontend-backend communication with HTMX.

Smooth interface animations (GSAP).

Dynamic display of results and confidence on a simulated screen.

Efficient serialization and loading of the model and vectorizer (joblib).

## Technologies Used

**Backend (Python):**

**Flask:** Web microframework.

**scikit-learn:** Implementation of the Logistic Regression model and TfidfVectorizer.

**pandas:** Dataset loading and manipulation.

**nltk:** Tools for Natural Language Processing (NLP).

**joblib**: Python object serialization.

**Frontend (Web):**

**HTML5, CSS3:** Structure and styling (layout with Flexbox/Grid, neumorphic/beveled design).

**HTMX:** Asynchronous interaction without complex JavaScript.

**GSAP:** Web animations.

**Bootstrap (partial):** Icons (Bootstrap Icons), layout utilities.

# How It Works (Simplified Pipeline Diagram)

```mermaid
%%{init: {
    'theme': 'dark',
    'themeVariables': {
        'lineColor': '#A9A9A9', // A slightly darker grey for lines
        'textColor': '#FFFFFF'
    },
    'flowchart': {
        'htmlLabels': true, // Allows use of <br> and other simple HTML in labels
        'nodeSpacing': 50,
        'rankSpacing': 60
    }
}}%%
graph TD
    A["Raw Text<br>(User Input)"] --> B["Preprocessing<br>(NLTK)"];
    B -- "Clean Text" --> C["TF-IDF Vectorization<br>(scikit-learn)"];
    C -- "Numeric Vector" --> D["Logistic Regression<br>Model (scikit-learn)"];
    D -- "Prediction (-1, 0, 1)<br>+ Probabilities" --> E["Map to Text<br>Label (Flask/Python)"];
    E -- "Formatted<br>Result" --> F["Flask Backend<br>generates HTML"];
    F -- "Partial HTML" --> G["HTMX<br>updates Screen"];
    G --> H["GSAP<br>animates Appearance"];

    classDef grey fill:#001f3f,stroke:#E0E0E0,color:#FFFFFF,font-family:Arial,font-size:12px;
    classDef primary fill:#0074D9,stroke:#E0E0E0,color:#FFFFFF,font-family:Arial,font-size:12px;

    class A,G,H primary;
    class B,C,D,E,F grey;
```

## Skills 


Applied Machine Learning: Implementation of a supervised text classification pipeline.

Predictive Models: Specific use of Logistic Regression for a categorical classification task (-1, 0, 1).

Natural Language Processing (NLP): Application of standard techniques such as tokenization, stopword handling, and stemming/lemmatization with nltk.

Textual Feature Engineering: Mastery of the TF-IDF vectorization technique (TfidfVectorizer) to represent textual data quantitatively.

Full-Stack Development: Connection and smooth communication between a Python/Flask backend and a dynamic web frontend.

Asynchronous Web Integration: Strategic use of HTMX to create a responsive user experience without the complexity of a traditional JS framework.

ML Model Serialization and Persistence: Management of the model and vectorizer lifecycle using joblib.

Advanced UI Design with CSS: Creation of complex layouts (Flexbox/Grid) and application of detailed visual effects like neumorphic/beveled design using box-shadow.

Performant Web Animation: Integration of GSAP for smooth visual effects.

Data Handling: Loading, cleaning, and transforming tabular data (pandas).

Systematic Debugging and Problem-Solving: Identification and resolution of errors across multiple layers of the stack (backend, frontend, ML data pipeline, serialization).

## Installation and Local Execution

Follow these steps to get the application running on your local machine:

1. Clone the Repository:

git clone ... https://github.com/Roska-x/Sentymental-Analysis-Machine/ ...
cd sentiment_analyzer


1. Create and Activate Virtual Environment: (Highly recommended to isolate dependencies)

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install Python Dependencies:

pip install -r requirements.txt


4. Download NLTK Data: Necessary for text preprocessing.

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"


(If the interactive NLTK downloader opens in your terminal, confirm the downloads.)

5. Obtain the Training Dataset:
The dataset used is "Sentiment Analysis on Multi-Source Social Media Text" by cosmos98 on Kaggle. It contains data from Reddit and Twitter. For this project, the Reddit_Data.csv file is specifically used.
Link to Original Dataset: https://www.kaggle.com/datasets/cosmos98/sentiment-analysis-on-multi-source-social-media

License: This dataset is under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) license. The work of the original author is acknowledged and respected. More details: https://creativecommons.org/licenses/by-nc-sa/4.0/

Note on Modifications: The Reddit_Data.csv file downloaded directly from Kaggle was not modified. Label cleaning operations (to convert '1', '0', '-1' to integer numerical values and remove rows with non-numeric labels like 'category') and text preprocessing are performed dynamically during the execution of the training script (train_model.py) using pandas and NLTK.

Action: Download the Reddit_Data.csv file from the Kaggle link and place it in the root directory of this project (sentiment_analyzer/).

6. Train and Save the Model:
Run the training script once. This will process the dataset and generate the vectorizer.pkl and sentiment_model.pkl files in the model/ folder. Ensure you've adjusted the language settings (stop_words_lang) and the regex in train_model.py (and in app.py!) if your dataset uses a language other than English or has a particular text format.

python train_model.py


(This script may take a few minutes depending on your dataset and CPU. It should print DEBUG messages about the process and evaluation.)

7. Run the Flask Application:

python app.py

(You will see Flask DEBUG messages in your terminal.)

8. Access the Application: Open your web browser and visit http://127.0.0.1:5000/.

## Project Structure
```text
sentiment_analyzer/
├── app.py                 # Main Flask application
├── train_model.py         # Script to train and save the ML model
├── requirements.txt       # Python dependencies
├── Reddit_Data.csv        # Training dataset file
├── model/                 # Folder to store model files
│   ├── vectorizer.pkl
│   └── sentiment_model.pkl
├── static/                # Static files (CSS, JS, fonts, images)
│   ├── css/
│   │   └── style.css      # UI styles and original theme
│   ├── fonts/             # (e.g., Orbitron-Regular.ttf if using digital font)
│   │   └── ...
│   ├── img/               # (e.g., favicon.ico, etc.)
│   │   └── ...
│   └── js/
│       └── main.js        # Frontend JavaScript logic (GSAP, HTMX)
└── templates/             # HTML templates (Jinja2)
    ├── index.html         # Main page with the UI
    └── _sentiment_result.html # HTML fragment for HTMX response
```

## Author

Franco Donati

https://linkedin.com/in/franco-donati/

https://github.com/Roska-x/Sentymental-Analysis-Machine/



        
