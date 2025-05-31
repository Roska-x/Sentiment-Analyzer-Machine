document.addEventListener('DOMContentLoaded', function () {
    console.log("DEBUG JS: main.js cargado y DOM listo.");

    // --- 1. Definición de todas las constantes y variables necesarias ---
    const visibleTextarea = document.getElementById('visibleCommentInput');
    const formTextarea = document.getElementById('formCommentText');
    const analyzeButton = document.getElementById('analyzeButtonVisible');
    const sentimentForm = document.getElementById('sentiment-form');
    const powerButton = document.querySelector('.machine-button.power');
    const resultDisplay = document.getElementById('sentiment-result'); 
    
    let heartIcon = null;
    if (analyzeButton) {
        heartIcon = analyzeButton.querySelector('i.bi-heart-fill');
    }
    let heartBeatAnim = null; // Animación del corazón
    let ecgPathAnim = null; 
    let corazonDeberiaLatir = false; 

    // --- Verificaciones de elementos críticos ---
    if (!visibleTextarea) console.error("ERROR JS: No se encontró visibleTextarea #visibleCommentInput");
    if (!formTextarea) console.error("ERROR JS: No se encontró formTextarea #formCommentText");
    if (!analyzeButton) console.error("ERROR JS: No se encontró analyzeButton #analyzeButtonVisible");
    if (!sentimentForm) console.error("ERROR JS: No se encontró sentimentForm #sentiment-form");
    if (!resultDisplay) console.error("ERROR JS: No se encontró resultDisplay #sentiment-result");
    if (!heartIcon && analyzeButton) console.error("ERROR JS: No se encontró i.bi-heart-fill dentro de #analyzeButtonVisible");

    // --- 2. Funciones para Animación del Latido del Corazón (GSAP) ---
    function startHeartbeatAnimation() { // <--- NOMBRE CORRECTO DE DEFINICIÓN
        console.log("%cAttempting to START heartbeat. Should beat: " + corazonDeberiaLatir, "color: blue; font-weight: bold;");
        if (!corazonDeberiaLatir) {
            console.warn("Start heartbeat called, but corazonDeberiaLatir is false. Aborting.");
            if (heartIcon) gsap.killTweensOf(heartIcon);
            if (heartBeatAnim) {
                heartBeatAnim.kill();
                heartBeatAnim = null;
            }
            if(heartIcon) {
                 gsap.set(heartIcon, { 
                    scale: 1, 
                    textShadow: "0 0 6px var(--heart-icon-color), 0 0 8px var(--heart-icon-color), 0 0 10px color-mix(in srgb, var(--heart-icon-color) 50%, transparent)"
                 });
            }
            return;
        }

        if (heartIcon && typeof gsap !== 'undefined') {
            if (heartBeatAnim && heartBeatAnim.isActive()) {
                console.log("DEBUG JS: heartBeatAnim ya está activo. No se crea una nueva.");
                return;
            }
            
            console.log("DEBUG JS: (startHeartbeatAnimation) Matando tweens previos de heartIcon antes de crear uno nuevo.");
            gsap.killTweensOf(heartIcon); 

            heartBeatAnim = gsap.to(heartIcon, {
                scale: 1.25,
                duration: 0.35,
                repeat: -1,
                yoyo: true,
                ease: 'power1.inOut',
                textShadow: "0 0 15px var(--heart-icon-color), 0 0 25px var(--heart-icon-color)",
                onStart: function() { console.log("GSAP heartBeatAnim INICIADA (onStart callback)"); },
                onKill: function() { console.log("GSAP heartBeatAnim MATADA (onKill callback)"); }
            });
            console.log("DEBUG JS: Nueva animación de latido del corazón creada y asignada a heartBeatAnim.");
        } else { /* logs de warning */ }
    }

    function stopHeartbeatAnimation() { // <--- NOMBRE CORRECTO DE DEFINICIÓN
        corazonDeberiaLatir = false; 
        console.error("DEBUG JS: --- stopHeartbeatAnimation() LLAMADA ---");
        
        if (heartIcon && typeof gsap !== 'undefined') {
            console.log("DEBUG JS: Estado de heartBeatAnim al entrar a stopHeartbeatAnimation:", heartBeatAnim);
            if (heartBeatAnim) {
                console.log("DEBUG JS: Matando instancia de heartBeatAnim. ¿Estaba activa?:", heartBeatAnim.isActive());
                heartBeatAnim.kill(); 
                heartBeatAnim = null; 
            } else {
                console.log("DEBUG JS: heartBeatAnim era null/undefined al entrar.");
            }
            console.log("DEBUG JS: Ejecutando gsap.killTweensOf(heartIcon) para matar TODAS las animaciones en el target.");
            gsap.killTweensOf(heartIcon); 
            console.log("DEBUG JS: Aplicando reseteo visual explícito a heartIcon.");
            gsap.set(heartIcon, {
                scale: 1,
                clearProps: "scale,textShadow,transform" 
            });
            gsap.to(heartIcon, { 
                duration: 0.05, 
                textShadow: "0 0 6px var(--heart-icon-color), 0 0 8px var(--heart-icon-color), 0 0 10px color-mix(in srgb, var(--heart-icon-color) 50%, transparent)",
                overwrite: "auto"
            });
            console.error("DEBUG JS: --- Latido del corazón TEÓRICAMENTE DETENIDO Y RESETEADO por stopHeartbeatAnimation() ---");
        } else { /* logs de warning */ }
    }
    // --- Funciones para el estado "Analizando" y ECG ---
    function showAnalyzingState() {
        if (!resultDisplay) return;
        console.log("DEBUG JS: Mostrando estado 'Analizando...'");
        resultDisplay.innerHTML = `
            <div class="ecg-line-container">
                <svg class="ecg-svg" viewBox="0 0 200 50" preserveAspectRatio="none">
                    <path class="ecg-path" d="M0,25 Q10,25 20,25 T40,25 Q45,5 50,25 T70,25 Q75,45 80,25 T100,25 Q105,15 110,25 T130,25 Q135,35 140,25 T160,25 Q165,10 170,25 T190,25 L200,25" stroke-width="2" fill="none"/>
                </svg>
            </div>
            <p class="sentiment-text analyzing-text">ANALIZANDO...</p>
        `;
        resultDisplay.className = 'sentiment-result-screen-content analyzing-state';

        const ecgPath = resultDisplay.querySelector('.ecg-path');
        if (ecgPath && typeof gsap !== 'undefined') {
            if (ecgPathAnim) ecgPathAnim.kill();
            gsap.set(ecgPath, { stroke: "var(--theme-positive-color)", strokeDasharray: 500, strokeDashoffset: 500 });
            ecgPathAnim = gsap.to(ecgPath, {
                strokeDashoffset: 0, duration: 1.5, ease: 'none', repeat: -1,
                onRepeat: function() { gsap.set(this.targets()[0], { strokeDashoffset: 500 }); }
            });
        }
    }

    function stopAnalyzingStateAndECG() {
        console.log("DEBUG JS: Deteniendo estado 'Analizando...' y ECG.");
        if (ecgPathAnim) {
            ecgPathAnim.kill();
            ecgPathAnim = null;
        }
        // El contenido de resultDisplay será reemplazado por HTMX, así que no necesitamos limpiarlo aquí.
    }

    // Eventos HTMX en el formulario
    if (sentimentForm) {
        sentimentForm.addEventListener('htmx:beforeRequest', function() {
            console.error("CRITICAL DEBUG: 'htmx:beforeRequest' en sentimentForm.");
            corazonDeberiaLatir = true; 
            showAnalyzingState();
            startHeartbeatAnimation(); // <--- CORREGIDO: LLAMAR A startHeartbeatAnimation
        });

        sentimentForm.addEventListener('htmx:afterSettle', function() { // El parámetro event no se usa aquí, es opcional.
            console.log("DEBUG JS: sentimentForm 'htmx:afterSettle'.");
            stopAnalyzingStateAndECG(); 
        });
    }

    // --- 3. Lógica de Interacción con el Usuario (Clic y Enter) ---
    function handleTextSubmit() {
        const textToAnalyze = visibleTextarea.value;
        formTextarea.value = textToAnalyze;
        if (textToAnalyze.trim() !== '') {
            console.log("DEBUG JS: Texto válido. Disparando 'submit' en formulario HTMX...");
            htmx.trigger(sentimentForm, 'submit');
        } else {
            console.log("DEBUG JS: Input vacío, no se envía.");
            // Opcional: mostrar mensaje de info directamente
        }
    }

    if (analyzeButton && visibleTextarea && formTextarea && sentimentForm) {
        analyzeButton.addEventListener('click', function () {
            console.log("DEBUG JS: Botón Corazón clickeado.");
            handleTextSubmit();
        });
    }

    if (visibleTextarea && formTextarea && sentimentForm) {
        visibleTextarea.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                console.log("DEBUG JS: Enter presionado.");
                handleTextSubmit();
            }
        });
    }

    // Limpiar textarea después de petición exitosa
    if (sentimentForm && visibleTextarea) {
        sentimentForm.addEventListener('htmx:afterRequest', (event) => {
            if (event.detail.successful) {
                console.log("DEBUG JS: Petición HTMX exitosa. Limpiando textarea visible.");
                visibleTextarea.value = '';
            }
        });
    }

    // --- 4. Animación del Resultado del Sentimiento (GSAP) ---
    document.body.addEventListener('htmx:afterSwap', (event) => {
        if (event.detail.target && event.detail.target.id === 'sentiment-result') {
            const resultContainer = event.detail.target; 
            if (resultContainer.classList.contains('analyzing-state')) {
                 console.log("DEBUG JS: 'htmx:afterSwap' detectó 'analyzing-state'. No se anima resultado.");
                 return; 
            }
            console.log("DEBUG JS: 'htmx:afterSwap' (RESULTADO FINAL). Animando Y DETENIENDO CORAZÓN.");
            stopAnalyzingStateAndECG(); 

            const iconElement = resultContainer.querySelector('.sentiment-icon');
            const sentimentTextElement = resultContainer.querySelector('.sentiment-text');
            const detailsTextElement = resultContainer.querySelector('.sentiment-details');

            gsap.set([iconElement, sentimentTextElement, detailsTextElement].filter(el => el), { autoAlpha: 0, y: 20 });
            if(iconElement) gsap.set(iconElement, { scale: 0.5 });

            const tl = gsap.timeline();
            if (iconElement) tl.to(iconElement, { autoAlpha: 1, scale: 1, y: 0, duration: 0.5, ease: 'back.out(1.7)'});
            if (sentimentTextElement) tl.to(sentimentTextElement, { autoAlpha: 1, y: 0, duration: 0.4, ease: 'power2.out' }, iconElement ? "-=0.3" : "+=0");
            if (detailsTextElement) tl.to(detailsTextElement, { autoAlpha: 1, y: 0, duration: 0.4, ease: 'power2.out' }, sentimentTextElement ? "-=0.2" : (iconElement ? "-=0.2" : "+=0"));
            
            if (sentimentTextElement && (sentimentTextElement.textContent.trim() === 'Espero tu texto...' || sentimentTextElement.classList.contains('info'))) {
                tl.fromTo(resultContainer, { filter: "blur(5px) brightness(0.5)" }, { filter: "blur(0px) brightness(1)", duration: 0.8, ease: "power2.out" }, 0);
            } else if (iconElement && (iconElement.classList.contains('positive') || iconElement.classList.contains('negative') || iconElement.classList.contains('neutral'))) {
                const color = getComputedStyle(iconElement).color || 'var(--machine-screen-text-color)';
                gsap.fromTo(iconElement, { filter: `drop-shadow(0 0 0px ${color})` }, { filter: `drop-shadow(0 0 15px ${color})`, duration: 0.4, yoyo: true, repeat: 1, ease: "power1.inOut"}, "-=0.3");
            }
        }
    });
    // --- FIN DE LA CORRECCIÓN --- EL BLOQUE FLOTANTE HA SIDO ELIMINADO ---

    // --- 5. Animaciones GSAP Iniciales de la Interfaz ---
    if (typeof gsap !== 'undefined') {
        gsap.from("#sentiment-machine", {
            duration: 1, opacity: 0, y: 50, ease: "power2.out", delay: 0.2
        });
        console.log("DEBUG JS: Animación GSAP inicial de #sentiment-machine configurada.");

        const initialResult = document.getElementById('sentiment-result');
        // Siempre intentar detener el corazón al inicio, haya o no resultado inicial
        console.log(">>>> (Carga Inicial) Intentando detener corazón por defecto <<<<");
        stopHeartbeatAnimation();

        if (initialResult && initialResult.innerHTML.trim() !== '') {
            // Solo simular el evento de swap si hay contenido Y no es el estado 'analyzing'
            if (!initialResult.classList.contains('analyzing-state')) {
                 console.log("DEBUG JS: Hay contenido inicial. Disparando 'htmx:afterSwap' simulado para animación inicial.");
                 const fakeEvent = new CustomEvent('htmx:afterSwap', { 
                    detail: { target: initialResult, elt: initialResult }
                 });
                 document.body.dispatchEvent(fakeEvent);
            } else {
                console.log("DEBUG JS: Contenido inicial es 'analyzing-state', no se simula afterSwap.");
            }
        } else {
            console.log("DEBUG JS: No hay contenido inicial o está vacío en #sentiment-result.");
        }
    } else {
        console.error("ERROR JS: GSAP no está definido. Funcionalidad limitada.");
    }

    // --- 6. Lógica del Botón de Encendido (Decorativo) ---
    // --- Lógica del Botón de Encendido (AHORA DETIENE EL CORAZÓN) ---
    if (powerButton) {
        powerButton.addEventListener('click', function() {
            console.warn("DEBUG JS: Botón Power clickeado - DETENIENDO CORAZÓN MANUALMENTE.");
            stopHeartbeatAnimation(); // <--- CORREGIDO: LLAMAR A stopHeartbeatAnimation
            if (typeof gsap !== 'undefined') {
                gsap.fromTo(this, 
                    { scale: 1, boxShadow: getComputedStyle(this).boxShadow }, 
                    { 
                        scale: 0.9, 
                        boxShadow: "inset 2px 2px 2px var(--dark-neumorphic-button), inset -2px -2px 2px var(--light-neumorphic-button)",
                        duration: 0.1, yoyo: true, repeat: 1, ease: 'power1.inOut',
                        onComplete: () => { gsap.set(this, { clearProps: "scale,boxShadow" }); }
                    }
                );
            }
        });
    }

}); // CIERRE DOMContentLoaded


// --- Listeners de depuración HTMX Globales (FUERA del DOMContentLoaded) ---
// Estos están bien aquí, ya que `document.body` siempre existe cuando se ejecuta el script.
document.body.addEventListener('htmx:beforeRequest', function(evt) {
    console.info('HTMX Event: beforeRequest', { path: evt.detail.pathInfo.path, element: evt.detail.elt, detail: evt.detail });
});
document.body.addEventListener('htmx:afterRequest', function(evt) {
    console.info('HTMX Event: afterRequest', { path: evt.detail.pathInfo.path, element: evt.detail.elt, successful: evt.detail.successful, detail: evt.detail });
    if (evt.detail.failed) {
        console.error('HTMX Error: Request failed.', { status: evt.detail.xhr.status, response: evt.detail.xhr.responseText, detail: evt.detail });
    }
});
document.body.addEventListener('htmx:sendError', function(evt) {
    console.error('HTMX Event: sendError', { path: evt.detail.pathInfo.path, element: evt.detail.elt, error: evt.detail.error, detail: evt.detail });
});
document.body.addEventListener('htmx:responseError', function(evt) { // Cuando el servidor responde con error (4xx, 5xx)
    console.error('HTMX Event: responseError.', { path: evt.detail.pathInfo.path, target: evt.detail.target, status: evt.detail.xhr.status, response: evt.detail.xhr.responseText, detail: evt.detail });
});
document.body.addEventListener('htmx:swapError', function(evt) { // Cuando hay un error aplicando el swap
    console.error('HTMX Event: swapError', { path: evt.detail.pathInfo.path, element: evt.detail.elt, xhr: evt.detail.xhr, detail: evt.detail });
});
// Considera añadir también:
// htmx:configRequest para modificar headers, etc.
// htmx:beforeSwap para inspeccionar el contenido antes de que se inserte.
// htmx:afterSwap (ya lo usas para animar el resultado)