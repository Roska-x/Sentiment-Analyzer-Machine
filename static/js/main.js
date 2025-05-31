// static/js/main.js
document.addEventListener('DOMContentLoaded', function () {
    console.log("DEBUG: main.js cargado y DOM listo.");

    const visibleTextarea = document.getElementById('visibleCommentInput');
    const formTextarea = document.getElementById('formCommentText');
    const analyzeButton = document.getElementById('analyzeButtonVisible');
    const sentimentForm = document.getElementById('sentiment-form');

    // Verificaciones de elementos críticos
    if (!visibleTextarea) {
        console.error("ERROR JS: No se encontró visibleTextarea #visibleCommentInput");
    }
    if (!formTextarea) {
        console.error("ERROR JS: No se encontró formTextarea #formCommentText");
    }
    if (!analyzeButton) {
        console.error("ERROR JS: No se encontró analyzeButton #analyzeButtonVisible");
    }
    if (!sentimentForm) {
        console.error("ERROR JS: No se encontró sentimentForm #sentiment-form");
    }

    // Lógica principal del botón de análisis
    if (analyzeButton && visibleTextarea && formTextarea && sentimentForm) {
        analyzeButton.addEventListener('click', function () {
            console.log("DEBUG JS: Botón Analizar clickeado!");

            // 1. Copiar el texto del textarea visible al textarea del formulario
            const textToAnalyze = visibleTextarea.value;
            formTextarea.value = textToAnalyze;
            console.log("DEBUG JS: Texto copiado al form textarea: ", textToAnalyze);

            // 2. Disparar el evento 'submit' en el formulario HTMX
            console.log("DEBUG JS: Disparando submit en el formulario HTMX...");
            htmx.trigger(sentimentForm, 'submit'); // HTMX interceptará esto
            // Alternativamente: sentimentForm.requestSubmit(); si htmx.trigger no funciona
            //                     asegúrate que el hx-trigger del form es 'submit'
            console.log("DEBUG JS: Evento submit disparado.");
        });
    } else {
        // Este mensaje es más útil si alguna de las verificaciones de arriba falló.
        console.error("ERROR JS: Uno o más elementos críticos no fueron encontrados (ver errores anteriores). La funcionalidad de análisis NO se activará.");
    }

    // Animación GSAP inicial
    // Asegúrate que GSAP está cargado (debería estarlo si el CDN está en index.html)
    if (typeof gsap !== 'undefined') {
        gsap.from("#sentiment-machine", {
            duration: 1.2, // Un poco más rápido
            opacity: 0,
            y: 60, // Un poco más de movimiento
            ease: "power2.out", // Un ease diferente
            delay: 0.3 // Menos delay
        });
        console.log("DEBUG JS: Animación GSAP configurada.");
    } else {
        console.error("ERROR JS: GSAP no está definido. La animación no se ejecutará.");
    }

    // Lógica del botón de encendido (placeholder)
    const powerButton = document.querySelector('.machine-button.power');
    if (powerButton) {
        powerButton.addEventListener('click', function() {
            console.log("DEBUG JS: Botón power clickeado (acción decorativa por ahora)");
            // Podrías añadir alguna animación o efecto aquí
        });
    }

});

// Listeners de eventos HTMX para depuración (¡muy útiles!)
document.body.addEventListener('htmx:beforeRequest', function(evt) {
    console.info('HTMX Event: beforeRequest', evt.detail.pathInfo.path, evt.detail);
});
document.body.addEventListener('htmx:afterRequest', function(evt) {
    console.info('HTMX Event: afterRequest', evt.detail.pathInfo.path, evt.detail);
    if (evt.detail.failed) {
        console.error('HTMX Error: Request failed. Status:', evt.detail.xhr.status, 'Response:', evt.detail.xhr.responseText, evt.detail);
    }
    if (evt.detail.successful) {
        console.log('HTMX Success: Request successful. Status:', evt.detail.xhr.status, evt.detail);
    }
});
document.body.addEventListener('htmx:sendError', function(evt) {
    console.error('HTMX Event: sendError', evt.detail.pathInfo.path, evt.detail.error, evt.detail);
});
document.body.addEventListener('htmx:responseError', function(evt) {
    console.error('HTMX Event: responseError. Path:', evt.detail.pathInfo.path, 'Status:', evt.detail.xhr.status, 'Server response:', evt.detail.xhr.responseText, evt.detail);
});