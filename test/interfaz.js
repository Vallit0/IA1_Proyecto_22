import * as tf from '@tensorflow/tfjs-node'; 
import fs from 'fs';
import readline from 'readline';

// ** Paso 1: Cargar el modelo previamente entrenado **
let model;

async function loadModel() {
    console.log('Cargando el modelo...');
    model = await tf.loadLayersModel('file://./chatbot_model/model.json');
    console.log('Modelo cargado correctamente.');
}

// ** Paso 2: Cargar el vocabulario usado para entrenar **
const trainingData = JSON.parse(fs.readFileSync('bilingual.json', 'utf8')).intents;

const vocabulary_es = [];
const vocabulary_en = [];
const labels = trainingData.map(item => item.tag);

// ** Reconstruir el vocabulario (debe coincidir con el proceso de entrenamiento) **
trainingData.forEach((item) => {
    item.patterns.forEach(pattern => {
        const isSpanish = /[áéíóúñ]/.test(pattern) || /¿|¡/.test(pattern); 
        const vocabulary = isSpanish ? vocabulary_es : vocabulary_en;
        const words = pattern.toLowerCase().split(' ');
        words.forEach(word => {
            if (!vocabulary.includes(word)) vocabulary.push(word);
        });
    });
});

console.log('Tamaño del vocabulario español:', vocabulary_es.length);
console.log('Tamaño del vocabulario inglés:', vocabulary_en.length);
console.log('Etiquetas:', labels);

// ** Paso 3: Convertir la entrada del usuario en un bag-of-words **
function createBagOfWords(inputText) {
    const isSpanish = /[áéíóúñ]/.test(inputText) || /¿|¡/.test(inputText); 
    const words = inputText.toLowerCase().split(' ');

    const bagOfWords = new Array(vocabulary_es.length + vocabulary_en.length).fill(0);
    const vocabulary = isSpanish ? vocabulary_es : vocabulary_en;

    words.forEach(word => {
        const index = vocabulary.indexOf(word);
        if (index > -1) bagOfWords[index] = 1;
    });

    // Se agrega la variable de idioma (1 para español, 0 para inglés)
    const languageFeature = isSpanish ? [1] : [0];
    const completeFeatures = bagOfWords.concat(languageFeature);

    return completeFeatures;
}

// ** Paso 4: Usar el modelo para hacer predicciones y generar una respuesta real **
async function predict(inputText) {
    if (!model) {
        console.log('El modelo aún no ha sido cargado.');
        return;
    }

    // ** Generar el bag-of-words para la entrada del usuario **
    const inputFeatures = createBagOfWords(inputText);
    const inputTensor = tf.tensor2d([inputFeatures]);
    
    console.log('Realizando la predicción...');
    const prediction = model.predict(inputTensor);
    
    // Obtener el índice de la etiqueta con la mayor probabilidad
    const predictionIndex = prediction.argMax(1).dataSync()[0];
    const predictedTag = labels[predictionIndex];
    
    // ** Imprimir la probabilidad para todas las etiquetas (opcional) **
    const probabilities = await prediction.data();
    probabilities.forEach((prob, idx) => {
        console.log(`Etiqueta: ${labels[idx]} - Probabilidad: ${prob.toFixed(4)}`);
    });

    // Liberar la memoria del tensor
    inputTensor.dispose();
    prediction.dispose();
    
    // ** Obtener la respuesta en base a la etiqueta predicha **
    const isSpanish = /[áéíóúñ]/.test(inputText) || /¿|¡/.test(inputText); 
    const intent = trainingData.find(item => item.tag === predictedTag);
    
    if (intent) {
        const responses = isSpanish ? intent.response_es : intent.response_en;
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        console.log(`🤖 Respuesta: ${randomResponse}\n`);
        return randomResponse;
    } else {
        console.log('🤖 No tengo una respuesta para eso.');
        return 'No tengo una respuesta para eso.';
    }
}

// ** Paso 5: Interfaz de línea de comandos (consola) **
function startChat() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    console.log('🤖 Bienvenido al chatbot desde la consola.');
    console.log('Escribe tu mensaje y presiona Enter (escribe "salir" para terminar).');

    rl.on('line', async (input) => {
        if (input.toLowerCase() === 'salir') {
            console.log('👋 ¡Hasta pronto!');
            rl.close();
            process.exit(0);
        } else {
            await predict(input);
        }
    });
}

// ** Cargar el modelo y comenzar la conversación **
(async () => {
    await loadModel();
    startChat();
})();

