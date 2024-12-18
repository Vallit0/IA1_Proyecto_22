import * as tf from '@tensorflow/tfjs-node'; 
import fs from 'fs';
import readline from 'readline';

let model;

// ** Paso 1: Cargar el modelo previamente entrenado **
async function loadModel() {
    console.log('Cargando el modelo...');
    model = await tf.loadLayersModel('file://./chatbot_model/model.json');
    console.log('Modelo cargado correctamente.');
}

const trainingData = JSON.parse(fs.readFileSync('bilingual.json', 'utf8')).intents;

const vocabulary_es = [];
const vocabulary_en = [];

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

// ** Paso 2: Crear el bag-of-words para la entrada del usuario **
function createBagOfWords(inputText) {
    const isSpanish = /[áéíóúñ]/.test(inputText) || /¿|¡/.test(inputText);
    const words = inputText.toLowerCase().split(' ');

    const bagOfWords = new Array(vocabulary_es.length + vocabulary_en.length).fill(0);
    words.forEach(word => {
        const index = (isSpanish ? vocabulary_es : vocabulary_en).indexOf(word);
        if (index > -1) bagOfWords[index] = 1;
    });

    // ** Agregar la variable de idioma (1 para español, 0 para inglés) **
    const languageFeature = isSpanish ? [1] : [0];
    const completeFeatures = bagOfWords.concat(languageFeature);

    // Asegúrate de que la longitud sea consistente con la entrada que espera el modelo
    while (completeFeatures.length < vocabulary_es.length + vocabulary_en.length + 1) {
        completeFeatures.push(0);
    }

    console.log('Tamaño de bag-of-words generado:', completeFeatures.length);
    return completeFeatures;
}

// ** Paso 3: Usar el modelo para hacer predicciones de intención e idioma **
async function predict(inputText) {
    if (!model) {
        console.log('El modelo aún no ha sido cargado.');
        return;
    }

    const inputFeatures = createBagOfWords(inputText);
    const inputTensor = tf.tensor2d([inputFeatures]);
    console.log('Forma del tensor de entrada:', inputTensor.shape); 

    // ** Realizar la predicción de la intención y el idioma **
    const [intentPrediction, languagePrediction] = model.predict(inputTensor);
    const intentIndex = intentPrediction.argMax(1).dataSync()[0];

    // ** Etiquetas de la intención (extraer del dataset) **
    const labels = trainingData.map(intent => intent.tag);
    const predictedTag = labels[intentIndex];

    // ** Detectar el idioma (0 = inglés, 1 = español) **
    const languageProb = languagePrediction.dataSync()[0];
    const isSpanish = languageProb >= 0.5; // español si probabilidad >= 0.5

    console.log(`Etiqueta predicha: ${predictedTag}`);
    console.log(`Idioma detectado: ${isSpanish ? 'Español' : 'Inglés'} (probabilidad: ${languageProb.toFixed(4)})\n`);

    inputTensor.dispose();
    return predictedTag;
}

// ** Paso 4: Interfaz de línea de comandos (consola) **
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

