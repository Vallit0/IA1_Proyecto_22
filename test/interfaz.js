import * as tf from '@tensorflow/tfjs-node'; 
import fs from 'fs';
import readline from 'readline';

let model;

// ** Cargar modelo y vocabulario **
async function loadModel() {
    console.log('Cargando el modelo...');
    model = await tf.loadLayersModel('file://./chatbot_model/model.json');
    console.log('Modelo cargado correctamente.');
    const vocabulary = JSON.parse(fs.readFileSync('vocabulary.json', 'utf8'));
    return vocabulary;
}

function createBagOfWords(inputText, vocabulary) {
    const words = inputText.toLowerCase().split(' ');
    const bagOfWords = new Array(vocabulary.length).fill(0);
    words.forEach(word => {
        const index = vocabulary.indexOf(word);
        if (index > -1) bagOfWords[index] = 1;
    });

    // ** Agregar la variable binaria (idioma) **
    bagOfWords.push(0); // No sabemos si es inglés o español en esta fase
    return bagOfWords;
}

async function predict(inputText, vocabulary) {
    const inputFeatures = createBagOfWords(inputText, vocabulary);
    const inputTensor = tf.tensor2d([inputFeatures]);
    
    const [intentPrediction, languagePrediction] = model.predict(inputTensor);

    // ** Predicción de la intención **
    const intentProbabilities = await intentPrediction.data();
    const intentIndex = intentPrediction.argMax(1).dataSync()[0];
    
    // ** Predicción del idioma **
    const languageProb = languagePrediction.dataSync()[0];
    const isSpanish = languageProb >= 0.5;

    // ** Imprimir la probabilidad de todas las intenciones **
    console.log('🔍 Probabilidades de cada intención:');
    intentProbabilities.forEach((prob, index) => {
        console.log(`Intención ${index}: ${prob.toFixed(4)}`);
    });

    // ** Imprimir la probabilidad de idioma **
    console.log('🌐 Probabilidad de español: ', languageProb.toFixed(4));
    console.log('🌐 Probabilidad de inglés: ', (1 - languageProb).toFixed(4));

    // ** Imprimir la etiqueta predicha **
    const response = isSpanish ? "Idioma detectado: Español" : "Idioma detectado: Inglés";
    console.log(`🤖 Intención predicha: ${intentIndex}`);
    console.log(`🤖 ${response}`);
    
    // ** Liberar la memoria de los tensores para evitar fuga de memoria **
    inputTensor.dispose();
    intentPrediction.dispose();
    languagePrediction.dispose();

    return response;
}

function startChat(vocabulary) {
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
            const response = await predict(input, vocabulary);
            console.log(`🤖 Respuesta: ${response}\n`);
        }
    });
}

(async () => {
    const vocabulary = await loadModel();
    startChat(vocabulary);
})();

