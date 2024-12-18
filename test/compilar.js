import * as tf from '@tensorflow/tfjs-node'; 
import fs from 'fs';

const trainingData = JSON.parse(fs.readFileSync('bilingual.json', 'utf8')).intents;

const vocabulary_es = [];
const vocabulary_en = [];
const labels = [];

trainingData.forEach((item) => {
    item.patterns.forEach(pattern => {
        const isSpanish = /[áéíóúñ]/.test(pattern) || /¿|¡/.test(pattern);
        const vocabulary = isSpanish ? vocabulary_es : vocabulary_en;
        const words = pattern.toLowerCase().split(' ');
        words.forEach(word => {
            if (!vocabulary.includes(word)) vocabulary.push(word);
        });
    });
    if (!labels.includes(item.tag)) labels.push(item.tag);
});

const xs = [];
const ys_intent = [];
const ys_language = [];

trainingData.forEach((item) => {
    item.patterns.forEach((pattern) => {
        const isSpanish = /[áéíóúñ]/.test(pattern) || /¿|¡/.test(pattern);
        const words = pattern.toLowerCase().split(' ');

        const bagOfWords = new Array(vocabulary_es.length + vocabulary_en.length).fill(0);
        words.forEach(word => {
            const index = (isSpanish ? vocabulary_es : vocabulary_en).indexOf(word);
            if (index > -1) bagOfWords[index] = 1;
        });

        const languageFeature = isSpanish ? [1] : [0];
        const completeFeatures = bagOfWords.concat(languageFeature);
        
        xs.push(completeFeatures);
        
        // ** Codificación One-Hot de la intención (tag) **
        const labelIndex = labels.indexOf(item.tag);
        const oneHotLabel = new Array(labels.length).fill(0);
        oneHotLabel[labelIndex] = 1;
        ys_intent.push(oneHotLabel);

        // ** Etiqueta de idioma (0 = inglés, 1 = español) **
        ys_language.push([isSpanish ? 1 : 0]);
    });
});

const xsTensor = tf.tensor2d(xs);
const ysIntentTensor = tf.tensor2d(ys_intent);
const ysLanguageTensor = tf.tensor2d(ys_language);

const totalVocabularySize = vocabulary_es.length + vocabulary_en.length + 1;

// ** Modelo con dos salidas: predicción de la intención y predicción del idioma **
const input = tf.input({ shape: [totalVocabularySize] });
const hidden1 = tf.layers.dense({ units: 32, activation: 'relu' }).apply(input);
const hidden2 = tf.layers.dense({ units: 32, activation: 'relu' }).apply(hidden1);
const hidden3 = tf.layers.dense({ units: 16, activation: 'relu' }).apply(hidden2);

const intentOutput = tf.layers.dense({ units: labels.length, activation: 'softmax', name: 'intent_output' }).apply(hidden3);
const languageOutput = tf.layers.dense({ units: 1, activation: 'sigmoid', name: 'language_output' }).apply(hidden3);

const model = tf.model({ inputs: input, outputs: [intentOutput, languageOutput] });

model.compile({
    optimizer: 'adam',
    loss: ['categoricalCrossentropy', 'binaryCrossentropy'],
    metrics: ['accuracy']
});

async function trainModel() {
    console.log('Entrenando el modelo...');
    await model.fit(xsTensor, { intent_output: ysIntentTensor, language_output: ysLanguageTensor }, { 
        epochs: 100, 
        batchSize: 16,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const intentAcc = logs.intent_output_acc !== undefined ? logs.intent_output_acc.toFixed(4) : 'N/A';
                const languageAcc = logs.language_output_acc !== undefined ? logs.language_output_acc.toFixed(4) : 'N/A';
                const intentLoss = logs.intent_output_loss !== undefined ? logs.intent_output_loss.toFixed(4) : 'N/A';
                const languageLoss = logs.language_output_loss !== undefined ? logs.language_output_loss.toFixed(4) : 'N/A';
                const totalLoss = logs.loss !== undefined ? logs.loss.toFixed(4) : 'N/A';
                
                console.log(`Época: ${epoch + 1} - Pérdida Total: ${totalLoss}`);
                console.log(`Intención - Precisión: ${intentAcc} - Pérdida: ${intentLoss}`);
                console.log(`Idioma - Precisión: ${languageAcc} - Pérdida: ${languageLoss}`);
            }
        }
    });
    await model.save('file://./chatbot_model');
    console.log('Modelo guardado correctamente.');
}

trainModel();

