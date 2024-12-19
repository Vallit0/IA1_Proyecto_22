import * as tf from '@tensorflow/tfjs-node'; 
import fs from 'fs';

// ** Leer los datos de entrenamiento desde el archivo JSON **
const trainingData = JSON.parse(fs.readFileSync('bilingual.json', 'utf8')).intents;

const vocabulary = [];
const labels = [];

trainingData.forEach((item) => {
    item.patterns.forEach(pattern => {
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
        const words = pattern.toLowerCase().split(' ');

        // ** Crear el bag-of-words basado en el vocabulario **
        const bagOfWords = new Array(vocabulary.length).fill(0);
        words.forEach(word => {
            const index = vocabulary.indexOf(word);
            if (index > -1) bagOfWords[index] = 1;
        });

        // ** Agregar la característica del idioma (1 para español, 0 para inglés) **
        const isSpanish = item.response_es && item.response_es.length > 0 ? 1 : 0;
        bagOfWords.push(isSpanish);
        
        xs.push(bagOfWords);

        // ** Codificación One-Hot de la intención (tag) **
        const labelIndex = labels.indexOf(item.tag);
        const oneHotLabel = new Array(labels.length).fill(0);
        oneHotLabel[labelIndex] = 1;
        ys_intent.push(oneHotLabel);

        // ** Etiqueta de idioma (0 = inglés, 1 = español) **
        ys_language.push([isSpanish]);
    });
});

const xsTensor = tf.tensor2d(xs);
const ysIntentTensor = tf.tensor2d(ys_intent);
const ysLanguageTensor = tf.tensor2d(ys_language);

const totalVocabularySize = vocabulary.length + 1;

const input = tf.input({ shape: [totalVocabularySize] });
const hidden1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(input);
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
    await model.fit(xsTensor, { intent_output: ysIntentTensor, language_output: ysLanguageTensor }, { epochs: 50 });
    await model.save('file://./chatbot_model');
    fs.writeFileSync('vocabulary.json', JSON.stringify(vocabulary));
}

trainModel();

