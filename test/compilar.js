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
const ys = [];

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
        
        const labelIndex = labels.indexOf(item.tag);
        const oneHotLabel = new Array(labels.length).fill(0);
        oneHotLabel[labelIndex] = 1;
        ys.push(oneHotLabel);
    });
});

const xsTensor = tf.tensor2d(xs);
const ysTensor = tf.tensor2d(ys);

const model = tf.sequential();
const totalVocabularySize = vocabulary_es.length + vocabulary_en.length + 1;

model.add(tf.layers.dense({ inputShape: [totalVocabularySize], units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));

model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

async function trainModel() {
    await model.fit(xsTensor, ysTensor, { epochs: 2000 });
    await model.save('file://./chatbot_model');
}

trainModel();

