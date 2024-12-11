import * as tf from '@tensorflow/tfjs-node'; // Para Node.js
import fs from 'fs';

const trainingData = JSON.parse(fs.readFileSync('content2.json', 'utf8')).intents;

const vocabulary = [];
const labels = [];

trainingData.forEach((item, index) => {
    item.patterns.forEach(pattern => {
        const words = pattern.toLowerCase().split(' ');
        words.forEach(word => {
            if (!vocabulary.includes(word)) {
                vocabulary.push(word);
            }
        });
    });
    labels.push(item.tag);
});

const xs = [];
const ys = [];

trainingData.forEach((item) => {
    item.patterns.forEach((pattern) => {
        const words = pattern.toLowerCase().split(' ');
        const bagOfWords = vocabulary.map(word => words.includes(word) ? 1 : 0);
        xs.push(bagOfWords);
        
        const labelIndex = labels.indexOf(item.tag);
        const oneHotLabel = new Array(labels.length).fill(0);
        oneHotLabel[labelIndex] = 1;
        ys.push(oneHotLabel);
    });
});

const xsTensor = tf.tensor2d(xs);
const ysTensor = tf.tensor2d(ys);

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [vocabulary.length], units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));

model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

async function trainModel() {
    console.log('Entrenando el modelo...');
    await model.fit(xsTensor, ysTensor, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Época: ${epoch + 1} Pérdida: ${logs.loss.toFixed(4)} Precisión: ${logs.acc.toFixed(4)}`);
            }
        }
    });
    console.log('Entrenamiento completado!');
    
    // Guardar el modelo
    try {
        await model.save('file://./chatbot_model');
        console.log('Modelo guardado!');
    } catch (err) {
        console.error('Error al guardar el modelo:', err);
    }
}

trainModel();

