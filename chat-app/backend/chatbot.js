import * as tf from '@tensorflow/tfjs';

let model;
let vocabulary = [];
let labels = [];
let trainingData = [];

// Cargar el modelo desde GitHub
async function loadModel() {
  try {
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/Vallit0/IA1_PROYECTO/main/model.json');
    console.log('Modelo cargado exitosamente');
  } catch (error) {
    console.error('Error al cargar el modelo:', error);
  }
}

// Cargar los datos de entrenamiento desde el archivo JSON
async function loadTrainingData() {
  try {
    const response = await fetch('https://raw.githubusercontent.com/Vallit0/IA1_PROYECTO/main/content2.json');
    const data = await response.json();
    trainingData = data.intents;

    // Construir vocabulario y etiquetas
    trainingData.forEach((item) => {
      item.patterns.forEach((pattern) => {
        const words = pattern.toLowerCase().split(' ');
        words.forEach((word) => {
          if (!vocabulary.includes(word)) {
            vocabulary.push(word);
          }
        });
      });
      if (!labels.includes(item.tag)) {
        labels.push(item.tag);
      }
    });

    console.log('Datos de entrenamiento cargados:', { vocabulary, labels });
  } catch (error) {
    console.error('Error al cargar el archivo JSON:', error);
  }
}

// Preprocesar la entrada del usuario
function preprocessInput(input) {
  const words = input.toLowerCase().split(' ');
  const bagOfWords = vocabulary.map((word) => (words.includes(word) ? 1 : 0));
  return tf.tensor2d([bagOfWords]);
}

// Hacer la predicción
function predictTag(input) {
  const inputTensor = preprocessInput(input);
  const prediction = model.predict(inputTensor);
  const predictedIndex = prediction.argMax(1).dataSync()[0];
  return labels[predictedIndex];
}

// Obtener una respuesta basada en la etiqueta
function getResponse(tag) {
  const intent = trainingData.find((intent) => intent.tag === tag);
  if (intent) {
    const responses = intent.response;
    return responses[Math.floor(Math.random() * responses.length)];
  }
  return "Lo siento, no entiendo tu pregunta.";
}

// Exportar la función del chatbot
export async function startChatbot(input = '') {
  if (!model) {
    throw new Error('El modelo no está cargado. Por favor, espere.');
  }
  if (!input) {
    throw new Error('El input está vacío o no es válido.');
  }

  const userInput = input.trim();
  if (!userInput) {
    throw new Error('El input después de aplicar trim() está vacío.');
  }

  try {
    const predictedTag = predictTag(userInput);
    const response = getResponse(predictedTag);
    return response;
  } catch (error) {
    console.error('Error al procesar el mensaje:', error);
    return 'Hubo un problema al procesar tu mensaje.';
  }
}

// Inicializar la aplicación
(async function initialize() {
  await loadModel();
  await loadTrainingData();
})();