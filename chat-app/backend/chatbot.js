import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { intents } from './intents.js';
import { responses } from './responses.js';

// Cargar el modelo de Universal Sentence Encoder
let model;
use.load().then((loadedModel) => {
  model = loadedModel;
  console.log('Modelo cargado');
});

// Verifica que input sea válido y que el modelo esté cargado
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

  const response = await generateResponse(userInput);
  return response;
}

async function recognizeIntent(userInput) {
  if (!model) {
    throw new Error('El modelo no está cargado. Por favor, espere.');
  }

  const userInputEmb = await model.embed([userInput]);
  let maxScore = -1;
  let recognizedIntent = null;

  for (const [intent, examples] of Object.entries(intents)) {
    const examplesEmb = await model.embed(examples);
    const scores = await tf.matMul(userInputEmb, examplesEmb, false, true).data();
    const maxExampleScore = Math.max(...scores);
    if (maxExampleScore > maxScore) {
      maxScore = maxExampleScore;
      recognizedIntent = intent;
    }
  }

  return recognizedIntent;
}

async function generateResponse(userInput) {
  const intent = await recognizeIntent(userInput);
  if (intent && responses[intent]) {
    return responses[intent];
  } else {
    return "Lo siento, no entiendo eso. ¿Puedes reformularlo?";
  }
}
