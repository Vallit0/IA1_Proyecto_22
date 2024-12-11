import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

// Obtener el nombre del archivo actual para construir las rutas correctamente
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Habilitar CORS para todas las solicitudes
app.use(cors());

// Servir archivos estáticos (HTML, JS, Modelo)
app.use(express.static(path.join(__dirname, '')));

// Iniciar el servidor en el puerto 8080
app.listen(8080, () => {
    console.log('Servidor corriendo en http://localhost:8080');
});

