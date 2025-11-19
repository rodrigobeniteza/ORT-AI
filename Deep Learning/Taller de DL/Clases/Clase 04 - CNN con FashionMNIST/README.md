# Clase 04 - Redes Neuronales Convolucionales con FashionMNIST

Este directorio contiene el material para la clase sobre Redes Neuronales Convolucionales (CNNs) utilizando el dataset FashionMNIST.

## Estructura del Directorio

```
.
├── CNNs_Intro-letra.ipynb    # Notebook principal con el contenido de la clase
├── utils.py                  # Funciones de utilidad para entrenamiento y visualización
├── assets/                   # Directorio con imágenes y recursos
└── data/                    # Directorio donde se almacenan los datasets
```

## Contenido Principal

### CNNs_Intro-letra.ipynb

Notebook interactivo que cubre:
- Introducción a las Redes Neuronales Convolucionales
- Conceptos fundamentales de convoluciones y pooling
- Implementación de LeNet-5 en PyTorch
- Entrenamiento y evaluación con FashionMNIST

### utils.py

Módulo de utilidades que proporciona funciones para:

1. **Entrenamiento y Evaluación**
   - `train()`: Función principal de entrenamiento con soporte para:
     - Early stopping
     - Logging personalizado
     - Métricas de entrenamiento y validación
   - `evaluate()`: Evaluación del modelo en conjuntos de datos
   - `EarlyStopping`: Clase para implementar early stopping

2. **Visualización**
   - `show_tensor_image()`: Visualización de imágenes individuales
   - `show_tensor_images()`: Visualización de múltiples imágenes
   - `plot_training()`: Gráficos de pérdida durante el entrenamiento

3. **Métricas**
   - `model_classification_report()`: Genera reportes de clasificación detallados

## Requisitos

- PyTorch
- torchvision
- matplotlib
- scikit-learn
- torchinfo

## Uso del Dataset FashionMNIST

El dataset FashionMNIST consiste en:
- 60,000 imágenes de entrenamiento
- 10,000 imágenes de prueba
- 10 clases de artículos de moda
- Imágenes en escala de grises de 28x28 píxeles

## Arquitectura LeNet

La implementación incluye una versión adaptada de LeNet-5 para FashionMNIST con:
- Capas convolucionales con kernels de 5x5
- Capas de pooling promedio
- Función de activación tanh
- Capas fully connected finales

## Utilización

1. Asegúrate de tener todas las dependencias instaladas
2. Ejecuta las celdas del notebook en orden
3. Los datos se descargarán automáticamente en el directorio `data/`
4. Sigue las instrucciones y ejercicios propuestos en el notebook

## Ejercicios Propuestos

1. Implementación de una CNN más profunda
2. Experimentación con Weight and Bias para seguimiento de experimentos

## Notas Adicionales

- El código está optimizado para ejecutarse en CPU, GPU (CUDA) o MPS según disponibilidad
- Incluye manejo de workers para carga de datos optimizada según el sistema operativo
- Implementa las mejores prácticas de PyTorch para entrenamiento de modelos
