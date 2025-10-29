# 🚀 Guía de uso del entorno con GPU

## ✅ Configuración completada

Se ha configurado un entorno virtual Python 3.12 con soporte CUDA para tu **NVIDIA GeForce RTX 3070**.

### Librerías instaladas:
- ✅ PyTorch 2.5.1 + CUDA 12.1
- ✅ torchvision, torchaudio
- ✅ torchinfo, pandas, matplotlib
- ✅ scikit-learn, nltk
- ✅ wandb (Weights & Biases)
- ✅ Jupyter Lab

---

## 🎯 Cómo usar el entorno

### Opción 1: Usar el script (MÁS FÁCIL)

Simplemente haz **doble clic** en:
```
start_jupyter_gpu.bat
```

Esto:
1. Activará el entorno virtual
2. Verificará que la GPU esté detectada
3. Abrirá Jupyter Lab automáticamente

---

### Opción 2: Manual (PowerShell)

1. **Abrir PowerShell** en esta carpeta

2. **Activar el entorno:**
```powershell
.\venv_gpu\Scripts\Activate.ps1
```

3. **Verificar GPU (opcional):**
```powershell
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

4. **Iniciar Jupyter:**
```powershell
jupyter lab
```

---

## 📓 Usar en un notebook existente

Cuando abras tu notebook en Jupyter:

1. Ve al menú superior derecho donde dice el kernel actual
2. Click en **"Kernel"** → **"Change kernel"**
3. Selecciona: **"Python 3.12 (ORT-AI GPU)"**
4. ¡Listo! Ahora tu notebook usará la GPU

---

## 🔍 Verificar que la GPU esté funcionando

En cualquier notebook, ejecuta:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No detectada'}")
print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

Deberías ver:
```
PyTorch version: 2.5.1+cu121
CUDA disponible: True
GPU: NVIDIA GeForce RTX 3070
Memoria GPU: 8.00 GB
```

---

## ⚠️ Importante

- **Siempre activa el entorno** antes de trabajar
- El entorno se llama: `venv_gpu`
- El kernel de Jupyter se llama: **"Python 3.12 (ORT-AI GPU)"**
- Si trabajas desde terminal, recuerda activar el entorno primero

---

## 🛠️ Comandos útiles

### Desactivar el entorno:
```powershell
deactivate
```

### Instalar nuevas librerías:
```powershell
# 1. Activar entorno
.\venv_gpu\Scripts\Activate.ps1

# 2. Instalar librería
python -m pip install nombre_libreria
```

### Listar librerías instaladas:
```powershell
pip list
```

---

## 🎓 Para tu tarea

Tu notebook **"Entrega_1_Lenet_&_DenseNet (funcional).ipynb"** ya está actualizado con mejor detección de GPU.

Cuando ejecutes la celda de configuración, verás:
```
✓ GPU detectada: NVIDIA GeForce RTX 3070
  - Número de GPUs disponibles: 1
  - Memoria total: 8.00 GB
  - CUDA Version: 12.1

→ Dispositivo seleccionado: CUDA
```

---

## ❓ Solución de problemas

### Si no detecta la GPU:
1. Verifica que activaste el entorno `venv_gpu`
2. Verifica que el kernel de Jupyter sea **"Python 3.12 (ORT-AI GPU)"**
3. Reinicia el kernel del notebook

### Si Jupyter no abre:
1. Asegúrate de estar en la carpeta correcta
2. Activa el entorno primero
3. Ejecuta: `jupyter lab`

---

¡Listo para entrenar tus modelos con GPU! 🚀

