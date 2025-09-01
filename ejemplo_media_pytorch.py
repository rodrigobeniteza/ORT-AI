import torch
import matplotlib.pyplot as plt

def modificar_media_tensor(tensor, nueva_media, dim=None):
    """
    Modifica la media de un tensor
    
    Args:
        tensor: Tensor de PyTorch
        nueva_media: Nueva media deseada
        dim: Dimensión a lo largo de la cual calcular la media (None para media global)
    
    Returns:
        Tensor con la nueva media
    """
    if dim is None:
        # Media global
        media_actual = tensor.mean()
        return tensor - media_actual + nueva_media
    else:
        # Media por dimensión específica
        media_actual = tensor.mean(dim=dim, keepdim=True)
        return tensor - media_actual + nueva_media

# Ejemplo 1: Tensor 1D
print("=== Ejemplo 1: Tensor 1D ===")
tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Tensor original: {tensor_1d}")
print(f"Media original: {tensor_1d.mean():.2f}")

tensor_1d_modificado = modificar_media_tensor(tensor_1d, 10.0)
print(f"Tensor modificado: {tensor_1d_modificado}")
print(f"Nueva media: {tensor_1d_modificado.mean():.2f}")

# Ejemplo 2: Tensor 2D
print("\n=== Ejemplo 2: Tensor 2D ===")
tensor_2d = torch.randn(3, 4)
print(f"Tensor original:\n{tensor_2d}")
print(f"Media original: {tensor_2d.mean():.2f}")

# Modificar media global
tensor_2d_global = modificar_media_tensor(tensor_2d, 5.0)
print(f"\nCon nueva media global (5.0):\n{tensor_2d_global}")
print(f"Nueva media global: {tensor_2d_global.mean():.2f}")

# Modificar media por filas
tensor_2d_filas = modificar_media_tensor(tensor_2d, 0.0, dim=1)
print(f"\nCon media 0 por filas:\n{tensor_2d_filas}")
print(f"Medias por fila: {tensor_2d_filas.mean(dim=1)}")

# Ejemplo 3: Normalización (media 0, std 1)
print("\n=== Ejemplo 3: Normalización ===")
tensor_random = torch.randn(100)
print(f"Media original: {tensor_random.mean():.2f}")
print(f"Std original: {tensor_random.std():.2f}")

# Normalizar: media 0, std 1
tensor_normalizado = (tensor_random - tensor_random.mean()) / tensor_random.std()
print(f"Media normalizada: {tensor_normalizado.mean():.2f}")
print(f"Std normalizada: {tensor_normalizado.std():.2f}")

# Cambiar a nueva media y std
nueva_media_deseada = 100.0
nueva_std_deseada = 15.0
tensor_escalado = tensor_normalizado * nueva_std_deseada + nueva_media_deseada
print(f"Media final: {tensor_escalado.mean():.2f}")
print(f"Std final: {tensor_escalado.std():.2f}")

# Ejemplo 4: Trabajando con gradientes
print("\n=== Ejemplo 4: Con gradientes ===")
tensor_grad = torch.randn(5, requires_grad=True)
print(f"Tensor con gradientes: {tensor_grad}")
print(f"Media original: {tensor_grad.mean():.2f}")

# Modificar media manteniendo gradientes
nueva_media = 3.0
tensor_grad_modificado = tensor_grad - tensor_grad.mean() + nueva_media
print(f"Tensor modificado: {tensor_grad_modificado}")
print(f"Nueva media: {tensor_grad_modificado.mean():.2f}")
print(f"¿Requiere gradientes?: {tensor_grad_modificado.requires_grad}")


