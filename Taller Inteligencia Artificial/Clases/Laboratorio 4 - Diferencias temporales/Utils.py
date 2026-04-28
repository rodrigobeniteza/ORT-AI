"""
Utilidades de visualización y selección de acciones para el notebook de
**Métodos de Diferencias Temporales** (Sutton & Barto, cap. 6).

Convenciones:
- `rewards` es la lista del retorno acumulado por episodio, G_0 = sum_t R_{t+1}.
- `epsilons` es la trayectoria de epsilon a lo largo del entrenamiento.
- Todas las funciones de plotting llaman a `plt.show()` al final, pensadas para
  uso interactivo dentro de Jupyter.
"""

import numpy as np
import matplotlib.pyplot as plt


def _moving_average(rewards, average_range):
    """
    Promedia los retornos en bloques no solapados de tamaño `average_range`.

    Devuelve `(avg, x)` donde `avg[i]` es la media del bloque i-ésimo y `x[i]`
    es el número de episodio en que comienza ese bloque (útil como eje X).

    Detalle: si `len(rewards)` no es múltiplo de `average_range`, se descarta
    la cola incompleta para que el `reshape` funcione.
    """
    rewards = np.asarray(rewards, dtype=float)
    n_full = (len(rewards) // average_range) * average_range
    if n_full == 0:
        # Caso borde: menos episodios que el tamaño de bloque -> un único punto.
        return np.array([np.mean(rewards)]), np.array([0])
    trimmed = rewards[:n_full].reshape(-1, average_range)
    avg = trimmed.mean(axis=1)
    x = np.arange(len(avg)) * average_range
    return avg, x


def plot_rewards(rewards, average_range=1000, title="Episode Accumulated Reward"):
    """
    Curva de aprendizaje: retorno acumulado por episodio, suavizado con media
    móvil de bloques no solapados de `average_range` episodios.

    Útil para ver la tendencia ignorando el ruido propio de la política
    epsilon-greedy.
    """
    avg, x = _moving_average(rewards, average_range)
    plt.plot(x, avg)
    plt.title(title)
    plt.xlabel("Episode Number")
    plt.ylabel(f"Reward (avg cada {average_range} eps.)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_rewards_comparison(
    rewards_list, labels, average_range=1000, title="Comparación de algoritmos"
):
    """
    Superpone las curvas de aprendizaje de varios algoritmos en un mismo eje.

    Pensado para comparaciones tipo Q-Learning vs Sarsa (Fig. 6.4 de S&B):
    todas las series se suavizan con la misma ventana para que la comparación
    sea justa.
    """
    plt.figure()
    for rewards, label in zip(rewards_list, labels):
        avg, x = _moving_average(rewards, average_range)
        plt.plot(x, avg, label=label)
    plt.title(title)
    plt.xlabel("Episode Number")
    plt.ylabel(f"Reward (avg cada {average_range} eps.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_epsilon(epsilons):
    """
    Grafica el decaimiento de epsilon a lo largo de los episodios.

    Es útil para verificar que el agente realmente termina explotando
    (i.e. epsilon -> epsilon_min) hacia el final del entrenamiento.
    """
    plt.plot(epsilons)
    plt.title(r"$\varepsilon$ a lo largo de los episodios")
    plt.xlabel("Episode Number")
    plt.ylabel(r"$\varepsilon$")
    plt.grid(True, alpha=0.3)
    plt.show()


def random_argmax(values):
    """
    `argmax` con **desempate aleatorio uniforme** entre las acciones óptimas.

    ¿Por qué no usar `np.argmax` directamente? Al inicio del entrenamiento la
    tabla Q(s,a) está inicializada en cero, por lo que **todas** las acciones
    empatan. `np.argmax` siempre devuelve el primer índice, lo que sesga la
    exploración hacia la acción 0 y rompe la garantía de cobertura uniforme
    de la política epsilon-greedy.

    Args:
        values: vector de Q-values para un estado fijo, shape `(n_actions,)`.

    Returns:
        Índice (int) de una acción óptima, elegido uniformemente entre las que
        empatan en el máximo.
    """
    values = np.asarray(values)
    max_value = values.max()
    # `flatnonzero` devuelve los índices donde la condición es True.
    candidates = np.flatnonzero(values == max_value)
    return int(np.random.choice(candidates))
