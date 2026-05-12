"""Replay buffer circular para DQN/DDQN.

Convencion de almacenamiento: guardamos los `state`/`next_state` como
arrays uint8 en CPU (lo que devuelven los wrappers de imagen). La
conversion a float32 [0,1] y el envio a GPU se hace recien al samplear
el minibatch (ver `Agent._sample_batch` en abstract_agent.py).

Por que: con obs 4x84x336 float32 GPU, 50k transiciones x 2 = 45 GB de
VRAM -> OOM seguro. Como uint8 CPU son 11 GB de RAM, manejable.
"""

import random
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "terminated", "next_state")
)

# Ejemplo de uso:
#   t = Transition(state, action, reward, terminated, next_state)
#
# Nota: el flag almacenado es `terminated` (estado terminal del MDP), NO
# `done = terminated or truncated`. El target de Bellman usa (1 - terminated)
# para anular el bootstrap solo cuando el episodio termino por estado
# terminal; un corte por timeout (truncated) deja un s' con valor futuro.


class ReplayMemory:
    def __init__(self, capacity):
        """
        Inicializa la memoria de repeticion con capacidad fija.

        Params:
          - capacity (int): numero maximo de transiciones a almacenar.
        """
        # TODO: almacenar capacity, inicializar lista de memoria y puntero
        # de posicion (para el indice circular).
        raise NotImplementedError("ReplayMemory.__init__ no esta implementado")

    def add(self, state, action, reward, terminated, next_state):
        """
        Agrega una transicion a la memoria.
        Si la memoria esta llena, sobrescribe la transicion mas antigua.

        Espera state/next_state como uint8 numpy.
        """
        # TODO: crear Transition y agregar/reemplazar en la lista segun capacity.
        # TODO: actualizar puntero de posicion circular ((position+1) % capacity).
        raise NotImplementedError("ReplayMemory.add no esta implementado")

    def sample(self, batch_size):
        """
        Devuelve un batch aleatorio de transiciones.

        Params:
          - batch_size (int): numero de transiciones a muestrear.
        Returns:
          - lista de Transition de longitud batch_size.
        """
        # TODO: verificar que batch_size <= len(self).
        # TODO: retornar una muestra aleatoria de self.memory (random.sample).
        raise NotImplementedError("ReplayMemory.sample no esta implementado")

    def __len__(self):
        """Devuelve el numero actual de transiciones en memoria."""
        # TODO: retornar el tamano de la lista de memoria.
        raise NotImplementedError("ReplayMemory.__len__ no esta implementado")

    def clear(self):
        """Elimina todas las transiciones de la memoria."""
        # TODO: resetear lista de memoria y puntero de posicion.
        raise NotImplementedError("ReplayMemory.clear no esta implementado")
