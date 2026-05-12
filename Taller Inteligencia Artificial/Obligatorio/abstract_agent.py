"""Clase base para agentes DQN/DDQN.

Centraliza el loop de entrenamiento, la politica epsilon-greedy y la
maquinaria de muestreo del replay buffer. Las subclases solo definen
`update_weights` (DQN: target con la propia red; DDQN: target con
target_net y argmax con online).

Convenciones:
- Las observaciones se guardan en el buffer como uint8 numpy (CPU).
  La conversion a float32 [0,1] y el .to(device) se difieren al sample
  para no inflar VRAM/RAM.
- `obs_processing_func` (Phi) acepta tanto una obs sola como un batch:
    * input shape == obs_shape          -> retorna (1, *obs_shape)
    * input shape == (B, *obs_shape)    -> retorna (B, *obs_shape)
  Devuelve un tensor float32 en [0,1] en CPU; el .to(device) lo hace el
  Agent. Asi `greedy_action` (1 obs) y `_sample_batch` (batch) comparten
  exactamente la misma logica de normalizacion.
- El checkpoint final se guarda en `self.checkpoint_path` (configurable
  via __init__).
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from replay_memory import ReplayMemory


class Agent(ABC):
    def __init__(
        self,
        gym_env,
        obs_processing_func,
        memory_buffer_size,
        batch_size,
        learning_rate,
        gamma,
        epsilon_i,
        epsilon_f,
        epsilon_anneal_steps,
        episode_block,
        device,
        learning_starts: int = 1_000,
        max_grad_norm: float | None = None,
        checkpoint_path: str = "agent.dat",
    ):
        self.device = device
        # Phi: convierte obs(s) uint8 numpy -> float32 tensor [0,1] en CPU,
        # con batch dim. Acepta una sola obs o un batch (ver docstring del
        # modulo). El .to(device) lo agrega el Agent.
        self.state_processing_function = obs_processing_func
        self.memory = ReplayMemory(memory_buffer_size)
        self.env = gym_env

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.episode_block = episode_block
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = checkpoint_path

        self.total_steps = 0

    def train(
        self, number_episodes=50_000, max_steps_episode=10_000, max_steps=1_000_000
    ):
        """
        Loop de entrenamiento estandar de DQN/DDQN.

        Esquema esperado por episodio:
            1) reset del env -> state inicial
            2) loop de hasta max_steps_episode pasos:
                a) seleccionar accion con `select_action(state, total_steps, train=True)`
                b) ejecutar accion: next_state, reward, terminated, truncated, _ = env.step(action)
                c) acumular reward y total_steps
                d) push a la memoria (state, action, reward, terminated, next_state)
                   con state/next_state como np.uint8. Ojo: lo que va al buffer
                   es `terminated` (estado terminal del MDP), NO `done`
                   (= terminated or truncated). El target de Bellman solo debe
                   anular el bootstrap cuando el episodio termino por estado
                   terminal, no por timeout.
                e) si len(memory) >= max(batch_size, learning_starts):
                       self.update_weights()
                f) state = next_state
                g) break si done o se alcanzo max_steps_episode
            3) registrar reward del episodio y actualizar metricas/pbar

        Al finalizar todos los episodios, guardar `policy_net.state_dict()`
        en `self.checkpoint_path`.

        Returns:
          - rewards: lista con el reward total de cada episodio.
        """
        rewards = []
        total_steps = 0
        metrics = {"reward": 0.0, "epsilon": self.epsilon_i, "steps": 0}

        pbar = tqdm(range(number_episodes), desc="Entrenando", unit="episode")

        for ep in pbar:
            if total_steps > max_steps:
                break

            # TODO: resetear el env, convertir state a np.uint8 (sin pre-procesar:
            #       el buffer guarda uint8 para no inflar VRAM).
            # TODO: inicializar acumuladores del episodio (reward, steps).

            for _ in range(max_steps_episode):
                raise NotImplementedError(
                    "Agent.train: cuerpo del loop por episodio no implementado"
                )

            # Registro de metricas y barra de progreso.
            rewards.append(...)  # TODO: agregar reward del episodio
            metrics["reward"] = float(np.mean(rewards[-self.episode_block :]))
            metrics["epsilon"] = self.compute_epsilon(total_steps)
            metrics["steps"] = total_steps
            pbar.set_postfix(metrics)

        # TODO: guardar self.policy_net.state_dict() en self.checkpoint_path.
        return rewards

    def compute_epsilon(self, steps_so_far):
        """Anneal lineal de epsilon entre epsilon_i y epsilon_f."""
        if steps_so_far < self.epsilon_anneal_steps:
            return self.epsilon_i - (self.epsilon_i - self.epsilon_f) * (
                steps_so_far / self.epsilon_anneal_steps
            )
        return self.epsilon_f

    def greedy_action(self, state):
        """
        Devuelve argmax_a Q(state, a) usando policy_net.

        Args:
          - state: una sola obs (numpy uint8 con shape obs_shape).
        Returns:
          - int: indice de la accion con mayor Q-value.
        """
        # TODO: aplicar Phi al state, mover al device, hacer forward, argmax.
        raise NotImplementedError("Agent.greedy_action no esta implementado")

    def select_action(self, state, current_steps, train=True):
        """
        Politica epsilon-greedy.
          - Si train=True: epsilon = compute_epsilon(current_steps).
          - Si train=False: puramente greedy.

        Returns:
          - int: accion seleccionada.
        """
        raise NotImplementedError("Agent.select_action no esta implementado")

    def play(self, env, episodes=1):
        """
        Modo evaluacion: corre `episodes` episodios sin actualizar la red,
        usando la politica greedy.
        """
        raise NotImplementedError("Agent.play no esta implementado")

    def _sample_batch(self):
        """
        Helper interno: muestrea un minibatch del replay buffer y lo deja
        como tensores en device.

        states / next_states: pasan por Phi (batched) -> float32 [0,1] -> GPU.
        actions / rewards / terminateds: tensores en device.
        """
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, terminateds, next_states = zip(*transitions)

        # np.stack arma el batch (B, *obs_shape); Phi reconoce que ya tiene
        # batch dim y devuelve el tensor sin tocar la forma.
        states_t = self.state_processing_function(np.stack(states)).to(
            self.device, non_blocking=True
        )
        next_states_t = self.state_processing_function(np.stack(next_states)).to(
            self.device, non_blocking=True
        )
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float, device=self.device)
        terminateds_t = torch.tensor(terminateds, dtype=torch.float, device=self.device)
        return states_t, actions_t, rewards_t, terminateds_t, next_states_t

    @abstractmethod
    def update_weights(self):
        """Implementado en subclases (DQNAgent / DoubleDQNAgent)."""
        pass
