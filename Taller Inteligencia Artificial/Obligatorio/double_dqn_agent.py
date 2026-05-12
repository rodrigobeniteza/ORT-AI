"""Double DQN (van Hasselt et al., 2015) — arXiv:1509.06461.

Cambio respecto a DQN vanilla:
    target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - terminated)

Donde DQN vanilla usaba `max_a Q_target` (mismo argmax y evaluacion).
Separar la seleccion de la accion (online) de su evaluacion (target)
reduce el sesgo de sobreestimacion clasico de Q-learning.

Nota sobre fidelidad: el paper de DDQN (2015) usa Huber + RMSProp y
target network. Aca mantenemos MSE + RMSProp por consistencia con el
DQN del 2013, asi la comparacion DQN vs DDQN aisla solo el cambio del
target rule (y la presencia/ausencia de target network), sin mezclar
tambien el cambio de loss/optimizer entre experimentos.
"""

import torch
import torch.nn as nn

from abstract_agent import Agent


class DoubleDQNAgent(Agent):
    def __init__(
        self,
        env,
        model_a,
        model_b,
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
        sync_target: int = 1_000,
        learning_starts: int | None = None,
        checkpoint_path: str = "ddqn_agent.dat",
    ):
        if learning_starts is None:
            learning_starts = batch_size

        super().__init__(
            env,
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
            learning_starts=learning_starts,
            max_grad_norm=None,
            checkpoint_path=checkpoint_path,
        )
        # TODO: configurar las dos redes (model_a, model_b) y el optimizador
        raise NotImplementedError("DoubleDQNAgent.__init__ no esta implementado")

    def update_weights(self):
        raise NotImplementedError(
            "DoubleDQNAgent.update_weights no esta implementado"
        )
