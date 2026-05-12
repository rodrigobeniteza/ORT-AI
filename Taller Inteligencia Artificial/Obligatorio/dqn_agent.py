"""DQN vanilla, fiel al paper de Mnih et al. (2013) "Playing Atari with
Deep Reinforcement Learning" (arXiv:1312.5602).

Decisiones de fidelidad al paper de 2013 (no Nature 2015):
- Loss MSE (no Huber/SmoothL1: eso es de Mnih 2015).
- Optimizador RMSProp (no Adam).
- Sin target network: el target Q se calcula con la propia policy_net.
- Sin grad clipping (eso es de Mnih 2015).
"""

import torch
import torch.nn as nn

from abstract_agent import Agent


class DQNAgent(Agent):
    def __init__(
        self,
        env,
        model,
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
        learning_starts: int | None = None,
        checkpoint_path: str = "dqn_agent.dat",
    ):
        # Por defecto, paper 2013: empieza a entrenar en cuanto |D| >= batch_size
        # (Algorithm 1 no menciona warmup explicito).
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
            max_grad_norm=None,  # paper 2013 no clipea
            checkpoint_path=checkpoint_path,
        )
        raise NotImplementedError("DQNAgent.__init__ no esta implementado")

    def update_weights(self):
        raise NotImplementedError("DQNAgent.update_weights no esta implementado")
