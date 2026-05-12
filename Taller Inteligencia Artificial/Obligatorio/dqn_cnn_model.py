import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN_Model(nn.Module):
    def __init__(self, obs_shape, n_actions):
        """
        Args:
          - obs_shape: (stack, height, width) del input ya preprocesado.
          - n_actions: cantidad de acciones discretas del env.

        Sugerencia (arquitectura del paper de Mnih et al. 2013):
          - Conv1: in=stack, out=16, kernel=8, stride=4 + ReLU
          - Conv2: in=16, out=32, kernel=4, stride=2 + ReLU
          - FC1: in = (h_out * w_out * 32), out = 256 + ReLU
          - FC2: in = 256, out = n_actions
        """
        super(DQN_CNN_Model, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        # TODO: definir capas convolucionales basadas en obs_shape.
        # TODO: definir capas lineales (la primera necesita el tamano
        #       del flatten de la ultima capa conv).
        raise NotImplementedError("DQN_CNN_Model.__init__ no esta implementado")

    def forward(self, obs):
        """
        Args:
          - obs: tensor de shape (batch, stack, height, width), float32 en [0,1].
        Returns:
          - tensor de Q-values de shape (batch, n_actions).
        """
        # TODO:
        #   1) Aplicar las capas convolucionales con activaciones (ReLU).
        #   2) Aplanar la salida (x.view(x.size(0), -1)).
        #   3) Aplicar las capas lineales con activacion.
        #   4) Retornar el tensor de Q-values.
        raise NotImplementedError("DQN_CNN_Model.forward no esta implementado")
