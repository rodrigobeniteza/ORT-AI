# Material de Estudio — DQN y Double DQN en highway-env

**Materia:** Taller de Inteligencia Artificial 2026 | **Obligatorio — Parte 1**

**Referencias:** Mnih et al. (2013) · van Hasselt et al. (2015) · highway-env (Leurent, 2018)

---

## Índice

1. [Arquitectura general del proyecto](#1-arquitectura-general-del-proyecto)
2. [El entorno: highway-fast-v0](#2-el-entorno-highway-fast-v0)
3. [Preprocesamiento de observaciones (φ)](#3-preprocesamiento-de-observaciones-φ)
4. [Replay Memory](#4-replay-memory-replay_memorypy)
5. [Clase base Agent](#5-clase-base-agent-abstract_agentpy)
6. [DQN vanilla](#6-dqn-vanilla-mnih-2013-dqn_agentpy)
7. [Double DQN](#7-double-dqn-van-hasselt-2015-double_dqn_agentpy)
8. [La notebook: flujo de principio a fin](#8-la-notebook-flujo-de-principio-a-fin)
9. [Hiperparámetros y su significado](#9-hiperparámetros-y-su-significado)
10. [DQN vs DDQN: diferencias clave](#10-dqn-vs-ddqn-diferencias-clave)
11. [Preguntas frecuentes del docente](#11-preguntas-frecuentes-del-docente)
12. [Glosario](#12-glosario)
13. [La red en profundidad (dqn_cnn_model.py)](#13-la-red-en-profundidad-dqn_cnn_modelpy)
14. [Defensa oral — Parcial 14/07/2026 (respuestas directas)](#14-defensa-oral--parcial-14072026-respuestas-directas)
15. [Banco de preguntas de parciales anteriores (2020–2022)](#15-banco-de-preguntas-de-parciales-anteriores-20202022)
16. [Repaso rápido: Gymnasium, K-Bandits y PyTorch](#16-repaso-rápido-gymnasium-k-bandits-y-pytorch)

---

## 1. Arquitectura general del proyecto

El proyecto implementa dos algoritmos de Aprendizaje por Refuerzo profundo para entrenar un agente que conduce en una autopista simulada: **DQN** (Deep Q-Network) y **Double DQN** (DDQN).

### Mapa de archivos y relaciones

```
Notebook (Obligatorio_TIA_2026_Parte1.ipynb)   ← orquesta todo
│
├── utils.py              → make_env(): construye el entorno con wrappers de imagen
│
├── replay_memory.py      → ReplayMemory: buffer circular de experiencias pasadas
│
├── abstract_agent.py     → Agent (clase base abstracta):
│   ├── train()           ← loop de entrenamiento completo
│   ├── compute_epsilon() ← calcula epsilon para ε-greedy
│   ├── select_action()   ← política ε-greedy
│   ├── greedy_action()   ← argmax Q(s,a)
│   ├── play()            ← modo evaluación (sin entrenamiento)
│   └── _sample_batch()   ← muestrea minibatch del buffer
│
├── dqn_agent.py          → DQNAgent(Agent):
│   └── update_weights()  ← target con la MISMA red (sin target network)
│
└── double_dqn_agent.py   → DoubleDQNAgent(Agent):
    └── update_weights()  ← online elige acción, target evalúa (dos redes)
```

> **Idea central:** La clase `Agent` define TODO el loop de entrenamiento. `DQNAgent` y `DoubleDQNAgent` solo difieren en **cómo calculan el target de Bellman** dentro de `update_weights()`. Todo lo demás es idéntico.

### Flujo de datos en un paso de entrenamiento

```
env.reset() → obs (uint8, numpy)
      ↓
select_action(obs)
  └─ compute_epsilon() → ε-greedy → env.step(action)
                                          ↓
                              obs, reward, terminated, truncated
                                          ↓
                              memory.add(uint8_obs, action, reward,
                                         terminated, uint8_next_obs)
                                          ↓
                              si |memory| ≥ learning_starts:
                                update_weights()
                                  └─ _sample_batch() → tensores float32 en device
                                  └─ forward(net) → Q-values
                                  └─ calcular target (Bellman)
                                  └─ MSELoss → backward → optimizer.step()
```

---

## 2. El entorno: highway-fast-v0

**highway-env** es una biblioteca de entornos de conducción autónoma basada en Farama Gymnasium. `highway-fast-v0` es una variante optimizada del entorno estándar de autopista: menos vehículos y menor frecuencia de simulación física, lo que lo hace ~15× más rápido sin perder la dinámica esencial.

### Acciones disponibles (DiscreteMetaAction)

| ID | Acción | Descripción |
|----|--------|-------------|
| 0 | `LANE_LEFT` | Cambiar al carril de la izquierda |
| 1 | `IDLE` | Mantener velocidad y carril actuales |
| 2 | `LANE_RIGHT` | Cambiar al carril de la derecha |
| 3 | `FASTER` | Acelerar |
| 4 | `SLOWER` | Frenar |

### Simulation frequency vs Policy frequency

| Parámetro | Qué controla | Valor usado |
|-----------|-------------|-------------|
| `policy_frequency` | Con qué frecuencia el agente toma decisiones (Hz en tiempo simulado) | 1 (default) |
| `simulation_frequency` | Con qué frecuencia avanza la física del simulador | 5 (train) / 30 (eval) |

Cada `env.step()` ejecuta **k = sim_freq / policy_freq** sub-pasos de física manteniendo la misma acción, equivalente al *frame skip* de los entornos Atari.

> **¿Por qué dos valores (5 train, 30 eval)?** Durante el entrenamiento el tiempo de cómputo es el cuello de botella: usamos 5 para entrenar rápido. En evaluación corremos pocos episodios, así que podemos usar 30 para generar videos fluidos.

### Reward function

`HIGH_SPEED_REWARD = 0.8` (subido del default 0.4) rompe el "óptimo local" de quedarse detrás de un auto lento: ir rápido vale el doble, incentivando al agente a adelantar y mantener velocidad alta.

---

## 3. Preprocesamiento de observaciones (φ)

### ¿Por qué imágenes y no la tabla de posiciones?

highway-env por default entrega una `KinematicObservation`: una matriz con posición, velocidad y atributos de cada vehículo cercano. Eso es un *oráculo simbólico*: la red aprende política sin aprender percepción visual. El obligatorio fuerza entrada visual para replicar el paper de Mnih: la red tiene que aprender features visuales (carriles, vehículos, geometría) además de la política.

### El pipeline de wrappers (aplicados en orden)

📍 `utils.py:86-175` (función `make_env()`; hay una copia casi idéntica en `utils_3.py` usada por `vector_utils.py` para envs vectorizados)

| Paso | Wrapper | Fuente | Resultado | ¿Por qué? |
|------|---------|--------|-----------|-----------|
| 1 | `_RenderAsObservation` | custom, `utils.py:22-40` | Frame RGB 600×150 | Reemplaza `KinematicObservation` por `env.render()`; highway-env no expone RGB plano nativo |
| 2 | `ResizeObservation` | `gymnasium.wrappers` (estándar) | 336×84 (aspecto 4:1) | Reduce el input; mantener 4:1 preserva la geometría horizontal |
| 3 | `GrayscaleObservation` | `gymnasium.wrappers` (estándar) | 336×84 × 1 canal | Luma BT.601, `Y' = 0.299R + 0.587G + 0.114B`. El color no aporta información útil; reduce input 3× |
| 4 | `FrameStackObservation(stack_size=4)` | `gymnasium.wrappers` (estándar) | `(4, 84, 336)` uint8 | Un frame no tiene velocidad. 4 frames permiten inferir movimiento por diferencia |
| 5 | `RecordEpisodeStatistics` | `gymnasium.wrappers` (estándar) | no toca la obs | Agrega stats de episodio al `info` |

Importados en el header del archivo (📍 `utils.py:13-19`):

```python
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    ResizeObservation,
)
```

Solo `_RenderAsObservation` es custom (además de `_SafeRecordVideo`, un parche sobre `RecordVideo` en `utils.py:43-54` para evitar un crash de highway-env al grabar video fuera de un episodio grabado). El resto es 100% wrappers estándar de `gymnasium.wrappers` — nada específico de Atari, a diferencia de lo que sugiere el paper original.

> **Resultado final:** tensor de shape `(4, 84, 336)` en `uint8`. Se almacena así en el buffer. La conversión a `float32 [0,1]` ocurre recién al momento de entrenar. Ese `4` (`stack_frames`, paso 4) es el mismo valor que `DQN_CNN_Model` usa como `in_channels` de `conv1` (`obs_shape[0]`, ver sección 13).

### ¿Por qué uint8 en el buffer y no float32?

```
Con observaciones (4, 84, 336) y buffer de 50k transiciones × 2:

float32 → 50k × 2 × 4 × 84 × 336 × 4 bytes ≈ 45 GB de VRAM  → Out Of Memory
uint8   → 50k × 2 × 4 × 84 × 336 × 1 byte  ≈ 11 GB de RAM   → manejable
```

### La función `process_state` (φ en código)

📍 `Obligatorio_TIA_2026_Parte1.ipynb` (celda 41 — no vive en un `.py`, se define directo en la notebook)

```python
def process_state(obs):
    arr = np.asarray(obs, dtype=np.uint8)
    t = torch.from_numpy(arr).float().div_(255.0)  # normaliza [0,255] → [0.0, 1.0]

    if t.ndim == 3:        # obs individual: (4, 84, 336)
        t = t.unsqueeze(0) # → (1, 4, 84, 336)  agrega batch dim

    return t               # si ya era (B, 4, 84, 336), lo deja igual
```

Maneja dos casos con el mismo código:
- **Una obs sola** (al elegir acción): llega `(4,84,336)` → sale `(1,4,84,336)`
- **Un batch** (al entrenar): llega `(B,4,84,336)` → sale `(B,4,84,336)` sin cambios

---

## 4. Replay Memory (`replay_memory.py`)

### ¿Por qué existe el replay buffer?

En Q-learning online, cada transición se usa una sola vez. Dos problemas:

1. **Correlación temporal:** muestras consecutivas son muy similares → sobreajuste local.
2. **Olvido catastrófico:** aprender algo nuevo destruye lo aprendido antes.

El buffer guarda experiencias pasadas y permite **muestreo aleatorio**, rompiendo la correlación temporal y reutilizando cada experiencia múltiples veces.

### La namedtuple `Transition`

📍 `replay_memory.py:15`

```python
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'terminated', 'next_state'))
```

> **⚠️ Se guarda `terminated`, NO `done`**
>
> `done = terminated OR truncated`
> - `terminated`: fin real del MDP (colisión, meta alcanzada)
> - `truncated`: corte artificial por timeout
>
> La ecuación de Bellman usa `(1 - terminated)` para anular el valor futuro solo en estados terminales reales. Con `done`, se anularía también al final de episodios por timeout, sesgando el aprendizaje.

### `ReplayMemory.__init__`

📍 `replay_memory.py:24-32`

```python
def __init__(self, capacity):
    self.capacity = capacity  # tamaño máximo del buffer
    self.memory = []          # lista de Transitions almacenadas
    self.position = 0         # puntero circular: dónde escribir la próxima
```

### `ReplayMemory.add` — El buffer circular

📍 `replay_memory.py:35-51`

```python
def add(self, state, action, reward, terminated, next_state):
    if len(self.memory) < self.capacity:
        self.memory.append(Transition(...))       # buffer no lleno → agregar
    else:
        self.memory[self.position] = Transition(...)  # buffer lleno → sobreescribir

    self.position += 1
    if self.position >= self.capacity:
        self.position = 0  # vuelve al principio → comportamiento circular
```

**Visualización con capacidad = 3:**

```
add(tran1): [tran1, -----, -----]  position=1
add(tran2): [tran1, tran2, -----]  position=2
add(tran3): [tran1, tran2, tran3]  position=0  ← vuelve al inicio
add(tran4): [tran4, tran2, tran3]  position=1  ← sobreescribió tran1 (la más vieja)
```

### `ReplayMemory.sample`

📍 `replay_memory.py:54-67` (también `__len__` en 70-72 y `clear` en 75-78)

```python
def sample(self, batch_size):
    if batch_size <= len(self):
        return random.sample(self.memory, batch_size)  # muestra aleatoria SIN repetición
    return None
```

`random.sample` devuelve `batch_size` elementos aleatorios sin reposición: en un mismo minibatch nunca aparece la misma transición dos veces.

### `batch_size` vs canales (`stack`): dos ejes distintos del tensor

Son dos ejes del tensor sin relación entre sí — uno es cuántas muestras entrenás juntas, el otro es una propiedad fija de cada observación individual.

- **`batch_size` (eje 0):** cuántas `Transition` samplea `ReplayMemory.sample()` de una vez. Es un hiperparámetro (32 en la config final) que no tiene nada que ver con canales — es solo cuántas filas independientes metés en un paso de gradiente.
- **`stack` / canales (eje 1):** cada `state` guardado en el buffer ya tiene shape `obs_shape = (4, 84, 336)` — ese `4` es `NUM_STACKED_FRAMES` (ver sección 3, paso 4 del pipeline de wrappers), 4 frames consecutivos apilados para que la red vea movimiento. `ReplayMemory` no sabe ni le importa qué es ese `4`: para ella `state` es solo un array que guarda tal cual.

Se juntan en `_sample_batch()` (📍 `abstract_agent.py:284`, ver sección 5):

```python
states_t = self.state_processing_function(np.stack(states)).to(self.device, ...)  # (B, *obs_shape)
```

`states` es una tupla de `batch_size` arrays `(4, 84, 336)`; `np.stack` los apila en un **eje nuevo**, dando `(batch_size, 4, 84, 336)`:

| Dim | Significado | Quién lo fija | Valor típico |
|---|---|---|---|
| 0 | `batch_size` — cuántas transiciones sampleaste | vos, en el config del agente | 32 |
| 1 | `stack`/canales — cuántos frames apilados por observación | `NUM_STACKED_FRAMES`, fijo en `obs_shape` | 4 |
| 2, 3 | `H, W` — tamaño de cada frame preprocesado | el wrapper de resize | 84, 336 |

Es la misma lógica que una imagen RGB `(B, 3, H, W)` donde 3 = canales de color — acá, en vez de color, el "canal" es tiempo (4 frames pasados), pero para `Conv2d` es matemáticamente idéntico: convoluciona sobre todos los canales/frames a la vez (de ahí que `conv1` tome `in_channels=obs_shape[0]`, sección 13).

---

## 5. Clase base Agent (`abstract_agent.py`)

Clase abstracta que implementa todo lo **común** a DQN y DDQN. Las subclases solo implementan `update_weights()`.

### Hiperparámetros del `__init__`

📍 `abstract_agent.py:33-81` (`Agent.__init__`)

| Parámetro | Significado |
|-----------|-------------|
| `gamma` | Factor de descuento γ: cuánto valen las recompensas futuras (0.99 = 99%) |
| `epsilon_i` / `epsilon_f` | Epsilon inicial/final para la política ε-greedy |
| `epsilon_anneal_steps` | Pasos para bajar epsilon de ini a fin |
| `learning_starts` | Pasos de warmup antes de empezar a entrenar |
| `batch_size` | Transiciones por minibatch de entrenamiento |
| `episode_block` | Ventana para el reward promedio mostrado en pantalla |
| `checkpoint_path` | Ruta donde se guardan los pesos de la red al finalizar |

### `compute_epsilon` — Anneal lineal

📍 `abstract_agent.py:181-185`

```python
def compute_epsilon(self, steps_so_far):
    if steps_so_far < self.epsilon_anneal_steps:
        return self.epsilon_i - (self.epsilon_i - self.epsilon_f) * (steps_so_far / self.epsilon_anneal_steps)
    return self.epsilon_f  # después del anneal, se queda fijo en epsilon_f
```

```
epsilon
 1.0 |\
     | \
     |  \
     |   \
0.05 |    \_____________________________
     +----+------------------------------→ pasos
          20k (epsilon_anneal_steps)
```

Al principio el agente explora casi todo al azar. Gradualmente confía más en lo aprendido y explora menos.

### `greedy_action` — Acción greedy (sin exploración)

📍 `abstract_agent.py:187-205`

```python
def greedy_action(self, state):
    with torch.no_grad():  # no calcular gradientes (solo inferencia)
        state_t = self.state_processing_function(state).to(self.device)
        q_values = self.policy_net(state_t)   # forward pass → (1, 5) Q-values
        return int(q_values.argmax(dim=1))    # índice de la acción con mayor Q
```

`torch.no_grad()` desactiva el grafo de gradientes: hace la inferencia más rápida y usa menos memoria. Se usa siempre que no vamos a llamar a `backward()`.

### `select_action` — Política ε-greedy

📍 `abstract_agent.py:207-220`

```python
def select_action(self, state, current_steps, train=True):
    if train:
        eps = self.compute_epsilon(current_steps)
        if np.random.random() < eps:
            return self.env.action_space.sample()  # con prob ε: acción ALEATORIA
    return self.greedy_action(state)               # resto: acción GREEDY
```

En modo evaluación (`train=False`), siempre greedy: aprovecha todo lo aprendido sin explorar.

### `train` — El loop de entrenamiento

📍 `abstract_agent.py:83-179` (versión real más completa: agrega `print_every`, `update_every`, y el registro de `q_values_history`/`loss_history`/`steps_history`/`epsilon_history` para graficar después — acá simplificado a lo esencial)

```python
def train(self, number_episodes, max_steps_episode, max_steps):
    rewards = []
    total_steps = 0

    for ep in tqdm(range(number_episodes)):
        if total_steps > max_steps:
            break                           # límite global de pasos alcanzado

        episode_reward = 0
        state, _ = self.env.reset()

        for _ in range(max_steps_episode):
            action = self.select_action(state, total_steps, train=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            episode_reward += reward
            total_steps += 1
            done = terminated or truncated

            # guardar en buffer como uint8 (ahorra memoria)
            self.memory.add(np.uint8(state), action, reward, terminated, np.uint8(next_state))

            # entrenar solo si hay suficientes muestras
            if len(self.memory) >= max(self.batch_size, self.learning_starts):
                self.update_weights()   # ← implementado en subclases

            state = np.uint8(next_state)
            if done:
                break

        rewards.append(episode_reward)

    # guardar pesos al finalizar
    torch.save(self.policy_net.state_dict(), self.checkpoint_path)
    return rewards
```

> `torch.save(self.policy_net.state_dict(), path)` guarda un diccionario `{nombre_capa: tensor_de_pesos}`. Es la práctica recomendada en PyTorch: más pequeño, portable y compatible entre versiones.

### ¿Por qué warmup (`learning_starts`) antes de entrenar?

📍 `abstract_agent.py:148` — línea real: `if len(self.memory) >= max(self.batch_size, self.learning_starts) and total_steps % update_every == 0:`

Al principio del entrenamiento el buffer está vacío o casi vacío, y todo lo que hay adentro fue recolectado bajo una política casi puramente aleatoria (epsilon arranca en 1.0), en una porción muy chica y poco representativa del espacio de estados. Entrenar de inmediato significaría muestrear minibatches de un pool minúsculo y poco diverso — exactamente lo que el replay buffer existe para evitar (correlación temporal, sobreajuste a lo último que pasó). `learning_starts` es un período de warmup puro: durante esos primeros pasos el agente solo actúa y llena el buffer (`memory.add(...)`) sin tocar los pesos de la red; recién con experiencia diversa acumulada arranca `update_weights()`. En la config final de la notebook `learning_starts=1_000` (bastante más que `batch_size=32`), para forzar ese margen de diversidad.

### ¿Por qué `max(batch_size, learning_starts)` y no solo uno de los dos?

Son dos condiciones distintas combinadas, y las dos tienen que cumplirse:

1. **`len(memory) >= batch_size` — no es opcional, es para que el código no explote.** `ReplayMemory.sample()` (`replay_memory.py:54-67`) devuelve `None` si pedís más elementos de los que hay:
   ```python
   def sample(self, batch_size):
       if batch_size <= len(self):
           return random.sample(self.memory, batch_size)
       return None
   ```
   Si `update_weights()` llamara a `_sample_batch()` (`abstract_agent.py:271-289`) con el buffer todavía más chico que `batch_size`, la línea `zip(*transitions)` haría `zip(*None)` → `TypeError` inmediato.
2. **`len(memory) >= learning_starts` — la decisión de diseño (warmup) explicada arriba.** Es independiente de `batch_size`: se puede pedir un warmup más largo que lo estrictamente necesario para samplear.

Se usa `max()` porque cualquiera de las dos puede ser la más restrictiva según la config: por default en `dqn_agent.py`/`double_dqn_agent.py`, si no se pasa `learning_starts` explícito, vale `learning_starts = batch_size` (empatados) — pero en la notebook final se sube a `1_000`, haciendo que el warmup "de diseño" mande por encima del mínimo "para no crashear".

> La segunda mitad de esa misma línea — `total_steps % update_every == 0` — es un control relacionado pero distinto: una vez que ya arrancó el entrenamiento, no se llama a `update_weights()` en cada step sino cada `update_every` steps, para ir más rápido y no sobreajustar a transiciones muy correlacionadas entre sí.

### `_sample_batch` — Muestrear el buffer para entrenar

📍 `abstract_agent.py:271-289`

```python
def _sample_batch(self):
    transitions = self.memory.sample(self.batch_size)

    # zip(*lista_de_tuplas) transpone:
    # [(s1,a1,...), (s2,a2,...)] → ([s1,s2,...], [a1,a2,...], ...)
    states, actions, rewards, terminateds, next_states = zip(*transitions)

    # np.stack arma (B, 4, 84, 336); Phi normaliza a float32 [0,1]
    states_t      = self.state_processing_function(np.stack(states)).to(self.device)
    next_states_t = self.state_processing_function(np.stack(next_states)).to(self.device)
    actions_t     = torch.tensor(actions,     dtype=torch.long,  device=self.device)
    rewards_t     = torch.tensor(rewards,     dtype=torch.float, device=self.device)
    terminateds_t = torch.tensor(terminateds, dtype=torch.float, device=self.device)

    return states_t, actions_t, rewards_t, terminateds_t, next_states_t
```

### `play` — Modo evaluación

📍 `abstract_agent.py:222-249`

```python
def play(self, env, episodes=1, max_steps=10_000):
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = self.greedy_action(state)   # siempre greedy
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
```

Corre episodios sin actualizar la red. Se usa después del entrenamiento para grabar los videos demostrativos.

---

## 6. DQN vanilla (Mnih 2013) (`dqn_agent.py`)

### Decisiones de fidelidad al paper 2013

- **Loss: MSE** (Huber/SmoothL1 es del paper Nature 2015)
- **Optimizador: RMSProp** (Adam es posterior a 2013)
- **Sin target network:** el target Q se calcula con la misma red
- **Grad clipping: no está en el paper 2013**, pero la implementación lo soporta como *opt-in* (`max_grad_norm`, default `None`). En la config final de la notebook se activa con `max_grad_norm=10.0` (ver sección 15, punto 1.5) porque sin clipping la gráfica de Q promedio mostraba picos de ruido grandes — es una desviación deliberada del paper, no un descuido.

### `__init__`

📍 `dqn_agent.py:19-78` (la asignación de `policy_net`/`optimizer` está en 63-66)

```python
self.policy_net = model.to(device)   # UNA sola red neuronal
self.optimizer  = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
```

### `update_weights` — La ecuación de Bellman

📍 `dqn_agent.py:91-116`

```python
def update_weights(self):
    states_t, actions_t, rewards_t, terminateds_t, next_states_t = self._sample_batch()

    # 1) Target: sin gradientes (es la "etiqueta correcta")
    with torch.no_grad():
        next_q_values = self.policy_net(next_states_t)   # Q(s', a) para todas las acciones
        max_next_q    = next_q_values.max(dim=1).values  # max_a Q(s', a)
        #
        #  Ecuación de Bellman: y_i = r_i + γ · max_a' Q(s'_i, a') · (1 - terminated_i)
        targets = rewards_t + self.gamma * max_next_q * (1 - terminateds_t)

    # 2) Q actual para las acciones que SE tomaron
    current_q_values = self.policy_net(states_t)                          # → (B, 5)
    current_q        = current_q_values.gather(1, actions_t.unsqueeze(1)) # → (B, 1)
    targets          = targets.unsqueeze(1)                                # → (B, 1)

    # 3) Backpropagation
    self.optimizer.zero_grad()
    loss = nn.MSELoss()(targets, current_q)
    loss.backward()
    if self.max_grad_norm is not None:
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
    self._ep_losses.append(loss.item())  # para graficar loss medio (utils.plot_losses)
    self.optimizer.step()
```

**Ecuaciones:**

```
y_i   = r_i  +  γ · max_a' Q(s'_i, a'; θ) · (1 - terminated_i)
Loss  = MSE(y_i,  Q(s_i, a_i; θ))
      = (y_i - Q(s_i, a_i; θ))²
```

### ¿Qué hace `torch.gather`?

Selecciona, de cada fila de Q-values, el elemento en la columna correspondiente a la acción tomada:

```
current_q_values (3×5):          actions: [[1],[2],[3]]
  [[0.1, 0.8, 0.3, 0.5, 0.2],
   [0.6, 0.2, 0.9, 0.1, 0.4],   .gather(1, actions) →   [[0.8],
   [0.3, 0.7, 0.4, 0.8, 0.1]]                             [0.9],
                                                           [0.8]]
```

Es la forma vectorizada de `current_q[i][action[i]]` para cada elemento del batch.

### `actions_t` NO es Q-values ni probabilidades — es un índice entero

Confusión clásica a evitar: `actions_t` y `current_q_values` son cosas completamente distintas.

- **`current_q_values`** (`(B, 5)`): la salida cruda de la red — un valor real (no probabilidad, no suma 1, puede ser negativo) por cada acción posible, para cada estado del batch. DQN es un método basado en **valor**, no en política probabilística: en ningún lado del código existe "la probabilidad de la acción 2" (eso sería un método de policy gradient como REINFORCE/PPO, no DQN).
- **`actions_t`** (`(B,)`): **no tiene Q-value ni probabilidad adentro**. Es solo un entero por transición — qué acción se ejecutó *realmente* en el entorno cuando esa transición se recolectó (elegida en su momento por `select_action`, greedy o al azar por exploración). Es un hecho histórico ya grabado en el buffer, no una cantidad que la red esté prediciendo ahora.

📍 `abstract_agent.py:124,144` (se ejecuta y se guarda el int) → `replay_memory.py:15` (`Transition.action`) → `abstract_agent.py:286` (`actions_t = torch.tensor(actions, dtype=torch.long, ...)`)

`gather` usa `actions_t` **puramente como índice de columna** — selección dura (como una máscara one-hot), no una mezcla ponderada por probabilidad:

```python
one_hot = F.one_hot(actions_t, num_classes=5).float()   # ej. [[0,0,1,0,0], [1,0,0,0,0]]
current_q = (current_q_values * one_hot).sum(dim=1)       # mismo resultado que gather, más lento
```

El resultado es "el Q-value de la acción que efectivamente se ejecutó" — el término que se necesita para comparar contra `y_i` (que también describe el valor de *esa* acción específica, no de todas).

### ¿Por qué `no_grad()` para el target?

El target `y_i` se trata como una **constante** — la respuesta correcta que queremos alcanzar. Si dejáramos fluir el gradiente también por el target, estaríamos diferenciando los pesos en ambas partes del loss simultáneamente, lo que desestabiliza el entrenamiento.

### `actions_t.unsqueeze(1)` — obligatorio por la API de `gather`

`gather(dim, index)` exige que `index` tenga **la misma cantidad de dimensiones** que el tensor fuente. `current_q_values` es `(B, n_actions)` (2D); `actions_t` sale de `_sample_batch()` como `(B,)` (1D). Sin el `unsqueeze(1)`, `gather` tira `RuntimeError` directamente — no es opcional, es un requisito duro de la API.

### `targets.unsqueeze(1)` — evita un broadcasting silencioso e incorrecto

`targets` sale de la ecuación de Bellman como `(B,)` (1D); `current_q` (tras el `gather`) es `(B, 1)` (2D). Si se comparara `MSELoss()(targets, current_q)` sin ese `unsqueeze`, PyTorch **no tira error** — hace *broadcasting*: alinea `(B,)` como `(1, B)` y lo compara contra `(B, 1)`, dando un resultado `(B, B)` en vez de `(B,)`. Es decir, en vez de comparar `target_i` contra `Q_i` elemento a elemento, terminaría comparando cada target contra el Q de **todos** los demás elementos del batch (una matriz de diferencias cruzadas), promediando la loss sobre `B²` términos en vez de `B`. El entrenamiento correría sin errores mostrando una loss numérica válida, pero aprendería algo incorrecto — PyTorch emite un `UserWarning` para este caso exacto (*"Using a target size that is different to the input size... This will likely lead to incorrect results due to broadcasting"*). Por eso se fuerzan ambas formas a `(B, 1)` explícitamente.

### El resto del `update_weights`: contabilidad y estabilidad

- `self.optimizer.zero_grad()`: limpia los gradientes acumulados de la iteración anterior — sin esto, PyTorch los suma en cada `.backward()` y la actualización queda corrompida.
- `if self.max_grad_norm is not None: nn.utils.clip_grad_norm_(...)`: gradient clipping opt-in (no está en el paper 2013). Recorta la norma global de los gradientes para que ningún update sea desproporcionadamente grande — mitiga los picos de inestabilidad que aparecen sin target network.
- `self._ep_losses.append(loss.item())`: `.item()` extrae el escalar Python del tensor de loss (que tiene un solo elemento) para acumularlo en una lista simple sin arrastrar el grafo computacional. Alimenta `loss_history` (ver `train()` en `abstract_agent.py`), usado para graficar la loss media por episodio y diagnosticar inestabilidad/underfitting.

---

## 7. Double DQN (van Hasselt 2015) (`double_dqn_agent.py`)

### El problema: sobreestimación en DQN vanilla

En DQN, la misma red elige la mejor acción Y evalúa su Q-value:

```
y^DQN = r + γ · max_a' Q(s', a'; θ)
```

El `max` sobre Q-values ruidosos (al inicio, la red no conoce los valores reales) tiende a **sobreestimar**: siempre elige el más alto aunque sea por ruido. Este sesgo se acumula y lleva a políticas subóptimas.

### La solución: separar selección y evaluación

Double DQN usa **dos redes**:
- **Red online (A):** se entrena con backpropagation → *elige* la mejor acción en s'
- **Red target (B):** copia periódica de A → *evalúa* el Q-value de esa acción

```
y^DDQN = r + γ · Q_B(s',  argmax_a' Q_A(s', a')) · (1 - terminated)
                  ↑ evalúa          ↑ elige
```

### `__init__`

📍 `double_dqn_agent.py:25-95` (las dos redes se asignan en 73-79). El `__init__` real guarda **cuatro** atributos redundantes para las mismas dos redes: `self.model_a`/`self.model_b` (referencia directa a los modelos recibidos) y `self.policy_net_a`/`self.policy_net_b` (los mismos modelos, tras `.to(device)`) — el propio código lo marca como redundancia heredada del esqueleto del constructor. `update_weights()` mezcla ambos nombres: usa `self.policy_net_a` para la selección de acción y `self.model_b` para evaluarla (ver abajo) — no es un bug, ambos apuntan al mismo objeto en memoria (`.to()` muta el módulo in-place), pero vale la pena tenerlo claro para no confundirse leyendo el archivo real.

```python
self.policy_net_a = model_a.to(device)    # red online (A) — se entrena
self.policy_net_b = model_b.to(device)    # red target (B) — copia periódica de A
self.policy_net   = self.policy_net_a     # la clase base usa self.policy_net en greedy_action()
self.optimizer    = torch.optim.RMSprop(self.policy_net_a.parameters(), lr=learning_rate)
```

> `self.policy_net = self.policy_net_a` es necesario para que `Agent.greedy_action()` (clase base) use la red online sin tener que reescribir el método.

### `update_weights` — El cambio clave respecto a DQN

📍 `double_dqn_agent.py:106-134`

```python
def update_weights(self):
    states_t, actions_t, rewards_t, terminateds_t, next_states_t = self._sample_batch()

    with torch.no_grad():
        # 1) Red A ELIGE la mejor acción en s'
        best_action = self.policy_net_a(next_states_t).argmax(dim=1).unsqueeze(1)  # → (B, 1)

        # 2) Red B EVALÚA el Q de esa acción
        next_q = self.model_b(next_states_t).gather(1, best_action).squeeze(1)  # → (B,)

        # 3) Target de Bellman
        targets = rewards_t + self.gamma * next_q * (1 - terminateds_t)

    # 4) Q actual (red A) para las acciones tomadas
    current_q = self.policy_net_a(states_t).gather(1, actions_t.unsqueeze(1))
    targets   = targets.unsqueeze(1)

    self.optimizer.zero_grad()
    loss = nn.MSELoss()(targets, current_q)
    loss.backward()
    if self.max_grad_norm is not None:
        nn.utils.clip_grad_norm_(self.policy_net_a.parameters(), self.max_grad_norm)
    self._ep_losses.append(loss.item())  # para graficar loss medio
    self.optimizer.step()

    # 5) Sincronización dura (hard update): copiar A -> B cada sync_target updates
    self.update_counter += 1
    if self.update_counter % self.sync_target == 0:
        self.model_b.load_state_dict(self.policy_net_a.state_dict())
```

> Los mismos motivos de `actions_t.unsqueeze(1)` (requisito de `gather`) y `targets.unsqueeze(1)` (evitar el broadcasting `(B,)` vs `(B,1)` → `(B,B)` de `MSELoss`) explicados en la [sección 6](#6-dqn-vanilla-mnih-2013-dqn_agentpy) aplican exactamente igual acá.

### La cadena `gather → squeeze → Bellman → unsqueeze` de `next_q`

DDQN tiene un paso extra que DQN no tiene: `next_q` pasa por un `.squeeze(1)` antes de entrar a la ecuación de Bellman, y recién más tarde el resultado final pasa por `.unsqueeze(1)`. No se cancelan entre sí — cada uno actúa sobre un tensor distinto, en un momento distinto, por una razón distinta:

```
Q_B(s',·)                          (B, n_actions)
  .gather(1, best_action)     →    (B, 1)      # gather deja la columna elegida por la red online
  .squeeze(1)                 →    (B,)        # se saca la columna para poder operar con rewards_t/terminateds_t (1D)
  ... r + γ·next_q·(1-term) ... →  (B,)        # aritmética elemento-a-elemento, todo en 1D, sin broadcasting raro
  .unsqueeze(1)               →    (B, 1)      # recién acá se vuelve a agregar, para el MSELoss contra current_q
```

**Por qué el `.squeeze(1)` no es cosmético:** `rewards_t` y `terminateds_t` son `(B,)` (1D), tal como salen de `_sample_batch()`. Si `next_q` se dejara en `(B, 1)` (la forma que devuelve `gather`), la línea `rewards_t + self.gamma * next_q * (1 - terminateds_t)` mezclaría un `(B,)` con un `(B,1)` **dentro del cálculo del target**, no solo en la loss final — PyTorch haría el mismo broadcasting silencioso a `(B, B)` que ya vimos, pero esta vez corrompiendo el propio target de Bellman antes de llegar siquiera a comparar contra `current_q`. El `.squeeze(1)` evita este bug llevando `next_q` de vuelta a `(B,)` antes de esa cuenta.

**Por qué el `.unsqueeze(1)` va después, y no en el mismo lugar:** una vez que el target ya se calculó correctamente en `(B,)` (todo 1D, aritmética elemento-a-elemento válida), recién ahí se le agrega la dimensión de columna para que coincida con `current_q` (que es `(B, 1)`, por venir de otro `gather`) — el mismo motivo de siempre: que `MSELoss` compare forma contra forma idéntica, sin dejarle a PyTorch la posibilidad de adivinar una forma "compatible" por broadcasting.

**Sincronización de la red target:** `self.update_counter` cuenta llamadas a `update_weights()` (no steps de entorno). Cada `sync_target` llamadas (default `1_000`), se copian los pesos completos de la red online (`policy_net_a`) a la red target (`model_b`) vía `load_state_dict`. Es un **hard update** (copia completa y discreta), no un promedio móvil tipo Polyak/soft-update — la red target se queda "congelada" con esos pesos hasta la próxima sincronización, lo cual es justamente lo que la hace un target más estable que evaluar con la misma red que se está entrenando.

---

## 8. La notebook: flujo de principio a fin

### Setup y semillas

```python
SEED = 23
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False  # no sacrifica velocidad por determinismo
torch.backends.cudnn.benchmark     = True   # cuDNN elige la implementación más rápida
np.random.seed(SEED)
random.seed(SEED)
```

### Detección de dispositivo

```python
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"   # GPU NVIDIA
elif torch.backends.mps.is_available():
    DEVICE = "mps"    # Apple Silicon (M1/M2/M3)
```

El entrenamiento con GPU es entre 5× y 20× más rápido que en CPU para redes convolucionales.

### Episodio aleatorio (verificación del entorno)

```python
env = make_env(ENV_NAME, video_folder="videos/random_agent", record_every=1, ...)
obs, info = env.reset(seed=SEED)

done = False
while not done:
    action = env.action_space.sample()  # acción completamente aleatoria
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
Video("videos/random_agent/rl-video-episode-0.mp4", embed=True)
```

Sirve para verificar que el entorno funciona y ver el comportamiento de un agente sin entrenar.

### Entrenamiento DQN

```python
net       = DQN_CNN_Model(env.observation_space.shape, env.action_space.n).to(DEVICE)
dqn_agent = DQNAgent(env, net, process_state, ...)
dqn_rewards = dqn_agent.train(EPISODES, STEPS_PER_EPISODE, TOTAL_STEPS)
```

### Evaluación con video

```python
env = make_env(ENV_NAME, video_folder="./videos/dqn_validation",
               simulation_frequency=SIM_FREQUENCY_EVAL, ...)  # 30 Hz para video fluido

dqn_agent.play(env, episodes=3)   # 3 episodios greedy grabados en video
Video("./videos/dqn_validation/rl-video-episode-0.mp4", embed=True, width=600)
```

### Entrenamiento DDQN

```python
modelo_a    = DQN_CNN_Model(env.observation_space.shape, env.action_space.n).to(DEVICE)
modelo_b    = DQN_CNN_Model(env.observation_space.shape, env.action_space.n).to(DEVICE)

ddqn_agent  = DoubleDQNAgent(env, modelo_a, modelo_b, process_state,
                              sync_target=SYNC_TARGET, ...)
ddqn_rewards = ddqn_agent.train(EPISODES, STEPS_PER_EPISODE, TOTAL_STEPS)
```

---

## 9. Hiperparámetros y su significado

| Hiperparámetro | Valor | Qué controla | ¿Qué pasa si lo cambiás? |
|----------------|-------|-------------|--------------------------|
| `TOTAL_STEPS` | 100,000 | Tope global de pasos de entrenamiento | Más pasos → más aprendizaje (hasta convergencia) |
| `GAMMA` | 0.99 | Factor de descuento: valor de recompensas futuras | Muy alto → planifica largo plazo pero puede ser inestable |
| `EPSILON_INI` | 1.0 | Exploración inicial: 100% aleatorio | Más bajo → explora menos, puede quedar en óptimos locales |
| `EPSILON_MIN` | 0.05 | Mínimo de exploración (nunca es 0%) | Muy bajo → puede sobreajustar lo aprendido |
| `EPSILON_ANNEAL_STEPS` | 20,000 | Pasos para bajar epsilon de 1.0 a 0.05 | Muy rápido → explota antes de aprender bien |
| `BATCH_SIZE` | 32 | Transiciones por actualización | Más grande → gradientes más estables pero más lento |
| `BUFFER_SIZE` | 10,000 | Capacidad del replay buffer | Más grande → más diversidad de muestras pero más RAM |
| `LEARNING_RATE` | 1e-5 | Tasa de aprendizaje de RMSProp | Muy alto → inestable. Muy bajo → aprende lento |
| `LEARNING_STARTS` | 1,000 | Pasos de warmup antes de entrenar | Asegura variedad en el buffer antes del primer update |
| `SYNC_TARGET` | 1,000 | (Solo DDQN) Cada cuántos pasos sincronizar B←A | Muy frecuente → B ≈ A, reduce el beneficio de tener dos redes |

---

## 10. DQN vs DDQN: diferencias clave

| Aspecto | DQN (Mnih 2013) | Double DQN (van Hasselt 2015) |
|---------|-----------------|-------------------------------|
| Número de redes | 1 (`policy_net`) | 2 (online + target) |
| Selección de acción en s' | `policy_net` | Red online (A): `Q_A(s').argmax()` |
| Evaluación Q en s' | `policy_net` (misma) | Red target (B): `Q_B(s').gather(best_action)` |
| Fórmula del target | `r + γ · max_a' Q(s',a'; θ)` | `r + γ · Q_B(s', argmax_a' Q_A(s',a'))` |
| Sesgo de sobreestimación | Presente | Reducido |
| Estabilidad del entrenamiento | Menor | Mayor (targets más estables) |
| Q-values esperados | Sobreestimados | Más calibrados con el reward real |

### ¿Cuándo es visible la diferencia?

La diferencia se vuelve notable en entornos complejos o con muchas acciones. Para verla en las gráficas, comparar los **Q-values medios**: DQN tenderá a sobreestimar (Q-values más altos que el reward real obtenido), DDQN estará más calibrado.

---

## 11. Preguntas frecuentes del docente

**¿Por qué `terminated` y no `done` en el buffer?**

`done = terminated OR truncated`. Si el episodio se cortó por timeout (`truncated=True`, `terminated=False`), el estado s' tiene valor futuro real. Si guardáramos `done`, el factor `(1 - done)` anularía el bootstrap en esos casos, enseñándole al agente que esas situaciones no tienen valor futuro — lo cual es incorrecto.

---

**¿Por qué RMSProp y no Adam?**

Fidelidad al paper de Mnih 2013, que usó RMSProp. Adam fue propuesto en 2014. Al mantener el mismo optimizador en DQN y DDQN, la comparación aísla solo el cambio en la regla del target, sin mezclar variables adicionales.

---

**¿Por qué MSE y no Huber Loss?**

El paper de Mnih 2013 usa MSE. La pérdida Huber (SmoothL1) apareció en la versión Nature 2015. Se mantiene MSE por consistencia con el paper original y para aislar el efecto del cambio de algoritmo.

---

**¿Por qué `self.policy_net = self.policy_net_a` en `DoubleDQNAgent`?**

La clase base `Agent.greedy_action()` usa `self.policy_net` para seleccionar acciones. DDQN necesita que esa referencia apunte a la red online (A), la más actualizada. Es un puente para reusar el código de la clase base sin reescribir el método.

---

**¿Por qué normalizar la observación a [0,1]?**

Las redes neuronales aprenden mejor con inputs en rangos pequeños. Un pixel uint8 va de 0 a 255; dividir por 255 lo lleva a [0,1]. Sin esto, los gradientes serían 255 veces más grandes, desestabilizando el entrenamiento.

---

**¿Por qué frame stacking de 4 y no más?**

4 frames es el valor del paper de Mnih para Atari. Es suficiente para capturar movimiento (velocidad) y aceleración. Más frames implican más memoria y más parámetros en la primera capa convolucional. Mnih validó que 4 es un buen balance.

---

**¿Qué hace `torch.no_grad()` y por qué importa?**

Cuando PyTorch hace un forward pass normalmente construye un grafo computacional para poder calcular gradientes. `torch.no_grad()` desactiva esto: más rápido y usa menos memoria. Se usa cuando no se va a llamar a `backward()` (inferencia en `greedy_action` y cálculo del target en `update_weights`).

---

**¿Por qué el target se calcula con `no_grad()`?**

El target `y_i` es la "respuesta correcta" que queremos que aprenda la red — se trata como una constante. Si dejáramos fluir el gradiente también por el target, estaríamos diferenciando los pesos en ambas partes del loss simultáneamente. Esto crea gradientes que se anulan o explotan, desestabilizando el entrenamiento.

---

## 12. Glosario

| Término | Definición |
|---------|-----------|
| **MDP** | Markov Decision Process. Framework del RL: (S, A, P, R, γ). El agente está en estado s, toma acción a, recibe reward r, pasa a s'. |
| **Q-value / Q(s,a)** | Valor esperado de recompensa acumulada al tomar la acción a en el estado s y luego seguir la política óptima. |
| **Ecuación de Bellman** | Q*(s,a) = E[r + γ · max_a' Q*(s',a')]. Relación recursiva que define el Q óptimo. |
| **Política ε-greedy** | Con prob. ε: acción aleatoria (exploración). Con prob. 1-ε: argmax Q(s,a) (explotación). |
| **Replay Buffer** | Buffer circular de transiciones pasadas para muestreo aleatorio. Rompe correlación temporal. |
| **Target network** | Copia de la red online cuyos pesos se actualizan periódicamente. Estabiliza el entrenamiento al proveer targets más estables. |
| **Backpropagation** | Algoritmo para calcular gradientes del loss respecto a todos los parámetros usando la regla de la cadena. |
| **RMSProp** | Optimizador adaptativo: ajusta la tasa de aprendizaje por parámetro según el promedio de cuadrados de gradientes recientes. |
| **Frame stacking** | Apilar los últimos k frames como input a la red para poder inferir velocidad y aceleración. |
| **state_dict()** | Diccionario PyTorch con `{nombre_capa: tensor_de_pesos}`. Forma recomendada de guardar y cargar pesos de una red. |
| **terminated vs truncated** | `terminated`: fin real del MDP (colisión). `truncated`: corte por tiempo máximo. Solo `terminated` anula el bootstrap de Bellman. |
| **Bootstrap** | Usar la estimación actual de Q para calcular el target: `r + γ · max Q(s')` "bootstrapea" sobre la estimación actual. |
| **torch.gather** | Selecciona elementos de un tensor según índices. Usado para extraer Q(s,a) de la fila de Q-values de un estado. |
| **Warmup / learning_starts** | Período inicial donde el agente solo recolecta experiencias sin entrenar, para que el buffer tenga suficiente variedad. |
| **Anneal** | Reducción gradual de un parámetro (como epsilon) a lo largo del tiempo de entrenamiento. |
| **Sobreestimación** | Sesgo de DQN vanilla: el operador max sobre Q-values ruidosos tiende a elegir el más alto aunque sea por ruido, inflando los Q-values estimados. |

---

## 13. La red en profundidad (`dqn_cnn_model.py`)

Esta es la única pieza del proyecto que **no** es un `Agent`: es el aproximador de función `Q(s,·;θ)` que reemplaza a la Q-table de los labs anteriores (Q-learning tabular en MountainCar/FrozenLake).

### ¿Por qué hace falta una red (y no una tabla)?

En los labs de TD, la observación se discretizaba a `NUMBER_STATES × NUMBER_STATES` para poder indexar `Q[pos, vel, action]`. Acá la observación es una imagen — un tensor `(4, 84, 336)` de píxeles — con un espacio de estados astronómico (cada combinación de valores de píxel es técnicamente un estado distinto). Es imposible tabular eso: hace falta un aproximador que **generalice** (dos imágenes parecidas deben tener Q-values parecidos, sin haber visto exactamente ese estado antes).

### ¿Por qué convolucional y no un MLP sobre los píxeles?

La entrada tiene estructura espacial 2D. Una `Conv2d` explota dos propiedades que un `Linear` sobre los píxeles crudos no aprovecha:

- **Localidad:** un filtro pequeño (8×8) mira una región acotada — coherente con que las features relevantes (un auto, una línea de carril) son locales.
- **Invarianza a la traslación (parcial):** el mismo filtro se aplica en toda la imagen, así que un auto se reconoce igual en cualquier posición, sin aprender el patrón por separado para cada lugar posible.

Esto reduce drásticamente los parámetros comparado con conectar cada píxel a cada neurona, y es el mismo argumento que usó Mnih et al. (2013) para Atari.

### El helper `h_w()` — por qué se calcula y no se hardcodea

📍 `dqn_cnn_model.py:10-14`

```python
def h_w(self, h_in, w_in, padding, kernel, stride, dilatation=1):
    h_out = ((h_in + 2*padding - dilatation*(kernel - 1) - 1) // stride) + 1
    w_out = ((w_in + 2*padding - dilatation*(kernel - 1) - 1) // stride) + 1
    return h_out, w_out
```

Es la fórmula estándar de PyTorch para el tamaño de salida de una `Conv2d`. Se usa para encadenar el cálculo de alto/ancho hasta llegar a `fc1`, de forma que si cambia `obs_shape` (por ejemplo, al pasar de highway-fast-v0 a otro entorno de la Parte 2), **no hay que recalcular a mano** el `in_features` de la primera capa densa — el propio `__init__` lo deriva.

### Arquitectura completa, con los números reales del proyecto

📍 `dqn_cnn_model.py:16-42` (`DQN_CNN_Model.__init__`, donde se instancian `conv1`, `conv2`, `fc1`, `fc2`)

Con `obs_shape = (4, 84, 336)` (4 frames apilados, 84 alto, 336 ancho) y `n_actions = 5`:

| Capa | Args | Entrada | Salida | Cálculo de `h_w` |
|------|------|---------|--------|------------------|
| `conv1` | `Conv2d(in=4, out=16, kernel=8, stride=4)` | `(B, 4, 84, 336)` | `(B, 16, 20, 83)` | `h1=((84-8)//4)+1=20` · `w1=((336-8)//4)+1=83` |
| ReLU | — | `(B, 16, 20, 83)` | igual | — |
| `conv2` | `Conv2d(in=16, out=32, kernel=4, stride=2)` | `(B, 16, 20, 83)` | `(B, 32, 9, 40)` | `h2=((20-4)//2)+1=9` · `w2=((83-4)//2)+1=40` |
| ReLU | — | `(B, 32, 9, 40)` | igual | — |
| flatten | `view(B, -1)` | `(B, 32, 9, 40)` | `(B, 11520)` | `32×9×40 = 11.520` |
| `fc1` | `Linear(11520, 256)` | `(B, 11520)` | `(B, 256)` | — |
| ReLU | — | `(B, 256)` | igual | — |
| `fc2` | `Linear(256, n_actions)` | `(B, 256)` | `(B, 5)` | **capa de salida: 5 neuronas** (una por acción de `DiscreteMetaAction`) |

**Total: 2.962.997 parámetros** (confirmado con `torchinfo.summary()` en la notebook final).

> **Capa de salida:** `fc2` no tiene activación — los Q-values son valores reales sin acotar (pueden ser negativos), a diferencia de una clasificación donde la última capa suele ir a softmax/sigmoid. Poner una ReLU o sigmoid en la salida sería un error: recortaría Q-values negativos que son perfectamente válidos.

### El `forward()` completo

📍 `dqn_cnn_model.py:45-66`

```python
def forward(self, obs):
    result = self.conv1(obs)               # (B, 4, 84, 336) -> (B, 16, 20, 83)
    result = F.relu(result)
    result = self.conv2(result)            # (B, 16, 20, 83) -> (B, 32, 9, 40)
    result = F.relu(result)
    result = result.view(result.size(0), -1)  # (B, 32, 9, 40) -> (B, 11520)
    result = F.relu(self.fc1(result))         # (B, 11520) -> (B, 256)
    return self.fc2(result)                   # (B, 256) -> (B, n_actions)
```

Un único forward pass devuelve el Q-value de **todas** las acciones a la vez (vector de tamaño `n_actions`), no un escalar por par `(s,a)`. Esto es clave para la eficiencia: tanto `argmax_a Q(s,a)` (selección de acción) como `max_a Q(s,a)` (target de Bellman) se resuelven con un solo forward pass por estado, no uno por cada acción posible.

### Nota sobre la arquitectura simplificada respecto al paper

El paper completo de Mnih 2013 usa 3 capas convolucionales y una FC de 512. Acá se sugiere una versión reducida (2 conv + FC de 256), justificada por las restricciones de tiempo/cómputo de un entorno de curso (Colab/notebook, pocas horas por agente). Además, como el input preserva el aspect ratio 4:1 de highway (336×84) en vez del cuadrado 84×84 de Atari, el vector aplanado que entra a `fc1` (11.520) es bastante más grande y asimétrico que en la arquitectura clásica — la primera capa `Linear` termina con muchos más parámetros de entrada de los que tendría con la geometría cuadrada original.

---

## 14. Defensa oral — Parcial 14/07/2026 (respuestas directas)

Estas son las 4 preguntas de defensa del parcial más reciente, con la respuesta lista para decir en voz alta.

### A. ¿Cuál es el principal parámetro al crear la Replay Memory? ¿Qué guarda y cuáles son sus principales métodos?

📍 `replay_memory.py:15` (`Transition`) · `24-32` (`__init__`) · `35-51` (`add`) · `54-67` (`sample`) · `70-78` (`__len__`/`clear`)

- **Parámetro principal:** `capacity` — el único argumento de `ReplayMemory.__init__(self, capacity)`. Define el tamaño máximo del buffer circular.
- **Qué guarda:** instancias de la namedtuple `Transition = (state, action, reward, terminated, next_state)`. Importante aclarar en la defensa: guarda `terminated`, **no** `done` (`done = terminated or truncated`) — el target de Bellman solo debe anular el bootstrap cuando el episodio terminó por estado terminal real, no por timeout.
- **Métodos principales:**
  - `__init__(capacity)`: inicializa `memory=[]` y el puntero circular `position=0`.
  - `add(state, action, reward, terminated, next_state)`: agrega si hay espacio; si está lleno, sobrescribe en `position` (FIFO circular) y avanza el puntero.
  - `sample(batch_size)`: `random.sample(self.memory, batch_size)` — muestreo uniforme sin reposición.
  - `__len__()`: cantidad actual de transiciones almacenadas.
  - `clear()`: vacía la memoria y resetea el puntero.

### B. Describa la arquitectura de la red utilizada (capas y su orden). ¿Cuántas neuronas tiene de salida?

📍 `dqn_cnn_model.py:16-42` (`__init__`, layers) · `45-66` (`forward`)

`Conv2d(4→16, k=8, s=4) → ReLU → Conv2d(16→32, k=4, s=2) → ReLU → Flatten (32×9×40=11.520) → Linear(11.520→256) → ReLU → Linear(256→n_actions)`.

**Neuronas de salida: `n_actions` = 5** (una por cada acción de `DiscreteMetaAction`: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`). Ver la [tabla completa con los cálculos de `h_w`](#13-la-red-en-profundidad-dqn_cnn_modelpy) en la sección 13.

### C. Describa la diferencia en la implementación de DQN y DDQN, especialmente en `update_weights`

📍 `dqn_agent.py:91-116` vs `double_dqn_agent.py:106-134`

| | `dqn_agent.py` | `double_dqn_agent.py` |
|---|---|---|
| Redes | 1 (`policy_net`) | 2 (`policy_net_a` online + `model_b` target) |
| Quién elige la mejor acción en `s'` | `policy_net` | `policy_net_a` (online): `argmax(dim=1)` |
| Quién evalúa esa acción | la misma `policy_net` | `model_b` (target): `.gather(1, best_action)` |
| Fórmula del target | `r + γ·max_a Q(s',a;θ)·(1-terminated)` | `r + γ·Q_target(s', argmax_a Q_online(s',a))·(1-terminated)` |
| Sincronización de redes | no aplica | `model_b.load_state_dict(policy_net_a.state_dict())` cada `sync_target` llamadas a `update_weights()` (hard update) |

```python
# DQN — dqn_agent.py: una sola red hace todo
with torch.no_grad():
    max_next_q = self.policy_net(next_states_t).max(dim=1).values
    targets = rewards_t + self.gamma * max_next_q * (1 - terminateds_t)

# DDQN — double_dqn_agent.py: selección (A) y evaluación (B) separadas
with torch.no_grad():
    best_action = self.policy_net_a(next_states_t).argmax(dim=1).unsqueeze(1)   # elige A
    next_q = self.model_b(next_states_t).gather(1, best_action).squeeze(1)     # evalúa B
    targets = rewards_t + self.gamma * next_q * (1 - terminateds_t)
```

El resto de `update_weights` (armar `current_q` con `gather`, `MSELoss`, `zero_grad/backward/step`, grad clipping opcional) es **idéntico** en ambos — es intencional, para que la comparación DQN vs DDQN aísle únicamente el cambio en la regla del target.

### D. ¿Por qué es importante el método `gather` de PyTorch? Nombre todos los lugares donde se usa

**Por qué importa:** `current_q_values` (o `next_q_values`) es un tensor `(B, n_actions)` con el Q-value de **todas** las acciones para cada elemento del batch. Pero para calcular la loss solo interesa el Q-value de **la acción que efectivamente se tomó** (o la elegida por la red online, en DDQN). `gather(dim, index)` selecciona, para cada fila, el valor en la columna indicada por `index` — es la forma vectorizada de `current_q[i][action[i]]` para todo el batch a la vez, sin loop en Python.

**Los 3 usos concretos en el proyecto:**

1. `dqn_agent.py:107` → `update_weights()`: `current_q_values.gather(1, actions_t.unsqueeze(1))` — extrae `Q(s,a)` de la acción tomada, para compararla contra el target en la `MSELoss`.
2. `double_dqn_agent.py:115` → `update_weights()`: `self.model_b(next_states_t).gather(1, best_action)` — evalúa con la red target el Q-value de la acción que eligió la red online (`best_action`, obtenido con `argmax` en la línea 114, no con `gather`).
3. `double_dqn_agent.py:120` → `update_weights()`: `current_q_values.gather(1, actions_t.unsqueeze(1))` — igual que en DQN, extrae `Q(s,a)` de la acción tomada usando `policy_net_a`.

`greedy_action()` **no** usa `gather`: ahí se quiere la mejor acción global del estado (un escalar), así que alcanza con `argmax(dim=1)` directamente sobre el vector de Q-values.

---

## 15. Banco de preguntas de parciales anteriores (2020–2022)

Preguntas de defensa de años previos que giran sobre exactamente estos 5 archivos. El docente tiende a repetir el patrón (adaptación del paper, manejo de las 2 redes, función del replay, por qué CNN), así que vale la pena tenerlas ensayadas.

### Parcial 2020 — "¿Qué modificaciones hicieron para adaptar el algoritmo de DeepMind (Atari) a highway-env?"

Resumen de las adaptaciones (todas ya cubiertas en detalle en las secciones 2 y 3, acá el listado corto para repasar rápido):

a. **Observación forzada a pixel-based:** highway-env da nativamente `KinematicObservation` (oráculo simbólico); se agregó `_RenderAsObservation` para reemplazarla por el frame RGB, igual que en Atari.
b. **Resize a 336×84** (no 84×84 cuadrado de Atari) para preservar el aspect ratio 4:1 de la autopista.
c. **Frame-skip vía config del simulador** (`simulation_frequency`/`policy_frequency`), no vía wrapper `AtariPreprocessing(frame_skip=k)` — mismo efecto, distinta implementación.
d. **Espacio de acciones distinto:** `DiscreteMetaAction` (5 meta-acciones de alto nivel) en vez de acciones de joystick crudas de Atari; no cambia la arquitectura de salida (sigue siendo `Linear` con `n_actions` salidas).
e. **Red simplificada:** 2 capas conv + FC de 256 (en vez de 3 capas + FC de 512), por restricciones de cómputo.
f. **Hiperparámetros reducidos:** `BUFFER_SIZE=10.000` (vs 1M del paper), `TOTAL_STEPS≈100k-300k` (vs ~50M frames de Atari) — adaptación práctica, no algorítmica.
g. **Reward shaping:** `HIGH_SPEED_REWARD` subido para romper el óptimo local de "ir a la velocidad del vecino".
h. **Sin reward clipping:** el paper de Atari clippea porque las magnitudes varían mucho entre juegos distintos; acá hay un solo problema con reward ya bien definido.

### Parcial 2021 — "¿Cómo manejaron las dos redes (target/online) en su Obligatorio?"

- **DQN (`dqn_agent.py`):** fiel al paper 2013 — **sin** target network. Una sola `policy_net` que elige la acción (ε-greedy) *y* calcula el target de Bellman. Es justo lo que el paper de 2015 identifica como fuente de inestabilidad (el objetivo se mueve al mismo tiempo que se entrena la red que lo genera), pero se mantiene así deliberadamente por fidelidad a la versión de 2013.
- **DDQN (`double_dqn_agent.py`):** dos redes explícitas. `policy_net_a` (online, se entrena por gradiente) elige la acción tanto en el entorno como en el target; `model_b` (target) solo evalúa esa acción y se sincroniza por **hard update** completo cada `sync_target` pasos — no hay soft-update/Polyak.
- **Por qué se mantuvo MSE + RMSProp en ambos** (en vez de Huber + otro optimizador para DDQN): para que la comparación DQN vs DDQN aísle únicamente el cambio en la regla del target, sin mezclar además un cambio de loss/optimizador entre los dos experimentos.
- **Por qué el reward por episodio es ruidoso pero el Q promedio (sobre un conjunto fijo de estados) es suave:** el reward depende de simular una trayectoria completa y estocástica (pequeños cambios en los pesos alteran decisiones tempranas que se propagan y se amplifican durante todo el episodio). El Q promedio sobre estados fijos, en cambio, es una función determinística de los pesos actuales (`Q(s,·;θ)`), sin rollouts estocásticos de por medio — por eso sube de forma mucho más monótona y es preferible para monitorear el progreso real del entrenamiento.
- **Ojo:** una curva de Q que sube consistentemente no garantiza ausencia de sobreestimación — de hecho esa es la motivación original de DDQN: el propio DQN puede inflar sistemáticamente esos valores.
- **¿Qué función cumple la experience replay?** Tres funciones relacionadas, todas explícitas en `replay_memory.py`: (1) rompe la correlación temporal entre transiciones consecutivas (SGD asume muestras ~i.i.d.); (2) reutiliza cada transición varias veces (sample efficiency: interactuar con el simulador es la parte cara); (3) estabiliza la distribución de entrenamiento, evitando que la red "olvide" (catastrophic forgetting) situaciones que ya no se visitan tanto al mezclar experiencia vieja y nueva.

### Parcial 2022 — "¿Qué problema de DQN soluciona Double DQN?" / "¿Qué función resuelve la red neuronal?"

- **Problema (overestimation bias):** en DQN vanilla, `Y = R + γ·max_a Q(S',a)` usa la misma red para elegir *y* evaluar la mejor acción. Como las estimaciones de Q siempre tienen ruido, tomar un `max` sobre estimaciones ruidosas tiende a sobreestimar sistemáticamente el valor real (consecuencia del tipo desigualdad de Jensen), y ese sesgo se amplifica episodio tras episodio vía bootstrapping.
- **Solución de DDQN:** desacoplar selección (`argmax` con la red online) de evaluación (Q con la red target) — como ambas redes rara vez coinciden exactamente en cuál es "la mejor" acción, el sesgo se reduce.
- **Qué observar empíricamente:** graficar `max_a Q(s,a)` promedio sobre un conjunto fijo de estados de validación para ambos agentes — DQN debería mostrar valores sistemáticamente más altos (a veces poco realistas frente al reward medio real) que DDQN.
- **Conexión de la replay memory con Dyna-Q (lab 6):** ambas técnicas atacan el mismo problema (sample efficiency) de formas distintas — Dyna-Q *inventa* experiencia extra simulando con un modelo aprendido del entorno; el replay buffer *reutiliza* experiencia real ya recolectada, sin aprender ningún modelo.
- **Conexión con el carácter off-policy de Q-learning:** el replay buffer solo es seguro de usar porque Q-learning aprende sobre la política greedy independientemente de qué política generó los datos — permite mezclar transiciones recolectadas bajo distintos valores históricos de ε. Un algoritmo on-policy como Sarsa no podría reusar el buffer tan directamente.
- **Por qué hace falta una CNN (no cualquier red densa):** la entrada tiene estructura espacial (2D); una convolucional explota localidad e invarianza a la traslación, reduciendo drásticamente los parámetros frente a un MLP directo sobre los píxeles.
- **Sobre las dimensiones `(32, 4, 84, 336)` de un batch:** `32` = batch_size (transiciones muestreadas del buffer), `4` = frames apilados (no color — es historia temporal para poder inferir velocidad), `84×336` = alto×ancho del frame preprocesado. La salida es `(32, n_actions)`: un vector de Q-values por cada elemento del batch.

---

## 16. Repaso rápido: Gymnasium, K-Bandits y PyTorch

Preguntas generales del curso que suelen combinarse con las de DQN/DDQN en la defensa. Respuestas cortas, sin repetir lo ya cubierto en las secciones anteriores.

### Gymnasium

- **Los 5 retornos de `env.step()`:** `observation, reward, terminated, truncated, info`. `terminated`: fin real del MDP. `truncated`: corte externo (típicamente `TimeLimit`). `info`: dict de diagnóstico, nunca debería usarse para decidir la política.
- **`env.reset()` devuelve `(observation, info)`:** un estado inicial válido (no cualquier estado — sigue la lógica interna de inicialización del entorno) y reinicia completamente el estado interno (el entorno "olvida" el episodio anterior).
- **Bug clásico:** resetear solo cuando `terminated` es `True` e ignorar `truncated` dejaría el bucle llamando `step()` sobre un entorno ya terminado por timeout. Corrección: `if terminated or truncated: env.reset()`.
- **Wrapper = patrón decorator:** envuelve un env para interceptar `step`/`reset`/`render`/espacios, sin tocar el entorno original. Ejemplos del proyecto: `RecordVideo` (graba episodios, solo intercepta `render()`), y el pipeline custom `_RenderAsObservation → GrayscaleObservation → FrameStackObservation`.

### K-Bandits (mismo patrón de `compute_epsilon` que `abstract_agent.py`)

- **Update incremental correcto:** `Q[arm] += (reward - Q[arm]) / N[arm]` — nunca `Q[arm] = reward` (perdería toda la historia acumulada).
- **α constante vs 1/N:** con 1/N, el peso de cada reward nuevo decrece con el tiempo (converge al promedio histórico); con α constante, el peso reciente no decae — mejor para entornos no estacionarios. El replay buffer del Obligatorio, al ser circular y de tamaño acotado, tiene una lógica de no-estacionariedad análoga: descarta transiciones viejas para no sobre-ponderar comportamiento ya obsoleto del agente.
- **Inicialización optimista:** `Q = np.ones(K) * optimistic_value` fuerza exploración temprana automática sin necesitar ε alto — el primer reward real (más bajo que la estimación inicial) "decepciona" ese brazo y empuja a probar los demás.

### PyTorch

- **`torch.Tensor` vs `np.ndarray`:** autograd (`requires_grad`, `.backward()`), soporte de device (`cpu`/`cuda`/`mps`), e integración nativa con `nn.Module`/`torch.optim`. Comparten memoria en CPU (`torch.from_numpy`).
- **`nn.Linear(in_features, out_features)`:** aplica `y = xW^T + b`; define el tamaño de la matriz de pesos entrenable.
- **`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`:** `in_channels` = canales de entrada (4 frames apilados en este proyecto, no color); `out_channels` = cantidad de filtros/feature maps de salida; `stride` controla el downsampling espacial; `padding` controla si se conserva el tamaño espacial.
- **`squeeze`/`unsqueeze`:** no mueven datos, solo reinterpretan la forma. `unsqueeze(0)` agrega la dimensión de batch cuando `process_state` recibe una sola observación (3D → 4D) para que la CNN reciba siempre `(B, C, H, W)`.
- **`torch.no_grad()`:** desactiva el grafo de gradientes — se usa en `greedy_action()` (inferencia pura) y al calcular el target de Bellman en `update_weights()` (el target se trata como constante, no debe propagar gradiente).
- **El ciclo estándar de entrenamiento** (`to(device)` → `zero_grad()` → forward → loss → `backward()` → `step()`): sin `zero_grad()`, PyTorch acumula gradientes de batches anteriores y corrompe la actualización; sin `backward()`, no hay gradientes que aplicar; sin `step()`, calcular el gradiente no serviría de nada — es el paso que efectivamente actualiza los pesos.

---

*Material de estudio — Taller de Inteligencia Artificial 2026 | Obligatorio Parte 1*
