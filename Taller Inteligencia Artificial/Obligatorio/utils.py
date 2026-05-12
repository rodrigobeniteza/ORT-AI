"""Wrappers y utilidades de entorno para highway-env.

El pipeline de observacion arma una imagen 4x84x336 grayscale stackeada a
partir del rendering de highway-env. Mantenemos el aspect ratio original
del scene (4:1) para no deformar autos y lineas.

make_env() devuelve un env listo para usar; los wrappers que aplica son
todos de gymnasium estandar (no hay nada Atari-specific aca).
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    ResizeObservation,
)


class _RenderAsObservation(gymnasium.ObservationWrapper):
    """Reemplaza la obs por env.render() (RGB del scene).

    highway-env no expone una obs RGB plana sobre la que aplicar wrappers
    de imagen de gymnasium. Lo armamos nosotros: render() devuelve el frame
    RGB del scene, y los wrappers de afuera (resize/grayscale/stack) hacen
    el preprocesado de forma estandar.
    """

    def __init__(self, env):
        super().__init__(env)
        env.reset()
        h, w, c = env.render().shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def observation(self, _obs):
        return self.env.render()


class _SafeRecordVideo(RecordVideo):
    """RecordVideo que ignora _capture_frame fuera de un episodio grabado.

    highway-env llama a _capture_frame() durante _simulate() siempre que el
    viewer este inicializado, sin chequear si la episode_trigger esta activa.
    Sin este guard, episodios no grabados rompen con
    'Cannot capture a frame, recording wasn't started.'
    """

    def _capture_frame(self):
        if self.recording:
            super()._capture_frame()


def show_observation(observation):
    """Muestra una observacion (H, W) o (H, W, C) en matplotlib."""
    dim = observation.shape
    if len(dim) == 3:
        if dim[2] == 3:
            plt.imshow(observation)
        elif dim[2] == 1:
            plt.imshow(observation[:, :, 0], cmap="gray")
    elif len(dim) == 2:
        plt.imshow(observation, cmap="gray")
    else:
        raise ValueError(f"Invalid observation shape: {dim}")
    plt.show()


def show_observation_stack(observation):
    """Muestra cada frame del stack uno abajo del otro."""
    frames = observation.shape[0]
    fig, axes = plt.subplots(frames, 1, figsize=(10, 2 * frames))
    if frames == 1:
        axes = [axes]
    for i in range(frames):
        axes[i].imshow(observation[i], cmap="gray")
        axes[i].set_title(f"frame {i}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def make_env(
    env_name: str,
    *,
    video_folder: str | None = None,
    record_every: int | None = None,
    obs_height: int = 84,
    obs_width: int = 336,
    stack_frames: int = 4,
    simulation_frequency: int = 15,
    duration: int = 600,
    lanes_count: int = 6,
    vehicles_count: int = 50,
    vehicles_density: float = 2.0,
    high_speed_reward: float = 0.4,
    reward_speed_range: tuple[float, float] = (20.0, 30.0),
    seed: int | None = None,
) -> gymnasium.Env:
    """Crea un env de highway-env con el pipeline de imagen aplicado.

    Args:
        env_name: id del env (ej "highway-fast-v0").
        video_folder: si se pasa junto con record_every, graba videos.
        record_every: frecuencia (en episodios) con que se graba video.
        obs_height, obs_width: shape de la imagen de entrada a la red.
            Default 84x336 mantiene el aspect 4:1 del render original.
        stack_frames: cuantos frames apila FrameStackObservation.
        simulation_frequency: Hz de fisica por segundo simulado. Default 15
            da videos fluidos. Bajar a 5 (highway-fast default) acelera ~3x
            el entrenamiento pero deja video choppy.
        duration: largo maximo del episodio en segundos simulados. Default
            600 (10 min). El default highway-fast es 30s.
        lanes_count: numero de carriles. Default 6 (default highway-fast = 3).
        vehicles_count: cantidad de vehiculos al inicio del episodio. Default
            50 (default highway-fast = 20). highway-env spawnea solo en
            reset(), no hay re-spawn continuo: subir esto evita que el scene
            quede vacio en episodios largos.
        vehicles_density: densidad de spawn (vehiculos/spacing-unit). Default
            2.0 (default = 1.0). Mayor densidad = autos mas juntos al inicio.
        high_speed_reward: peso del componente de velocidad en la recompensa.
            Default 0.4 (highway-env). Subir a 0.8-1.0 hace que adelantar
            valga claramente mas que cruisear detras del vehiculo lento — fix
            tipico para el local optimum "match neighbor speed". Es reward por
            ESTAR a alta velocidad (no por la accion FASTER); se computa cada
            step como `peso * scaled_speed` donde scaled_speed mapea linealmente
            `reward_speed_range` a [0, 1] (clipeado).
        reward_speed_range: rango (m/s) en que se mapea linealmente la reward
            de velocidad. Default (20, 30). Subir el lower bound (ej. (25, 30))
            obliga al agente a perseguir velocidad activa: por debajo del minimo
            no cobra nada por velocidad.
        seed: semilla para env.reset().
    """
    config = {
        # Render a Surface en RAM, sin abrir ventana SDL.
        "offscreen_rendering": True,
        "simulation_frequency": simulation_frequency,
        "duration": duration,
        "lanes_count": lanes_count,
        "vehicles_count": vehicles_count,
        "vehicles_density": vehicles_density,
        "high_speed_reward": high_speed_reward,
        "reward_speed_range": list(reward_speed_range),
    }
    env = gymnasium.make(env_name, render_mode="rgb_array", config=config)
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    # Pipeline de preprocesado, todo wrappers de gymnasium estandar:
    #   render RGB (H, W, 3) → resize → grayscale (H, W) → stack (S, H, W)
    env = _RenderAsObservation(env)
    env = ResizeObservation(env, (obs_height, obs_width))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=stack_frames)
    env = RecordEpisodeStatistics(env)

    if video_folder is not None and record_every is not None:
        env = _SafeRecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep: ep % record_every == 0,
            disable_logger=True,
        )
        # highway-env captura solo 1 frame por env.step() por defecto, lo
        # que produce videos saltarines. Registrar el wrapper le dice a
        # AbstractEnv._simulate() que llame a _capture_frame() por cada
        # sub-frame de simulacion (a simulation_frequency Hz).
        env.unwrapped.set_record_video_wrapper(env)

    return env
