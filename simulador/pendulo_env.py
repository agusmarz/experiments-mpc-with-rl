"""Planta no lineal carro-péndulo como entorno de Gymnasium.

Replica la planta de Gazebo del repo (URDF + world.sdf) para entrenar y
evaluar en Python. Estado [x, x_dot, theta, theta_dot] con theta = 0 en la
vertical y theta > 0 inclinando la punta hacia +x (eje +y de la junta).
La observación que devuelven step y reset es ese mismo estado completo.

Ecuaciones (1) del enunciado, más términos opcionales para el eje B
    (M + m) x_dd + m l cos(th) th_dd - m l th_d^2 sin(th) = F + F_cte - b_c x_d
    J th_dd + m l cos(th) x_dd - m g l sin(th)          = tau - b_j th_d
con J = I + m l^2 la inercia respecto del pivote.
"""

from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


@dataclass
class ParametrosPlanta:
    # Geometría y masas (inverted_pendulum.model.xacro)
    M: float = 3.0          # masa del carro [kg]
    m: float = 1.0          # masa del péndulo [kg]
    L: float = 0.5          # largo total del péndulo [m]
    r: float = 0.01         # radio del cilindro [m]
    g: float = 9.8          # gravedad de world.sdf (el LQR original usaba 9.81)

    # Límites de la planta
    fuerza_max: float = 100.0       # effort del cart_rail_joint [N]
    x_max: float = 0.5              # tope del riel [m]
    theta_corte: float = np.deg2rad(46.0)  # ángulo donde se da el episodio por perdido

    # Tiempos
    dt: float = 0.02                # período de control (mpc_params.yaml)
    dt_fisica: float = 0.001        # paso de física de Gazebo (world.sdf)

    # Términos que Gazebo NO tiene, en cero por defecto (pensados para el eje B)
    b_carro: float = 0.0            # fricción viscosa en el riel [N s/m]
    b_junta: float = 0.0            # fricción viscosa en la junta [N m s/rad]
    fuerza_constante: float = 0.0   # fuerza no modelada sobre el carro [N]

    # Costo de etapa, el mismo del MPC (mpc_params.yaml)
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 1.0, 100.0, 1.0]))
    R: float = 0.1
    penalizacion_falla: float = 1000.0

    @property
    def l(self):
        return self.L / 2.0

    @property
    def J(self):
        I = (self.m / 12.0) * (3.0 * self.r**2 + self.L**2)
        return I + self.m * self.l**2


def aceleraciones(estado, F, tau, p, lib=np):
    """Resuelve el sistema 2x2 de las ecuaciones de movimiento.

    Escrito con Cramer y lib.sin/lib.cos para que la misma función sirva con
    numpy o con casadi (pasando lib=casadi) al armar el MPC no lineal.
    """
    _, x_d, th, th_d = estado[0], estado[1], estado[2], estado[3]
    s, c = lib.sin(th), lib.cos(th)
    ml = p.m * p.l

    # Matriz de masa [[a11, a12], [a12, a22]] y lado derecho [b1, b2]
    a11 = p.M + p.m
    a12 = ml * c
    a22 = p.J
    b1 = F + p.fuerza_constante - p.b_carro * x_d + ml * th_d**2 * s
    b2 = tau - p.b_junta * th_d + ml * p.g * s

    det = a11 * a22 - a12**2
    x_dd = (b1 * a22 - a12 * b2) / det
    th_dd = (a11 * b2 - a12 * b1) / det
    return x_dd, th_dd


def derivada(estado, F, p, tau=0.0, lib=np):
    """f(x, u) en tiempo continuo."""
    x_dd, th_dd = aceleraciones(estado, F, tau, p, lib)
    return [estado[1], x_dd, estado[3], th_dd]


class PenduloEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, parametros: ParametrosPlanta | None = None, render_mode=None):
        self.p = parametros if parametros is not None else ParametrosPlanta()
        self.sub_pasos = int(round(self.p.dt / self.p.dt_fisica))
        assert abs(self.sub_pasos * self.p.dt_fisica - self.p.dt) < 1e-12, "dt tiene que ser múltiplo de dt_fisica"

        # Acción, la fuerza horizontal sobre el carro en Newtons
        self.action_space = spaces.Box(-self.p.fuerza_max, self.p.fuerza_max, shape=(1,), dtype=np.float32)

        # Observación, el estado completo [x, x_d, th, th_d] sin ruido, en el
        # mismo orden que arma joint_state_callback en el nodo de ROS 2.
        # El problema es totalmente observable, así que V*(x) depende solo de x.
        # Estos son solo los rangos de cada componente, no lo que se observa.
        #   x     hasta el doble del riel, para que la observación del paso en
        #         que te pasás de x_max siga adentro del espacio (como el CartPole)
        #   x_d   sin cota física, va en infinito
        #   th    ±pi alcanza de sobra porque el episodio corta en theta_corte
        #   th_d  sin cota física, va en infinito
        alto = np.array([2 * self.p.x_max, np.inf, np.pi, np.inf])
        self.observation_space = spaces.Box(-alto, alto, dtype=np.float64)

        # Torque sobre la junta, equivalente a publicar en /pendulum_force_cmd
        self.torque_perturbacion = 0.0

        self.render_mode = render_mode
        self._dibujante = None
        self.estado = None

    # Dinámica

    def _paso_fisica(self, F):
        """Un paso de 1 ms con Euler semi implícito, el integrador de DART."""
        x, x_d, th, th_d = self.estado
        x_dd, th_dd = aceleraciones(self.estado, F, self.torque_perturbacion, self.p)
        x_d = x_d + self.p.dt_fisica * x_dd
        th_d = th_d + self.p.dt_fisica * th_dd
        x = x + self.p.dt_fisica * x_d
        th = th + self.p.dt_fisica * th_d
        self.estado = np.array([x, x_d, th, th_d])

    def costo_etapa(self, estado, F):
        return float(estado @ self.p.Q @ estado + self.p.R * F**2)

    def step(self, action):
        assert self.estado is not None, "Llamá a reset antes de step"
        F = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -self.p.fuerza_max, self.p.fuerza_max))

        costo = self.costo_etapa(self.estado, F)

        # Retención de orden cero, la fuerza queda fija durante todo el período
        for _ in range(self.sub_pasos):
            self._paso_fisica(F)

        x, _, th, _ = self.estado
        fuera_riel = abs(x) > self.p.x_max
        caido = abs(th) > self.p.theta_corte
        terminated = bool(fuera_riel or caido)

        recompensa = -costo - (self.p.penalizacion_falla if terminated else 0.0)

        if self.render_mode == "human":
            self.render()

        info = {"costo": costo, "fuerza": F, "fuera_riel": fuera_riel, "caido": caido}
        return self.estado.copy(), recompensa, terminated, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "estado" in options:
            # Estado inicial exacto, para barrer la región de recuperación en (theta0, x_d0)
            self.estado = np.asarray(options["estado"], dtype=np.float64).copy()
        else:
            self.estado = self.np_random.uniform(-0.05, 0.05, size=4)
        self.torque_perturbacion = 0.0

        if self.render_mode == "human":
            self.render()
        return self.estado.copy(), {}

    # Render, reutilizando el dibujo del CartPole de Gymnasium

    def render(self):
        if self.render_mode is None:
            return None
        if self._dibujante is None:
            self._dibujante = CartPoleEnv(render_mode=self.render_mode)
            self._dibujante.x_threshold = self.p.x_max + 0.1  # ancho visible del mundo
            self._dibujante.length = self.p.l                 # el CartPole guarda el medio largo
        self._dibujante.state = self.estado
        return self._dibujante.render()

    def close(self):
        if self._dibujante is not None:
            self._dibujante.close()
            self._dibujante = None


gym.register(id="PenduloGazebo-v0", entry_point=PenduloEnv, max_episode_steps=500)
