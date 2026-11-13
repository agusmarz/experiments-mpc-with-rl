from concurrent.futures import ProcessPoolExecutor
import os
from typing import Callable, Sequence
import numpy as np
import torch
import torch.nn as nn

# Evita que PyTorch sature hilos dentro de cada proceso paralelo
torch.set_num_threads(1)


class RedEstimadora(nn.Module):
    """Red neuronal MLP en PyTorch que estima correcciones de parámetros del MPC."""

    def __init__(
        self,
        dim_entrada: int = 12,
        capas_ocultas: Sequence[int] = (10, 10),
        dim_salida: int = 4,
        activacion: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        self.dim_entrada = int(dim_entrada)
        self.capas_ocultas = tuple(int(c) for c in capas_ocultas)
        self.dim_salida = int(dim_salida)
        self.activacion_cls = activacion

        capas: list[nn.Module] = []
        dim_prev = self.dim_entrada

        for dim_oculta in self.capas_ocultas:
            capas.append(nn.Linear(dim_prev, dim_oculta, bias=True))
            capas.append(activacion())
            dim_prev = dim_oculta

        capas.append(nn.Linear(dim_prev, self.dim_salida, bias=True))
        capas.append(nn.Tanh())

        self.red = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)

    @torch.no_grad()
    def predecir(
        self,
        params_actuales: np.ndarray,
        estado_observado: np.ndarray,
        error_prediccion: np.ndarray,
    ) -> np.ndarray:
        """Normaliza las entradas a orden O(1) y devuelve el delta de corrección."""
        p_norm = np.asarray(params_actuales, dtype=np.float32).reshape(4) / np.array(
            [5.0, 2.0, 1.0, 0.05], dtype=np.float32
        )
        x_norm = np.asarray(estado_observado, dtype=np.float32).reshape(4) / np.array(
            [0.5, 1.0, 0.5, 2.0], dtype=np.float32
        )
        e_norm = np.asarray(error_prediccion, dtype=np.float32).reshape(4) / np.array(
            [0.001, 0.05, 0.001, 0.05], dtype=np.float32
        )

        entrada_np = np.concatenate([p_norm, x_norm, e_norm])
        entrada_t = torch.from_numpy(entrada_np).unsqueeze(0)
        salida_t = self.forward(entrada_t)
        return salida_t.squeeze(0).cpu().numpy().astype(np.float64)

    def cantidad_pesos(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def cargar_desde_fenotipo(self, fenotipo: np.ndarray) -> None:
        vector = np.asarray(fenotipo, dtype=np.float32).reshape(-1)
        esperados = self.cantidad_pesos()
        if vector.size != esperados:
            raise ValueError(
                f"El fenotipo tiene {vector.size} elementos, pero la red requiere {esperados}."
            )

        puntero = 0
        for param in self.parameters():
            n_elem = param.numel()
            trozo = vector[puntero : puntero + n_elem].reshape(param.shape)
            param.copy_(torch.from_numpy(trozo))
            puntero += n_elem

    @torch.no_grad()
    def exportar_a_fenotipo(self) -> np.ndarray:
        bloques = [param.detach().cpu().numpy().reshape(-1) for param in self.parameters()]
        return np.concatenate(bloques).astype(np.float64)


# --- Funciones auxiliares para ejecución paralela por proceso ---
_WORKER_EVALUADOR = None
_WORKER_RED = None


def _inicializar_worker(fabrica_evaluador: Callable, arq_red: dict):
    """Cada proceso hijo crea su propia instancia de Ipopt y PyTorch una sola vez."""
    global _WORKER_EVALUADOR, _WORKER_RED
    torch.set_num_threads(1)
    _WORKER_EVALUADOR = fabrica_evaluador()
    _WORKER_RED = RedEstimadora(**arq_red)
    _WORKER_RED.eval()


def _evaluar_fenotipo_worker(tarea: tuple[np.ndarray, int, Sequence[dict]]) -> float:
    fenotipo, pasos_episodio, escenarios = tarea
    _WORKER_RED.cargar_desde_fenotipo(fenotipo)
    score = float(_WORKER_EVALUADOR.evaluar(_WORKER_RED, pasos_episodio, escenarios))
    return -np.inf if np.isnan(score) else score


class AlgoritmoGenetico:
    """Algoritmo Genético paralelo para entrenar `RedEstimadora`."""

    def __init__(
        self,
        fabrica_evaluador: Callable,
        generador_escenarios: Callable[[int, np.random.Generator], Sequence[dict]],
        num_casos: int = 3,
        n_workers: int | None = None,
        dim_entrada: int = 12,
        capas_ocultas: Sequence[int] = (10, 10),
        dim_salida: int = 4,
        activacion: type[nn.Module] = nn.Tanh,
        pasos_episodio: int = 120,
        tamano_poblacion: int = 24,
        tamano_torneo: int = 3,
        num_elites: int = 2,
        prob_mutacion_ruido: float = 0.8,
        tasa_genes_ruido: float = 0.25,
        sigma_ruido: float = 0.05,
        prob_permutacion: float = 0.3,
        num_permutaciones: int = 5,
        escala_inicial: float = 0.05,
        max_generaciones: int = 150,
        max_gen_sin_mejora: int = 25,
        seed: int | None = None,
    ):
        assert tamano_poblacion >= 2
        assert 1 <= num_elites < tamano_poblacion, "num_elites debe ser >= 1 para conservar al mejor."
        assert 1 <= tamano_torneo <= tamano_poblacion

        self.fabrica_evaluador = fabrica_evaluador
        self.generador_escenarios = generador_escenarios
        self.num_casos = int(num_casos)
        self.n_workers = n_workers if n_workers is not None else max(1, (os.cpu_count() or 2) - 1)

        self.arq_red = {
            "dim_entrada": int(dim_entrada),
            "capas_ocultas": tuple(int(c) for c in capas_ocultas),
            "dim_salida": int(dim_salida),
            "activacion": activacion,
        }
        self.pasos_episodio = int(pasos_episodio)

        # Hiperparámetros del AG
        self.tamano_poblacion = int(tamano_poblacion)
        self.tamano_torneo = int(tamano_torneo)
        self.num_elites = int(num_elites)
        self.prob_mutacion_ruido = float(prob_mutacion_ruido)
        self.tasa_genes_ruido = float(tasa_genes_ruido)
        self.sigma_ruido = float(sigma_ruido)
        self.prob_permutacion = float(prob_permutacion)
        self.num_permutaciones = int(num_permutaciones)
        self.max_generaciones = int(max_generaciones)
        self.max_gen_sin_mejora = int(max_gen_sin_mejora)

        self.rng = np.random.default_rng(seed)

        red_base = self.construir_red()
        self.num_pesos = red_base.cantidad_pesos()

        self.poblacion = self.rng.normal(
            loc=0.0,
            scale=escala_inicial,
            size=(self.tamano_poblacion, self.num_pesos),
        ).astype(np.float64)

        self.scores = np.full(self.tamano_poblacion, -np.inf, dtype=np.float64)
        self.mejor_fenotipo: np.ndarray | None = None
        self.mejor_score: float = -np.inf

        self.historial_mejor_score: list[float] = []
        self.historial_promedio_score: list[float] = []
        self.generacion_actual: int = 0
        self.gen_sin_mejora: int = 0
        self._pool: ProcessPoolExecutor | None = None

    def construir_red(self, fenotipo: np.ndarray | None = None) -> RedEstimadora:
        red = RedEstimadora(**self.arq_red)
        if fenotipo is not None:
            red.cargar_desde_fenotipo(fenotipo)
        return red

    def obtener_mejor_red(self) -> RedEstimadora:
        if self.mejor_fenotipo is None:
            raise RuntimeError("Aún no se ha evaluado ningún individuo.")
        red_optima = self.construir_red(self.mejor_fenotipo)
        red_optima.eval()
        return red_optima

    def _evaluar_poblacion(self, escenarios_generacion: Sequence[dict]) -> None:
        tareas = [
            (self.poblacion[i], self.pasos_episodio, escenarios_generacion)
            for i in range(self.tamano_poblacion)
        ]

        if self.n_workers > 1 and self._pool is not None:
            resultados = list(self._pool.map(_evaluar_fenotipo_worker, tareas))
            self.scores = np.asarray(resultados, dtype=np.float64)
        else:
            if _WORKER_EVALUADOR is None:
                _inicializar_worker(self.fabrica_evaluador, self.arq_red)
            self.scores = np.asarray(
                [_evaluar_fenotipo_worker(t) for t in tareas], dtype=np.float64
            )

    def _seleccion_torneo(self) -> np.ndarray:
        indices = self.rng.choice(self.tamano_poblacion, size=self.tamano_torneo, replace=False)
        ganador = indices[np.argmax(self.scores[indices])]
        return self.poblacion[ganador].copy()

    def _cruzar(self, padre_1: np.ndarray, padre_2: np.ndarray) -> np.ndarray:
        mascara = self.rng.random(self.num_pesos) < 0.5
        return np.where(mascara, padre_1, padre_2)

    def _mutar(self, individuo: np.ndarray) -> np.ndarray:
        hijo = individuo.copy()
        if self.num_permutaciones > 0 and self.rng.random() < self.prob_permutacion:
            for _ in range(self.num_permutaciones):
                idx_a, idx_b = self.rng.choice(self.num_pesos, size=2, replace=False)
                hijo[idx_a], hijo[idx_b] = hijo[idx_b], hijo[idx_a]

        if self.rng.random() < self.prob_mutacion_ruido:
            mascara = self.rng.random(self.num_pesos) < self.tasa_genes_ruido
            ruido = self.rng.normal(loc=0.0, scale=self.sigma_ruido, size=self.num_pesos)
            hijo[mascara] += ruido[mascara]

        return hijo

    def paso_generacion(self) -> dict:
        escenarios_generacion = self.generador_escenarios(self.num_casos, self.rng)
        self._evaluar_poblacion(escenarios_generacion)

        # En gen > 0, los índices 0..num_elites-1 son los élites heredados (0 es el mejor anterior)
        score_elite_previo = float(self.scores[0]) if self.generacion_actual > 0 else -np.inf

        orden = np.argsort(self.scores)[::-1]
        self.poblacion = self.poblacion[orden]
        self.scores = self.scores[orden]

        mejor_gen = float(self.scores[0])
        scores_validos = self.scores[np.isfinite(self.scores)]
        promedio_gen = float(np.mean(scores_validos)) if scores_validos.size > 0 else -np.inf

        # Hay mejora real si es la Gen 1 o si un individuo nuevo superó al élite anterior
        # bajo los mismos escenarios de esta generación
        if self.generacion_actual == 0 or (orden[0] >= self.num_elites and mejor_gen > score_elite_previo):
            self.mejor_fenotipo = self.poblacion[0].copy()
            self.gen_sin_mejora = 0
        else:
            self.gen_sin_mejora += 1

        self.mejor_score = mejor_gen
        self.historial_mejor_score.append(mejor_gen)
        self.historial_promedio_score.append(promedio_gen)

        # Reemplazo generacional con elitismo
        nueva_poblacion = np.empty_like(self.poblacion)
        nueva_poblacion[: self.num_elites] = self.poblacion[: self.num_elites]

        for idx in range(self.num_elites, self.tamano_poblacion):
            padre_1 = self._seleccion_torneo()
            padre_2 = self._seleccion_torneo()
            hijo = self._cruzar(padre_1, padre_2)
            hijo = self._mutar(hijo)
            nueva_poblacion[idx] = hijo

        self.poblacion = nueva_poblacion
        self.generacion_actual += 1

        return {
            "generacion": self.generacion_actual,
            "mejor_score_generacion": mejor_gen,
            "promedio_score": promedio_gen,
            "gen_sin_mejora": self.gen_sin_mejora,
        }

    def ejecutar(self, verbose: bool = True) -> RedEstimadora:
        if verbose:
            print(f"Iniciando AG con {self.n_workers} procesos paralelos en CPU...")

        try:
            if self.n_workers > 1:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.n_workers,
                    initializer=_inicializar_worker,
                    initargs=(self.fabrica_evaluador, self.arq_red),
                )

            for _ in range(self.max_generaciones):
                info = self.paso_generacion()
                if verbose:
                    print(
                        f"Gen {info['generacion']:03d} | "
                        f"Mejor Gen: {info['mejor_score_generacion']:.2f} | "
                        f"Prom: {info['promedio_score']:.2f} | "
                        f"Sin mejora: {info['gen_sin_mejora']}/{self.max_gen_sin_mejora}"
                    )

                if self.gen_sin_mejora >= self.max_gen_sin_mejora:
                    if verbose:
                        print(f"Criterio de parada: {self.max_gen_sin_mejora} generaciones sin superar al élite.")
                    break
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None

        return self.obtener_mejor_red()


if __name__ == "__main__":
    import numpy as np
    from evaluador_mpc import EvaluadorSimulacionMPC

    def crear_evaluador():
        return EvaluadorSimulacionMPC(
            params_iniciales_mpc=(3.0, 1.0, 0.50, 0.01),
            rangos_params_min=(1.5, 0.4, 0.30, 0.005),
            rangos_params_max=(5.5, 2.0, 0.90, 0.030),
            horizonte_mpc=12,  # Horizonte más corto en entrenamiento = solver 3x-4x más rápido
            escala_delta=(0.04, 0.015, 0.015, 0.0005),
        )

    # Instancia ligera solo para generar escenarios en el proceso principal
    eval_dummy = crear_evaluador()

    ag = AlgoritmoGenetico(
        fabrica_evaluador=crear_evaluador,
        generador_escenarios=eval_dummy.generar_escenarios,
        num_casos=5,
        n_workers=None,          # Usa automáticamente todos los núcleos disponibles - 1
        dim_entrada=12,
        capas_ocultas=(10, 10),
        dim_salida=4,
        activacion=nn.Tanh,
        pasos_episodio=200,
        tamano_poblacion=30,
        tamano_torneo=3,
        num_elites=2,
        prob_mutacion_ruido=0.8,
        tasa_genes_ruido=0.25,
        sigma_ruido=0.05,
        prob_permutacion=0.3,
        num_permutaciones=5,
        escala_inicial=0.05,
        max_generaciones=200,
        max_gen_sin_mejora=10,
        seed=42,
    )
    eval_dummy.close()

    mejor_red_pytorch = ag.ejecutar(verbose=True)
    torch.save(mejor_red_pytorch.state_dict(), "mejor_red_estimadora.pt")
    print("Red óptima guardada:", type(mejor_red_pytorch))