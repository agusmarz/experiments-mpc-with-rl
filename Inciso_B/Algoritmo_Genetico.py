import os

# Una sola hebra de BLAS/OpenMP por proceso. Tiene que ir ANTES de importar numpy,
# torch o casadi: cada librería reserva buffers por hebra al cargarse y, con 16
# hebras, un worker comprometía ~3.3 GB de memoria (contra ~0.4 GB con 1 hebra).
# Con 15 workers eso supera el límite de memoria virtual de Windows y aparece
# "MemoryError: bad allocation". El paralelismo ya lo dan los procesos del pool.
# Los procesos hijos heredan estas variables y además reimportan este módulo.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
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


def _evaluar_fenotipo_worker(tarea: tuple[np.ndarray, int, Sequence[dict]]) -> tuple[float, tuple[bool, ...]]:
    """Devuelve (score, fallos_inicio): fallos_inicio[j] es True si el MPC no pudo
    resolver el paso 0 del escenario j (ver AlgoritmoGenetico._verificar_fallos_inicio)."""
    fenotipo, pasos_episodio, escenarios = tarea
    _WORKER_RED.cargar_desde_fenotipo(fenotipo)
    score = float(_WORKER_EVALUADOR.evaluar(_WORKER_RED, pasos_episodio, escenarios))
    fallos = tuple(bool(f) for f in getattr(_WORKER_EVALUADOR, "fallos_inicio", ()))
    return (-np.inf if np.isnan(score) else score), fallos


def _guardar_atomico(objeto, ruta: str) -> None:
    """torch.save a un temporal y después renombra: si el proceso muere a mitad de
    la escritura, el archivo anterior queda intacto."""
    tmp = ruta + ".tmp"
    torch.save(objeto, tmp)
    os.replace(tmp, ruta)


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
        # Validación: escenarios fijos (los mismos durante todo el entrenamiento)
        escenarios_validacion: Sequence[dict] | None = None,
        cada_validacion: int = 10,
        top_k_validacion: int = 5,
        # Guardado durante el entrenamiento
        ruta_checkpoint: str | None = None,
        ruta_mejor_red: str | None = None,
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

        self.escenarios_validacion = list(escenarios_validacion) if escenarios_validacion else []
        self.cada_validacion = max(1, int(cada_validacion))
        self.top_k_validacion = max(1, min(int(top_k_validacion), self.tamano_poblacion))
        self.mejor_fenotipo_validacion: np.ndarray | None = None
        self.mejor_score_validacion: float = -np.inf
        self.gen_mejor_validacion: int = 0
        self.historial_validacion: list[tuple[int, float]] = []  # (generación, mejor del top-k)

        self.ruta_checkpoint = ruta_checkpoint
        self.ruta_mejor_red = ruta_mejor_red
        self._red_entregable_cambio = False

    def construir_red(self, fenotipo: np.ndarray | None = None) -> RedEstimadora:
        red = RedEstimadora(**self.arq_red)
        if fenotipo is not None:
            red.cargar_desde_fenotipo(fenotipo)
        return red

    def obtener_mejor_red(self) -> RedEstimadora:
        """La mejor red en validación si ya hubo alguna; si no, el último élite ganador."""
        fenotipo = (self.mejor_fenotipo_validacion if self.mejor_fenotipo_validacion is not None
                    else self.mejor_fenotipo)
        if fenotipo is None:
            raise RuntimeError("Aún no se ha evaluado ningún individuo.")
        red_optima = self.construir_red(fenotipo)
        red_optima.eval()
        return red_optima

    # ------------------------------------------------------------------ evaluación

    def _mapear(self, tareas: list) -> list[tuple[float, tuple[bool, ...]]]:
        if self.n_workers > 1 and self._pool is not None:
            return list(self._pool.map(_evaluar_fenotipo_worker, tareas))
        if _WORKER_EVALUADOR is None:
            _inicializar_worker(self.fabrica_evaluador, self.arq_red)
        return [_evaluar_fenotipo_worker(t) for t in tareas]

    def _verificar_fallos_inicio(self, fallos: np.ndarray, contexto: str, verbose: bool) -> None:
        """Alarma. `fallos` es (individuos x escenarios): True si el MPC no resolvió el paso 0.

        El paso 0 se resuelve con los parámetros nominales, antes de que la red actúe,
        así que su resultado depende solo del escenario: o falla para todos los
        individuos o para ninguno. Si falla en todo, o falla para unos individuos y
        para otros no, el problema es del entorno (memoria, solver), no de las redes,
        y seguir entrenando solo produce puntajes basura.
        """
        if fallos.size == 0 or not fallos.any():
            return
        inconsistente = not np.all(fallos == fallos[0])
        if fallos.all() or inconsistente:
            detalle = ("de forma distinta según el individuo (no reproducible)" if inconsistente
                       else "en todos los escenarios para todos los individuos")
            raise RuntimeError(
                f"{contexto}: el MPC no pudo resolver el paso 0 {detalle} "
                f"({int(fallos.sum())}/{fallos.size} evaluaciones). Ese paso no depende de la red: "
                "suele ser falta de memoria del sistema (revisá otros procesos de Python) o un "
                "problema de Ipopt. Se corta el entrenamiento"
                + (f"; el último checkpoint está en '{self.ruta_checkpoint}'." if self.ruta_checkpoint else ".")
            )
        if verbose:
            n = int(fallos[0].sum())
            print(f"  Aviso ({contexto}): {n} escenario(s) infactibles desde el paso 0 para "
                  "todos los individuos; no discriminan entre redes.")

    def _evaluar_poblacion(self, escenarios_generacion: Sequence[dict], verbose: bool = False) -> None:
        tareas = [
            (self.poblacion[i], self.pasos_episodio, escenarios_generacion)
            for i in range(self.tamano_poblacion)
        ]
        resultados = self._mapear(tareas)
        self.scores = np.asarray([s for s, _ in resultados], dtype=np.float64)
        fallos = np.asarray([f for _, f in resultados], dtype=bool)
        self._verificar_fallos_inicio(fallos, f"Generación {self.generacion_actual + 1}", verbose)

    def _validar(self, candidatos: np.ndarray, verbose: bool) -> dict:
        """Evalúa cada candidato en los escenarios fijos de validación. Una tarea por
        (candidato, escenario) para repartir el trabajo entre todos los workers."""
        n_val = len(self.escenarios_validacion)
        tareas = [
            (candidatos[i], self.pasos_episodio, [esc])
            for i in range(len(candidatos))
            for esc in self.escenarios_validacion
        ]
        resultados = self._mapear(tareas)
        scores = np.asarray([s for s, _ in resultados], dtype=np.float64).reshape(len(candidatos), n_val)
        fallos = np.asarray([f[0] for _, f in resultados], dtype=bool).reshape(len(candidatos), n_val)
        self._verificar_fallos_inicio(fallos, f"Validación gen {self.generacion_actual + 1}", verbose)

        medias = scores.mean(axis=1)
        mejor = int(np.argmax(medias))
        mejoro = bool(medias[mejor] > self.mejor_score_validacion)
        if mejoro:
            self.mejor_score_validacion = float(medias[mejor])
            self.mejor_fenotipo_validacion = candidatos[mejor].copy()
            self.gen_mejor_validacion = self.generacion_actual + 1
            self._red_entregable_cambio = True
        self.historial_validacion.append((self.generacion_actual + 1, float(medias[mejor])))
        return {"scores": medias, "mejor": float(medias[mejor]), "mejoro": mejoro}

    def _seleccion_torneo(self) -> np.ndarray:
        indices = self.rng.choice(self.tamano_poblacion, size=self.tamano_torneo, replace=False)
        ganador = indices[np.argmax(self.scores[indices])]
        return self.poblacion[ganador].copy()

    
    def _cruzar(self, padre1: np.ndarray, padre2: np.ndarray) -> np.ndarray:
        # Con probabilidad 0.5 clona a padre1 sin cruzar para explotar soluciones locales
        if np.random.rand() >= 0.5:
            return padre1.copy()

        hijo = padre1.copy()
        capas = [
        self.arq_red["dim_entrada"],
        *self.arq_red["capas_ocultas"],
        self.arq_red["dim_salida"], ]
       # Ej: [n_entradas, h1, h2, ..., n_salidas]

        # 1. Extraer vistas matriciales dinámicas según la arquitectura
        Ws_h, bs_h = [], []
        Ws_p2, bs_p2 = [], []
        idx = 0
        for n_in, n_out in zip(capas[:-1], capas[1:]):
            tam_w = n_out * n_in
            Ws_h.append(hijo[idx : idx + tam_w].reshape(n_out, n_in))
            Ws_p2.append(padre2[idx : idx + tam_w].reshape(n_out, n_in))
            idx += tam_w

            bs_h.append(hijo[idx : idx + n_out])
            bs_p2.append(padre2[idx : idx + n_out])
            idx += n_out

        # 2. Cruzar neurona por neurona capa a capa
        num_transiciones = len(Ws_h)
        for l in range(num_transiciones):
            n_neuronas = capas[l + 1]
            mask = np.random.rand(n_neuronas) < 0.5

            # Pesos entrantes y bias de cada neurona seleccionada en la capa l+1
            Ws_h[l][mask, :] = Ws_p2[l][mask, :]
            bs_h[l][mask] = bs_p2[l][mask]

            # Si es capa oculta, también hereda sus pesos salientes hacia la capa siguiente
            if l + 1 < num_transiciones:
                Ws_h[l + 1][:, mask] = Ws_p2[l + 1][:, mask]

        return hijo

    def _mutar(self, individuo: np.ndarray) -> np.ndarray:
        mutado = individuo.copy()
        if np.random.rand() < self.prob_mutacion_ruido:
            gen = getattr(self, "generacion", 0)
            max_gens = getattr(self, "n_generaciones", getattr(self, "max_generaciones", 100))
            progreso = min(1.0, gen / max(1, max_gens - 1))
            sigma = 0.10 * (1.0 - progreso) + 0.01 * progreso

            mascara = np.random.rand(len(mutado)) < 0.25
            mutado[mascara] += np.random.normal(0.0, sigma, size=np.sum(mascara))
        return mutado
    
    def paso_generacion(self, verbose: bool = False) -> dict:
        escenarios_generacion = self.generador_escenarios(self.num_casos, self.rng)
        self._evaluar_poblacion(escenarios_generacion, verbose)

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
            if self.mejor_fenotipo_validacion is None:
                self._red_entregable_cambio = True
        else:
            self.gen_sin_mejora += 1

        self.mejor_score = mejor_gen
        self.historial_mejor_score.append(mejor_gen)
        self.historial_promedio_score.append(promedio_gen)

        # Validación cada `cada_validacion` generaciones y siempre en la última,
        # sobre los top-k de esta generación (ya ordenados, antes del reemplazo)
        validacion = None
        numero_gen = self.generacion_actual + 1
        es_ultima = (numero_gen >= self.max_generaciones
                     or self.gen_sin_mejora >= self.max_gen_sin_mejora)
        if self.escenarios_validacion and (numero_gen % self.cada_validacion == 0 or es_ultima):
            validacion = self._validar(self.poblacion[: self.top_k_validacion], verbose)

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
            "validacion": validacion,
        }

    # ------------------------------------------------------------------ checkpoint

    def guardar_checkpoint(self, ruta: str) -> None:
        """Todo lo necesario para reanudar el AG en la generación siguiente. Solo tensores
        y tipos básicos, para que cargue con torch.load(weights_only=True)."""
        def a_tensor(arr):
            return None if arr is None else torch.from_numpy(np.array(arr, dtype=np.float64))

        _guardar_atomico({
            "generacion": self.generacion_actual,
            "num_pesos": self.num_pesos,
            "poblacion": a_tensor(self.poblacion),
            "scores": a_tensor(self.scores),
            "mejor_fenotipo": a_tensor(self.mejor_fenotipo),
            "mejor_score": self.mejor_score,
            "gen_sin_mejora": self.gen_sin_mejora,
            "mejor_fenotipo_validacion": a_tensor(self.mejor_fenotipo_validacion),
            "mejor_score_validacion": self.mejor_score_validacion,
            "gen_mejor_validacion": self.gen_mejor_validacion,
            "historial_mejor_score": list(self.historial_mejor_score),
            "historial_promedio_score": list(self.historial_promedio_score),
            "historial_validacion": [list(t) for t in self.historial_validacion],
            "rng": self.rng.bit_generator.state,
        }, ruta)

    def cargar_checkpoint(self, ruta: str) -> None:
        """Restaura el estado guardado por `guardar_checkpoint` (misma arquitectura de red)."""
        c = torch.load(ruta, map_location="cpu", weights_only=True)
        if c["num_pesos"] != self.num_pesos:
            raise ValueError(f"El checkpoint es de una red con {c['num_pesos']} pesos y esta tiene {self.num_pesos}.")

        def a_numpy(t):
            return None if t is None else t.numpy().copy()

        self.generacion_actual = int(c["generacion"])
        self.poblacion = a_numpy(c["poblacion"])
        self.tamano_poblacion = self.poblacion.shape[0]
        self.scores = a_numpy(c["scores"])
        self.mejor_fenotipo = a_numpy(c["mejor_fenotipo"])
        self.mejor_score = float(c["mejor_score"])
        self.gen_sin_mejora = int(c["gen_sin_mejora"])
        self.mejor_fenotipo_validacion = a_numpy(c["mejor_fenotipo_validacion"])
        self.mejor_score_validacion = float(c["mejor_score_validacion"])
        self.gen_mejor_validacion = int(c["gen_mejor_validacion"])
        self.historial_mejor_score = list(c["historial_mejor_score"])
        self.historial_promedio_score = list(c["historial_promedio_score"])
        self.historial_validacion = [(int(g), float(s)) for g, s in c["historial_validacion"]]
        self.rng.bit_generator.state = c["rng"]

    def _guardar_progreso(self) -> bool:
        """Checkpoint en cada generación; la red entregable solo cuando cambia.
        Devuelve True si se reescribió la red."""
        if self.ruta_checkpoint:
            self.guardar_checkpoint(self.ruta_checkpoint)
        if self.ruta_mejor_red and self._red_entregable_cambio:
            _guardar_atomico(self.obtener_mejor_red().state_dict(), self.ruta_mejor_red)
            self._red_entregable_cambio = False
            return True
        return False

    # ------------------------------------------------------------------ bucle

    def ejecutar(self, verbose: bool = True) -> RedEstimadora:
        if verbose:
            print(f"Iniciando AG con {self.n_workers} procesos paralelos en CPU...")
            if self.escenarios_validacion:
                print(f"Validación: {len(self.escenarios_validacion)} escenarios fijos, top-{self.top_k_validacion} "
                      f"cada {self.cada_validacion} generaciones.")
            if self.generacion_actual > 0:
                print(f"Reanudando desde la generación {self.generacion_actual}.")

        try:
            if self.n_workers > 1:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.n_workers,
                    initializer=_inicializar_worker,
                    initargs=(self.fabrica_evaluador, self.arq_red),
                )

            while self.generacion_actual < self.max_generaciones:
                info = self.paso_generacion(verbose)
                guardo_red = self._guardar_progreso()
                if verbose:
                    print(
                        f"Gen {info['generacion']:03d} | "
                        f"Mejor Gen: {info['mejor_score_generacion']:.2f} | "
                        f"Prom: {info['promedio_score']:.2f} | "
                        f"Sin mejora: {info['gen_sin_mejora']}/{self.max_gen_sin_mejora}"
                    )
                    val = info["validacion"]
                    if val is not None:
                        print(
                            f"  Validación | top-{len(val['scores'])}: "
                            + " ".join(f"{s:.3f}" for s in val["scores"])
                            + f" | mejor histórico: {self.mejor_score_validacion:.3f} "
                            f"(gen {self.gen_mejor_validacion:03d})"
                            + ("  <- nuevo mejor" if val["mejoro"] else "")
                        )
                    if guardo_red:
                        print(f"  Red guardada en {self.ruta_mejor_red}")

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
    import functools
    import numpy as np
    from evaluador_mpc import EvaluadorSimulacionMPC

    # La fábrica tiene que poder viajar por pickle a los procesos hijos. En Windows
    # (spawn) los hijos importan este archivo como `__mp_main__` sin ejecutar este
    # bloque, así que una función definida acá no existe para ellos. Un partial de
    # la clase se serializa como referencia a `evaluador_mpc.EvaluadorSimulacionMPC`,
    # que sí pueden importar.
    # Se pueden pisar con variables de entorno (útil en Kaggle, sin editar el archivo):
    #   AG_CONTROLADOR = mpc | lqr | lqr_adaptativo
    #   AG_WORKERS     = cantidad de procesos (por defecto, núcleos - 1)
    #   AG_REANUDAR    = 1 para seguir desde RUTA_CHECKPOINT
    CONTROLADOR = os.environ.get("AG_CONTROLADOR", "lqr")  # "lqr" es ~20x más rápido que "mpc"
    N_WORKERS = int(os.environ["AG_WORKERS"]) if os.environ.get("AG_WORKERS") else None
    print(f"Controlador de entrenamiento: {CONTROLADOR}")

    crear_evaluador = functools.partial(
        EvaluadorSimulacionMPC,
        controlador=CONTROLADOR,
        params_iniciales_mpc=(3.0, 1.0, 0.50, 0.01),
        rangos_params_min=(1.5, 0.4, 0.30, 0.005),
        rangos_params_max=(5.5, 2.0, 0.90, 0.030),
        horizonte_mpc=12,  # Horizonte más corto en entrenamiento = solver 3x-4x más rápido
        escala_delta=(0.04, 0.015, 0.015, 0.0005),
    )

    RUTA_CHECKPOINT = "checkpoint_ag.pt"        # estado completo del AG, se pisa cada generación
    RUTA_MEJOR_RED = "mejor_red_estimadora.pt"  # la red que carga el notebook
    REANUDAR = os.environ.get("AG_REANUDAR", "0") == "1"  # True: sigue desde RUTA_CHECKPOINT (si existe)

    # Instancia ligera solo para generar escenarios en el proceso principal
    eval_dummy = crear_evaluador()

    # Validación: siempre los mismos 10 escenarios (semilla propia, independiente de la
    # del AG), así los puntajes de distintas generaciones se pueden comparar entre sí.
    escenarios_validacion = eval_dummy.generar_escenarios(10, np.random.default_rng(2024))

    ag = AlgoritmoGenetico(
        fabrica_evaluador=crear_evaluador,
        generador_escenarios=eval_dummy.generar_escenarios,
        num_casos=5,
        n_workers=N_WORKERS,     # None: todos los núcleos disponibles - 1
        dim_entrada=12,
        capas_ocultas=(32, 32),
        dim_salida=4,
        activacion=nn.Tanh,
        pasos_episodio=300,      # 3 tramos de 100 pasos: centro, izquierda, derecha
        tamano_poblacion=30,
        tamano_torneo=3,
        num_elites=2,
        prob_mutacion_ruido=0.5,
        tasa_genes_ruido=0.25,
        sigma_ruido=0.5,
        prob_permutacion=0.3,
        num_permutaciones=5,
        escala_inicial=0.05,
        max_generaciones=200,
        max_gen_sin_mejora=10,
        #seed=42,
        escenarios_validacion=escenarios_validacion,
        cada_validacion=5,
        top_k_validacion=5,
        ruta_checkpoint=RUTA_CHECKPOINT,
        ruta_mejor_red=RUTA_MEJOR_RED,
    )
    eval_dummy.close()

    if REANUDAR and os.path.exists(RUTA_CHECKPOINT):
        ag.cargar_checkpoint(RUTA_CHECKPOINT)

    mejor_red_pytorch = ag.ejecutar(verbose=True)
    _guardar_atomico(mejor_red_pytorch.state_dict(), RUTA_MEJOR_RED)
    print(f"Red óptima guardada en {RUTA_MEJOR_RED} "
          f"(validación: {ag.mejor_score_validacion:.3f}, gen {ag.gen_mejor_validacion})")