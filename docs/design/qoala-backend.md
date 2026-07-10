# Diseño: backend Qoala para NetQMPI

> Estado: **implementado y validado** (v1, alcance mínimo). Ver §10.
> Autor: exploración asistida. Fecha: 2026-07-10.
> Backend de referencia estudiado: [qoala-sim](https://github.com/QuTech-Delft/qoala-sim),
> paper arXiv:2502.17296 *"Qoala: an Application Execution Environment for
> Quantum Internet Nodes"*.

## 0. TL;DR

- Qoala es un **simulador NetSquid** (no hay ejecución en hardware real); el backend
  se documenta explícitamente como *simulation-only*.
- El backend encaja en el contrato de NetQMPI **sin tocar el SDK**
  (`Environment`, `Circuit`, `OperationContainer`, `QMPICommunicator` abstracto).
- Se implementan tres piezas nuevas siguiendo el patrón de los backends existentes:
  `QoalaExecutorAdapter`, `QoalaCircuitAdapter`, `QoalaCommunicator`, más el flag
  `--qoala` en la CLI.
- **Alcance de la primera versión (acordado):** puertas locales (`h`, `x`, `rx`, …),
  `measure`, `qsend`/`qrecv` (teleportación). `qscatter`/`qgather`/`expose`/`unexpose`
  lanzan `NotImplementedError`, igual que el backend CUNQA.
- **Bloqueo de entorno (acordado):** Qoala requiere Python 3.10–3.12 y `netqasm >= 2.0`;
  el entorno `squidasm` actual usa Python 3.8 y `netqasm 1.0`. El backend Qoala vive en un
  **entorno conda separado**. `--netqasm` y `--qoala` no coexisten en el mismo intérprete.

---

## 1. Recordatorio del contrato NetQMPI

Un backend de NetQMPI se compone de tres adaptadores inyectados por el runtime
(patrón Adapter + inyección de dependencias). Referencias al código real:

| Pieza | Clase abstracta | Responsabilidad |
|---|---|---|
| Executor | [`Executor`](../../netqmpi/runtime/executor.py) | `create_circuit()`, `build_apps()`, `run()` |
| Circuit  | [`Circuit`](../../netqmpi/sdk/circuit.py) | `_translate_*()` para cada `Operation` |
| Communicator | [`QMPICommunicator`](../../netqmpi/sdk/communicator.py) | `__init__(rank, size)`, `__enter__`/`__exit__` |

### 1.1. Modelo de ejecución (patrón del backend NetQASM)

El punto sutil del runtime es **cuándo** se dispara la simulación conjunta de los N ranks.
En [`NetQASMCommunicator.__exit__`](../../netqmpi/runtime/adapters/netqasm/netqasm_communicator.py):

1. `main(env)` se ejecuta una vez por rank, en el **mismo proceso** (secuencialmente).
2. Cada rank construye su `Circuit` dentro de `with comm:` (solo se *graban* operaciones).
3. Al salir del `with`, cada rank añade su programa a una lista de clase.
4. **Cuando la lista alcanza `size == N`**, se construye una única `ApplicationInstance`
   y se llama a `simulate_application` (NetSquid ejecuta los N programas a la vez con
   entrelazamiento vía EPR sockets).
5. Los resultados quedan en `comm.results` (histograma `{bitstring: count}`).

Es decir: **un solo proceso de simulación, N contextos lógicos** — no N procesos del SO.
Este modelo es el que Qoala generaliza, así que lo reproducimos casi idéntico.

### 1.2. Aplicación de referencia (contrato exacto a cumplir)

[`examples/netqmpi/send_recv.py`](../../examples/netqmpi/send_recv.py) — "superposición
distribuida":

```python
def main(env):
    comm = env.comm
    rank = comm.rank
    next_rank = comm.get_next_rank(rank)
    prev_rank = comm.get_prev_rank(rank)
    with comm:
        if rank == 0:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.h(0)
            comm.qsend(circuit, [0], next_rank)
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            comm.qrecv(circuit, [0], prev_rank)
            circuit.measure(0, 0)
    results = comm.results
```

El backend Qoala **debe ejecutar este archivo sin cambiar una sola línea**.

---

## 2. Arquitectura de Qoala relevante para el mapeo

### 2.1. Formato de programa (tres capas)

Un programa Qoala (`.iqoala`) tiene exactamente las tres capas del paper:

1. **Host code** — bloques básicos `^bN {type = CL|CC|QL|QC}`:
   - `CL` = clásico local (`assign_cval`, control de flujo).
   - `CC` = clásico con comunicación (`send_cmsg`, `recv_cmsg`).
   - `QL` = cuántico local (`run_subroutine` → invoca una *local routine*).
   - `QC` = cuántico con comunicación (`run_request` → genera EPR).
2. **Local routines** — `SUBROUTINE … NETQASM_START/END`: ensamblador NetQASM 2.0
   (`init`, `rot_x`, `cnot`, `h`, `meas`, `store … @output`).
3. **Request routines** — `REQUEST …`: generación de EPR con `role: create|receive`,
   `remote_id`, `num_pairs`, `virt_ids`, `typ: create_keep`.

Metadatos por programa (`META_START … META_END`): `name`, `parameters`, y —clave para el
mapeo a `QMPICommunicator`— `csockets: <id> -> <nodo_remoto>` y
`epr_sockets: <id> -> <nodo_remoto>`.

### 2.2. API programática (existe y es limpia)

No hace falta trabajar solo con archivos: Qoala expone construcción por objetos.

- `QoalaProgram(meta, blocks, local_routines, request_routines)` es directamente
  instanciable (`qoala/lang/program.py`). También hay `QoalaParser(text).parse()` para
  construir un `QoalaProgram` desde texto `.iqoala`.
- **Runner multi-nodo listo para usar:** `qoala/util/runner.py::run_n_node_app_separate_inputs(...)`:
  ```python
  network = build_network_from_config(network_cfg)      # N ProcNodes
  for name in names:                                     # 1 batch por nodo
      batch = procnode.submit_batch(create_batch(program, unit_module, inputs, num_iterations))
  for name in names:                                     # emparejar PIDs remotos (entrelazamiento)
      procnode.initialize_processes(remote_pids)
  network.start(); ns.sim_run()                          # una sola simulación NetSquid
  results[name] = procnode.scheduler.get_batch_results()[0]
  ```
- Configuración de red por objetos (sin YAML/JSON):
  `ProcNodeNetworkConfig.from_nodes_perfect_links(nodes, link_duration)` genera links EPR
  *todos-con-todos* (`itertools.combinations`), y `cconns` (canales clásicos) se añaden a mano.

### 2.3. El ejemplo `teleport` de Qoala == `qsend`/`qrecv` de NetQMPI

`examples/teleport/teleport_alice.iqoala` y `teleport_bob.iqoala` son, literalmente, el
protocolo que implementan `qsend`/`qrecv`:

- **Alice (emisor / `qsend`)**: `assign_cval` (csocket) → `run_subroutine: prepare_qubit`
  (`QL`) → `run_request: epr_gen` (`QC`, role=create) → `run_subroutine: bsm`
  (`QL`: `cnot; h; meas; meas` → `m1, m2`) → `send_cmsg(m1); send_cmsg(m2)` (`CC`).
- **Bob (receptor / `qrecv`)**: `run_request: epr_gen` (`QC`, role=receive) →
  `recv_cmsg → m1`; `recv_cmsg → m2` (`CC`) → `run_subroutine: bsm_corrections`
  (`QL`: `z/x` condicionadas a `m1/m2`) → `measure`.

El adaptador de circuito de NetQMPI genera este mismo esqueleto a partir de la lista de
operaciones grabada.

---

## 3. Mapeo operación → Qoala

`QoalaCircuitAdapter` funciona como un **mini-compilador** de la lista plana de
`Operation` (grabada por el SDK) al modelo bloques + local routines + request routines de
Qoala. Regla general:

- Acumula puertas locales consecutivas en una **local routine** (subrutina NetQASM),
  emitida desde un bloque `{QL}`.
- En cada frontera de comunicación (`qsend`/`qrecv`) **cierra el bloque local** y emite
  los bloques `{QC}` (request EPR) + `{QL}` (BSM/correcciones) + `{CC}` (mensajes clásicos).

### 3.1. Tabla de traducción

| Operación NetQMPI | Traducción a Qoala | Estado v1 |
|---|---|---|
| `h(q)` | `h Q{q}` en local routine (`{QL}`) | ✅ |
| `x/y/z(q)` | `x/y/z Q{q}` | ✅ |
| `s/t(q)` | `rot_z` equivalente (`s = rot_z Q 8 4`, `t = rot_z Q 4 4`) | ✅ |
| `rx/ry/rz(θ, q)` | `rot_x/rot_y/rot_z Q n d` con `θ = n·π/2^d` | ✅ |
| `cx/cz(c,t)` | `cnot/cphase Q{c} Q{t}` | ✅ |
| `measure(q, c)` | `meas Q{q} M{c}` + `store M{c} @output[c]` + `return_result` | ✅ |
| `qsend([q], dst)` | `{QC}` `REQUEST`(role=create, remote=dst) + `{QL}` BSM(`cnot;h;meas;meas`) + `{CC}` `send_cmsg(m1); send_cmsg(m2)` | ✅ |
| `qrecv([q], src)` | `{QC}` `REQUEST`(role=receive, remote=src) + `{CC}` `recv_cmsg→m1,m2` + `{QL}` correcciones(`z/x`) | ✅ |
| `qscatter/qgather` | descomposición en `qsend`/`qrecv` (como el adapter NetQASM) | ⛔ `NotImplementedError` en v1 |
| `expose/unexpose` | telegate vía GHZ compartido | ⛔ `NotImplementedError` en v1 |
| `reset/barrier` | `init Q{q}` / no-op | ⛔ `NotImplementedError` en v1 (como CUNQA) |

### 3.2. Codificación de rotaciones (NetQASM 2.0)

En ensamblador NetQASM `rot_x Q n d` aplica una rotación de ángulo `n·π/2^d`. El adapter
convierte `θ` a `(n, d)` con un `d` fijo (p. ej. `d=4`, resolución `π/16`) y
`n = round(θ / (π/2^d)) mod 2^(d+1)`. Se documenta la pérdida de precisión.
> Nota: el `NetASMCircuitAdapter` actual usa `rot_X(n=round(θ), d=16)` del **SDK** de
> netqasm 1.0, que **no** es reutilizable aquí (ver §5).

### 3.3. Estrategia de construcción: emitir texto `.iqoala` y parsear

Aunque `QoalaProgram` es construible por objetos, las *local routines* envuelven un objeto
`netqasm.lang.subroutine.Subroutine` cuya construcción manual es verbosa y frágil. La vía
recomendada es **generar el texto `.iqoala` por nodo desde Python y parsearlo con
`QoalaParser`** (análogo a emitir OpenQASM). Ventajas: reutiliza el parser robusto de Qoala,
es legible y depurable (se puede volcar el `.iqoala` generado a `log/`).

---

## 4. Diseño de las tres piezas

Estructura de carpetas (idéntica al patrón de `netqasm/`, `cunqa/`, `aer/`):

```
netqmpi/runtime/adapters/qoala/
├── __init__.py              # exporta las 4 clases públicas
├── qoala_executor.py        # QoalaExecutorAdapter + QoalaRunConfig
├── qoala_communicator.py    # QoalaCommunicator
└── qoala_circuit.py         # QoalaCircuitAdapter
```

### 4.1. `QoalaExecutorAdapter`

Análogo a `NetQASMExecutorAdapter`: **un único proceso de simulación con N contextos**
(no N procesos). Responsabilidades:

- `__init__(size, config: QoalaRunConfig)`. `QoalaRunConfig(RunConfig)` añade campos de
  simulador: `num_qubits_per_node`, `link_duration`, `qnos_instr_time`, `seed`,
  `scheduler_type` (default `NO_SCHED`/lineal), `netschedule` opcional.
- `create_circuit(num_qubits, num_clbits, comm)` → `QoalaCircuitAdapter`.
- `build_apps(file, size)`: `load_main(file)`; por cada rank construye
  `QoalaCommunicator(rank, size, config, self)` + `Environment` y envuelve `main(env=env)`.
  (El descubrimiento de "recursos" aquí es trivial: `size` viene de `-n N`.)
- `run(apps)`: `for app in apps: app()` — ejecuta cada rank; la simulación real la dispara
  `QoalaCommunicator.__exit__` del último rank.
- Helper interno `run_simulation(programs_by_rank)`: construye N `ProcNodeConfig`
  (IDs `0..N-1`, `TopologyConfig.perfect_config_uniform_default_params(num_qubits)`),
  `ProcNodeNetworkConfig.from_nodes_perfect_links(...)`, añade `cconns` todos-con-todos, y
  reproduce `run_n_node_app_separate_inputs` (submit_batch → initialize_processes → sim_run).

### 4.2. `QoalaCommunicator`

- `__init__(rank, size, config, executor)`: mapea `rank → node_id/name` (`rank_{i}` como en
  el resto de backends) y precrea el mapa de `epr_sockets`/`csockets` (id → nombre de rank
  remoto) que se volcará al `META` del `.iqoala`.
- `__enter__`: devuelve `self`.
- `__exit__`: patrón NetQASM — cada rank traduce su circuito a un `QoalaProgram` y lo añade
  a una lista de clase; **cuando hay N**, llama a `executor.run_simulation(...)`, ejecuta la
  simulación única y reparte los `BatchResult` agregados a `comm.results` de cada rank.
- Provee a `QoalaCircuitAdapter` los helpers de nombres/sockets (`get_rank_name`,
  `remote_id`), pero **no** ejecuta puertas (a diferencia del NetQASM eager): en Qoala el
  circuito es diferido, así que el communicator solo orquesta.

### 4.3. `QoalaCircuitAdapter`

- Implementa los `_translate_*`. En vez de ejecutar, **acumula** en estructuras internas:
  `List[str]` de líneas NetQASM del bloque local actual y `List[BasicBlock/str]` del host
  code, cerrando el bloque local en cada `qsend`/`qrecv`.
- `translate(ops)` recorre el contenedor y devuelve el `QoalaProgram` final (texto → parser).
- `measure` registra qué qubit/clbit va a `@output` para el `return_result` final; el
  resultado se agregará a histograma en el communicator.
- Operaciones fuera de alcance (`qscatter`, `qgather`, `expose`, `unexpose`, `reset`,
  `barrier`, `ClassicalControlledGate`) → `NotImplementedError` con mensaje claro
  ("not implemented for the Qoala backend yet").

### 4.4. CLI

En [`cli.py`](../../netqmpi/runtime/cli.py), añadir al grupo mutuamente excluyente:

```python
backend_group.add_argument("--qoala", action="store_true", help="Use Qoala backend (simulation only)")
...
elif args.qoala:
    from netqmpi.runtime.adapters.qoala import QoalaExecutorAdapter, QoalaRunConfig
    config = QoalaRunConfig(shots=(args.shots or 1))
    executor = QoalaExecutorAdapter(args.num_procs, config=config)
```

El import es **perezoso** (dentro del `elif`), coherente con los otros backends, así que
`--netqasm`/`--cunqa`/`--aer` siguen funcionando sin tener Qoala instalado.

---

## 5. ¿Se reutiliza el adaptador NetQASM existente?

**No.** Aunque Qoala y NetQMPI-NetQASM comparten origen (QuTech) y conceptos (EPR sockets,
teleportación), el código no es reutilizable:

- El `NetASMCircuitAdapter` actual usa el **SDK de `netqasm` 1.0** (`qubit.H()`,
  `epr_socket.create_keep()`), ejecución *eager*.
- Qoala usa **`netqasm` 2.0 a nivel de ensamblador** dentro de local routines, ejecución
  *diferida* orquestada por su scheduler.
- Las dos versiones de `netqasm` son **incompatibles en el mismo intérprete**.

Se transfiere el *conocimiento* (protocolo de teleportación, roles create/receive), no las
líneas de código. `QoalaCircuitAdapter` se escribe desde cero.

---

## 6. Impacto en el SDK: ninguno

Se confirma la restricción del encargo: **no se modifica el SDK**.

- El modelo SPMD de NetQMPI (un programa por rank) se cubre con **1 batch de 1 programa por
  nodo** (`num_iterations = shots`). No se necesita exponer la multitarea de Qoala.
- La multitarea / scheduling conjunto de Qoala (`submit_batch`, `SchedulerType`,
  ejecución de varios programas por nodo) queda **encapsulada dentro del backend**. Si en el
  futuro se quisiera exponer (p. ej. varios "programas NetQMPI" compitiendo por un nodo),
  eso sí requeriría ampliar la interfaz abstracta — se deja como trabajo futuro y **fuera**
  de esta entrega para mantener el SDK agnóstico.

---

## 7. Plan de validación (test end-to-end)

- Reutilizar [`examples/netqmpi/send_recv.py`](../../examples/netqmpi/send_recv.py) sin cambios.
- `netqmpi -n 2 examples/netqmpi/send_recv.py --qoala` en el entorno conda `qoala`.
- Verificar que el rank receptor mide `1` de forma determinista si se prepara `|1⟩`, o una
  distribución ~50/50 con `h` (según lo que fije el ejemplo), y que **el mismo `app.py`**
  produce resultados coherentes con `--netqasm` (ejecutado en el entorno `squidasm`).
- Como el backend es *simulation-only*, el test compara distribuciones de medida, no timing.

> **Restricción de entorno:** al no poder coexistir `netqasm` 1.0 y 2.0, la comparación
> `--netqasm` vs `--qoala` se hace en **dos entornos conda distintos**, no en una sola
> invocación. Esto se documenta en el README del backend.

---

## 8. Incertidumbres / bloqueos abiertos

| # | Tema | Detalle | Mitigación |
|---|---|---|---|
| 1 | **Entorno (🔴)** | Qoala exige Python 3.10–3.12 y `netqasm ≥2.0`; instalar requiere cuenta NetSquid (`--extra-index-url=https://pypi.netsquid.org`). Incompatible con el env `squidasm` (py3.8/netqasm1.0). | Env conda separado `qoala`; imports perezosos en la CLI. |
| 2 | `NetworkScheduleConfig` | El ejemplo `teleport` fija un `netschedule` (bins de timing EPR); `pingpong`/`run_two_node_app` no. Falta confirmar si `from_nodes_perfect_links` basta sin netschedule para `qsend`/`qrecv`. | Verificar en implementación; si hace falta, generar un netschedule por defecto derivado del nº de comunicaciones. |
| 3 | Canales clásicos | `from_nodes_perfect_links` crea links EPR pero **no** `cconns`. Hay que añadir `ClassicalConnectionConfig` para cada par que use `send_cmsg`/`recv_cmsg`. | Generar `cconns` todos-con-todos en el executor. |
| 4 | Codificación de rotaciones | `rot_x/y/z` en ensamblador usa `(n, d)` discreto → pérdida de precisión en `rx/ry/rz(θ)`. | Documentar; usar `d` configurable. |
| 5 | Emparejado de PIDs | `initialize_processes(remote_pids)` requiere conocer los PIDs de los batches remotos antes de simular → orden de `submit_batch` importa. | Replicar el patrón exacto de `run_n_node_app_separate_inputs`. |
| 6 | Agregación de resultados | Qoala devuelve `ProgramResult.values` por iteración; NetQMPI espera `{bitstring: count}`. | Agregar las `num_iterations` iteraciones a histograma en el communicator. |
| 7 | Topología estática | Qoala construye la red completa (N nodos, IDs, links) antes de simular, frente al "descubrimiento dinámico" conceptual de NetQMPI. | No es problema real: `size` se conoce con `-n N`; se construye red all-to-all al vuelo. |
| 8 | Solo simulador | Qoala no despliega en hardware; no hay paridad física con NetQASM/CUNQA reales. | Marcado explícito *simulation-only* en docstrings, help de CLI y este doc. |

---

## 9. Próximos pasos (tras validar este diseño)

1. Crear `netqmpi/runtime/adapters/qoala/` con las 4 clases.
2. Añadir `--qoala` a la CLI (import perezoso).
3. Implementar el mapeo mínimo (§3.1, filas ✅) vía generación de texto `.iqoala` + parser.
4. Test e2e con `send_recv.py` en el entorno `qoala`.
5. Documentar en el README del adapter la restricción de entorno y el carácter
   *simulation-only*.

---

## 10. Estado de implementación (v1)

**Implementado y validado el 2026-07-10.** Ficheros:

- `netqmpi/runtime/adapters/qoala/qoala_circuit.py` — `QoalaCircuitAdapter`
  (compilador puro-Python de operaciones → texto `.iqoala`; sin imports de `qoala`).
- `netqmpi/runtime/adapters/qoala/qoala_communicator.py` — `QoalaCommunicator`.
- `netqmpi/runtime/adapters/qoala/qoala_executor.py` — `QoalaExecutorAdapter` +
  `QoalaRunConfig` + driver `run_simulation` (único módulo que importa `qoala`/`netsquid`,
  de forma perezosa).
- `netqmpi/runtime/adapters/qoala/__init__.py`, `README.md`.
- Flag `--qoala` en `netqmpi/runtime/cli.py` (import perezoso).
- `test/test_qoala_backend.py` — test e2e marcado `integration`, se salta si `qoala`
  no está instalado.

**Validación (env conda `qoala`, Python 3.11, netqasm 2.3.0, netsquid 1.1.8, qoala main
`f20ea00`):**

```
netqmpi -n 2 examples/netqmpi/send_recv.py --qoala --shots 20
→ rank_0: teleportation complete
→ rank_1: measure: {'0': 7, '1': 13}          # ~50/50 esperado (H teleportado, medido en Z)
pytest test/test_qoala_backend.py -m integration  → 1 passed
```

El **mismo `app.py` sin cambios** produce resultados coherentes. El SDK **no se modificó**.

### Incertidumbres del §8 ya resueltas

- **#1 Entorno:** resuelto con env conda `qoala` separado. Instalación documentada en el
  README del adapter (netsquid + qoala editable desde clon + netqmpi editable).
- **#2 `NetworkScheduleConfig`:** **no es necesario**. Verificado con los ejemplos GHZ
  (3 nodos) y con `send_recv` (2 nodos): `from_nodes_perfect_links` + `cconns` basta.
- **#3 Canales clásicos:** confirmado — se generan `cconns` todos-con-todos en el executor.
- **#5 Emparejado de sockets/PIDs:** el `EntDist` empareja peticiones EPR por
  **node_id + PID** (`EntDistRequest.is_opposite`), **no** por `epr_socket_id`. Por eso cada
  programa asigna sus ids de socket localmente (`id = rank remoto`) y el emparejado lo hace
  `initialize_processes(remote_pids)`. `remote_id` se pasa como parámetro-plantilla
  (`{peer_<r>_id}`) instanciado vía `ProgramInput`.
- **#6 Resultados:** `run_simulation` agrega las `shots` iteraciones a un histograma
  `{bitstring: count}` en `comm.results`, ordenado por índice de bit clásico ascendente.

### Deviaciones / notas

- `shots` → `num_iterations` del batch Qoala (default de la CLI: 10).
- Qubits físicos por nodo = `max(qubits de circuito) + 1` (un slot scratch para el EPR del
  `qsend`), o `QoalaRunConfig.num_qubits_per_node` si se fija.
- Codificación de rotaciones con `d = 4` (unidad `π/16`); `S/T` mapeadas a `rot_z`.
- Limitación v1: **un circuito por rank** (lanza `NotImplementedError` si hay más).
- **Cross-check con `--netqasm`:** el mismo `app.py` construye correctamente en el env
  `squidasm`, pero la simulación NetSquid de netqasm 1.0 para este ejemplo es extremadamente
  lenta/se cuelga en esta máquina (>7 min sin salida). Es comportamiento **preexistente** del
  backend NetQASM, ajeno a este trabajo (no se tocó ni el SDK ni el path de netqasm).
