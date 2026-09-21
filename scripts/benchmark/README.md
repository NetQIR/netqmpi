# Benchmark multi-backend de NetQMPI

> Requiere los **cuatro** entornos: `qiskit-1-3` (Aer), `qoala` (Qoala),
> `squidasm` (NetQASM) y la imagen Docker `netqmpi:cunqa` (CUNQA + Slurm).
> Ninguno puede compartir intérprete con otro: netqasm 1.x y 2.x son
> incompatibles, y CUNQA solo existe dentro de su contenedor.

## Objetivo

Dos preguntas, una por cada mitad de la promesa de NetQMPI.

1. **«Un código, varios backends»: ¿se cumple?** El mismo `app.py` sin tocar
   se ejecuta en los cuatro backends y se comprueba si sigue significando lo
   mismo: si termina, si el adaptador soporta las primitivas que usa, y si el
   resultado es el correcto. La respuesta es la **matriz de portabilidad**.

2. **¿Cuánto cuesta la abstracción?** Un programa distribuido pasa por cinco
   fases y solo dos son NetQMPI. Se mide cada una por separado, en tiempo y
   en memoria, y se ajusta un **modelo de coste** que permite extrapolar:
   dado un programa, ¿cuánto va a costar la capa NetQMPI y a partir de qué
   tamaño deja de importar?

No se busca novedad algorítmica. Se busca cuantificar, con números
reproducibles, algo que hasta ahora solo se afirmaba.

## Las cinco fases de una ejecución

| Fase | Qué es | ¿Es NetQMPI? |
|---|---|---|
| `import` | Importar el adaptador y el simulador que arrastra | no |
| `setup` | Construir el executor y `build_apps`: descubrimiento de recursos, `qraise` de vQPUs, construcción de topología | no |
| `trace` | Ejecutar el `main()` del usuario; el SDK graba operaciones en el `OperationContainer` | **sí — coste del SDK** |
| `translate` | `Circuit.translate`: convertir esas operaciones en instrucciones nativas | **sí — coste del adaptador** |
| `backend` | El backend ejecutando de verdad: enviar, simular, recoger | no |
| `sync` | Esperas en barrera entre ranks y fontanería de resultados | no |

`t_netqmpi = trace + translate` es el precio de la abstracción;
`setup + backend + sync` es el precio de la plataforma que hay debajo.

**Cómo se obtiene la separación.** `translate` se mide envolviendo
`netqmpi.sdk.circuit.Circuit.translate`, que es el embudo por el que pasan
todos los adaptadores, contando solo la llamada más externa para no cobrar
dos veces la recursión sobre el `OperationContainer`. `backend` se mide
envolviendo una llamada por backend (`profiler.BACKEND_PROBES`).

Una pasada de traducción *conjunta* (la de Aer y la de CUNQA, que necesitan
todos los ranks a la vez para emparejar transferencias o expandir un
colectivo) hace parte de su trabajo fuera de `Circuit.translate`, así que se
envuelve también como ámbito externo de traducción, con la misma guarda de
reentrada para no contar dos veces las llamadas por operación que hace dentro.

`trace` se mide **directamente**, no por resta, y esto importa más de lo que
parece. Todo lo que hace un backend —traducir, simular y sincronizar los
ranks— ocurre dentro del `__exit__` del comunicador, así que cronometrar el
`main()` de cada rank y descontarle el tiempo que pasó en ese `__exit__` deja
exactamente el grabado de operaciones. Tomar `trace` como «lo que sobra de
`executor.run()`» metería dentro las esperas en barrera de los backends que
ejecutan un hilo por rank, y esas esperas son latencia de concurrencia, no
coste de la abstracción: **en Aer llegaban a superar al trazado real en un
orden de magnitud** y variaban al azar entre ejecuciones. Lo que queda dentro
del `__exit__` tras quitar traducción y ejecución se reporta aparte como
`sync`.

En un backend que ejecuta sus ranks concurrentemente las cifras por rank se
suman, así que pueden sumar más que el reloj de pared; `total` es siempre el
tiempo transcurrido real.

## Los algoritmos

Cuatro sondas, elegidas para cubrir **patrones de comunicación distintos**,
no para cubrir dominios de aplicación distintos. Lo que cambia el coste de
un programa distribuido es su topología de comunicación y su volumen, no si
calcula una transformada o un estado de prueba.

| App | Patrón | Transferencias | Primitiva |
|---|---|---|---|
| [`cascade`](apps/cascade.py) | cadena punto a punto | `q · (size−1)` | `qsend`/`qrecv` |
| [`ghz`](apps/ghz.py) | estrella uno-a-muchos | `4 · (size−1)` | `qsend`/`qrecv` |
| [`qft`](apps/qft.py) | todos con todos | `2 · q · size · (size−1)` | `qsend`/`qrecv` |
| [`qft_telegate`](apps/qft_telegate.py) | todos con todos | `size − q` ventanas | `expose`/`unexpose` |

`qft` y `qft_telegate` calculan **la misma transformada** de dos maneras:
moviendo el control (*teledata*) o compartiéndolo (*telegate*). Ejecutar las
dos sobre el mismo backend es lo que aísla el coste de una estrategia frente
a la otra.

### Por qué todas son *echo*

Cada sonda aplica una transformación y luego la deshace, de modo que el
resultado sin ruido es **todo ceros en todos los ranks**. Es deliberado:

- Una expectativa de todo ceros **no depende del orden de bits** con que
  cada backend etiqueta su histograma, lo que elimina una clase entera de
  errores silenciosos de comparación.
- Es un criterio que **cada rank puede puntuar por separado**, y eso importa
  porque **CUNQA devuelve un histograma por rank — marginales, no la
  distribución conjunta**. De marginales no se puede reconstruir la
  probabilidad conjunta de éxito, así que la métrica portable es el producto
  de las probabilidades por rank. Donde el backend sí expone la conjunta
  (Aer guarda los bits clásicos de todos los ranks en un registro) se
  reporta también, y ambas coinciden si los errores por rank son
  independientes.
- Un QFT *sin* deshacer no serviría: aplicado a un estado base, su
  histograma en Z es uniforme haga lo que haga con las rotaciones, así que
  un backend que se comiera todas las rotaciones puntuaría 1.0. El eco es
  sensible a **cada** rotación.

Por eso el QFT usa `NQB_INPUT=0` por defecto. El eco es sensible a las
rotaciones sea cual sea la entrada, y con entrada 0 el criterio de éxito es
independiente del orden de bits. Con `NQB_INPUT=x` distinto de cero la
comprobación pasa a ser «cada rank lee su rodaja de x», útil como test de
corrección pero ya dependiente de la convención de cada backend.

## Métricas

**Tiempo.** Las cinco fases, por separado, en segundos de reloj de pared.
La primera repetición de cada proceso paga los imports y lo que cada
librería inicializa perezosamente; se reporta aparte como *cold start* en
lugar de mezclarse en el modelo.

**Memoria.** Dos cifras, porque ninguna es honesta por sí sola:

- `py_peak` — pico de `tracemalloc`, solo asignaciones de Python. Es lo que
  captura de verdad la huella de NetQMPI: los objetos `Operation` que
  construye el trazado. **No ve** un vector de estado en C++.
- `rss_peak` — pico de RSS del proceso entero (`getrusage`), que sí incluye
  la memoria del simulador — salvo en CUNQA, donde los vQPUs son procesos
  aparte y quedan fuera.

`tracemalloc` distorsiona mucho los tiempos, así que **nunca se miden a la
vez**: las repeticiones de tiempo van sin él y se añade una pasada de
memoria aparte.

**Fidelidad.** `P(todo ceros)` por rank, y su producto como métrica global.

## Modelo de coste

El trazado y la traducción deberían ser **afines** en el número de
operaciones grabadas: un precio fijo por entrar en la maquinaria más un
precio por operación. El modelo ajustado es

    t_netqmpi(G) = α + β · G

con `G` las operaciones que grabó el trazado, `α` en milisegundos y `β` en
microsegundos por operación. A la memoria de Python se le ajusta la misma
forma.

Tratar todas las operaciones como igual de caras es el primer modelo obvio y
uno malo: una puerta local se traduce en una instrucción nativa, mientras que
un `qsend` se expande en un protocolo de teleportación entero —entrelazamiento,
medida de Bell y sus correcciones—. Por eso se ajusta también un modelo con
dos regresores,

    t_netqmpi = α + β · G_local + γ · G_comm

donde el cociente `γ/β` es un número directamente útil: cuántas puertas
locales cuesta traducir un solo acto de comunicación.

Los ajustes se hacen sobre **una mediana por configuración**, no sobre cada
repetición: repetir más veces una configuración no debe inclinar la regresión,
y el jitter entre repeticiones no es lo que el modelo intenta explicar.

Como **α y β son constantes** mientras el coste de un simulador crece con la
anchura del registro, el modelo se puede resolver para el punto en que la
abstracción deja de importar: el **cruce** en que NetQMPI cae por debajo del
1% de la ejecución. El coste del backend se ajusta como exponencial en el
número total de qubits, `t_backend = a · exp(b · Q)`, regresando su
logaritmo — el comportamiento de vector de estado que hace que ese cruce
exista.

Todos los ajustes son mínimos cuadrados con `R²` reportado
([`analyze.py`](analyze.py)).

## Cómo ejecutar

```sh
# Rejilla completa en los cuatro backends (NetQASM es con diferencia el más lento)
./scripts/benchmark/run_all.sh

# Solo algunos backends, o una rejilla mínima de humo
BACKENDS="aer cunqa" ./scripts/benchmark/run_all.sh
./scripts/benchmark/run_all.sh --quick

# Una sola configuración
python scripts/benchmark/run_benchmark.py --backend aer --app qft \
    --ranks 4 --qubits 2 --shots 1024 --reps 3 --memory --out results/raw.jsonl

# Ajuste del modelo, tablas y gráficas
python scripts/benchmark/analyze.py
```

Cada configuración corre en **un proceso propio**: el pico de RSS es una
cifra monótona de proceso y el coste de importar un simulador solo se paga
una vez por proceso, así que compartir proceso entre configuraciones
emborronaría ambas cosas.

Ninguna ejecución se pierde. Un backend que no implementa una primitiva
queda registrado con `status="unsupported"` — eso *es* el dato del que sale
la matriz de portabilidad. Dos de los backends ejecutan sus ranks en hilos y
un fallo ahí deja al hilo principal esperando en una barrera para siempre,
por lo que cada ejecución lleva un *watchdog* armado que registra la causa
real antes de matar el proceso.

## Salidas

| Fichero | Qué es |
|---|---|
| `results/raw.jsonl` | Un registro por repetición, con todo lo medido |
| `results/report.md` | Tablas: portabilidad, overhead por backend, modelo |
| `results/model.json` | Coeficientes ajustados, por backend y agregados |
| `results/phase_breakdown.png` | Dónde se va el tiempo, por backend y app |
| `results/overhead_model.png` | Coste de NetQMPI frente a `G`, con el ajuste |
| `results/overhead_share.png` | Cuota de NetQMPI frente al total, con la línea del 1% |
| `results/memory_model.png` | Pico de memoria Python frente a `G` |
| `results/fidelity.png` | Fidelidad de cada app en cada backend |

---

# Resultados

> **Estos números son los de la rama `fix-aer-backend`**, con los defectos
> del adaptador de Aer descritos en §1 corregidos y con `expose`/`unexpose`
> implementados. La medición contra el adaptador tal y como estaba se
> conserva en la rama `benchmark-multibackend`. La medición
> previa, con el adaptador tal y como estaba, se conserva en la rama
> `benchmark-multibackend`; las diferencias se señalan abajo.

Rejilla de esta ejecución: Aer 2–6 ranks × 1–3 qubits/rank (1024 shots),
CUNQA 2–5 ranks × 1–2 qubits/rank (1024 shots), Qoala 2–4 ranks (100 shots),
NetQASM 2–3 ranks (10 shots). 3 repeticiones de tiempo más una pasada de
memoria por configuración; **324 registros** en
[`results/raw.jsonl`](results/raw.jsonl). Tablas completas en
[`results/report.md`](results/report.md), coeficientes en
[`results/model.json`](results/model.json).

## 1. Matriz de portabilidad

| app | aer | cunqa | qoala | netqasm |
|---|---|---|---|---|
| `cascade` | OK | OK | OK | OK |
| `ghz` | OK | OK | n/i | OK |
| `qft` | OK | OK | n/i | n/i |
| `qft_telegate` | OK | OK | n/i | n/i |

`OK` = eco exacto en **todas** las configuraciones probadas. `n/i` = el
adaptador lanza `NotImplementedError`. `WRONG` = terminó pero devolvió la
respuesta equivocada en un backend **sin modelo de ruido**, donde cualquier
cosa por debajo de un eco perfecto es un fallo de traducción, no
decoherencia.

**Los cuatro backends ejecutan ya la misma sonda teledata** (`cascade`), que
es el mínimo que hacía falta para que «un código, varios backends» signifique
algo medible. Aer y CUNQA ejecutan las cuatro, así que son donde se puede
comparar telegate contra teledata sobre el mismo programa; NetQASM ejecuta
las dos teledata; Qoala sigue limitada a una. Backend por backend:

- **CUNQA** — ejecuta las cuatro sondas con F = 1.0000. Es el único que
  implementa `expose`/`unexpose`, y por tanto el único donde se puede
  comparar telegate contra teledata.
- **Aer** — tenía cuatro defectos, **los cuatro corregidos en esta rama**:
  1. *Resultados silenciosamente incorrectos.* El adaptador traducía **rank
     por rank** en orden de rank, y como Aer ejecuta todos los ranks dentro
     de un único `QuantumCircuit`, el orden de emisión *es* el orden de
     ejecución: todas las puertas de rank 0 antes que ninguna de rank 1. Eso
     solo funciona si las dependencias del programa siguen el orden de rank.
     `cascade` (cadena monótona 0→1→2) sobrevivía por casualidad; `ghz`
     (estrella, el control vuelve a rank 0 entre saltos) devolvía la
     respuesta equivocada **sin lanzar ningún error**: F = 1.000 solo en el
     caso trivial 2 ranks × 1 qubit, y 0.12–0.23 en todo lo demás.

     Ahora `translate_group` intercala los ranks como se ejecutarían de
     verdad: cada uno avanza por sus operaciones hasta llegar a una
     transferencia, y cuando las dos mitades están esperando se mueve el
     qubit y ambos continúan. Las tres sondas portables dan **F = 1.0000 en
     todas las configuraciones** (2–6 ranks × 1–3 qubits/rank).
  2. *El receptor no elegía dónde aterrizaba el qubit.* `_translate_qrecv`
     era un **no-op** y `_translate_qsend` intercambiaba al **mismo índice
     local** del rank destino, ignorando el índice que pedía el receptor.
     Emparejar `qsend` con su `qrecv` aporta justo el dato que faltaba, así
     que la transferencia va ahora del hueco del emisor al que pidió el
     receptor. Las apps de este benchmark hacen viajar el control por una
     ranura *scratch* con el mismo índice en ambos lados para sortear el
     fallo antiguo; **ese rodeo ya no hace falta**, pero se conserva para
     que las cifras sigan siendo comparables con la medición previa.
  3. *Las rodajas de los ranks se solapaban cuando no medían lo mismo.* El
     circuito global se dimensionaba con la anchura del **primer** rank que
     llamaba a `create_circuit`, pero el desplazamiento de cada rank se
     calculaba con **su propia** anchura. Mientras todos los ranks pedían el
     mismo número de qubits nadie lo notaba; en cuanto no (un root de
     `qscatter` tiene un qubit por receptor y los receptores uno cada uno),
     las rodajas se pisaban. `3_scatter` devolvía la respuesta equivocada y
     `4_gather` reventaba con `duplicate qubit arguments`.

     El circuito global se construye ahora **al terminar el trazado**, no
     durante él: en ese momento se conocen todas las anchuras y las rodajas
     se colocan grupo a grupo y, dentro de cada grupo, en orden de rank — lo
     que además hace determinista el orden de bits del histograma, que antes
     dependía de qué hilo llegase primero.
  4. *`transfer_mode="teleport"` documentado pero inexistente.* Se ha
     corregido la **documentación**, no añadido el modo: sobre un simulador
     sin ruido un circuito de teleportación devuelve exactamente lo mismo
     que el SWAP, solo que con más puertas y dos ancillas por transferencia.
     Implementarlo únicamente tiene sentido junto con un modelo de ruido de
     Aer, que hoy no existe en `AerSimulatorConfig`.

  **Y además se le han añadido `expose`/`unexpose`**, que antes lanzaban
  `NotImplementedError`. Un backend real comparte el control mediante un
  estado GHZ —el *cat-entangler* de CUNQA— para que cada receptor tenga una
  copia en base computacional que usar como control local y la devuelva
  intacta. Aer ejecuta todos los ranks dentro de un mismo circuito, así que
  la copia sobra: la puerta controlada del receptor se emite directamente
  contra el qubit del root.

  Eso es **exacto, no aproximado**: un telegate lee su control solo en la
  base computacional, que es justamente la razón por la que la copia
  cat-entangled puede sustituir al original; hacerlo al revés es igual de
  fiel. Y es no físico en el mismo sentido que el `qsend` de este adaptador
  —no se consume entrelazamiento ni se envía corrección alguna—, lo que
  mantiene a Aer como referencia de corrección y no como modelo de red.

  En la práctica un qubit de comunicación deja de pertenecer a ninguna
  rodaja: `_global()` lo resuelve al qubit de datos del root mientras la
  ventana esté abierta, y lo rechaza con un error claro si se usa fuera de
  ella. `translate_group` bloquea ahora también en los colectivos y expande
  la ventana cuando todos sus participantes han llegado, igual que hace con
  las transferencias.

  Verificado con el eco telegate de [`qft_telegate`](apps/qft_telegate.py),
  que sí ejercita cada rotación: **F = 1.0000** en todas las configuraciones
  de 2 a 6 ranks, con ventanas anidadas y fan-out a varios receptores.

  Además, un fallo dentro del hilo designado (transferencia sin pareja,
  puerta no soportada) dejaba a los demás ranks esperando en una barrera
  **para siempre**: el proceso se colgaba en vez de decir qué había pasado.
  Ahora se captura, se libera la barrera y el executor lo relanza.

  Como efecto colateral, Aer detecta ahora desemparejamientos que antes
  pasaban desapercibidos. `examples/1_send_recv.py` ejecutado con `-n 3`
  (está documentado para `-n 2`) deja un `qrecv` de rank 2 sin `qsend` que
  lo alimente; antes el no-op lo ignoraba y rank 2 medía su propio `|0⟩`,
  ahora se reporta con los ranks y el tag implicados. Es el mismo contrato
  que ya aplicaba `check_transfers` en CUNQA.

- **Qoala** — solo `cascade`. Faltan `expose`/`unexpose`, `SWAP` y
  controlled-P. Además **`cx` y `cz` son inalcanzables**: el SDK emite
  `ControlledGate(control, Gate('X'))` pero el adaptador compara contra
  `"RX"`/`"RZ"` ([`qoala_circuit.py:313`](../../netqmpi/runtime/adapters/qoala/qoala_circuit.py#L313)),
  así que la única puerta de dos qubits que se puede usar hoy es la que
  nadie escribe a mano.
- **NetQASM/SquidASM** — **no arrancaba en absoluto**; esta rama lo pone en
  marcha. Lo que había detrás del primer error, en orden de aparición:
  1. *`program_inputs` vacío.* SquidASM hace `program_inputs[party]` para
     cada programa y luego escribe el `AppConfig` dentro de ese mapa, así
     que cada parte necesita una entrada **propia y mutable**. Se construía
     `{}`, de modo que `run_app` lanzaba `KeyError: 'rank_0'` antes de
     arrancar ningún programa.
  2. *`translate()` reasignaba el registro entero en cada llamada.* Como
     `translate` recursa sobre el `OperationContainer` a través de sí mismo,
     cada operación anidada recibía un registro nuevo y abandonaba aquel
     contra el que se habían escrito las anteriores.
  3. *Las correcciones se enviaban sin resolver.* `qsend` metía en el socket
     los *futures* de las dos medidas de Bell sin hacer `flush()` antes, así
     que el receptor recibía objetos, no bits.
  4. *Un qubit enviado quedaba muerto en su ranura.* Medir libera el qubit en
     NetQASM, pero el SDK promete que tras un `qsend` la ranura sigue siendo
     un qubit en `|0⟩`; volver a tocarla abortaba la ejecución.
  5. *`qrecv` retenía tres qubits donde bastaba uno*: metía la mitad EPR ya
     corregida en un qubit nuevo mediante tres CNOT y dejaba vivos tanto el
     EPR como el ocupante original de la ranura.
  6. *`shots` se ignoraba.* `num_rounds` estaba fijado a 1, así que toda
     ejecución devolvía **una sola muestra** y un resultado 50/50 salía como
     una certeza.
  7. *Ninguna puerta controlada había funcionado nunca.* El adaptador leía
     `op.name` sobre un `ControlledGate`, que no tiene ese atributo, así que
     `cx` y `cz` lanzaban `AttributeError`. Y `SWAP`, que el SDK registra
     como `Gate` de dos qubits, estaba en la tabla de puertas controladas —
     inalcanzable— y además implementado como un único CNOT.
  8. *Las puertas desconocidas se descartaban en silencio.* Ambas tablas
     hacían `if nombre in tabla: emitir`, sin `else`: el circuito salía sin
     la puerta y el histograma parecía razonable.

  Las ranuras se asignan ahora **de forma perezosa** —solo al usarlas—, lo
  que elimina el qubit de relleno que `qrecv` tenía que liberar y con él una
  carrera que abortaba aproximadamente una ejecución de cada cuatro. Cada
  shot es una simulación completa sobre una red construida de nuevo
  (`num_rounds` no sirve: los sockets se cierran entre rondas), y la
  traducción ocurre **antes** de arrancar el simulador, de modo que una
  puerta no soportada se reporta al instante en lugar de matar un hilo de
  programa y dejar la ejecución esperándolo para siempre.

  Verificado contra un control **nativo** en NetQASM puro, sin NetQMPI
  (teleportación de dos partes por `simulate_application`), y con sondas
  deterministas: `|1⟩` teleportado llega como `1` en todos los shots, y el
  eco en base X de `cascade` da **todo ceros en 2 y 3 ranks**. `ghz` también
  pasa a ejecutarse (F = 1.0). Siguen sin implementarse la fase controlada
  —que NetQASM no tiene como instrucción— y `expose`.

## 2. Cuánto cuesta la abstracción

| backend | configs | total en frío | total en caliente | NetQMPI | cuota |
|---|---|---|---|---|---|
| aer | 60 | 0.311 s | 0.0132 s | 1.19 ms | **12.69%** |
| cunqa | 24 | 4.795 s | 4.6838 s | 3.42 ms | **0.07%** |
| qoala | 3 | 3.137 s | 2.3921 s | 0.16 ms | **0.01%** |
| netqasm | 4 | 29.416 s | 29.4159 s | 0.32 ms | **0.00%** |

En términos absolutos NetQMPI cuesta **entre 0.16 y 3.4 ms** por ejecución.
Lo que cambia entre backends no es ese coste, sino contra qué se compara:
Aer termina en 15 ms, CUNQA y Qoala tardan segundos.

![Cuota de NetQMPI frente al total](results/overhead_share.png)

Los puntos se separan en dos regímenes a ambos lados de la línea del 1%.
Con Aer sobre circuitos diminutos la abstracción es el **2.8–35.8%** del
reloj; en cuanto hay una simulación de verdad detrás cae a **0.006–0.93%**
en CUNQA y **0.006–0.01%** en Qoala. No hay ningún punto intermedio en esta
rejilla: el cruce ya ha ocurrido para cualquier carga no trivial.

El coste **en frío** es otra cosa y conviene no esconderlo: la primera
ejecución de un proceso paga 0.32 s en Aer (dominado por importar Qiskit) y
4.8 s en CUNQA (dominado por el `qraise`/SLURM que levanta los vQPUs). Eso no
es NetQMPI, pero es lo que paga quien ejecuta un programa una sola vez.

> Esos imports de Qiskit se hacían **perezosamente dentro de
> `create_circuit`**, es decir en mitad del `main()` del usuario, así que el
> profiler los leía como ~0.8 s de *trazado*. Se han subido a nivel de módulo
> en `aer_executor.py`: el paquete solo se importa cuando la ejecución ya ha
> elegido Aer, así que no había nada que ganar difiriéndolos, y diferirlos
> falseaba la atribución de fases.

## 3. El modelo de coste

![Coste de NetQMPI frente al tamaño de la carga](results/overhead_model.png)

| backend | α (ms) | β (µs/op) | R² | configs |
|---|---|---|---|---|
| aer | 0.423 | 13.40 | **0.975** | 60 |
| cunqa | −1.589 | 135.45 | 0.767 | 24 |
| qoala | 0.018 | 16.12 | 1.000 † | 3 |

† 3 configuraciones contra 2 parámetros libres: los coeficientes son
indicativos, el R² no es evidencia.

El ajuste de Aer confirma la forma esperada con **R² = 0.975** sobre 60
configuraciones: un coste fijo de **0.42 ms** más **13.4 µs por operación**. El de CUNQA es más ruidoso
(R² = 0.77, α negativo, que es un artefacto del ajuste) por una razón
mecánica: se está midiendo una cantidad de ~3 ms dentro de una ejecución de
~4.7 s, es decir el 0.07%, al borde de la resolución y con el jitter de SLURM
y ZMQ encima; su recta además la arrastra una única configuración de mucha
varianza (`qft_telegate`, 5 ranks × 2 qubits). **No se ajusta una recta
agregada sobre los tres backends**: sus pendientes difieren en un orden de
magnitud, así que un ajuste conjunto no describe a ninguno.

Separando puertas locales de primitivas de comunicación:

| backend | α (ms) | β (µs/op local) | γ (µs/op comm) | γ/β | R² |
|---|---|---|---|---|---|
| aer | 0.389 | 5.73 | 22.86 | **4.0×** | 0.983 |

**Traducir un acto de comunicación cuesta 4 veces lo que una puerta local**
(Aer, R² = 0.983), que es lo que cabe esperar: una puerta local es una
instrucción, mientras que un `qsend` hay que emparejarlo con su `qrecv` y
ordenarlo contra los demás ranks, y un `expose` hay que expandirlo una vez
que los participantes han llegado todos. El ajuste de CUNQA para este
modelo sale degenerado (γ negativo): sus dos regresores están demasiado
correlacionados en una rejilla de un solo qubit por rank.

## 4. Fidelidad

![Fidelidad por app y backend](results/fidelity.png)

Los tres backends que ejecutan devuelven el eco exacto (F = 1.0000) en todo
lo que ejecutan, Aer incluido en las cuatro sondas. Con el adaptador de Aer sin corregir, `ghz` daba
F = 0.12–0.23 (§1). No hay aquí una comparación de *ruido* entre backends: Aer y
CUNQA simulan sin modelo de ruido en esta configuración, y Qoala solo llegó a
ejecutar una de las cuatro sondas. Medir degradación por decoherencia exige
activar los modelos de hardware de Qoala — que es justo lo que hacen los
[experimentos de Qoala](../experiments/README.md) ya existentes.

## Limitaciones

- **Fidelidad marginal, no conjunta.** CUNQA devuelve un histograma por
  rank. De marginales no se reconstruye la probabilidad conjunta de éxito,
  así que la métrica portable es el producto de las probabilidades por rank.
  Coinciden si los errores por rank son independientes; donde el backend
  expone la conjunta (Aer) se reporta también.
- **`rss_peak` no ve los vQPUs de CUNQA**, que son procesos aparte. Para ese
  backend la única cifra de memoria válida es `py_peak` (lado Python).
- **La rejilla de CUNQA se queda en 1 qubit/rank** en la mayoría de sondas:
  la definición de vQPU por defecto es estrecha y dos qubits de datos más la
  ranura scratch más los de comunicación ya la desbordan («Not enough data
  qubits in the QPU»). Ampliarla requiere un fichero de definición de vQPU
  como ajuste `backend`, que es propiedad del despliegue, no de NetQMPI.
- **CUNQA local no pasa de 5 ranks**: a partir de 6 el Slurm del contenedor
  responde `sbatch submission failed` con los recursos de una sola máquina.
  No es un límite de NetQMPI.
- **`translate` no se puede aislar en NetQASM**: su adaptador devuelve
  *callables* desde `translate` y los ejecuta dentro de la simulación, así
  que las dos fases están entrelazadas. El campo `translate_isolated` de cada
  registro lo indica.
- **Los tiempos por fase se suman por rank.** En un backend concurrente
  (Aer) pueden sumar más que el reloj de pared; `total` es siempre el tiempo
  transcurrido real.
- **Una sola máquina, un solo despliegue.** Estos números caracterizan *esta*
  instalación. La forma del modelo (α + β·G) debería trasladarse; los valores
  de α y β no.
