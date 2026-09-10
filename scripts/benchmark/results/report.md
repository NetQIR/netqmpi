## Portability matrix

| app | aer | cunqa | qoala | netqasm |
|---|---|---|---|---|
| cascade | OK | OK | OK | error |
| ghz | **WRONG** 0.10–1.00 | OK | n/i | – |
| qft | OK | OK | n/i | – |
| qft_telegate | n/i | OK | n/i | – |

`OK` = echo exact on every configuration tried. `n/i` = the adapter raises `NotImplementedError`. `WRONG` = completed but returned the wrong answer on a backend that has no noise model. A bare `F=` range is decoherence on a backend that models hardware.

## Overhead by backend

| backend | configs | cold total | warm total | warm NetQMPI | warm share | median ops |
|---|---|---|---|---|---|---|
| aer | 45 | 0.326 s | 0.0165 s | 1.20 ms | 8.11% | 47 |
| cunqa | 24 | 4.804 s | 4.6261 s | 0.64 ms | 0.02% | 32 |
| qoala | 3 | 3.137 s | 2.3921 s | 0.16 ms | 0.01% | 9 |

## Fitted cost model  t_netqmpi = alpha + beta * G

| backend | alpha (ms) | beta (us/op) | R^2 | configs |
|---|---|---|---|---|
| aer | 0.619 | 10.32 | 0.974 | 45 |
| cunqa | -0.720 | 40.43 | 0.714 | 24 |
| qoala | 0.018 | 16.12 | 1.000 † | 3 |
| pooled | 0.537 | 12.00 | 0.612 | 72 |

† fitted on fewer than 5 configurations against 2 free parameters: the coefficients are indicative, the R^2 is not evidence.

Splitting local gates from communication primitives, `t = alpha + beta*G_local + gamma*G_comm`:

| backend | alpha (ms) | beta (us/local op) | gamma (us/comm op) | gamma/beta | R^2 |
|---|---|---|---|---|---|
| aer | 0.614 | 6.57 | 14.57 | 2.2x | 0.976 |
| cunqa | -0.642 | 94.58 | -11.64 | -0.1x | 0.750 |
| pooled | 0.519 | 4.85 | 20.07 | 4.1x | 0.616 |
