# AutoPlace

AutoPlace is a GPU global placer for standard-cell designs, derived from
[Xplace](https://github.com/cuhk-eda/Xplace) (Liu et al., DAC 2022). It reads LEF + DEF and writes a DEF with every
movable cell placed (wirelength-driven global placement; no legalization or detailed placement).

It keeps Xplace's placement model (electrostatic density, WA wirelength, Nesterov's method) and changes how a run
is carried out. First, the density grid is chosen from the design, with about one cell per free bin, so the
density force can see clumps that a coarse grid would hide. Next, the cells are merged into clusters of about five
cells by matching them along their nets, and the clusters are placed first. The placement then starts from
wirelength alone: the density is switched on only once the wirelength has settled, with a first density weight
set in units of the wire tension, and the weight grows each iteration by how far the cells moved. When the
clusters are spread enough, every cell starts inside its cluster and the run continues on the cells with the same
weight, until the overflow is reached. During the run the step size, the density weight, the smoothing and the
stop are decided on the GPU, so the host never waits on an iteration.

On the ISPD 2015 / 2019 benchmarks it uses about half of Xplace's iterations and a third of its global-placement
time, at an HPWL within 1 % ([Results](#results)). Details of each step are in [How it works](#how-it-works).

## Contents

- [Build](#build)
- [Run](#run)
- [Options](#options)
- [Outputs](#outputs)
- [How it works](#how-it-works)
- [Results](#results)
- [Acknowledgement](#acknowledgement)

## Build

Requirements: Python 3, PyTorch with CUDA, `nvcc`, a C++17 compiler. Tested with Python 3.8, PyTorch 2.0.1 +
CUDA 11.8, g++ 11, on an RTX 3070.

```bash
cpp_to_py/build.sh        # about 1 minute
# options: PYTHON=python NVCC=nvcc CXX=g++ CUDA_ARCH=86 cpp_to_py/build.sh
```

This builds two modules into `cpp_to_py/`: `gp_cuda*.so` (the CUDA kernels) and `def_io*.so` (the DEF reader and
writer). `CUDA_ARCH` defaults to the compute capability of GPU 0. The modules store no absolute path, so the folder
can be moved without rebuilding; a different GPU architecture needs a rebuild.

Without the build AutoPlace still runs: it falls back to the PyTorch operators and the Python DEF reader, which
give the same placement up to floating-point order, several times slower and not reproducible bit for bit. The log
then starts with a warning.

## Run

```bash
python main.py --lef tech.lef cells.lef --def floorplan.def --out_dir out/
python main.py --lef tech.lef cells.lef --def floorplan.def --eval_def placed.def   # only evaluate a placement
```

The DEF gives the netlist, the rows, the die area and the fixed objects (macros, pads, placement blockages). Every
unplaced or placed core cell is moved; fixed components and placed components that are not core cells stay where
they are.

The same input, options and GPU give the same `gp.def`.

## Options

`python main.py -h` lists them all.

| Option | Default | Meaning |
|---|---|---|
| `--lef` | (required) | LEF files: technology, cells, macros |
| `--def` | (required) | DEF with the netlist, rows and fixed objects |
| `--out_dir` | none | output folder; nothing is written without it |
| `--eval_def` | none | evaluate this placed DEF (same netlist as `--def`) instead of placing |
| `--write_def` | true | write `gp.def` |
| `--seed` | 0 | random seed (start, clustering, fillers) |
| `--gpu` | 0 | GPU id; -1 = CPU |
| `--target_density` | 1.0 | density every bin may reach (raised to the utilization if lower) |
| `--num_bin` | 0 | grid of the overflow (stop, schedule, reported overflow): `num_bin x num_bin`; 0 = the largest power of 2 not above the row count, at most 512 |
| `--force_num_bin` | 0 | grid of the density force; 0 = from the cells (about one cell per free bin), never coarser than `--num_bin` |
| `--stop_overflow` | 0.07 | stop when the overflow is below this |
| `--settle_iter` | 30 | iterations run after reaching `--stop_overflow` (the lowest HPWL is kept) |
| `--stall_iter` | 200 | stop when nothing improved for this many iterations |
| `--min_iter` | 100 | no stop before this iteration |
| `--inner_iter` | 10000 | maximum iterations |
| `--ignore_net_degree` | 100 | nets with more pins are not optimized (still counted in the HPWL) |
| `--start_tol` | 1e-3 | the wirelength-only start ends when the WA wirelength falls by less than this share per iteration |
| `--density_weight` | 1e-3 | first density weight, as a share of the full wire tension |
| `--ref_motion` | 5.0 | cell motion per iteration (RMS, in cell pitches) at which the density weight stops growing |
| `--wa_coeff` | 4.0 | WA smoothing, in bin sizes (as Xplace) |
| `--cluster_ratio` | 5 | cells per cluster for the two-level placement; 1 = one level |
| `--handover_overflow` | 0.7 | overflow of the clusters at which the cells take over |
| `--lr` | 0.01 | length of the probe step that sets the first step size |
| `--log_freq` | 100 | log every this many iterations |

## Outputs

In `--out_dir`:

- `gp.def`: the input DEF with every moved cell `PLACED` at its global-placement position (only the `COMPONENTS`
  statements of the moved cells change). Not written when the placement is not finite.
- `metrics.json`: the design (`num_movable`, `num_nets`, `num_pins`, `num_fillers`), the grids (`num_bin`,
  `force_num_bin`, `coarse_num_bin`), the run (`stop_reason`: converged / stalled / diverged / max_iter, `iters`,
  `start_iters`, `coarse_iters`, `clusters`, `grad_calls`, `backtracks`), the result (`hpwl_sites`, `hpwl_um`,
  `overflow`: exact overflow of the real cell sizes on the `--num_bin` grid) and the times (`parse_time_s`,
  `cluster_time_s`, `gp_time_s`, `per_iter_s`, `write_time_s`, `total_time_s`).
- `placer.log`: the options, the design, the progress and warnings.

## How it works

The objective is the one of ePlace / Xplace: WA wirelength + density weight x electrostatic density energy,
minimized by Nesterov's method with a Barzilai-Borwein step and backtracking, each node's gradient divided by
max(1, pins + coefficient x density weight x area). Fillers take the free area the cells do not need. These parts
follow Xplace; what AutoPlace changes is how a run is driven:

1. **Wirelength-only start.** All cells start in a small cloud at the centre. The density weight is 0 (no density
   map, no field) until the WA wirelength falls by less than `--start_tol` per iteration; the cells first find
   their order from the nets alone, instead of waiting at the centre for the density weight to grow by 10^3 to
   10^5 as in Xplace.
2. **First density weight in tension units.** The first weight is `--density_weight` x (number of pins / L1 norm of
   the density gradient of the cells): the weight at which the density force on the cells equals every pin pulling
   at full tension. It means the same on any cell library or length unit.
3. **Weight growth by cell motion.** Every iteration the weight is multiplied by 1.05^(1 - motion / `--ref_motion`)
   (at least 1.05 x 0.95), motion being the RMS move of the nodes in the last iteration in cell pitches; by
   1.05 x 0.98 when the HPWL fell. The weight grows fast while the cells move little and slows when the optimizer
   falls behind. There is no length constant and no slow window.
4. **Two levels.** The cells are merged into clusters of about `--cluster_ratio` cells (star matching on the GPU: in
   each round half of the clusters are heads and every other cluster joins the head it shares the most net weight
   with per unit of merged area; a net of d pins gives 1 / (d - 1) to each pair; macros stay alone). The clusters
   are placed first, with the steps above, until their overflow is under `--handover_overflow`; then every cell
   starts at a random point inside the square of its cluster, with the density weight carried over in tension
   units, and the run continues on the cells.
5. **Density grid from the design.** The force is computed on its own grid with about one cell per free bin
   (2^round(log2 sqrt(cells / free fraction)) bins per side, never coarser than `--num_bin`); the clusters use the
   same rule with their own count. With many small cells per bin, the smoothed density (each cell widened to
   sqrt(2) bins) cannot see clumps of bin size and the field does not push them apart. The overflow (stop,
   schedule, report) stays on the `--num_bin` grid, so it means the same as with a single grid.
6. **The run controlled on the GPU.** Once the density weight is on, the step length, the backtracking, the density
   weight, the WA smoothing, the preconditioner and the stop rules are decided by one GPU thread, in double
   precision; the kernels read their parameters from device memory and the host only looks every 16 iterations
   whether the run has stopped.
7. **Own CUDA kernels and DEF reader / writer.** The density map and the wirelength are summed in a fixed order or
   in fixed point, so the same input gives the same bits.

Stops: `converged` (overflow under `--stop_overflow`, then `--settle_iter` more iterations, lowest HPWL kept),
`stalled` (no improvement for `--stall_iter` iterations while the density term dominates; lowest overflow kept),
`diverged` (a step that is not finite is not taken; the last finite placement is kept), `max_iter`.

The WA smoothing is Xplace's: `--wa_coeff` x (bin width + bin height) x 10^((overflow - 0.1) x 20 / 9 - 1), in
bins of the force grid.

## Results

Against Xplace (upstream, same LEF / DEF, target density and overflow grid, each placer at its own stop at overflow
0.07, both placements evaluated by `--eval_def`), geometric means:

| Benchmark | Designs | HPWL / Xplace | Iterations / Xplace | GP time / Xplace |
|---|---|---|---|---|
| ISPD 2019 | 9 (test1-4, 6-10; 8 k to 900 k cells) | 1.0014 | 0.535 | 0.33 |
| ISPD 2015 | 18 (fences removed as in Xplace; 30 k to 1.3 M cells) | 1.0069 | 0.519 | 0.29 |

GP time: global placement only (for AutoPlace, clustering and both levels), median of 3 runs on an RTX 3070.

## Acknowledgement

AutoPlace is based on [Xplace](https://github.com/cuhk-eda/Xplace) (Lixin Liu et al., CUHK), which is released under
the BSD 3-Clause License, and on the methods of ePlace / RePlAce and DREAMPlace.
