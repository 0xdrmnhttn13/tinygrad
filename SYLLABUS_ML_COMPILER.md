# ML Compiler Syllabus — learning by reading tinygrad

A step-by-step path from "I can use a tensor library" to "I can build an ML compiler", using **this** repository as the textbook. Every step lists the exact files and line numbers to read, what to look for, and a small runnable example.

> Run examples from the repo root with `PYTHONPATH=. python3 -c "..."` (or save to a file). Use `DEBUG=2/4/5` to see kernels, `VIZ=1` to open the rewrite visualizer, `BEAM=2` to search.

---

## Mental model first (read this once)

The compiler has 4 stages. Keep this picture in your head — every step below maps to one of them.

| Stage | What it does | Main directory |
| --- | --- | --- |
| **Frontend** | Tensor ops build a lazy UOp graph | `tinygrad/tensor.py`, `tinygrad/uop/` |
| **Scheduler** | Cuts the graph into kernels (`CALL`s) | `tinygrad/schedule/` |
| **Codegen / Lowering** | Per-kernel: optimize, devectorize, render to source | `tinygrad/codegen/`, `tinygrad/renderer/` |
| **Runtime** | Compile source → binary, alloc memory, launch | `tinygrad/runtime/`, `tinygrad/engine/realize.py` |

The whole compiler is built on **one data structure**: the `UOp` (a typed DAG node). Every transformation is a `PatternMatcher` rewrite over that DAG — `graph_rewrite(sink, pm, ...)`. Once you understand `UOp` + `PatternMatcher`, every other file is "just more rewrites".

Reference docs already in the repo:
- `docs/developer/developer.md` — 1-page summary of the 4 stages
- `docs/abstractions3.py` — front-to-back tour (54 lines)
- `docs/abstractions4.py` — five abstraction levels of the **same** `sum` kernel (high-level → assembly)

---

## Phase 0 — Prerequisites (≈ 1 week)

Before tinygrad-specific reading, make sure you have:

1. **Python**: dataclasses, `enum`, `functools.cache`, `weakref`. Read `tinygrad/helpers.py` lines 1–200 to see the style used.
2. **A bit of compiler theory**: SSA form, dataflow graph, dead-code elimination, common subexpression elimination, loop tiling. Don't go deep — just know the words.
3. **GPU vocabulary**: thread / warp (lane) / block (workgroup) / grid, global vs local vs register memory, coalesced loads, occupancy. Read `docs/developer/speed.md`.
4. **One linear-algebra kernel by hand**: write a CUDA or Metal `gemm` or `reduce_sum` yourself once. You will see this exact `reduce_sum` recurring through `docs/abstractions4.py`.

Checkpoint: can you explain *why* fusing `relu(x+y)` into one kernel is faster than two? If yes, continue.

---

## Phase 1 — The frontend: Tensors are syntactic sugar (≈ 3–5 days)

**Goal:** prove to yourself that a `Tensor` is just a wrapper that builds a `UOp` graph.

### Read

- `tinygrad/tensor.py:80` — `class Tensor`. Skim until `tensor.py:235` `schedule_linear` and `tensor.py:242` `realize`.
- `tinygrad/uop/__init__.py:13` — `class Ops(FastEnum)`. **Read every member.** This is the entire IR vocabulary. Notice the groups at `__init__.py:111` (`Unary`, `Binary`, `Ternary`, `ALU`, `Movement`, `Buffer`, ...).
- `tinygrad/uop/ops.py:129` — `class UOp`. Read the constructor and the property accessors (`shape`, `dtype`, `src`, `arg`).
- `tinygrad/uop/ops.py:1027` — `class UPat` and `tinygrad/uop/ops.py:1162` — `class PatternMatcher`. This is *the* abstraction the whole compiler uses.
- `tinygrad/uop/ops.py:1447` — `def graph_rewrite`. The rewrite engine.

### Example

Save as `learn_01_frontend.py` and run with `DEBUG=4 PYTHONPATH=. python3 learn_01_frontend.py`:

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([10.0, 20.0, 30.0, 40.0])
c = (a + b).sum()

# Frontend has done nothing concrete yet — it built a UOp graph:
print(c.uop)              # pretty-printed UOp DAG
print("op:", c.uop.op)    # an Ops member
print("shape:", c.uop.shape)
print("src:", c.uop.src)  # children — keep walking these by hand

# Now force execution. DEBUG=4 will dump the generated kernel source.
print("value =", c.item())
```

### Exercises

1. Add a `.relu()` and re-print `c.uop`. Which `Ops` members appeared?
2. Replace `+` with `*`. Notice `Ops.ADD` becomes `Ops.MUL` — that is **the entire** "frontend codegen for multiply".
3. Write a 5-line `PatternMatcher` that turns `x * 1` into `x` and call `graph_rewrite` on a small `UOp`. (Hint: pattern is `UPat(Ops.MUL, src=(UPat(name="x"), UPat.cvar("c", val=1)))`.)

### Checkpoint

You can answer: *"What kind of object does `Tensor.__add__` return, and what does it do under the hood?"*

---

## Phase 2 — The scheduler: how the graph is cut into kernels (≈ 1 week)

**Goal:** understand what makes a kernel boundary, and why fusion is non-trivial.

### Read

- `docs/abstractions3.py` — runs end-to-end and **prints the schedule**. Run it once: `PYTHONPATH=. python3 docs/abstractions3.py`.
- `tinygrad/tensor.py:235` — `Tensor.schedule_linear`. This is the entry point that returns a `LINEAR` UOp whose `.src` is a list of `CALL` UOps (= one per kernel).
- `tinygrad/schedule/rangeify.py` — the modern scheduler. Start at top of file; read `bufferize_to_store` (`rangeify.py:384`), `remove_bufferize` (`rangeify.py:235`), `split_reduceop` (`rangeify.py:92`).
- `tinygrad/schedule/memory.py` — buffer reuse / memory planning.
- `tinygrad/schedule/multi.py` and `tinygrad/schedule/allreduce.py` — multi-GPU sharding (skim only on first pass).

### Example

```python
from tinygrad import Tensor
from tinygrad.engine.realize import run_linear

a = Tensor.randn(64, 64).realize()
b = Tensor.randn(64, 64).realize()
c = (a @ b + a).relu().sum()

linear = Tensor.schedule_linear(c)
print(f"{len(linear.src)} kernels:")
for i, call in enumerate(linear.src):
    print(i, str(call)[:120])   # CALL UOps, one per kernel

run_linear(linear)
print(c.item())
```

### Exercises

1. Add a `.contiguous()` somewhere mid-expression — the kernel count changes. Why?
2. Compare `(a+b).sum()` with `(a+b).sum().sum()`: when does the second sum fuse vs. require a new kernel?
3. Run with `VIZ=1` and explore the rewrite trace in your browser.

### Checkpoint

You can read a printed `LINEAR.src` and explain, for each `CALL`, what compute happens and which buffers it reads/writes.

---

## Phase 3 — The IR core: UOps, UPats and rewrites (≈ 1–2 weeks, the big one)

**Goal:** become fluent in the `UOp`/`PatternMatcher` style. Every later phase is just "more PatternMatchers".

### Read (in this order)

1. `tinygrad/uop/ops.py:129` — re-read `class UOp` properly now. Pay attention to `op`, `dtype`, `src`, `arg`, `shape`. `arg` is op-specific metadata (shape for `RESHAPE`, axis for `REDUCE_AXIS`, etc.).
2. `tinygrad/uop/ops.py:1027` — `UPat`. Patterns can match by op set, by dtype, by `src` structure, by name (to capture).
3. `tinygrad/uop/ops.py:1162` — `PatternMatcher`. A list of `(UPat, fxn)`; `fxn` returns a new `UOp` or `None`.
4. `tinygrad/uop/ops.py:1338` — `RewriteContext`, then `tinygrad/uop/ops.py:1447` — `graph_rewrite`. Two modes: top-down and `bottom_up=True`.
5. `tinygrad/uop/symbolic.py` — the **symbolic algebra** rewrites. ~470 lines of pure rewrite rules: simplify `x*0`, fold constants, `x < x+1 → True`, etc. This is your model for "what a rewrite rule looks like".
6. `tinygrad/uop/spec.py` — the **type system** for UOps. `kernel_spec` and `program_spec` are the two grammars valid UOps must obey at the kernel boundary and right before rendering. Run with `SPEC=1` to enforce them.
7. `tinygrad/uop/decompositions.py` — examples of expressing one op as a tree of others (e.g. `EXP2` via polynomial when device lacks it).

### Example: write your own pass

```python
from tinygrad.uop.ops import UOp, UPat, PatternMatcher, graph_rewrite, Ops
from tinygrad.dtype import dtypes

# Build a tiny UOp expression: (x + 0) * 1
x = UOp(Ops.DEFINE_VAR, dtype=dtypes.float, arg=("x", -1e9, 1e9))
zero = UOp.const(dtypes.float, 0.0)
one  = UOp.const(dtypes.float, 1.0)
expr = (x + zero) * one

print("before:", expr)

# A two-rule mini-pass: x*1 -> x, x+0 -> x
pm = PatternMatcher([
    (UPat(Ops.MUL, src=(UPat(name="a"), UPat.cvar("c"))),
     lambda a, c: a if c.arg == 1.0 else None),
    (UPat(Ops.ADD, src=(UPat(name="a"), UPat.cvar("c"))),
     lambda a, c: a if c.arg == 0.0 else None),
])

simplified = graph_rewrite(expr, pm, name="my_first_pass")
print("after :", simplified)
```

(Note: `tinygrad/uop/symbolic.py` already has these rules — you are reinventing them on purpose.)

### Exercises

1. Read `tinygrad/uop/symbolic.py:159` `lt_folding` and explain in plain English what it proves.
2. Add a rule that turns `x - x` into `0`. Verify it only fires on float dtypes.
3. Run any tensor program with `VIZ=1` and find the symbolic rewrite step in the trace.

### Checkpoint

You can read a `PatternMatcher` rule and predict what graph fragments it will rewrite without running it.

---

## Phase 4 — Codegen: AST → linear list of UOps (≈ 1–2 weeks)

**Goal:** understand the per-kernel pipeline that turns a `SINK` AST into a renderable straight-line sequence.

### Read — this is the spine of the compiler

- `tinygrad/codegen/__init__.py:23` — **`full_rewrite_to_sink`**. Read this function top to bottom. It is a sequence of ~15 named `graph_rewrite` calls. Each one is a phase. Annotate them on paper:

  ```
  early movement ops      → push movement ops down toward loads
  load collapse           → fuse gather-style loads
  split ranges            → cut loops along factors
  initial symbolic        → x+0, x*1, ...
  simplify ranges         → eliminate dead axes
  apply_opts              → BEAM / hand-coded opts choose tile sizes etc.
  postopt symbolic
  expander                → unroll UNROLL ranges
  add local buffers       → introduce shared memory
  remove_reduce
  add gpudims             → bind ranges to threadIdx / blockIdx
  add loads
  devectorize             → break vec4 ops into scalar ops if needed
  lower_index_dtype       → choose int32 vs int64 per index
  decompositions          → ops the device can't do natively
  transcendental          → polynomial expansions for exp/log/sin
  pm_render               → final cleanups before rendering
  add_control_flow        → insert RANGE/END (the loops)
  ```

- `tinygrad/codegen/simplify.py` — range/loop simplification. `simplify.py:21` `simplify_merge_adjacent` and `simplify.py:77` `reduce_unparented` are the showcase rules.
- `tinygrad/codegen/late/expander.py` — turns `UNROLL` ranges into many UOps.
- `tinygrad/codegen/late/devectorizer.py` — vectorization / devectorization decisions; load/store folding.
- `tinygrad/codegen/late/linearizer.py` — flattens the DAG into a list with `RANGE`/`END` boundaries.
- `tinygrad/codegen/__init__.py:126` `do_linearize`, `__init__.py:140` `do_render`, `__init__.py:144` `do_compile`, `__init__.py:158` `do_to_program` — the orchestration.
- `tinygrad/codegen/opt/` — the optimizer:
  - `opt/heuristic.py` — hand-coded opts (upcast, local, group, ...).
  - `opt/search.py` — **BEAM search** over those opts. Run with `BEAM=2`.
  - `opt/tc.py` — tensor-core matching.

### Example: see every phase

```python
import os
os.environ["DEBUG"] = "5"   # dumps UOps after each rewrite
os.environ["VIZ"]   = "1"   # opens the visualizer in your browser
from tinygrad import Tensor
((Tensor.randn(256, 256) @ Tensor.randn(256, 256)).relu().sum()).item()
```

Open the VIZ tab and walk the named rewrites in order. Each box is one of the rewrites in `full_rewrite_to_sink`.

### Exercises

1. Re-run with `BEAM=0` then `BEAM=2`. Compare reported kernel time and the chosen opts.
2. Pick one rewrite, e.g. `simplify_merge_adjacent`. Construct a tiny UOp DAG that triggers it, run only that rewrite, and inspect the result.
3. Trace a single `EXP2` from frontend through `decompositions` and observe what the device-specific renderer ends up emitting.

### Checkpoint

Given any `full_rewrite_to_sink` line in the source, you can sketch what shape change happens to the IR.

---

## Phase 5 — Renderers: UOps → device source (≈ 3–5 days)

**Goal:** see the last mile — turning the linearized UOp list into actual C / Metal / PTX / WGSL.

### Read

- `tinygrad/renderer/__init__.py:134` — `class Renderer`. The base class; lists `code_for_op`, `extra_matcher`, `pre_matcher` knobs.
- `tinygrad/renderer/__init__.py:65` — `class ProgramSpec`. The thing the runtime consumes (name, source, binary, global/local sizes).
- `tinygrad/renderer/cstyle.py` — base C-style renderer + concrete CUDA / OpenCL / Metal / HIP subclasses. **Start here.** The key method is the per-op string template.
- `tinygrad/renderer/llvmir.py` — same idea but emitting LLVM IR instead of C.
- `tinygrad/renderer/ptx.py` — direct NVIDIA PTX assembly emission.
- `tinygrad/renderer/wgsl.py`, `tinygrad/renderer/nir.py`, `tinygrad/renderer/amd/` — more targets to compare.

### Example: see what each renderer emits

```python
import os
os.environ["DEBUG"] = "4"   # 4 = print rendered source
from tinygrad import Tensor, Device
print("Default device:", Device.DEFAULT)
((Tensor.randn(64) + Tensor.randn(64)).sum()).item()
```

Then try `CPU=1`, `CL=1`, `METAL=1`, `LLVM=1` (whichever are available on your machine) — same Python, different rendered text.

### Exercises

1. Write the C string a `Renderer` would emit for `c[i] = a[i] + b[i]` (a 1-D add). Compare to actual `DEBUG=4` output.
2. Find where `Ops.SQRT` is mapped to `"sqrt(...)"` in `cstyle.py`. Add a fake op or change the format string and re-run.

### Checkpoint

You can predict, for a tiny kernel, roughly what source the C renderer will print.

---

## Phase 6 — Engine + runtime: source → running on metal (≈ 1 week)

**Goal:** understand how compiled code actually reaches the GPU.

### Read

- `tinygrad/engine/realize.py:48` — `class Runner` and `realize.py:72` `class CompiledRunner` (kernels), plus `realize.py:150` `exec_view`, `realize.py:157` `exec_copy`, `realize.py:172` `exec_kernel` — the dispatcher functions for each `CALL` kind.
- `tinygrad/engine/realize.py:243` — `def run_linear`. The actual top-level "run a schedule" loop.
- `tinygrad/engine/realize.py:100` — `def get_runner` — caches `CompiledRunner`s per (device, ast).
- `tinygrad/engine/jit.py:247` — `class TinyJit`. How a Python function is captured into a single fast graph.
- `tinygrad/runtime/ops_cpu.py` (small + simple) — read first.
- `tinygrad/runtime/ops_metal.py` or `tinygrad/runtime/ops_cuda.py` (whichever you have). They each implement: `Compiler`, `Allocator`, `Program`. That's the runtime API.
- `tinygrad/device.py` — the `Buffer` and `Device` objects everyone uses.
- `docs/developer/runtime.md` and `docs/developer/hcq.md` — high-level overview of the runtime API and the lower HCQ API used by AMD/NV userspace drivers (`runtime/ops_amd.py`, `runtime/ops_nv.py`).

### Example

```python
import os
os.environ["DEBUG"] = "2"   # 2 = print kernel timings + sizes
os.environ["JIT"]   = "1"
from tinygrad import Tensor, TinyJit

@TinyJit
def step(x):
    return (x @ x.T).relu().sum()

x = Tensor.randn(128, 128).realize()
for _ in range(3):
    print(step(x).item())
```

First call compiles and captures; subsequent calls replay the captured graph — note the timing difference.

### Checkpoint

You can describe the path of one tensor op from `Tensor.__add__` to a memory write on the GPU, naming each function in order.

---

## Phase 7 — Optimization: BEAM, opts, tensor cores (≈ 1–2 weeks)

**Goal:** understand how performance happens.

### Read

- `tinygrad/codegen/opt/__init__.py` — what an `Opt` is (UPCAST, LOCAL, GROUP, UNROLL, ...).
- `tinygrad/codegen/opt/heuristic.py` — hand-coded "sane defaults" for kernels.
- `tinygrad/codegen/opt/search.py` — BEAM search engine over opts. Read `class BeamSearch`.
- `tinygrad/codegen/opt/tc.py` — tensor-core (HMMA / WMMA) matching.
- `tinygrad/codegen/opt/postrange.py:apply_opts` — applies the chosen list of `Opt`s by inserting splits/upcasts in the IR.
- `docs/developer/speed.md` — the philosophy.

### Example

```python
import os; os.environ["DEBUG"] = "2"
from tinygrad import Tensor, Context

a, b = Tensor.randn(1024, 1024).realize(), Tensor.randn(1024, 1024).realize()

with Context(BEAM=0):
    (a @ b).realize()           # baseline

with Context(BEAM=2):
    (a @ b).realize()           # BEAM-searched
```

Compare the timing lines.

### Exercises

1. Print the chosen opts list for a matmul (find where opts are logged in `opt/search.py`).
2. Force a single specific opt via `KOPT=...` (see `docs/env_vars.md`) and observe the kernel difference.

---

## Phase 8 — Putting it all together: `docs/abstractions4.py` (≈ 1 week)

This file shows the **same `sum` kernel at 5 abstraction levels**. By now you should be able to read every line.

- Example 1: `Tensor.sum()` — frontend.
- Example 2: hand-written **HIP** source injected as a custom kernel via `Ops.PROGRAM`.
- Example 3: hand-written **UOp graph** (no source string) — you build the IR yourself with `UOp.range`, `UOp.placeholder`, `UOp.store`, `.end(...)`.
- Example 4: stock tinygrad with `BEAM=2`.
- Example 5: hand-written **RDNA3 assembly** instructions, assembled directly into a `BINARY`.

Run them (on a supported GPU; emulator works on Mac with `DEV=MOCKPCI+AMD`):

```bash
PYTHONPATH="." DEV=MOCKPCI+AMD python3 docs/abstractions4.py
```

For each example, answer: where does my code plug into the pipeline? At which UOp `Ops.*` boundary?

---

## Phase 9 — Capstone projects (pick one)

You are now an ML compiler engineer. Prove it:

1. **New rewrite rule.** Find a missed simplification in `tinygrad/uop/symbolic.py` (e.g. an algebraic identity not yet handled), add a `PatternMatcher` rule, write a test in `test/unit/`, send a PR.
2. **New backend.** Add a renderer for a toy target (e.g. NumPy code generator, or RISC-V scalar code) by subclassing `Renderer` like `cstyle.py` does.
3. **A new fusion pass.** Pick a fusion the scheduler currently misses (e.g. fuse a `softmax` numerator/denominator into one kernel) and implement it as a `PatternMatcher`.
4. **A new `Opt`.** Add an opt to `codegen/opt/__init__.py` and integrate it into BEAM search.
5. **Docs / tutorial.** Re-write phase 4 as a blog post with diagrams. Teaching is the highest test of understanding.

---

## Cheatsheet of files (bookmark this)

```
Frontend          tinygrad/tensor.py:80              class Tensor
                  tinygrad/uop/__init__.py:13        class Ops (the IR vocabulary)
                  tinygrad/uop/ops.py:129            class UOp
                  tinygrad/uop/ops.py:1027,1162      class UPat, PatternMatcher
                  tinygrad/uop/ops.py:1447           graph_rewrite

Symbolic / type   tinygrad/uop/symbolic.py           algebraic simplifications
                  tinygrad/uop/spec.py               kernel_spec, program_spec
                  tinygrad/uop/decompositions.py     polynomial expansions

Scheduler         tinygrad/schedule/rangeify.py      bufferize / fusion
                  tinygrad/schedule/memory.py        buffer reuse
                  tinygrad/schedule/multi.py         multi-GPU sharding

Codegen           tinygrad/codegen/__init__.py:23    full_rewrite_to_sink (the spine)
                  tinygrad/codegen/simplify.py       range simplification
                  tinygrad/codegen/late/*.py         expander, devectorizer, linearizer
                  tinygrad/codegen/opt/search.py     BEAM
                  tinygrad/codegen/opt/tc.py         tensor cores

Renderers         tinygrad/renderer/cstyle.py        CUDA / OpenCL / Metal / HIP
                  tinygrad/renderer/llvmir.py        LLVM IR
                  tinygrad/renderer/ptx.py           PTX assembly

Engine + runtime  tinygrad/engine/realize.py:243     run_linear
                  tinygrad/engine/jit.py:247         TinyJit
                  tinygrad/device.py                 Buffer, Device
                  tinygrad/runtime/ops_*.py          per-device runtimes

End-to-end demos  docs/abstractions3.py              front-to-back tour
                  docs/abstractions4.py              5 abstraction levels of one kernel
                  docs/developer/developer.md        official 1-pager
```

## Useful debug knobs

| Env var | Effect |
| --- | --- |
| `DEBUG=2` | per-kernel timing & sizes |
| `DEBUG=4` | dump rendered source |
| `DEBUG=5` | dump UOp DAG after every rewrite |
| `VIZ=1` | open browser visualizer of all rewrites |
| `BEAM=2` | search for fast kernels |
| `NOOPT=1` | disable optimizer (useful when comparing) |
| `JIT=0` | disable JIT capture |
| `SPEC=1` | enforce `kernel_spec` / `program_spec` after rewrites |

External resources used by this repo's own docs:
- Di Zhu's tinygrad-notes tutorials (linked from `docs/developer/developer.md`).
- `docs/developer/speed.md` — performance philosophy.

Good luck. The shortest path to "ML compiler engineer" is: write a `PatternMatcher` rule today, then another tomorrow.
