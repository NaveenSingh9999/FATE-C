# FATE-C  
Universal Neuron → Layer → Network Designer & Compiler

> Design custom neurons, assemble layers, build simple or complex (sequential / graph / multi-model) neural systems in pure Python in 10–20 lines. Run them in the lightweight FATE-C math engine or export to PyTorch / TensorFlow / JAX / ONNX — without rewriting.

---

## 1. Mission
Make neural architecture creation as simple as writing plain Python functions while keeping a clear runway to advanced research (custom activations, novel optimizers, multi-network systems) and seamless portability across major ML backends.

## 2. Philosophy (Pillars)
| Pillar | Meaning | Practical Outcome |
|--------|---------|-------------------|
| Neuron-first | Start at the smallest unit | Users invent new activation / compute atoms easily |
| Progressive Complexity | Ladder: Sequential → Custom forward → Graph/DAG → Multi-model | Beginners stay productive; experts not boxed in |
| Universal Portability | One definition → compile/export anywhere | Reduce reimplementation & vendor lock |
| Research Agility | Low ceremony for experimental math | Faster iteration on novel ideas |
| Ergonomic Defaults | Inferred shapes, auto loss/metrics, safe inits | 10–20 line usable models |
| Transparency | No heavy magic, explicit when complex | Debuggable & teachable |

## 3. Quick Glance
```python
import fatec as fc

# 11 lines: 3-layer classifier
model = fc.seq([
	fc.Dense(128, act="relu"),
	fc.Dense(64, act="relu"),
	fc.Dense(10, act="softmax")
])

model.compile()              # infers loss = cross_entropy, opt = adam
model.fit(train_x, train_y, epochs=5, val=(val_x, val_y))
model.save("mlp.ftc")
```

Custom neuron + reuse:
```python
@fc.neuron
def mish(x):
	return x * fc.tanh(fc.softplus(x))

net = fc.seq([
	fc.Dense(256, act=mish),
	fc.Dropout(0.2),
	fc.Dense(10, act="logits")
])
net.compile(lr=3e-4)
net.fit(x, y, epochs=3)
```

Branch graph:
```python
g = fc.graph()
inp = g.input("x", shape=(None, 128))
h = g.dense(inp, 128, act="relu")
cls = g.dense(h, 10, act="logits", name="class_logits")
aux = g.dense(h, 1, act="sigmoid", name="confidence")
model = g.build()
model.compile(loss={"class_logits":"ce", "confidence":"bce"}, weights={"confidence":0.2})
model.fit(x, {"class_logits": y_cls, "confidence": y_conf}, epochs=4)
```

Multi-model (GAN skeleton):
```python
G = fc.seq([fc.Dense(128, act="relu"), fc.Dense(784, act="tanh")])
D = fc.seq([fc.Dense(128, act="relu"), fc.Dense(1, act="sigmoid")])

gan = fc.group({"G": G, "D": D}).as_gan(gen_input_shape=(100,))
gan.compile(gen_opt="adam", disc_opt="adam")
gan.fit(dataset, steps_per_epoch=200, epochs=5)
```

## 4. Progressive Abstraction Ladder
1. Sequential (fc.seq([...])) – fastest ramp.
2. Custom forward class – subclass BaseModel, write forward().
3. GraphBuilder – explicit DAG with branching/merging.
4. ModelGroup – orchestrate multiple cooperating models (GAN, dual encoders, distillation).

## 5. Core User-Facing API Surface
Flat namespace (import fatec as fc):
| Category | Key Objects / Functions | Notes |
|----------|-------------------------|-------|
| Neurons | @fc.neuron decorator, built-ins (relu, gelu, mish, swish, etc.) | Auto-grad support |
| Layers | Dense, Conv, Embedding, RNNCell, Dropout, Norm, Lambda | Lazy init on first call |
| Models | seq(), graph(), group(), BaseModel | Compose arbitrarily |
| Training | compile(), fit(), evaluate(), predict() | Auto loss/metric inference |
| Export | export("torch"|"tf"|"jax"|"onnx"), save(), load() | Canonical form normalizer |
| Inspection | summary(), describe(), ascii_graph() | Shape + param counts |
| Utilities | dataset.auto(), seed(), set_precision(), quickfit() | Remove boilerplate |

## 6. Neuron Design Flow
```python
@fc.neuron(params=dict(beta=1.5))
def swish(x, beta):
	return x / (1 + fc.exp(-beta * x))
```
Features:
- Expression or pure Python function.
- Optional params dict for learnable or fixed hyper-parameters.
- Auto registration + derivative (symbolic or numeric fallback with caching).
- Validation utilities: gradient check, saturation report, domain warnings.

## 7. Layer & Shape Inference
Lazy parameter creation: first forward pass traces input shapes → allocates weights (initializer auto-chosen based on activation & fan-in/out). Shape errors raise early with a diagnostic including path: `Dense(1).weight shape mismatch at graph.node3`.

## 8. Graph / DAG Builder
Minimal builder with explicit inputs & node names; merges via operations (concat, add, multiply). Provides:
- Static shape inference & validation pass (`g.validate()`).
- Topological schedule caching.
- Activation / gradient flow probes.
- ASCII & (optional) rich visualization.

## 9. Multi-Model Orchestration (ModelGroup)
Supports patterns: GAN, teacher–student, multi-task, dual encoders.
Capabilities:
- Independent / shared optimizers.
- Step scheduling recipes (e.g., {"gen":1, "disc":1}).
- Freeze/unfreeze patterns by name glob.
- Unified save/load with namespaced state keys.

## 10. Training Engine
Goals: clarity > cleverness, but optionally fast.
Features:
- AutoLossResolver: infers standard loss from target shape/dtype (e.g., (N,) ints → sparse CE; (N,C) probs → CE; regression → MSE).
- Metrics auto pick (accuracy for classification, MAE for regression) with override.
- Mixed precision optional; unsafe ops flagged.
- Gradient clipping, early stopping, learning rate schedulers, checkpoint callbacks.
- Determinism report (seeds, backend versions).

## 11. Optimizer Lab
Built-ins: SGD, Momentum, Adam, AdamW, RMSProp, Adagrad, Lookahead wrapper, EMA.
Composable: fc.opt.adam(lr=1e-3) | fc.opt.with_ema(decay=0.999).
Custom: implement step(param, grad, state) and register.

## 12. Export & Interop
`model.export("torch")` → Torch module stub with identical forward.
`model.export("onnx")` → ONNX graph after canonical lowering.
Coverage checker warns on unsupported ops & suggests shims.

## 13. Visualization & Debugging
- summary(): hierarchical param table.
- ascii_graph(): textual graph.
- activation_probe(layer_name) / gradient_report().
- saturation_heatmap() (optional dependency for plotting).

## 14. Experiment Utilities
- Run logger (JSONL + YAML config snapshot).
- Hash of neuron/layer definitions for reproducibility.
- quickfit(): one-shot baseline training for sanity.
- compare(models, metric): tabular result.

## 15. Error Design Principles
| Situation | Message Style |
|-----------|---------------|
| Shape mismatch | Shows expected vs got, node path, preceding node output shape |
| Unsupported export op | Suggests nearest composite alternative |
| Diverging loss | Detects NaN/Inf source op via backward trace |
| Dead activations | Reports % dead units per layer (ReLU-like) |

## 16. CLI (Optional Future)
```
fatec new neuron --name MishLike --template rational
fatec quickfit --layers 128,64,10 --epochs 3 --dataset iris.csv
fatec export model.ftc --to onnx
```

## 17. Minimal Internal Architecture (Conceptual Modules)
```
core/
  tensor.py      # thin wrapper (NumPy / internal)
  autograd.py    # reverse-mode engine
  ops.py         # primitive ops + safe numerics
neurons/
layers/
graph/
model/
training/
optim/
export/
viz/
utils/
```

## 18. Roadmap (Indicative)
| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| 0.1 | MVP Sequential | Dense, Dropout, compile/fit/save, auto loss, export(torch stub) |
| 0.2 | Neuron & Autograd | @neuron decorator, gradient checker, custom activations |
| 0.3 | Graph Builder | DAG support, ascii_graph, shape validator |
| 0.4 | Multi-Model | ModelGroup, GAN recipe, schedules |
| 0.5 | Export Suite | ONNX + TF + JAX lowering, op coverage checker |
| 0.6 | Optimizer Lab | Composable optimizers, EMA, Lookahead |
| 0.7 | Visualization | Activation / gradient maps, saturation diagnostics |
| 0.8 | Performance | Kernel fusion (internal), mixed precision, caching |
| 0.9 | CLI & Docs | CLI scaffold, doc site, examples gallery |
| 1.0 | Stable Release | API freeze, tests, benchmarks |

## 19. Contribution Guide (Brief)
1. Fork & branch (`feature/short-name`).
2. Add or update tests (target: >90% for modified paths). 
3. Run lint & test suite.
4. Update README/docs for new public APIs.
5. Submit PR with concise rationale + design notes.

Coding principles:
- Keep public API small & flat.
- Fail loud & early; helpful diagnostics.
- No hidden global state (explicit registries only).
- Prefer composable primitives over giant classes.

## 20. FAQ (Conceptual)
**Q: Do I have to learn a DSL?**  
No—pure Python is enough; optional expr helper may come later.

**Q: Is there a heavy runtime dependency?**  
Core aims to lean on NumPy (or optional minimal backend) + optional extras.

**Q: How are custom neurons differentiated?**  
Either symbolic (if built from primitive ops) or numeric grad fallback with caching.

**Q: How does export ensure parity?**  
Intermediate canonical graph + operator equivalence tests per backend.

**Q: Can I mix backends?**  
Design in FATE-C, export each variant; hybrid execution may come later.

## 21. Status
Early concept blueprint. Implementation phases pending. Sections marked as future may evolve.

## 22. License
TBD (recommend MIT or Apache-2.0 for openness + adoption). Add SPDX header once chosen.

## 23. Inspiration & Prior Art
Draws ideas from: Keras (API ergonomics), PyTorch (eager flexibility), JAX (function-first), ONNX (interop), tinygrad / micrograd (minimal autograd clarity). FATE-C focuses specifically on neuron-first design + multi-model orchestration simplicity.

## 24. Next Steps (Actionable)
- Decide initial license.
- Scaffold core directories.
- Implement MVP (Phase 0.1) with tests.
- Draft CONTRIBUTING.md & CODE_OF_CONDUCT.md.
- Add examples: mnist_mlp.py, custom_neuron.py, gan_toy.py.

---

Feel free to adapt, trim, or split sections into dedicated docs under `docs/` as the codebase grows.

