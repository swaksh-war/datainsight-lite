# DataInsight Framework — Pre-Development Roadmap

> **Framework Vision**: A unified, tiered ML framework where beginners drop a dataset and get trained models, intermediate users build custom pipelines, and experts wire in custom layers and preprocessors — all backed by a single adapter system supporting scikit-learn, PyTorch, and TensorFlow.

---

## Current State Audit

### ✅ Already Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `quick_report()` — one-liner EDA | ✅ Done | CSV / Excel / JSON → HTML report |
| HTML report generator (Jinja2) | ✅ Done | Template-based, embeds charts |
| `Reporter` — stats engine | ✅ Done | Summary, outliers, correlation, missing values |
| `UniVariateAnalysis` | ✅ Done | Histogram, boxplot, KDE, bar, pie (matplotlib/seaborn) |
| `BiVariateAnalysis` | ✅ Done | Scatter, violin, grouped bar, line plot |
| `MultiVariateAnalysis` | ✅ Done | Pairplot, correlation heatmap |
| `LabelEncoder` | ✅ Done | Fit/transform/inverse/save/load, handles missing + unknown |
| `OrdinalEncoder` | ✅ Done | Auto or custom-order encoding |
| `CountFrequencyEncoder` | ✅ Done | Count or frequency mode |
| `LabelBinarizer` | ✅ Done | One-hot / multi-hot, NumPy output |
| `BaseEncoder` (ABC) | ✅ Done | JSON serialization pattern |
| `datatypes.py` utility aliases | ✅ Done | Type alias layer |
| Basic utility functions | ✅ Done | `dataset_summary`, `describe_numerical`, `detect_outliers` |

### ⚠️ Partially Implemented

| Module | Status | Gap |
|--------|--------|-----|
| `OneHotEncoder` | 🟡 Skeleton | `__transform()` and public methods missing |
| `BaseScaler` | 🟡 Abstract only | No concrete scalers (StandardScaler, MinMaxScaler, etc.) |
| `eda_agent.py` | 🟡 Empty file | Placeholder, nothing built |
| Test suite | 🟡 Empty files | Both test files are completely blank |

### ❌ Not Yet Started

- ML model training of any kind
- Framework adapter system
- Image / video data ingestion and preprocessing
- Training dashboard
- Deployment layer
- Pipeline builder (configurable or visual)
- Custom layer / custom preprocessor API
- Model registry, weight export

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          User Interface Layer                         │
│   Tier 1: AutoML API    │  Tier 2: Pipeline API  │  Tier 3: Low-level │
└──────────────┬──────────┴──────────┬─────────────┴─────────┬─────────┘
               │                     │                        │
┌──────────────▼─────────────────────▼────────────────────────▼────────┐
│                         Pipeline Orchestrator                          │
│          (DAG-based pipeline with registry + validation)               │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
          ┌─────────────────────────┼──────────────────────┐
          │                         │                       │
┌─────────▼──────┐       ┌──────────▼────────┐   ┌────────▼──────────┐
│  Data Ingestion │       │   Preprocessing    │   │    Model Layer     │
│  CSV/Image/Video│       │  Scalers/Encoders  │   │ (Adapter Pattern)  │
└─────────────────┘       └───────────────────┘   └───────────────────┘
                                                            │
                                       ┌────────────────────┼──────────────────┐
                                       │                     │                  │
                                ┌──────▼──────┐   ┌─────────▼──────┐   ┌──────▼──────┐
                                │  sklearn     │   │   PyTorch       │   │ TensorFlow  │
                                │  Adapter     │   │   Adapter       │   │  Adapter    │
                                └─────────────┘   └────────────────┘   └─────────────┘
                                                            │
                          ┌─────────────────────────────────┼─────────────────────────┐
                          │                                  │                         │
                 ┌────────▼────────┐              ┌──────────▼────────┐     ┌──────────▼───┐
                 │ Training Dashboard│              │  Model Registry    │     │  Deployment   │
                 │ (per-model live   │              │  + Weight Export   │     │  Layer        │
                 │  metrics UI)      │              └───────────────────┘     └──────────────┘
                 └──────────────────┘
```

---

## User Tiers

### Tier 1 — No ML Knowledge (AutoML)
> "Drop your file, get trained models and a comparison report."

```python
from datainsight import AutoML

pipeline = AutoML("my_data.csv", task="classification")
pipeline.run()
# → preprocesses, trains N models, shows comparison, saves weights
```

### Tier 2 — Some ML Knowledge (Pipeline Builder)
> "I know what I want, let me configure the pipeline."

```python
from datainsight import Pipeline
from datainsight.preprocessing import StandardScaler, OneHotEncoder
from datainsight.models import RandomForestAdapter, SVMAdapter

pipeline = Pipeline()
pipeline.add_step("scaler", StandardScaler())
pipeline.add_step("encoder", OneHotEncoder(columns=["city"]))
pipeline.add_model(RandomForestAdapter(n_estimators=200))
pipeline.add_model(SVMAdapter(kernel="rbf"))
pipeline.fit(X_train, y_train)
pipeline.compare()
```

### Tier 3 — Expert / Pro (Full Custom)
> "I'm building my own layers, preprocessors, and training loops."

```python
from datainsight.core import BaseLayer, BasePreprocessor, Trainer
from datainsight.adapters import TorchAdapter

class MyAttentionLayer(BaseLayer):
    def forward(self, x): ...

class MyNormalizer(BasePreprocessor):
    def fit(self, data): ...
    def transform(self, data): ...

trainer = Trainer(adapter=TorchAdapter(), dashboard=True)
trainer.fit(model, train_loader, val_loader, epochs=50)
```

---

## Development Phases

---

### Phase 0 — Fix & Harden Current Codebase
**Goal**: Close open gaps before building new things on top.

| Task | File(s) | Priority |
|------|---------|----------|
| Implement `OneHotEncoder.__transform()` and public methods | `categorical/one_hot_encoder.py` | High |
| Implement concrete numerical scalers: `StandardScaler`, `MinMaxScaler`, `RobustScaler` | `numerical/` | High |
| Implement `MaxAbsScaler`, `Normalizer` | `numerical/` | Medium |
| Write unit tests for all existing encoders | `tests/test_core.py` | High |
| Write unit tests for all analysis modules | `tests/test_visualizations.py` | Medium |
| Fix `__init__.py` import paths in `feature_analysis/` (currently uses bare imports) | `feature_analysis/__init__.py` | High |
| Finalize `BaseScaler` abstract interface (mirror `BaseEncoder` pattern) | `numerical/base_scaler.py` | High |

---

### Phase 1 — Core Infrastructure & Adapter System
**Goal**: Build the backbone everything else depends on.

#### 1.1 Adapter Abstraction Layer
```
datainsight_lite/
  adapters/
    __init__.py
    base_adapter.py        ← Abstract BaseAdapter(ABC)
    sklearn_adapter.py     ← Wraps sklearn models
    torch_adapter.py       ← Wraps nn.Module, DataLoader
    tensorflow_adapter.py  ← Wraps Keras model
```

- `BaseAdapter` defines: `fit()`, `predict()`, `evaluate()`, `save()`, `load()`, `get_params()`, `set_params()`
- Each adapter translates the common interface to backend-specific calls
- Adapters expose `backend` property: `"sklearn"` | `"torch"` | `"tensorflow"`
- Adapters handle device management (CPU/GPU/MPS) internally

#### 1.2 Data Ingestion Layer
```
datainsight_lite/
  ingestion/
    __init__.py
    base_ingester.py       ← Abstract BaseIngester
    tabular_ingester.py    ← CSV, Excel, JSON, Parquet
    image_ingester.py      ← JPEG, PNG, folder of images
    video_ingester.py      ← MP4, AVI → frame extraction
```

- Unified `DataBundle` dataclass: holds raw data, metadata, detected modality
- Auto-detects modality from file extension or directory structure
- Image ingester supports: single file, directory, zip archive
- Video ingester: configurable frame sampling rate, resize

#### 1.3 Pipeline Orchestrator
```
datainsight_lite/
  pipeline/
    __init__.py
    pipeline.py            ← DAG-based Pipeline class
    step.py                ← PipelineStep wrapper
    registry.py            ← StepRegistry — maps name → class
    validator.py           ← Schema + type compatibility checks
```

- Ordered list of named steps (fit → transform chain)
- Supports branching for multi-modal inputs
- `Pipeline.add_step(name, transformer)`
- `Pipeline.add_model(adapter)` — can add multiple for comparison
- `Pipeline.fit(X, y)` / `Pipeline.predict(X)` / `Pipeline.compare()`
- Serialization: save/load entire pipeline to JSON + weight files

#### 1.4 Model Registry
```
datainsight_lite/
  models/
    __init__.py
    registry.py            ← ModelRegistry singleton
    model_card.py          ← Metadata per model (params, metrics, backend)
```

- Central catalog of all available models across all backends
- Auto-discovers adapters via registration decorator `@registry.register`
- Supports querying by task: `registry.get_models(task="classification", backend="sklearn")`

---

### Phase 2 — Preprocessing Completion
**Goal**: Full preprocessing suite matching sklearn's API but with own base classes.

#### 2.1 Numerical Preprocessing (complete `preprocessing/numerical/`)
| Class | Purpose |
|-------|---------|
| `StandardScaler` | Zero-mean, unit-variance |
| `MinMaxScaler` | Scale to [0, 1] or custom range |
| `RobustScaler` | Median/IQR based, outlier resistant |
| `MaxAbsScaler` | Scale by max absolute value |
| `Normalizer` | Row-wise L1/L2/max normalization |
| `PowerTransformer` | Yeo-Johnson / Box-Cox |
| `QuantileTransformer` | Uniform or normal output distribution |

All inherit `BaseScaler`, support fit/transform/inverse_transform/save/load.

#### 2.2 Categorical Preprocessing (complete existing + new)
| Class | Status |
|-------|--------|
| `LabelEncoder` | ✅ Done |
| `OrdinalEncoder` | ✅ Done |
| `CountFrequencyEncoder` | ✅ Done |
| `LabelBinarizer` | ✅ Done |
| `OneHotEncoder` | 🟡 Complete transform |
| `TargetEncoder` | ❌ New |
| `BinaryEncoder` | ❌ New |
| `HashingEncoder` | ❌ New |

#### 2.3 Image Preprocessing
```
datainsight_lite/utils/preprocessing/image/
  base_image_transform.py
  resize_transform.py
  normalize_transform.py
  augmentation_transform.py   ← flip, rotate, crop, color jitter
  pipeline_transform.py       ← Compose multiple transforms
```

- Backend-agnostic API; internally uses PIL/torchvision or tf.image
- Compatible with both PyTorch `Dataset` and TF `tf.data.Dataset`

#### 2.4 Text Preprocessing (optional, medium term)
- Tokenizer, stopword removal, TF-IDF vectorizer — only if NLP use cases are scoped

---

### Phase 3 — AutoML Tier (Tier 1 Users)
**Goal**: Drop a file, get trained models and a comparison report with zero code.

#### 3.1 Task Detector
- Auto-infer task type from target column: classification / regression / clustering
- Modality detection: tabular / image / video
- Class imbalance detection + recommended strategy

#### 3.2 Auto Preprocessor
- Tabular: missing value imputation → encoding → scaling (auto-selected per column dtype)
- Image: resize → normalize (mean/std auto-computed from dataset)
- Video: frame sampling → per-frame image pipeline

#### 3.3 Auto Model Selector
```
datainsight_lite/
  automl/
    __init__.py
    task_detector.py
    auto_preprocessor.py
    model_selector.py       ← Selects candidate models per task
    trainer.py              ← Orchestrates training + evaluation
    comparator.py           ← Generates comparison table + plots
```

**Default model pools per task:**

| Task | sklearn models | PyTorch models | TF models |
|------|----------------|----------------|-----------|
| Classification (tabular) | LogReg, RFC, SVM, GBM, KNN | MLP | MLP |
| Regression (tabular) | LinearReg, Ridge, Lasso, RFC, GBM | MLP | MLP |
| Classification (image) | — | CNN (ResNet18 pretrained) | CNN (MobileNetV2) |
| Clustering | KMeans, DBSCAN, Agglomerative | — | — |

#### 3.4 AutoML Output
- Trained weight files per model (`.pkl` for sklearn, `.pt` for torch, `.h5`/`SavedModel` for tf)
- `comparison_report.html` — table of metrics + training curves per model
- `best_model.json` — metadata about the best performing model
- Optional: ONNX export for the best model

---

### Phase 4 — Configurable Pipeline (Tier 2 Users)
**Goal**: Let users with some ML knowledge build and run their own pipelines.

#### 4.1 Pipeline Builder API
- Fluent interface: `Pipeline().add_step(...).add_model(...).fit(...)`
- Support for conditional branches (e.g., numerical branch vs. categorical branch)
- Support `ColumnTransformer`-style per-column step specification
- Cross-validation wrapper: `pipeline.cross_validate(X, y, cv=5)`

#### 4.2 Hyperparameter Configuration
- YAML / JSON-based pipeline config file support
- `Pipeline.from_config("pipeline.yaml")`
- Grid search and random search wrappers around any adapter
- Optional: Bayesian optimization via `optuna` integration

#### 4.3 Evaluation & Comparison
- Unified `Evaluator` class: computes relevant metrics based on task type
- Metrics: accuracy, F1, ROC-AUC, precision, recall (classification); MAE, MSE, R² (regression)
- `Comparator.plot()` — side-by-side bar chart of model metrics (plotly or matplotlib, user-selectable)
- Confusion matrix, ROC curve, feature importance plots

---

### Phase 5 — Expert / Pro Tier (Tier 3 Users)
**Goal**: Full extensibility — custom layers, preprocessors, and training loops.

#### 5.1 Custom Layer API
```
datainsight_lite/
  core/
    __init__.py
    base_layer.py          ← Abstract BaseLayer
    base_block.py          ← Composable block of layers
    layer_registry.py      ← Register custom layers by name
```

- `BaseLayer` is backend-agnostic; subclass with `@torch_layer` or `@tf_layer` decorator
- Layers auto-register into `LayerRegistry` when defined
- Compatible with Pipeline's model steps

#### 5.2 Custom Preprocessor API
- Extend `BaseEncoder` or `BaseScaler` with own `_fit()` / `_transform()` / `_inverse_transform()`
- `@register_preprocessor("my_normalizer")` decorator for auto-discovery
- Save/load inherited from base classes

#### 5.3 Custom Trainer & Callbacks
```
datainsight_lite/
  training/
    __init__.py
    trainer.py             ← Trainer orchestrator
    callbacks/
      base_callback.py     ← on_epoch_start/end, on_batch_start/end
      early_stopping.py
      model_checkpoint.py
      lr_scheduler.py
      dashboard_callback.py  ← Feeds metrics to live dashboard
```

- `Trainer(adapter, callbacks=[...], dashboard=True)`
- Callbacks follow Keras-style lifecycle hooks
- Custom training loops via `trainer.train_step()` override

---

### Phase 6 — Training Dashboard
**Goal**: Per-model live training visualization; one dashboard per run.

#### 6.1 Dashboard Design
```
datainsight_lite/
  dashboard/
    __init__.py
    server.py              ← Lightweight local HTTP server (Flask or Dash)
    dashboard.py           ← Dashboard state manager
    metrics_store.py       ← Thread-safe in-memory metrics buffer
    templates/
      dashboard.html       ← Real-time charts with Plotly.js
```

#### 6.2 Features
- Launches automatically on `trainer.fit()` if `dashboard=True`
- Opens in browser at `http://localhost:PORT`
- Live charts (WebSocket or SSE): loss, accuracy (or other metrics) per epoch
- Side-by-side panels when training multiple models
- Logs panel: timestamped epoch summaries
- Export dashboard as static HTML snapshot after training

#### 6.3 Multi-model Comparison View
- Separate tab per model during training
- Final comparison tab: overlay all models' learning curves
- Highlight best model per metric

---

### Phase 7 — Deployment Layer
**Goal**: Test and deploy trained models.

#### 7.1 Model Export
```
datainsight_lite/
  deployment/
    __init__.py
    exporter.py            ← Export to ONNX, TorchScript, SavedModel, pickle
    importer.py            ← Load any exported model
```

| Format | Supported Backends |
|--------|--------------------|
| `.pkl` (joblib) | sklearn |
| `.pt` (TorchScript) | PyTorch |
| `SavedModel` / `.h5` | TensorFlow |
| ONNX | PyTorch, TensorFlow, sklearn (via skl2onnx) |

#### 7.2 Local Serving & Testing
```
datainsight_lite/
  deployment/
    server.py              ← REST API wrapper (FastAPI)
    test_client.py         ← CLI and Python client for predictions
```

- `deploy(model, host="localhost", port=8000)` — spins up a FastAPI endpoint
- `POST /predict` — accepts JSON or multipart (for images)
- Auto-generates OpenAPI schema from model input spec
- `test_client.predict(data)` — Python helper for quick testing

#### 7.3 Downstream Task Hooks
- Post-training evaluation on held-out test set
- A/B comparison between two deployed model versions
- Batch prediction from a CSV file

---

## Dependency Plan

| Dependency | Purpose | When Added |
|------------|---------|------------|
| `pandas` | Tabular data | ✅ Current |
| `matplotlib` | Static visualization | ✅ Current |
| `seaborn` | Statistical plots | ✅ Current |
| `jinja2` | HTML templating | ✅ Current |
| `numpy` | Array ops | Phase 0 |
| `scikit-learn` | sklearn adapter + metrics | Phase 1 |
| `torch` | PyTorch adapter | Phase 1 |
| `tensorflow` | TF adapter | Phase 1 |
| `plotly` | Interactive charts, dashboard | Phase 4/6 |
| `flask` or `dash` | Dashboard server | Phase 6 |
| `fastapi` + `uvicorn` | Deployment server | Phase 7 |
| `onnx` + `onnxruntime` | ONNX export/import | Phase 7 |
| `pillow` | Image loading | Phase 1 |
| `optuna` | Hyperparameter search (optional) | Phase 4 |
| `pyyaml` | Config file support | Phase 4 |

All new framework dependencies should be **optional extras** to avoid bloating the base install:
```
pip install datainsight           # base (EDA only)
pip install datainsight[sklearn]  # + sklearn adapter
pip install datainsight[torch]    # + PyTorch adapter
pip install datainsight[tf]       # + TensorFlow adapter
pip install datainsight[all]      # everything
```

---

## Naming & Package Restructure Plan

Current package `datainsight_lite/` nesting is redundant (`datainsight_lite/datainsight/`). Proposed target layout for the full framework:

```
datainsight/
  __init__.py                    ← top-level: quick_report, AutoML, Pipeline
  adapters/                      ← sklearn / torch / tf adapters
  automl/                        ← Tier 1 automl engine
  core/                          ← BaseLayer, BaseBlock, layer_registry
  dashboard/                     ← training dashboard server
  deployment/                    ← export, serving, test client
  eda/                           ← current datainsight/ content (reporter, feature_analysis, reporting)
  ingestion/                     ← tabular, image, video ingesters
  models/                        ← model registry, model cards
  pipeline/                      ← pipeline orchestrator, step, validator
  training/                      ← Trainer, callbacks
  utils/
    preprocessing/
      categorical/               ← existing encoders + new ones
      numerical/                 ← scalers (complete)
      image/                     ← new
    datatypes.py
    basic.py
```

---

## Phase Summary & Priority Order

| Phase | Name | Tier Dependency | Estimated Scope |
|-------|------|----------------|----------------|
| **0** | Fix & Harden Current Code | All | Small |
| **1** | Adapter System + Data Ingestion + Pipeline | All | Large |
| **2** | Preprocessing Completion | All | Medium |
| **3** | AutoML (Tier 1) | Tier 1 | Large |
| **4** | Configurable Pipeline (Tier 2) | Tier 2 | Medium |
| **5** | Expert API (Tier 3) | Tier 3 | Medium |
| **6** | Training Dashboard | All | Medium |
| **7** | Deployment Layer | All | Medium |

**Recommended build order**: 0 → 1 → 2 → 6 → 3 → 4 → 5 → 7

Reason: the adapter (Phase 1), preprocessing (Phase 2), and dashboard (Phase 6) are shared infrastructure needed by all three tiers before any tier-specific work begins.

---

## Open Design Decisions (To Resolve Before Implementation)

1. **Dashboard framework**: Flask + Plotly.js (lightweight, fewer deps) vs. Dash (richer but heavier) vs. a terminal-based TUI (no browser needed)?
2. **Config format for Tier 2 pipelines**: YAML vs. JSON vs. Python dict — YAML is most human-friendly.
3. **Multi-backend model in one pipeline**: Should one pipeline be able to mix sklearn and torch steps, or is each pipeline single-backend? Recommendation: single-backend per pipeline, but the `Comparator` can compare pipelines across backends.
4. **Image/video support scope for v1**: Full support (ingestion + preprocessing + CNN training) or tabular-only for v1 with image/video as v2?
5. **Package rename**: Stay as `datainsight-lite` or rename to `datainsight` as scope expands?
6. **Adapter granularity**: Should adapters wrap individual models (e.g., `RandomForestAdapter`) or backends as a whole (e.g., `SklearnAdapter(model=RandomForest(...))`)? Recommendation: backend-level adapter with model passed in, to keep the adapter count manageable.
