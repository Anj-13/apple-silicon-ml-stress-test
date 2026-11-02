# Apple Silicon ML Stress Test

![Language](https://img.shields.io/badge/Language-Python%203.11-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-informational)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Six ML, RAG, and backend workloads run on a 16 GB Apple Silicon MacBook to measure what memory pressure really looks like.**

This project ran gradient boosting, GPU training, vector search, and a database-backed API to see whether real workloads need all 16 GB. The technically interesting result is that the answer depends on the tool: `psutil` reported about 6–7 GB used during training, while Activity Monitor peaked at 13.78 GB once swap and compression were involved.

**Links:** [XGBoost results](ml-training/XGBoost/training_results.md) · [PyTorch results](ml-training/pytorchMPStest/training_results.md)

## Contributors

Developed by Anj-13 ([@Anj-13](https://github.com/Anj-13)).

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Implementation](#technical-implementation)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture & Project Structure](#system-architecture--project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing & Quality](#testing--quality)
- [Challenges & Solutions](#challenges--solutions)
- [Results & What I Learned](#results--what-i-learned)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

**Problem:** Would realistic ML and backend workloads use all the memory on a 16 GB MacBook?

**Solution:** Run a series of heavy workloads, then compare script-level `psutil` readings against system-level Activity Monitor readings.

**Audience / Context:** Personal hands-on weekend project.

**Project Status:** Completed. The code is kept as a record and is not actively maintained.

## Technical Implementation

**Anj-13 — Developer**

Designed and ran the experiments independently.

### Core Contributions

**ML training**
- Ran an XGBoost `GridSearchCV` across 36 fits on 500,000 synthetic rows with 50 features.
- Trained a CIFAR-10 CNN with 3 convolutional and 2 fully connected layers on the Apple GPU through PyTorch MPS.

**RAG pipeline**
- Embedded 1,000 synthetic documents with `all-MiniLM-L6-v2` and queried them from a persistent ChromaDB store.

**Full-stack API**
- Built a FastAPI service using SQLAlchemy and PostgreSQL, with read-through Redis caching.
- Ran PostgreSQL and Redis as native macOS services.

**Profiling and measurement**
- Used `cProfile` and `pstats` to locate the training hot path.
- Allocated 12 GB of NumPy arrays to compare reported memory use with system behaviour.

> **Contribution summary:** Built and ran ML training, RAG, and FastAPI workloads on a 16 GB Apple Silicon MacBook, then used profiling and memory measurements to show how macOS handled the load.

## Key Features

- **Gradient boosting at scale:** Grid search across tree depth, tree count, and learning rate on 500,000 rows.
- **Apple GPU training:** CNN training on MPS, with automatic CPU fallback on other machines.
- **Vector search:** Sentence embeddings stored in ChromaDB and queried for the top 5 matches.
- **Cached API:** Paper records stored in PostgreSQL, with titles cached in Redis for one hour.
- **Live memory endpoint:** `GET /stats` returns current system memory usage.
- **Memory stress test:** Six 2 GB arrays are held in memory until you release them.

## Tech Stack

| Layer | Technologies | Purpose |
| --- | --- | --- |
| Interface | CLI scripts, FastAPI interactive docs | Running workloads and calling the API |
| Backend / Core | Python 3.11, FastAPI, Uvicorn, Pydantic, SQLAlchemy | API and request validation |
| ML / AI | XGBoost, scikit-learn, PyTorch (MPS), sentence-transformers | Training and embeddings |
| Database / Storage | PostgreSQL, Redis, ChromaDB | Records, caching, and vector search |
| Measurement | `psutil`, `cProfile`, `pstats`, Activity Monitor | Memory and performance analysis |
| Tools | Git, venv, Docker Compose (written, not run) | Environment and packaging |

## System Architecture & Project Structure

The workloads are independent, and each one runs from its own script. Only the API and RAG pipeline have a multi-step data flow:

```text
API:  Client -> FastAPI -> Redis (hit: return title)
                             |
                             +-- miss -> PostgreSQL -> cache title for 1 hour

RAG:  Synthetic documents -> all-MiniLM-L6-v2 embeddings -> ChromaDB -> top 5 results
```

<details>
<summary>View project folder structure</summary>

```text
apple-silicon-ml-stress-test/
├── fullstack-app/
│   ├── app.py                     # FastAPI, PostgreSQL, and Redis service
│   ├── docker-compose.yml         # Container setup (never run)
│   ├── Dockerfile                 # API image (never run)
│   └── requirements.txt           # API dependencies
├── ml-training/
│   ├── XGBoost/
│   │   ├── train_large_model.py   # 500k-row grid search with cProfile
│   │   └── training_results.md    # Run log
│   ├── pytorchMPStest/
│   │   ├── pytorch_mps_test.py    # CIFAR-10 CNN on MPS
│   │   └── training_results.md    # Run log
│   ├── memory_stress.py           # Allocates six 2 GB arrays
│   └── profile_me.py              # Python loop vs NumPy timing demo
├── rag-system/
│   └── rag_pipeline.py            # Embeddings and ChromaDB search
├── requirements.txt               # Shared dependencies
├── LICENSE                        # MIT License
└── README.md
```

</details>

### Notable Design Decisions

- **Native services instead of containers:** PostgreSQL and Redis ran directly on macOS. This kept setup light, but the Compose setup remains unproven.
- **One shared virtual environment:** A single venv kept setup simple and saved disk space.
- **Parallelism inside each XGBoost fit:** `GridSearchCV` uses `n_jobs=1` while XGBoost uses every core, so grid search doesn't compete with XGBoost for CPU threads.
- **Synthetic data:** Generated data kept the project self-contained. It suits load testing, but says little about real-world model quality.

## Getting Started

This project was only run on macOS with Apple Silicon. Windows and Linux are untested. The setup commands in this README have not been verified from a clean install.

### Prerequisites

- Python 3.11 (developed on 3.11.7)
- An Apple Silicon Mac for MPS. Other machines fall back to CPU and train more slowly.
- OpenMP on macOS, which XGBoost needs: `brew install libomp`
- PostgreSQL and Redis, for the API only
- At least 16 GB of RAM for `memory_stress.py`, which allocates 12 GB

### 1. Clone

```bash
git clone https://github.com/Anj-13/apple-silicon-ml-stress-test.git
cd apple-silicon-ml-stress-test
```

### 2. Configure

Create a virtual environment and install both dependency files. The second file provides `redis` and `psutil`, which the root file doesn't list.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt -r fullstack-app/requirements.txt
```

For the API, start PostgreSQL and Redis, then create the role and database the app expects by default:

```bash
brew services start postgresql
brew services start redis
psql postgres -c "CREATE ROLE \"user\" WITH LOGIN PASSWORD 'password';"
psql postgres -c "CREATE DATABASE appdb OWNER \"user\";"
```

These credentials are local development defaults only. To use different connection settings, set these environment variables:

| Variable | Default |
| --- | --- |
| `DATABASE_URL` | `postgresql://user:password@localhost/appdb` |
| `REDIS_HOST` | `localhost` |

### 3. Build and Run

There is no build step. Run each workload from its own folder:

| Workload | Folder | Command | Creates |
| --- | --- | --- | --- |
| XGBoost grid search | `ml-training/XGBoost` | `python train_large_model.py` | `xgboost_model.pkl` |
| PyTorch CNN | `ml-training/pytorchMPStest` | `python pytorch_mps_test.py` | `data/` (CIFAR-10 download) |
| RAG pipeline (see [known issues](#testing--quality)) | `rag-system` | `python rag_pipeline.py` | `chroma_db/` |
| Memory stress | `ml-training` | `python memory_stress.py` | Nothing |
| NumPy timing demo | `ml-training` | `python profile_me.py` | Nothing |
| API | `fullstack-app` | `uvicorn app:app --reload` | `papers` table in PostgreSQL |

**Expected result:** Each script prints its progress and results to the console. The API runs at `http://127.0.0.1:8000`, with interactive docs at `/docs`.

Models, datasets, the vector store, and profiling output are all git-ignored.

## Usage

With the API running:

1. Create a paper. The response contains the full record.

   ```bash
   curl -X POST http://127.0.0.1:8000/papers \
     -H "Content-Type: application/json" \
     -d '{"title":"Example paper","author":"me","year":2026}'
   ```

2. Read it back. `POST` has already cached the title, so this request is served from Redis.

   ```bash
   curl http://127.0.0.1:8000/papers/1
   # {"id":1,"title":"Example paper","cached":true}
   ```

3. Check current memory usage.

   ```bash
   curl http://127.0.0.1:8000/stats
   ```

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service message and timestamp |
| `POST` | `/papers` | Create a paper and cache its title |
| `GET` | `/papers/{id}` | Read a paper, using the cache when possible |
| `GET` | `/stats` | Report system memory usage |

`memory_stress.py` holds its arrays until you press Enter, which leaves time to inspect Activity Monitor.

## Testing & Quality

- **Tests:** Automated tests are not implemented. `pytest` is listed in `requirements.txt`, but the repository has no test files.
- **Validation:** Pydantic validates `POST /papers` request bodies, and `GET /papers/{id}` returns `404` for an unknown ID. Both paper endpoints and the Redis cache hit were checked manually.

**Known issues**

- `rag-system/rag_pipeline.py` uses `metadata=`, `query_text=`, and `results['metadata']`. ChromaDB's API uses `metadatas=`, `query_texts=`, and `results['metadatas']`, so these calls need updating before the script can complete.
- Every RAG document embeds the same repeated text, because the generated abstract is never stored. The run proves the pipeline works, but its result ranking is meaningless.
- `docker-compose.yml` sets `REDIS_URL`, but `app.py` reads `REDIS_HOST`. The API container would look for Redis on `localhost` and fail to reach it.
- `GET /papers/{id}` returns only `id`, `title`, and `cached`, even on a cache miss.
- `train_large_model.py` prints `xgboosst_model.pkl`, but the file it actually saves is `xgboost_model.pkl`.
- `profile_me.py` isn't a like-for-like comparison: the loop runs 1M iterations, the others run 10M, and `fast_function` computes a different expression.

## Challenges & Solutions

| Challenge | Approach | Outcome |
| --- | --- | --- |
| `import xgboost` failed with a missing `libomp.dylib` | Installed the OpenMP runtime | XGBoost loaded and trained |
| macOS `spawn` multiprocessing broke data loading | Added an `if __name__ == '__main__':` guard | CNN training ran with 4 `DataLoader` workers |
| PostgreSQL reported `role "user" does not exist` | Created a role and database matching the connection string | The API connected to the database |
| Library API changes in SQLAlchemy 2.0, Pydantic v2, and ChromaDB | Moved to `sqlalchemy.orm.declarative_base`, `model_dump()`, and `SentenceTransformerEmbeddingFunction` | The API endpoints worked on current library versions |
| `psutil` and Activity Monitor reported very different memory use | Compared both views across the sessions | Showed that swap and compression explain much of the gap |

## Results & What I Learned

- **Outcome:** The API, XGBoost, and PyTorch workloads all ran to completion. Memory pressure reached yellow at peak but never red, so the machine was stressed without being exhausted.

| Workload | Result |
| --- | --- |
| XGBoost (500k rows, 36 fits) | About 103–107 s per run; best 3-fold CV accuracy **0.9879** on synthetic data |
| XGBoost profiling | 6,575,523 function calls, with about 88–91 s of a ~103 s run in native `xgboost.update` |
| PyTorch CNN on MPS | 3 epochs in 107.50–111.03 s. MPS worked, but the model did not learn: loss stayed near 2.30, about random guessing for 10 classes |
| RAG pipeline | 1,000 documents inserted and a top-5 query run, with no runtime recorded |
| Memory stress | 12 GB allocated; `psutil` reported at most 9.74 GB used of 17.18 GB total |

Activity Monitor snapshots:

| Snapshot | Memory used | Share of 16 GB | Swap | Compressed | Largest Python process (Memory / Real Mem) |
| --- | --- | --- | --- | --- | --- |
| Idle baseline | 7.98 GB | 50% | 3.09 GB | 473.8 MB | None visible |
| PyTorch MPS training | 11.52 GB | 72% | 3.06 GB | 383.6 MB | 1.92 GB / 1.04 GB |
| Sustained load | 12.16 GB | 76% | 2.98 GB | 412.5 MB | 1.94 GB / 1.06 GB |
| Peak | **13.78 GB** | **86%** | **6.48 GB** | **8.13 GB** | **11.20 GB / 1.55 GB** |

- **Skills gained:** Reading macOS memory metrics, profiling native-code hot paths, MPS training, vector search with ChromaDB, read-through caching, and adapting to library API changes.
- **Key finding:** No single number describes memory use on macOS. At peak, the largest Python process showed 11.20 GB in Activity Monitor's Memory column but only 1.55 GB of Real Mem, because the rest had been compressed or swapped out.
- **Reflection:** Logging memory for each workload as it ran would have been more reliable than matching system-wide readings to sessions afterwards.

## Future Improvements

- [ ] Fix the CNN training recipe (learning rate, momentum, or a schedule) so loss actually falls.
- [ ] Correct the ChromaDB calls and embed distinct documents so search results mean something.
- [ ] Add a real dataset and retrieval evaluation to measure RAG quality.
- [ ] Align the Redis settings and test the Compose stack.
- [ ] Cache and return full paper records so the API response stays consistent.
- [ ] Add `pytest` coverage for the API to catch regressions.
- [ ] Use a larger dataset that genuinely approaches the 16 GiB limit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
