# Controlled Validation Harness

Automated benchmarking suite for validating latency, memory stability (RSS and free RAM), page faults, and concurrency throughput (`PIPELINE_SEM=4`) across 4 core retrieval paths under randomized execution order:
1. `unfiltered` (3 movie seeds, dual-signal fusion)
2. `plot_text` (3 movie seeds + natural language query)
3. `55k_filter` (Exact path over ~55k matches, e.g. Japanese language catalog)
4. `295k_filter` (Exact path over ~295k matches, e.g. Drama genre catalog)

Each case is executed twice in a deterministic, randomized sequence (8 sequential stages total), followed by an 8-request simultaneous burst through `PIPELINE_SEM=4` (single-threaded BLAS). All per-request latencies, memory footprints, and system counters are captured into a single JSON artifact: `validation_results.json`.

---

## Instructions for Running on a 16 GB VM

These steps assume a dedicated Linux VM (Ubuntu 22.04/24.04 LTS or Debian 12) with **16 GB RAM**, **4+ vCPUs**, and at least **25 GB free disk space**.

### Step 1: Clone and Checkout Branch

```bash
git clone <REPO_URL> movie_recommendation_system
cd movie_recommendation_system
git checkout latencyFix
```

---

### Step 2: System Packages & PostgreSQL Setup

1. Install system prerequisites:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-venv python3-pip git curl docker.io docker-compose-v2
   sudo systemctl enable --now docker
   ```

2. Start the PostgreSQL database container:
   ```bash
   docker compose up -d database
   ```

3. If starting from a blank database, restore TMDB data from `database/moviedb.sql.gz`:
   ```bash
   # Ensure DVC has pulled database/moviedb.sql.gz (see Step 4 below)
   gunzip -c database/moviedb.sql.gz | docker compose exec -T database psql -U postgres -d moviedb
   ```

---

### Step 3: Python Environment & Dependencies

1. Navigate to `backend` and initialize a virtual environment:
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Configure environment file (`backend/.env`):
   ```bash
   cat << 'EOF' > .env
   DATABASE_URL=postgresql://postgres:postgres@localhost:15432/moviedb
   APP_ENV=dev
   ALLOWED_HOSTS=["localhost","127.0.0.1"]
   ARTIFACTS_DIR=artifacts
   ARTIFACT_VERSION=v1
   CHROMA_DIR=chroma_data
   CORS_ORIGINS=["http://localhost:5173"]
   GEMINI_API_KEY=dummy_key_for_bench
   EOF
   ```
   *(Note: `GEMINI_API_KEY` is not required for retrieval/ranking benchmarks).*

---

### Step 4: Pull ML Artifacts with DVC

From the repository root, pull the precomputed artifacts:
```bash
cd ..  # return to repo root
dvc pull
```

Verify that the following directories and files are present:
- `backend/artifacts/v1/` (`matrix.npz`, `meta_index.npz`, `row_norms.npy`, `plot_embeddings.bin`, `plot_ids.npy`, `inv_norms.npy`)
- `backend/chroma_data/`
- `database/moviedb.sql.gz`

---

### Step 5: Run the Validation Benchmark

Activate the backend virtual environment:
```bash
cd backend
source .venv/bin/activate
```

#### A. Quick Smoke Test (Verifies complete pipeline in ~15 seconds)
```bash
python bench/run_validation.py --smoke
```

#### B. Full Validation Run
```bash
# Standard run (25 distinct queries per stage × 8 stages + 2 concurrency bursts):
python bench/run_validation.py --queries-per-case 25

# Or extended run (50 queries per stage):
python bench/run_validation.py --queries-per-case 50
```

CLI Options:
- `--queries-per-case N`: Number of distinct queries per run (default: `25`).
- `--concurrency-iterations N`: Number of 8-request concurrent bursts (default: `2`).
- `--seed N`: Fixed RNG seed (default: `42`).
- `--output PATH`: JSON destination (default: `backend/bench/validation_results.json`).
- `--smoke`: Minimal dry-run (1 query per case, 1 concurrency burst).

---

### Step 6: Verify and Send Back Results

1. Verify that the output JSON file was created:
   ```bash
   ls -lh bench/validation_results.json
   ```

2. Review the printed summary table in the terminal:
   - Sequential run p50/p95/max/mean latency per case
   - Free RAM at start vs end (must remain > 1.0 GB on 16 GB VM)
   - Concurrency throughput (`req/s`) under `PIPELINE_SEM=4`
   - RSS growth and page fault count

3. Transfer the file back:
   ```bash
   # Example: copy via scp or attach to report
   scp user@vm-ip:~/movie_recommendation_system/backend/bench/validation_results.json ./
   ```
