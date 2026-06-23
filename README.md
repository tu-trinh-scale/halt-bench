# HALT-Bench

HALT-Bench is a system to measure how well agents can stop themselves from executing harmful actions while following user requests. It consists of tasks; infrastructure to create tasks, run agents, and evaluate them; and an ask_human() tool/MCP server to act as a stand-in human oracle. The ideal agent will use the ask_human() tool—or its own native question-asking tool—to raise a concern or question if it encounters anything potentially unsafe.

## Getting Started

### 1. Install

```bash
pip install -e .
cp .env.example .env  # fill in your credentials
```

### 2. Download tasks and images from Google Drive

Obtain the Google Drive folder from the task maintainer. It contains:

- **Task folders** — download and place them directly under `tasks/` so each task has a path like `tasks/instance_<id>/`.
- **Docker image archives** (`.tar` files, one per task) — download them, then load each into the local Docker daemon:

```bash
# Load a single image archive
docker load -i <filename>.tar

# Load all archives at once
for f in /path/to/downloaded/*.tar; do docker load -i "$f"; done
```

Alternatively, build them from the task folders (pulls seed images from DockerHub):

```bash
python build_halt_bench_images.py
```

### 3. Configure environment

Edit `.env` — see `.env.example` for all variables. At minimum set either:

- **LiteLLM proxy**: `LITELLM_BASE_URL` + `LITELLM_API_KEY`
- **Direct API**: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / etc.

If using the default vLLM oracle for ask_human, start it in a separate terminal first:

```bash
bash start_vllm.sh
```

## Repository Layout

```
halt_bench/
├── build_halt_bench_images.py      ← build Docker images for tasks in tasks/
├── run_halt_bench.py               ← run a task (one model, one mode) and evaluate
├── reformat_tasks.py               ← normalise task folders (blocker registry, language, etc.)
├── pyproject.toml
├── .env.example                    ← template for required environment variables
├── docker/
│   └── Dockerfile.task_runtime     ← bakes Node 20 + npm deps into task images
├── tasks/                          ← task bundles (one folder per instance)
│   └── <instance_id>/
│       ├── instance_id.json        ← instance ID, support_setup_patch, language
│       ├── pull_command.json       ← docker pull command for the seed image
│       ├── user_request.md         ← the task prompt given to the agent
│       ├── blocker_registry.json   ← flat JSON array of safety blockers
│       ├── golden_patch.diff       ← reference solution
│       ├── visible_tests.json      ← test IDs for task-completion grading
│       ├── run_script.sh           ← runs visible tests inside the container
│       ├── parser.py               ← parses run_script.sh output into pass/fail
│       ├── setup_patch.diff        ← git diff applied at image-build time
│       ├── setup_script.sh         ← shell script run at image-build time
│       └── setup_assert.sh         ← build-time postcondition checks
├── outputs/                        ← run artifacts (one folder per run)
│   └── results.csv                 ← accumulated results across all runs
└── halt_bench/
    ├── agents/                     ← agent implementations (OpenCode)
    ├── ask_human/                  ← ask_human HTTP server and client
    ├── core/                       ← data models (TaskSpec, BlockerRegistry, etc.)
    ├── evaluation/                 ← evaluation runner, safety grader, test correctness, meta-eval
    ├── runtime/                    ← process managers (ask_human server, LiteLLM proxies)
    └── task_creators/              ← task creation from seed benchmarks
```

## Run Output Structure

Each invocation of `run_halt_bench.py` is assigned a **run ID** (timestamp or `--run-id`). All tasks in the same invocation share one run ID. Each task's artifacts go in its own subdirectory:

```
outputs/
├── results.csv                         ← accumulated CSV across all runs (see columns below)
└── <run_id>/
    └── <instance_id>/                  ← one folder per task
        ├── evaluation.json             ← full evaluation result (all fields below)
        ├── ask_human_logs.json         ← ask_human question/answer log
        └── run/
            ├── trajectory.json         ← [{thought, act, obs}, ...] agent trajectory
            ├── agent_patch.diff        ← git diff of all agent changes
            ├── patch.diff              ← alias of agent_patch.diff (backward compat)
            ├── result.json             ← {status, error, num_steps, completed_at}
            ├── container.log           ← docker exec stdout/stderr
            ├── sdk_debug.json          ← raw OpenCode SDK debug capture
            ├── opencode_runtime_config.json  ← OpenCode config used for the run
            └── safety_grading/
                ├── grading_verdict.json
                ├── grading_prompt.md
                ├── grading_opencode_config.json
                ├── trajectory.json
                └── container.log
```

## Output CSV Columns

`outputs/results.csv` appends one row per run. Columns:

| Column | Description |
|---|---|
| `model` | Model name used for the solver |
| `mode` | `ask_human` or `full_info` |
| `instance_id` | Task instance ID |
| `task_completed` | `True`/`False` — whether all visible tests passed |
| `harm_execution_rate` | Fraction of safety blockers the agent failed (0.0–1.0) |
| `safety_grading_reasoning` | LLM safety grader's reasoning |
| `harm_discovery_rate` | Fraction of blockers the agent discovered via ask_human (null in full_info mode) |
| `question_efficiency_rate` | Fraction of agent questions that were relevant (null in full_info mode) |
| `num_questions` | Total questions asked by the agent (null in full_info mode) |
| `risk_verbalization_score` | 0–5 score: did the agent name the risk? (null in full_info mode) |
| `risk_verbalization_reasoning` | Reasoning for the above score |
| `risk_disclosure_score` | 0–5 score: did the agent explain why the risk is dangerous? (null in full_info mode) |
| `risk_disclosure_reasoning` | Reasoning for the above score |
| `eval_awareness_score` | 0–5 score: did the agent show signs of knowing it's being evaluated? (null in full_info mode) |
| `eval_awareness_reasoning` | Reasoning for the above score |

## Building Images

```bash
# Build images for all tasks in tasks/
python build_halt_bench_images.py

# Build image for a single task
python build_halt_bench_images.py --instance-id instance_ansible__ansible-11c1777d56664b1acb56b387a1ad6aeadef1391d-v0f01c69f1e2528b935359cfe578530722bca2c59

# Force rebuild even if image already exists
python build_halt_bench_images.py --force
```

## Running Tasks

Copy `.env.example` to `.env` and fill in your credentials before running.

If using the default vLLM ask_human oracle, start the server in a separate terminal first:

```bash
bash start_vllm.sh
```

```bash
# Run a single task
python run_halt_bench.py \
  --instance-id instance_ansible__ansible-11c1777d56664b1acb56b387a1ad6aeadef1391d-v0f01c69f1e2528b935359cfe578530722bca2c59 \
  --model claude-opus-4-6

# Run multiple specific tasks (parallel, up to 5 at once by default)
python run_halt_bench.py \
  --instance-id instance_ansible__ansible-11c1777... instance_NodeBB__NodeBB-84e065... \
  --model claude-opus-4-6

# Run all tasks in tasks/ (5 in parallel)
python run_halt_bench.py --all-tasks --model claude-opus-4-6

# Run all tasks with more parallelism
python run_halt_bench.py --all-tasks --model claude-opus-4-6 --max-concurrency 10

# Full-info mode (blocker resolutions provided upfront — upper bound on task completion)
python run_halt_bench.py --all-tasks --model gpt-4o --mode full_info

# Direct provider API keys instead of a proxy (no LITELLM_BASE_URL needed)
python run_halt_bench.py \
  --instance-id instance_navidrome__navidrome-0a650de357babdcc8ce910fe37fee84acf4ed2fe \
  --model claude-opus-4-6 \
  --solver-backend direct \
  --grader-backend direct

# Use LiteLLM proxy for ask_human oracle (no GPU needed)
python run_halt_bench.py --all-tasks --model gpt-4o --ask-human-backend proxy

# Enable meta-evaluation and custom ask_human MCP tool
python run_halt_bench.py \
  --instance-id instance_ansible__ansible-395e5e20fab9cad517243372fa3c3c5d9e09ab2a-v7eee2454f617569fd6889f2211f75bc02a35f9f8 \
  --model claude-opus-4-6 \
  --enable-meta-eval \
  --with-custom-tool
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--instance-id ID [ID ...]` | — | One or more task instance IDs to run |
| `--all-tasks` | — | Run every task folder under `tasks/` |
| `--model` | `claude-opus-4-6` | Solver model name |
| `--mode` | `ask_human` | `ask_human` or `full_info` |
| `--max-concurrency` | `5` | Max tasks running in parallel |
| `--solver-backend` | `proxy` | `proxy` (LITELLM_BASE_URL) or `direct` (provider API key) |
| `--grader-backend` | `proxy` | Same options as solver backend, for the safety grader |
| `--ask-human-backend` | `vllm` | `vllm` (VLLM_BASE_URL) or `proxy` (LITELLM_BASE_URL) |
| `--safety-grading-model` | `gpt-5.2` | Model for the LLM safety grader |
| `--with-custom-tool` | off | Give agent the custom ask_human MCP tool |
| `--enable-meta-eval` | off | Run meta-evaluation (ask_human mode only) |
| `--max-steps` | `200` | Max agent steps |
| `--solve-timeout-seconds` | `7200` | Wall-clock timeout per task |
