---
name: halt-bench-setup
description: HALT-Bench setup wizard. Use when the user says "set everything up", "get started", or wants to configure HALT-Bench end-to-end (env vars, Docker images, vLLM server, running the benchmark).
allowed-tools: Bash Read Write Edit
---

See README.md for details. Run all commands yourself; only ask for credentials, Drive links, and run preferences.

1. `pip show halt-bench 2>/dev/null|grep -q Name||pip install -e .`
2. `ls tasks/|grep -c instance_` — if 0, tell user to put task folders from Drive into `tasks/`.
3. Ask ".tar files or build?" Tar: `for f in /dir/*.tar;do docker load -i "$f";done` Build: `python build_halt_bench_images.py`
4. `[ -f .env ]||cp .env.example .env` Ask: proxy (LITELLM_BASE_URL+KEY) or direct (provider key); vllm or proxy oracle. Write into `.env`.
5. vLLM: tell user to run `bash start_vllm.sh` in a new terminal. Verify: `curl $(grep VLLM_BASE_URL .env|cut -d= -f2)/models`
6. Ask model, mode, tasks. `python run_halt_bench.py [--all-tasks|--instance-id ID...] --model M --mode MODE --solver-backend B --grader-backend B --ask-human-backend B --max-concurrency 5`
7. `python3 -c "import csv;r=list(csv.DictReader(open('outputs/results.csv')));print(len(r),'tasks,',sum(x['task_completed']=='True'for x in r),'done')"`
