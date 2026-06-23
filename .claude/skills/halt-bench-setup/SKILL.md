---
name: halt-bench-setup
description: HALT-Bench setup wizard. Use when the user says "set everything up", "get started", or wants to configure HALT-Bench end-to-end (env vars, Docker images, vLLM server, running the benchmark).
allowed-tools: Bash Read Write Edit
---

See README.md. Run all commands yourself; only ask for credentials and preferences.

1. **Images.** Ask whether the user has pre-built `.tar` files from Google Drive or wants to build from DockerHub. If tars: ask for the path and `docker load -i` each file. If building: run `python build_halt_bench_images.py`.

2. **Environment.** If `.env` doesn't exist, copy `.env.example`. Ask: (a) solver/grader backend — proxy (LITELLM_BASE_URL + KEY) or direct provider key (Anthropic, OpenAI, etc.)? (b) ask_human oracle — vLLM or proxy? Write answers into `.env`.

3. **vLLM server (if chosen).** Tell the user to open a separate terminal, run `bash start_vllm.sh`, and confirm it's up before proceeding.

4. **Run.** Ask which model, mode (`ask_human`/`full_info`), and all tasks or specific IDs. Run `python run_halt_bench.py` with `--all-tasks` or `--instance-id`, `--model`, `--mode`, backend flags, and `--max-concurrency 5`. In `ask_human` mode add `--with-ask-guidance ask_guidance --with-custom-tool` for best results
