# Running inference on a remote machine

Clone PixelPolicy on a Linux box with an NVIDIA GPU and have a model play Wordle.
Example model: **`Qwen/Qwen3.5-8B`** (needs a ~24 GB GPU at fp16/bf16).

## 1. Clone + install

```bash
git clone <your-repo-url> PixelPolicy
cd PixelPolicy

# uv (one-time, if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync                      # resolves the whole workspace into one .venv
```

## 2. Configure the endpoint

```bash
cp .env.example .env
# edit .env:
#   OPENAI_BASE_URL=http://127.0.0.1:8000/v1
#   INFERENCE_MODEL=Qwen/Qwen3.5-8B
#   OPENAI_API_KEY=EMPTY
```

## 3. Start the model server (terminal A)

```bash
uv run --package inference python -m inference.server --model Qwen/Qwen3.5-8B
```

First start downloads the weights and compiles CUDA graphs (a few minutes).
Wait for `Application startup complete`, then verify (terminal B):

```bash
curl -s http://127.0.0.1:8000/v1/models      # should list Qwen/Qwen3.5-8B
```

The launcher auto-adds overridable defaults: `--max-model-len 8192`,
`--gpu-memory-utilization 0.85`, `--limit-mm-per-prompt '{"image":0,"video":0}'`,
`--attention-backend TRITON_ATTN` (no CUDA toolkit needed). Pass any flag yourself to
override — e.g. `--gpu-memory-utilization 0.90`, or `--limit-mm-per-prompt '{}'` for a
**text-only** model.

## 4. Run the agent (terminal B)

```bash
# watch one game on a colored board
uv run --package agents python -m agents.run --demo --word crane

# headless: many games, prints a win-rate summary
uv run --package agents python -m agents.run --episodes 20
```

If you serve a different model than `.env` says, add `--model <id>` to the run command
too (it must match the served id).

## Remote server, local client (host a big model elsewhere)

Run vLLM on a beefy GPU box and drive it from this machine — the agent client needs **no
GPU and no vLLM**, only the API URL. Good for hosting a large model (e.g. `Qwen/Qwen3.5-32B`)
while iterating on the agent locally.

**On the GPU box** — serve the large model, bound so it's reachable:

```bash
uv run --package inference python -m inference.server \
  --model Qwen/Qwen3.5-32B --host 0.0.0.0          # listen on all interfaces
```

**On this (client) machine** — just point the API at the remote and run the agent.

Safest is an SSH tunnel (keeps the server private, no open port):

```bash
# forwards local :8000 → remote :8000; leave running
ssh -N -L 8000:localhost:8000 user@<remote-host>
```

Then update the model API — either edit `.env`:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1     # tunnel; or http://<remote-host>:8000/v1 direct
INFERENCE_MODEL=Qwen/Qwen3.5-32B             # must match the served id
OPENAI_API_KEY=EMPTY
```

…or override per run without touching `.env`:

```bash
uv run --package agents python -m agents.run --demo --word crane \
  --base-url http://<remote-host>:8000/v1 --model Qwen/Qwen3.5-32B
```

> Binding `--host 0.0.0.0` exposes the server to the network — prefer the SSH tunnel, or
> firewall the port. vLLM accepts any non-empty API key, so don't expose it openly.

## Notes

- **VRAM too small?** Lower `--gpu-memory-utilization` or `--max-model-len`, or serve a
  smaller model (`--model Qwen/Qwen3.5-0.8B`).
- **Thinking mode** is on by default (`enable_thinking`); sampling lives in
  `agents/config.py`. See [README.md](README.md) for the per-round prompt/parse details.
- Stop the server with `Ctrl-C` (or `pkill -f "vllm serve"`).
```
