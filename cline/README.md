# Cline CLI Integration

Connect [Cline CLI](https://github.com/cline/cline) to your local vLLM server for AI-powered coding with zero cloud dependency.

## Setup

```bash
# Automatic (recommended) — auto-detects model and configures everything
make setup-cline

# Manual
cline auth --provider openai --apikey local-vllm --modelid openai/gpt-oss-120b --baseurl http://localhost:8000/v1
```

## How It Works

Cline CLI supports OpenAI-compatible API providers. Since vLLM exposes the same `/v1/chat/completions` API, we configure Cline to point at `localhost:8000` instead of `api.openai.com`.

The setup script:
1. Checks that vLLM is running and healthy
2. Auto-detects the model name from `/v1/models`
3. Patches `~/.cline/data/globalState.json` for both Act and Plan modes
4. Sets a dummy API key (vLLM doesn't require auth)
5. Verifies the configuration

## Configuration

The setup script configures both **Act mode** and **Plan mode** to use your local vLLM.

Key settings (stored in `~/.cline/data/globalState.json`):

| Setting | Value |
|---------|-------|
| `actModeApiProvider` | `openai` |
| `openAiBaseUrl` | `http://localhost:8000/v1` |
| `actModeOpenAiModelId` | Auto-detected (e.g., `openai/gpt-oss-120b`) |

See `cline_config_example.json` for a full example.

## Usage

```bash
# One-shot query
cline "Explain this error: CUDA out of memory"

# Interactive session
cline

# With a specific task
cline "Refactor the auth module to use JWT tokens"
```

## Costs

Running locally = **$0/token**. The DGX Spark handles all inference on-device.
