# PACE: Policy for Adaptive Compute Efficiency

**When to Stop Thinking: Adaptive Compute Allocation for Multi-Turn LLM Orchestration**

Aljumayli & Madisetti — Georgia Institute of Technology

---

## Prerequisites

- Python 3.10+
- An OpenAI API key (for GPT-4o agent + embeddings)
- An Anthropic API key (for Claude Sonnet agent in dual-agent S5 experiments)
- A Tavily API key (for web search tool — free tier: https://app.tavily.com)
- ~4GB disk for the DistilRoBERTa NLI model (downloads on first S5 run)
- No GPU required — NLI model runs on CPU in ~30ms per pair

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd pace

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install PACE + all dependencies
pip install -e .

# Install dev tools (pytest, ruff)
pip install -e ".[dev]"

# (Optional) Install τ²-Bench for that benchmark
pip install -e ".[tau2]"
```

### 2. Set API keys

```bash
cp .env.template .env
```

Edit `.env` with your actual keys:

```
OPENAI_API_KEY=sk-...          # Required — GPT-4o agent + embeddings
ANTHROPIC_API_KEY=sk-ant-...   # Required for dual-agent (S5) experiments
TAVILY_API_KEY=tvly-...        # Required — web search tool
```

The DistilRoBERTa NLI model (`cross-encoder/stsb-distilroberta-base`) downloads
automatically from HuggingFace on the first run that uses S5. No HF token needed
(it's a public model).

### 3. Verify installation

```bash
# Run the test suite (no API keys needed — uses mock embedder)
pytest tests/ -v

# Should see 19 tests pass covering:
#   - All 5 signals + derivatives
#   - Signal masking (ablation)
#   - Policy decisions (stop/continue/rewind/scale)
#   - Confident conflict detection
#   - Ablation condition generation (31 subsets)
```

### 4. Verify API connectivity

```bash
# Quick smoke test — one question, single agent, 3 turns
python -c "
from dotenv import load_dotenv; load_dotenv()
from agents.single_react import SingleReActAgent
from tools import get_frames_tools
from pace import SignalComputer, PACEPolicy, PolicyConfig

agent = SingleReActAgent(model='gpt-4o-mini', max_turns=3, tools=get_frames_tools())
answer, traj, decisions = agent.run('What is the capital of France?', pace_policy=PACEPolicy(PolicyConfig(convergence_window=1, signal_mask={'answer_similarity'})))
print(f'Answer: {answer}')
print(f'Turns used: {traj.current_turn}, Max Turns: 3')
for i, t in enumerate(traj.turns):
    print(f'  Turn {t.turn_number}: sim={t.answer_similarity:.3f}, entropy={t.token_entropy:.3f}')
    decision = decisions[i].decision.name if i < len(decisions) else "none"
    print(f'  Decision: {decision}')
"
```

If this prints an answer and signal values, you're good to go.


## Project Structure

```
pace/
├── pace/                        # Core library
│   ├── trajectory.py            #   TurnState + Trajectory (5 signals, derivatives)
│   ├── signals.py               #   SignalComputer (S1-S4, mask-aware)
│   ├── policy.py                #   PACEPolicy (relative thresholds, mask-aware)
│   └── embeddings.py            #   Embedder (OpenAI) + NLIScorer (DistilRoBERTa)
│
├── agents/                      # Agent implementations
│   ├── __init__.py              #   BaseAgent + FixedBudgetRunner
│   ├── single_react.py          #   Single-agent ReAct loop
│   ├── multi_agentflow.py       #   4-module team (Planner/Executor/Verifier/Generator)
│   └── dual_agent.py            #   DualAgentRunner (GPT-4o + Claude, computes S5)
│
├── benchmarks/                  # Benchmark harnesses
│   ├── frames.py                #   FRAMES dataset loader + evaluator
│   └── tau2.py                  #   τ²-Bench wrapper (post-hoc PACE analysis)
│
├── tools/                       # Tool implementations
│   └── __init__.py              #   web_search (Tavily), wikipedia, python executor
│
├── experiments/                 # Experiment scripts
│   ├── run_frames.py            #   Main FRAMES experiment (single/multi × fixed/PACE)
│   ├── run_tau2.py              #   τ²-Bench experiment
│   ├── run_ablation.py          #   Signal ablation (31 conditions)
│   ├── sweep_thresholds.py      #   Threshold sensitivity analysis
│   └── analyze_results.py       #   Paper tables + LaTeX output
│
├── configs/                     # YAML configurations
│   ├── pace_thresholds.yaml     #   All 5 signal thresholds + sweep ranges
│   ├── single_agent.yaml        #   Single-agent model + tool config
│   └── multi_agent.yaml         #   Multi-agent module models
│
├── tests/
│   └── test_pace.py             #   Full test suite (signals, policy, ablation)
│
├── results/                     #   Experiment outputs (gitignored)
├── pyproject.toml
├── .env.template
└── README.md
```


## Signals

| ID | Signal              | Source           | What it measures                        |
|----|---------------------|------------------|-----------------------------------------|
| S1 | answer_similarity   | OpenAI embed     | Is the answer still changing?           |
| S2 | info_gain           | OpenAI embed     | Is new context adding anything?         |
| S3 | token_entropy       | Token logprobs (fallback: output text) | Is the model confident or hedging?      |
| S4 | tool_entropy        | Tool history     | Is tool selection diversifying or stuck? |
| S5 | agent_agreement     | DistilRoBERTa NLI| Do two independent agents agree?        |

Each signal has level `s(t)`, velocity `Δs(t)`, and acceleration `Δ²s(t)`.


## Running Experiments

All experiments write JSON results to `results/`. Run them in this order:

### Step 1: Generate baseline trajectories

```bash
# Single-agent fixed budget (5 and 10 turns) — generates trajectories for ablation
python experiments/run_frames.py --mode single-fixed --num-questions 50 --model gpt-4o

# Single-agent with PACE
python experiments/run_frames.py --mode single-pace --num-questions 50 --model gpt-4o

# Multi-agent fixed budget
python experiments/run_frames.py --mode multi-fixed --num-questions 50 --model gpt-4o

# Multi-agent with PACE
python experiments/run_frames.py --mode multi-pace --num-questions 50 --model gpt-4o

# Or run everything at once:
python experiments/run_frames.py --mode all --num-questions 50 --model gpt-4o
```

### Step 2: Signal ablation study

```bash
# Replays saved trajectories through PACE with 31 signal combinations
# No new API calls — this is pure post-hoc analysis
python experiments/run_ablation.py --trajectories results/frames/
```

This produces a table showing which signals and combinations give the best turn savings.

### Step 3: Threshold sensitivity sweep

```bash
# Sweeps threshold values over saved trajectories
# Also no new API calls
python experiments/sweep_thresholds.py --trajectories results/frames/
```

### Step 4: τ²-Bench (requires tau2-bench installed)

```bash
pip install -e ".[tau2]"

# Run τ²-Bench simulations then analyze with PACE
python experiments/run_tau2.py --step all --domain airline --num-tasks 20
```

### Step 5: Generate paper tables

```bash
# Text tables
python experiments/analyze_results.py --input results/

# LaTeX tables (for the paper)
python experiments/analyze_results.py --input results/ --latex
```

### Step 6: Compare entropy signal variants (unigram vs logprobs)

```bash
# Turn-1 comparison for stop-confidence calibration across models
python experiments/compare_entropy_signals.py \
  --models gpt-4o,gpt-4o-mini \
  --num-questions 20 \
  --turns 1 \
  --agent-type single
```

Outputs:
- `results/entropy_compare/samples_*.csv` (per-question rows)
- `results/entropy_compare/summary_*.json` (AUC/effect-size metrics)
- `results/entropy_compare/entropy_comparison_*.png` (comparison graph)

### Step 7: Build 1000-task RL training mix

```bash
python experiments/build_training_dataset.py --out tasks_1000.json
```

Default composition:
- 334 HotpotQA bridge (easy)
- 333 MuSiQue (medium)
- 233 HARP levels 4-5 (hard)
- 100 GPQA Diamond (hard)

### Step 8: Run Qwen3.5-27B baseline trajectories (resumable)

```bash
python experiments/run_qwen27b_baseline.py \
  --tasks data/tasks_mixed.json \
  --output data/trajectories_1000.json \
  --max-turns 10
```

Requires `TOGETHER_API_KEY` and `together` package.

For a local vLLM server (recommended on your pod), launch:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-27B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

Then run the baseline against that endpoint:

```bash
python experiments/run_qwen27b_baseline.py \
  --backend openai_compat \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model Qwen/Qwen3.5-27B \
  --tasks data/tasks_mixed.json \
  --output data/trajectories_1000.json \
  --max-turns 10 \
  --disable-thinking
```

Or let the script start/stop local vLLM automatically:

```bash
python experiments/run_qwen27b_baseline.py \
  --backend openai_compat \
  --start-vllm \
  --vllm-python /path/to/python-with-vllm \
  --vllm-host 127.0.0.1 \
  --vllm-port 8000 \
  --vllm-tensor-parallel-size 1 \
  --model Qwen/Qwen3.5-27B \
  --tasks data/tasks_mixed.json \
  --output data/trajectories_1000.json \
  --max-turns 10 \
  --disable-thinking
```

### Step 9: Build per-turn signal dataset

```bash
python experiments/build_signal_dataset.py \
  --trajectories data/trajectories_1000.json \
  --labels data/labeled_1000.json \
  --output data/signal_dataset_1000.json
```
## Experiment Matrix

| Benchmark  | Single Agent + Fixed | Single Agent + PACE | Multi Agent + Fixed | Multi Agent + PACE |
|------------|---------------------|--------------------|--------------------|-------------------|
| FRAMES     | ✅ Run               | ✅ Run              | ✅ Run              | ✅ Run             |
| τ²-Bench   | ✅ Run               | ✅ Run              | ✅ Run              | ✅ Run             |
| HLE        | 📊 Reference         | —                  | —                  | 📊 Reference       |

## Cost Estimates

Approximate API costs for the full experiment suite at 50 questions:

| Condition                    | API Calls       | Est. Cost |
|------------------------------|-----------------|-----------|
| Single fixed-5 (50 Qs)      | 250 GPT-4o      | ~$3       |
| Single fixed-10 (50 Qs)     | 500 GPT-4o      | ~$6       |
| Single PACE (50 Qs)         | ~350 GPT-4o     | ~$4       |
| Multi fixed-5 (50 Qs)       | 750 GPT-4o      | ~$9       |
| Multi fixed-10 (50 Qs)      | 1500 GPT-4o     | ~$18      |
| Multi PACE (50 Qs)          | ~1000 GPT-4o    | ~$12      |
| Dual-agent PACE (50 Qs)     | ~700 GPT-4o + ~700 Claude | ~$15 |
| Embeddings (all runs)       | ~5000 embed calls | ~$0.50  |
| **Total**                   |                 | **~$68**  |

The ablation study and threshold sweep are free (post-hoc replay over saved trajectories).
NLI model runs locally on CPU.

Start with `--num-questions 10` to validate the pipeline before running the full 50.


## Key Design Decisions

**Why OpenAI embeddings for S1/S2 but DistilRoBERTa for S5?**
S1/S2 measure *topical drift* (is the answer changing?). Embedding cosine is perfect for this.
S5 measures *semantic agreement* (do two agents mean the same thing?). Embedding cosine fails here
because two opposite conclusions about the same topic have high cosine similarity. The NLI cross-encoder
is trained to distinguish agreement from topical overlap.

**Why relative thresholds?**
An absolute info_gain floor of 0.05 is arbitrary. Relative thresholds adapt: "gain dropped below
the recent mean minus 1 std dev" works regardless of the signal's natural scale for this episode.

**Why ablation is post-hoc?**
Running 31 signal combinations × 50 questions × live agents = thousands of API calls.
Instead, we run once at full budget, record all signals, then replay the PACE decision logic
with different masks active. Same result, zero extra API cost.

**Why unanimity for STOP?**
If 3 of 4 active signals say converged but one says "still changing," stopping is risky.
Requiring all active signals to agree prevents premature termination. The ablation study
then reveals which signals are the noisy ones that prevent convergence detection.
