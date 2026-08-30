# Adaptive Multi-LLM Coding Agent

## Project Title

**Adaptive Multi-LLM Coding Agent with Token-Aware Routing and Context Compression**

### Alternative Research-Oriented Title

**Token-Efficient Multi-LLM Routing for Autonomous Software Engineering Agents**

---

# 1. Project Overview

The goal is to build a terminal-based autonomous coding agent similar to Claude Code, but with an additional intelligence layer that dynamically selects between multiple cloud-based LLMs based on:

- Task difficulty
- Expected model performance
- Token usage
- Model cost
- Remaining quota/session limits
- Latency
- Verification results

The system should also minimize token consumption through:

- Conversation summarization
- Relevant-context selection
- Minimal-context/"Caveman" mode
- Automatic context compression
- Verification-driven escalation

The central research question is:

> **Can an autonomous coding agent maintain the performance of a strong frontier model while significantly reducing inference cost and token consumption through adaptive model routing and context optimization?**

---

# 2. Core Architecture

```text
                         ┌───────────────────┐
                         │    Terminal CLI   │
                         │                   │
                         │  "fix this bug"   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Task Analyzer   │
                         │                   │
                         │ difficulty        │
                         │ domain            │
                         │ files             │
                         │ context required  │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Model Router    │
                         │                   │
                         │ cost              │
                         │ tokens            │
                         │ capability        │
                         │ remaining quota   │
                         └─────────┬─────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
           Gemini Flash        DeepSeek          Claude/GPT
              cheap             medium             expensive
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   ▼
                         ┌───────────────────┐
                         │   Agent Loop     │
                         │                   │
                         │ read/edit/run     │
                         │ test/git          │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    Verifier       │
                         │                   │
                         │ tests / compiler  │
                         │ diff / errors     │
                         └─────────┬─────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                       PASS                 FAIL
                         │                   │
                         ▼                   ▼
                       DONE              ESCALATE
```

---

# 3. Context Management Architecture

Token optimization is a major component of the project.

```text
                    Conversation
                         │
                         ▼
                ┌─────────────────┐
                │ Context Manager │
                └───────┬─────────┘
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Recent     Summary     Relevant
          turns      memory       files
             │          │          │
             └──────────┼──────────┘
                        ▼
                   Model prompt
```

The system should avoid blindly sending the entire conversation and repository context to every model.

---

# 4. Model Strategy

The system should prioritize **cloud/non-local models**.

Initial providers:

- Gemini
- DeepSeek
- Claude
- OpenAI

The architecture should be provider-agnostic.

```text
providers/
    gemini.py
    deepseek.py
    anthropic.py
    openai.py
```

Each provider should implement a common interface:

```python
class ModelProvider:

    def generate(self, messages):
        ...

    def stream(self, messages):
        ...

    def get_usage(self):
        ...

    def get_limits(self):
        ...

    def estimate_cost(self, tokens):
        ...
```

This allows models to be added or removed without modifying the routing system.

---

# 5. Model Quota and Token Dashboard

The CLI should provide real-time usage information.

Example:

```text
╭──────────────────────────────────────────────────╮
│                 MODEL USAGE                      │
├────────────┬───────────┬───────────┬─────────────┤
│ Model      │ Used      │ Remaining │ Context     │
├────────────┼───────────┼───────────┼─────────────┤
│ Gemini     │ 12.4k     │ 87.6k      │ 128k        │
│ DeepSeek   │ 8.2k      │ 41.8k      │ 64k         │
│ Claude     │ 5.7k      │ 14.3k      │ 200k        │
│ GPT        │ 3.1k      │ 26.9k      │ 128k        │
╰────────────┴───────────┴───────────┴─────────────╯

Current session:
Input:       4,821 tokens
Output:      1,392 tokens
Total:       6,213 tokens

Estimated cost: $0.014
```

CLI commands:

```bash
agent usage
agent usage --today
agent usage --session
agent usage --model claude
```

## Important limitation

Not every provider exposes exact remaining quota information.

The system should therefore distinguish between:

```text
EXACT
ESTIMATED
UNKNOWN
```

For example:

```text
Claude
Tokens used: 12,430
Session limit: UNKNOWN
```

This is preferable to inventing quota information.

---

# 6. Token Accounting

Every model request should produce a usage record.

Example:

```json
{
  "model": "gemini",
  "input_tokens": 4812,
  "output_tokens": 1392,
  "total_tokens": 6204,
  "latency_ms": 1840,
  "success": true,
  "escalated": false
}
```

Store usage locally using SQLite:

```text
~/.adaptive-agent/usage.db
```

Suggested tables:

```text
sessions
requests
models
tasks
summaries
```

Track:

- Input tokens
- Output tokens
- Total tokens
- Model
- Task
- Latency
- Success/failure
- Estimated cost
- Escalation count
- Context compression ratio

---

# 7. Context Compression

Long coding-agent conversations can become extremely expensive.

For example:

```text
Turn 1: 2k tokens
Turn 2: 4k tokens
Turn 3: 6k tokens
Turn 4: 8k tokens
Turn 5: 7k tokens
Turn 6: 9k tokens
```

Instead of sending all 36k tokens again, summarize older context.

```text
Old conversation
       │
       ▼
Summary model
       │
       ▼
Compact state
```

Example summary:

```text
PROJECT STATE

Goal:
Fix memory corruption in BFS implementation.

Files modified:
- Solution.cpp
- test.cpp

Current hypothesis:
visited nodes are marked after enqueue rather than before.

Completed:
- identified race in queue insertion
- added regression test

Remaining:
- verify disconnected graph case
- run full test suite

Important constraints:
- O(V+E)
- C++17
```

Potential result:

```text
36k tokens → 1.5k tokens
```

The actual reduction should be measured experimentally.

---

# 8. Minimal Context / "Caveman Mode"

The user should be able to explicitly request an extremely minimal prompt.

CLI:

```bash
agent --minimal
```

Academic terminology:

> **Minimal Context Mode**

or

> **Token-Minimal Mode**

Instead of:

```text
system prompt
+ project context
+ conversation
+ relevant files
+ tool history
+ instructions
+ task
```

use:

```text
system prompt
+ task
+ relevant code
+ essential constraints
```

Example:

```text
Claude

Normal:
12,401 input tokens

Minimal:
4,182 input tokens

Reduction:
66.3%
```

Display this directly:

```text
TOKEN OPTIMIZATION

Before:     12,401
After:       4,182

Saved:       8,219 tokens
Reduction:    66.3%
```

---

# 9. Three Context Modes

Implement three modes.

## Normal

```text
Full conversation
Relevant project context
Tool history
Recent files
```

## Compressed

```text
Summary
+
Relevant files
+
Recent conversation
```

## Minimal

```text
Task
+
Minimum required code
+
Essential constraints
```

The system should benchmark all three modes.

---

# 10. Automatic Context Strategy

Context management should become automatic.

```text
                    Model
                      │
                      ▼
             Is model expensive?
                /           \
              YES           NO
               │             │
               ▼             ▼
         aggressive       normal
         compression      context
               │
               ▼
          token budget
               │
          ┌────┴────┐
          ▼         ▼
       enough    insufficient
          │         │
          ▼         ▼
         send     summarize
```

For example:

```text
Claude
↓
Minimal context
↓
4k tokens
```

while:

```text
Gemini Flash
↓
Normal context
↓
12k tokens
```

This creates a relationship between model cost and context size.

---

# 11. Difficulty Classifier

Use a cheap cloud model to analyze incoming tasks.

Example prompt:

```text
Classify this software engineering task.

Return JSON:

{
  "difficulty": 1-5,
  "reasoning_required": 1-5,
  "files_likely_affected": integer,
  "security_risk": 0-1,
  "architecture_change": 0-1
}
```

Initial routing policy:

```text
Difficulty 1–2 → Gemini Flash
Difficulty 3   → DeepSeek
Difficulty 4   → DeepSeek / Claude
Difficulty 5   → Claude / GPT
```

However, difficulty should not be the only routing feature.

A short security-sensitive task may still require a stronger model.

---

# 12. Model Routing

Each model should have metadata:

```python
Model(
    capability,
    input_cost,
    output_cost,
    context_limit,
    remaining_quota,
    latency
)
```

Each task should have:

```python
Task(
    difficulty,
    estimated_tokens,
    files,
    risk,
    domain
)
```

The router can initially use a utility function:

\[
Score(m) =
P(success|task,m)
-\lambda Cost(m)
-\mu Tokens(m)
-\gamma Latency(m)
\]

Select:

```python
best_model = max(models, key=score)
```

Initially, `P(success)` can be estimated heuristically.

Later, it can be learned from collected task results.

---

# 13. Verification-Driven Escalation

This should be one of the core features.

```text
             Task
              │
              ▼
          Cheap model
              │
              ▼
           Modify
              │
              ▼
          Run tests
              │
          ┌───┴───┐
        PASS     FAIL
          │        │
          ▼        ▼
        DONE     Escalate
                   │
                   ▼
                Stronger
                   │
                   ▼
                  Test
```

Example:

```text
Gemini Flash
   ↓
3,281 tokens
   ↓
FAIL
   ↓
DeepSeek
   ↓
5,821 tokens
   ↓
FAIL
   ↓
Claude
   ↓
4,192 tokens
   ↓
PASS
```

Record:

```text
initial model
escalation count
final model
tokens consumed
success/failure
```

Example:

```text
Task #43

Gemini Flash    3,281 tokens    FAIL
DeepSeek        5,821 tokens    FAIL
Claude          4,192 tokens    PASS

Total: 13,294 tokens
Escalations: 2
```

---

# 14. Adaptive Routing

Eventually the system should learn from its own results.

Example:

```text
Task 184
Gemini → failure

Task 185
Gemini → success

Task 186
DeepSeek → success

...
```

Estimate:

\[
P(success \mid task, model)
\]

Then choose:

\[
m^* =
\arg\max_m
[
P(success|x,m)
-\lambda C_m
-\mu L_m
]
\]

This transforms model selection into an ML decision problem.

---

# 15. Seven-Day Development Plan

# Day 1 — Agent Foundation

## Goal

Get this working:

```bash
agent "fix this bug"
```

Implement:

- CLI
- Repository detection
- File reading
- Cloud model API
- Response streaming
- Basic file modification
- Shell command execution
- Basic testing

Do not implement routing yet.

### Deliverable

```text
Terminal
    ↓
LLM
    ↓
Edit
    ↓
Test
```

---

# Day 2 — Multi-Model Abstraction

Implement:

```text
Provider
├── Gemini
├── DeepSeek
├── Claude
└── OpenAI
```

All providers should expose:

```python
generate()
stream()
estimate_tokens()
get_usage()
get_limits()
```

Add:

```bash
agent models
```

Example:

```text
MODEL             STATUS       CONTEXT
Gemini Flash      AVAILABLE    ...
DeepSeek          AVAILABLE    ...
Claude            AVAILABLE    ...
GPT               AVAILABLE    ...
```

---

# Day 3 — Token Accounting

Implement SQLite storage.

Track:

```text
Input tokens
Output tokens
Total tokens
Latency
Model
Task
Success
Cost
```

Add:

```bash
agent usage
agent usage --today
agent usage --session
agent usage --model claude
```

This provides the data required for later experiments.

---

# Day 4 — Model Routing

Implement:

```text
Task Analyzer
       ↓
Difficulty
       ↓
Model Router
```

Start with deterministic routing:

```text
Difficulty 1–2 → Gemini
Difficulty 3   → DeepSeek
Difficulty 4–5 → Claude
```

Then incorporate:

- Remaining quota
- Estimated tokens
- Cost
- Latency
- Model capability

---

# Day 5 — Token Optimization

Implement:

## A. Conversation Summarization

```text
old messages
      ↓
summary
      ↓
compact context
```

## B. Relevant Context Selection

Don't send the entire repository.

Send:

```text
task
+
relevant files
+
relevant symbols
```

## C. Minimal Context Mode

```bash
agent --minimal
```

Measure:

```text
Normal tokens
Minimal tokens
Reduction %
```

This should be a major focus of the project.

---

# Day 6 — Verification and Escalation

Implement:

```text
Model
 ↓
Modify
 ↓
Run tests
 ↓
Success?
 ├── YES → Done
 └── NO  → Escalate
```

Record:

```text
Initial model
Escalation count
Final model
Tokens consumed
Success/failure
```

Add:

```bash
agent history
```

---

# Day 7 — Evaluation and Polish

Run 30–100 coding tasks.

Compare:

### Baseline A

Always use cheapest model.

### Baseline B

Always use strongest model.

### Baseline C

Static difficulty router.

### Baseline D

Adaptive router.

### Baseline E

Adaptive router + context compression.

Measure:

| System | Success | Tokens | Cost | Latency |
|---|---:|---:|---:|---:|
| Cheapest model | | | | |
| Strongest model | | | | |
| Static router | | | | |
| Adaptive router | | | | |
| Adaptive + compression | | | | |

---

# 16. Main Research Experiment

The strongest experiment should answer:

> **How much coding-agent performance can be retained while reducing token consumption and inference cost?**

Example format:

```text
                         Success      Input Tokens

Always Claude              82%           100%

Adaptive Router             80%            61%

Adaptive +
Compression                 79%            38%
```

The numbers above are hypothetical.

Actual results should be obtained from experiments.

The important metrics are:

- Success rate
- Test pass rate
- Input tokens
- Output tokens
- Total tokens
- Cost
- Latency
- Number of escalations

---

# 17. Evaluation Dataset

Use public software-engineering benchmarks where practical.

Potential sources:

- SWE-bench
- HumanEval
- MBPP
- CodeRepair-style datasets
- Curated GitHub issues
- Custom coding tasks

For every task, record:

```text
task
difficulty
language
repository size
files touched
model used
input tokens
output tokens
latency
tests passed
final success
escalation count
```

---

# 18. Suggested Repository Structure

```text
adaptive-coding-agent/
│
├── agent/
│   ├── planner/
│   ├── executor/
│   ├── verifier/
│   ├── patcher/
│   └── context_manager/
│
├── router/
│   ├── features.py
│   ├── difficulty.py
│   ├── model_selector.py
│   └── policy.py
│
├── providers/
│   ├── gemini.py
│   ├── deepseek.py
│   ├── anthropic.py
│   └── openai.py
│
├── evaluation/
│   ├── benchmarks.py
│   ├── metrics.py
│   └── experiments.py
│
├── datasets/
│
├── dashboard/
│
├── cli/
│
├── tests/
│
└── README.md
```

---

# 19. Demo Interface

The README should show an example like:

```text
$ agent "Fix the failing BFS tests"

Analyzing task...
Difficulty: 3/5

Selected: Gemini Flash
Reason: low complexity + low estimated token cost

Context:
  Original:   11,842 tokens
  Compressed:  4,193 tokens
  Saved:        64.6%

Running agent...

✓ Modified Solution.cpp
✓ Tests: 18/18

Total:
  Input:       4,193
  Output:      1,027
  Total:       5,220
  Escalations: 0
```

Hard task:

```text
$ agent "Refactor authentication and fix concurrency issue"

Difficulty: 5/5

Selected: Claude

Context optimization:
  Original:   31,402
  Sent:         7,841
  Saved:        75.0%

✓ Changes applied
✓ Tests passed
```

---

# 20. Important Design Principle

Do not make the system:

```text
Easy → Gemini
Medium → DeepSeek
Hard → Claude
```

and stop there.

The stronger architecture is:

```text
                  Task
                   │
                   ▼
             Cheap analysis
                   │
                   ▼
            Select cheapest
            capable model
                   │
                   ▼
                Execute
                   │
                   ▼
              Run tests
             /         \
          PASS         FAIL
           │             │
           ▼             ▼
         DONE         Escalate
                         │
                         ▼
                    Stronger model
```

This is:

> **cheap → attempt → verify → escalate**

rather than:

> **classify → gamble → hope**

---

# 21. Research Contributions

The project should emphasize three contributions:

## 1. Adaptive Model Routing

Automatically select an LLM based on:

- Task difficulty
- Expected success
- Cost
- Token budget
- Latency
- Remaining quota

## 2. Token-Efficient Context Management

Reduce prompt size using:

- Conversation summarization
- Relevant-context extraction
- Context compression
- Minimal Context Mode

## 3. Verification-Driven Escalation

Use actual execution/test results to decide whether a stronger model is necessary.

---

# 22. Resume Positioning

Avoid:

> Built a CLI tool similar to Claude Code.

Avoid:

> Built an AI coding assistant using Gemini and Claude.

Instead, aim for:

> **Developed a multi-LLM autonomous coding agent with difficulty-aware model routing, verification-driven escalation, and token-budget optimization across Gemini, DeepSeek, Claude and OpenAI models.**

Second bullet:

> **Designed hierarchical context compression and minimal-context inference, reducing prompt token consumption by X% while retaining Y% task success across Z software-engineering tasks.**

Third bullet:

> **Implemented real-time per-model token/cost accounting and quota-aware routing using SQLite, with automatic escalation based on test failures and model confidence.**

Only insert actual numbers after benchmarking.

---

# 23. Final One-Week Deliverable

By the end of the week, the project should contain:

```text
✓ Terminal coding agent
✓ Multiple cloud LLM providers
✓ Automatic task difficulty estimation
✓ Cost-aware model routing
✓ Token accounting
✓ Session/quota tracking where APIs expose it
✓ Conversation summarization
✓ Relevant-context selection
✓ Minimal Context Mode
✓ Automatic escalation
✓ Test-based verification
✓ SQLite usage database
✓ Benchmarking framework
✓ Cost/token/performance comparison
✓ GitHub README
✓ Experimental results
```

---

# 24. Final Project Concept

The final system can be summarized as:

```text
                 TOKEN-AWARE
              AUTONOMOUS CODING
                    AGENT
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   MODEL ROUTING   CONTEXT          ESCALATION
                   COMPRESSION
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              COST / PERFORMANCE
                 OPTIMIZATION
```

The central objective is:

> **Achieve frontier-model coding performance under a constrained token and inference budget.**

This framing turns the project from a simple **Claude Code clone** into a **cost-aware LLM systems/ML research project**.