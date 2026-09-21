# Agentic Fraud Investigation Agent

An agentic fraud investigation system that combines **TigerGraph**, **LangGraph**, deterministic policy rules, and optional LLM reasoning to investigate suspicious financial activity.

The system takes a fraud investigation trigger, gathers evidence from a graph and historical cases, assesses risk and uncertainty, requests additional evidence when necessary, recommends or executes policy-approved actions, explains the decision, and writes the complete case history back to the graph.

> **Status:** Initial project scaffold. Data loading, graph queries, policy rules, assessment, agent workflow, evaluation, and UI are being built incrementally.

---

## Table of Contents

- [What We're Building](#what-were-building)
- [Core Idea](#core-idea)
- [Architecture](#architecture)
- [Investigation Flow](#investigation-flow)
- [Key Design Principles](#key-design-principles)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Assessment Model](#assessment-model)
- [Policy Engine](#policy-engine)
- [Evidence Loop](#evidence-loop)
- [Graph Model](#graph-model)
- [GraphRAG and Case Memory](#graphrag-and-case-memory)
- [Output Format](#output-format)
- [Evaluation](#evaluation)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Development Roadmap](#development-roadmap)
- [Important Constraints](#important-constraints)
- [License](#license)

---

## What We're Building

The application investigates potentially fraudulent transactions using an agentic workflow.

A case can be triggered by:

- a risk score,
- a customer report,
- an analyst request,
- or another supported investigation trigger.

The agent then:

1. Opens the case.
2. Retrieves relevant graph evidence.
3. Looks for similar historical cases.
4. Assesses fraud probability and uncertainty.
5. Determines whether additional evidence is required.
6. Requests or simulates additional evidence when appropriate.
7. Reassesses the case.
8. Applies deterministic policy rules.
9. Produces recommended actions and approval routes.
10. Explains what changed between the initial and final assessment.
11. Writes the investigation back to the graph as case memory.
12. Produces a validated JSON answer file.

The goal is not simply to classify a transaction as fraud or legitimate.

The goal is to build an **auditable investigation agent** that can explain:

> What happened → what evidence was found → what is still uncertain → what additional evidence was requested → how the assessment changed → what action the policy allows → why.

---

## Core Idea

The system is deliberately divided into two responsibilities.

### Deterministic layer

Facts and rules should be reproducible and testable:

- graph queries
- transaction features
- fraud-pattern detection
- probability/scoring logic
- stopping conditions
- policy rules
- action approval routes
- output validation

### LLM layer

The LLM is used where language and synthesis are useful:

- evidence summarization
- reasoning over retrieved evidence
- undocumented-pattern descriptions
- case explanations
- customer/SAR narrative generation

The LLM should **not independently decide which policy action or approval route is allowed**.

The policy engine remains the source of truth.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │    Case Trigger     │
                         │ risk / report /     │
                         │ analyst request     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Open Case       │
                         └──────────┬──────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────┐
              │          Evidence Gathering            │
              │                                        │
              │  TigerGraph │ Historical Cases │ Docs  │
              └──────────────────┬─────────────────────┘
                                 │
                                 ▼
                         ┌─────────────────────┐
                         │     Assessment      │
                         │                     │
                         │ probability         │
                         │ confidence          │
                         │ pattern             │
                         │ evidence            │
                         │ uncertainty         │
                         └──────────┬──────────┘
                                    │
                         enough evidence?
                           /               \
                         no                 yes
                         │                   │
                         ▼                   │
                ┌─────────────────┐          │
                │ Request Evidence│          │
                └────────┬────────┘          │
                         │                   │
                         ▼                   │
                ┌─────────────────┐          │
                │ Simulate/Receive│          │
                │ Response        │          │
                └────────┬────────┘          │
                         │                   │
                         └───────┬───────────┘
                                 ▼
                           ┌──────────────┐
                           │ Reassessment │
                           └──────┬───────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Policy Engine   │
                         │                  │
                         │ action + route   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Explain       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Write Case       │
                         │ Memory to Graph  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Validated JSON   │
                         └──────────────────┘
```

---

## Investigation Flow

The agent is implemented as a state machine.

```text
trigger
   ↓
open_case
   ↓
gather
   ↓
assess
   ↓
initial_actions
   ↓
┌───────────────────────────────┐
│ Is more evidence necessary?   │
└───────────────┬───────────────┘
                │
       ┌────────┴────────┐
       │                 │
      yes                no
       │                 │
       ▼                 ▼
request_evidence       explain
       │                 │
       ▼                 │
simulate                │
       │                 │
       └──────► assess   │
                         │
                         ▼
                   write_memory
                         │
                         ▼
                        END
```

The evidence loop is intentionally bounded so the agent cannot investigate indefinitely.

---

## Key Design Principles

### 1. Evidence before conclusions

The agent should investigate the connected entities and transaction history before reaching a final conclusion.

### 2. Uncertainty is explicit

An assessment contains:

- probability,
- verdict,
- pattern,
- evidence,
- independent evidence count,
- conflicting evidence,
- confidence/sufficiency information.

### 3. New evidence can change the recommendation

Every case has an `initial` and `final` stage.

The output records:

```text
initial assessment
       ↓
additional evidence
       ↓
final assessment
       ↓
what_changed
```

### 4. Policy is code

Rules and approval routes are implemented in deterministic code rather than delegated entirely to an LLM.

### 5. Everything is auditable

Important operations should produce:

- evidence references,
- tool-call counts,
- token usage,
- latency,
- action routes,
- case IDs,
- graph references.

### 6. Avoid over-blocking

The benchmark contains legitimate cases, so a system that treats every suspicious signal as fraud will perform poorly.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data processing | Pandas, NumPy |
| Columnar storage | PyArrow / Parquet |
| ML utilities | scikit-learn |
| Agent orchestration | LangGraph |
| Graph database | TigerGraph |
| TigerGraph Python client | pyTigerGraph |
| LLM | Anthropic API (optional) |
| Backend API | FastAPI |
| Frontend | React / Next.js |
| Testing | pytest |
| Configuration | `.env` |

### Backend modes

The application supports two intended backend modes:

```text
BACKEND=pandas
```

for local development without TigerGraph, and:

```text
BACKEND=tigergraph
```

for the full graph-backed system.

---

## Project Structure

```text
agentic-agent/
│
├── agent/
│   ├── __init__.py
│   ├── state.py
│   ├── tools.py
│   ├── assess.py
│   ├── policy.py
│   ├── simulator.py
│   ├── graph_flow.py
│   └── memory.py
│
├── cases/
│   └── *.json
│
├── data/
│   ├── *.csv
│   └── *.parquet
│
├── docs/
│   └── ...
│
├── eval/
│   ├── __init__.py
│   └── replay.py
│
├── graph/
│   ├── schema.gsql
│   └── queries/
│       ├── card_window.gsql
│       ├── card_history_profile.gsql
│       ├── region_history.gsql
│       ├── device_neighbors.gsql
│       ├── region_neighbors.gsql
│       ├── recurring_charge_check.gsql
│       └── similar_closed_cases.gsql
│
├── loader/
│   ├── __init__.py
│   ├── inspect_data.py
│   └── load.py
│
├── outputs/
│   ├── __init__.py
│   ├── writer.py
│   └── validate.py
│
├── tests/
│   ├── __init__.py
│   ├── test_policy.py
│   ├── test_scoring.py
│   └── test_validator.py
│
├── ui/
│   └── ...
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Assessment Model

The central assessment object stores the investigation state.

Conceptually:

```python
Assessment(
    p=0.0,
    verdict="uncertain",
    pattern="none",
    pattern_description="",
    affected_txn_ids=[],
    first_suspicious_txn_id="",
    connected_card_ids=[],
    connected_device_profiles=[],
    exposure=0.0,
    evidence=[],
    similar_prior_cases=[],
    signals={},
    n_independent=0,
)
```

It also stores policy-relevant flags:

```text
customer_response
customer_disputes
recurring_match
card_testing
cleared_over_100
single_signal
shared_element
coordinated
credentials_compromised
conflicting
trigger_type
```

This allows the assessment layer and policy layer to remain separate.

---

## Risk Assessment

The scoring system should use interpretable signals such as:

- card-testing sequence
- tiny authorization sequence
- new device
- proxy/device indicators
- first use of a billing region
- amount relative to card history
- shared device
- shared region
- shared email/recipient
- recurring-charge match
- transaction risk score
- connected historical fraud cases

The system should record the contribution of individual signals:

```python
signals = {
    "card_testing": 1.2,
    "new_device": 0.3,
    "shared_device": 1.1,
}
```

The assessment should also track independent evidence categories.

This is important because multiple correlated signals should not automatically be treated as multiple independent pieces of evidence.

---

## Policy Engine

Policy decisions are deterministic.

Conceptually:

```text
Assessment
    │
    ▼
Policy Engine
    │
    ├── action
    ├── route
    └── reason
```

Routes are:

```text
auto
L1
L2
```

Examples of action categories include:

```text
ALLOW_TRANSACTION
MONITOR_CARD
MONITOR_CONNECTED_CARDS
WARN_CUSTOMER
VERIFY_WITH_CUSTOMER
STEP_UP_AUTH
GENERATE_REPORT
CREATE_CASE
ESCALATE_TO_ANALYST
CLOSE_NO_FRAUD
DECLINE_TRANSACTION
BLOCK_CARD
BLOCK_ALL_CARDS
FILE_REPORT
```

The exact policy rules must match the benchmark README and should be covered by unit tests.

---

## Evidence Loop

When the initial evidence is insufficient, the agent can request additional evidence.

Example:

```text
Initial assessment
    │
    ├── suspicious card-testing sequence
    ├── new device
    └── uncertain customer intent
             │
             ▼
     Request verification
             │
             ▼
      Customer response
             │
       ┌─────┴─────┐
       │           │
    confirmed    denied
       │           │
       ▼           ▼
 legitimate      fraud
       │           │
       └─────┬─────┘
             ▼
        Reassessment
```

Each evidence request should record the assumed/simulated response.

Example:

```json
{
  "request_id": "REQ-001",
  "type": "customer_verification",
  "reason": "Customer confirmation is required to resolve uncertainty.",
  "assumed_response": "denied"
}
```

The simulator should be rule-based rather than randomly generating responses.

---

## Graph Model

The graph stores entities and relationships relevant to fraud investigation.

### Vertices

```text
Customer
Card
Transaction
DeviceProfile
BillingRegion
EmailDomain
ClosedCase
```

### Relationships

```text
Customer ──OWNS──────────────► Card
Card ──MADE──────────────────► Transaction
Transaction ──FROM_DEVICE────► DeviceProfile
Transaction ──BILLED_IN──────► BillingRegion
Transaction ──PURCHASER_EMAIL► EmailDomain

ClosedCase ──INVOLVES────────► Transaction
ClosedCase ──ON_CARD─────────► Card
ClosedCase ──CONNECTED_TO────► Card
```

### Example investigation

```text
Card
 │
 ├── Transaction A
 │      └── Device X
 │
 ├── Transaction B
 │      └── Device X
 │
 └── Transaction C
        └── Region Y

Device X
 │
 ├── Card A
 ├── Card B
 └── Card C
```

This makes shared devices and connected cards directly queryable.

---

## Graph Queries

The initial feature-query set includes:

### `card_window`

Returns transactions around a suspicious transaction.

Used for:

- velocity
- transaction sequences
- tiny authorizations
- temporal investigation

### `card_history_profile`

Builds a profile of normal card behavior.

Used for:

- typical transaction amounts
- regions
- devices
- product codes

### `region_history`

Checks whether the card has previously used a billing region.

### `device_neighbors`

Finds other cards connected through the same device.

### `region_neighbors`

Finds cards connected through a shared billing region.

### `recurring_charge_check`

Detects recurring transactions using merchant/amount/time patterns.

### `similar_closed_cases`

Retrieves relevant historical cases.

---

## GraphRAG and Case Memory

The application combines two retrieval mechanisms.

### Vector retrieval

Searches:

- closed-case narratives
- fraud-pattern descriptions
- policy documents
- regulatory references

### Graph retrieval

Expands from:

- cards
- transactions
- devices
- regions
- connected cases

Together:

```text
                    Case
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Vector Search        Graph Expansion
          │                     │
          └──────────┬──────────┘
                     ▼
             Evidence Packet
                     │
                     ▼
                    LLM
```

The LLM should receive a compact evidence packet rather than an entire raw dataset.

---

## Case Memory

After an investigation, the case is written back to the graph.

The resulting case can contain:

```text
Case
 ├── trigger
 ├── evidence
 ├── initial assessment
 ├── evidence requests
 ├── customer response
 ├── final assessment
 ├── actions
 ├── explanation
 └── metrics
```

Future cases can retrieve relevant historical investigations.

---

## Output Format

Each benchmark case produces an individual JSON file.

Example:

```text
cases/
├── HHG-001.json
├── HHG-002.json
├── HHG-003.json
└── ...
```

A case output should contain the information required by the benchmark, including:

- case ID
- initial assessment
- final assessment
- evidence
- evidence requests
- affected transaction IDs
- exposure
- actions
- approval routes
- SAR status
- explanation
- graph case ID
- tool-call metrics
- token usage
- latency

The output validator should verify:

- required fields exist
- action names are valid
- IDs exist
- SAR status agrees with `FILE_REPORT`
- exposure matches affected transaction amounts
- legitimate cases are not incorrectly assigned affected transactions
- graph case IDs exist

---

## Evaluation

The evaluation system should replay historical closed cases while hiding their final outcomes from the agent.

Evaluate:

- fraud/legitimate verdict
- probability calibration
- pattern identification
- affected transactions
- exposure
- connected cards
- policy actions
- approval routes
- SAR decision
- over-blocking of legitimate cases

The closed history should be treated as historical evidence rather than blindly used as a probability model, especially because the benchmark distribution differs from the closed-case distribution.

---

## Local Setup

### 1. Clone or enter the project

```bash
cd /home/claude/agentic-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your local configuration.

### 5. Verify Python dependencies

```bash
python3 - <<'PY'
import pandas
import numpy
import pyarrow
import sklearn
import langgraph
import pyTigerGraph
import anthropic

print("All imports OK")
PY
```

---

## Environment Variables

Example configuration:

```env
# Backend
BACKEND=pandas

# TigerGraph
TG_HOST=https://YOUR-WORKSPACE.i.tgcloud.io
TG_GRAPH=FraudGraph
TG_USER=tigergraph
TG_PASS=
TG_CLOUD=1
TG_TOKEN=

# LLM
ANTHROPIC_API_KEY=
LLM_MODEL=claude-sonnet-5
USE_LLM=1

# Card mapping
CARD_STRATEGY=
```

### Backend

```env
BACKEND=pandas
```

Use this during local data exploration.

```env
BACKEND=tigergraph
```

Use this for the full graph-backed application.

### LLM

The LLM is optional for development.

When disabled, the application should fall back to deterministic/template-based explanations.

---

## Data

Place the benchmark CSV files inside:

```text
data/
```

The repository intentionally ignores CSV and Parquet data files because benchmark datasets should not be committed to Git.

Before building the loader, inspect the actual dataset:

```bash
python -m loader.inspect_data
```

The inspection process should establish:

- available CSV files
- row counts
- column names
- transaction ID format
- customer/card relationship
- device fields
- billing-region fields
- case IDs
- relationships between datasets

This is especially important for resolving card IDs used by the case data against transaction records.

---

## Development Roadmap

### Phase 1 — Data

- [ ] Inspect dataset
- [ ] Confirm CSV schemas
- [ ] Confirm transaction ID format
- [ ] Resolve customer/card mapping
- [ ] Build local data loader
- [ ] Store full transaction rows in Parquet

### Phase 2 — TigerGraph

- [ ] Create graph schema
- [ ] Load customers
- [ ] Load cards
- [ ] Load transactions
- [ ] Load devices
- [ ] Load regions
- [ ] Load email domains
- [ ] Load historical cases
- [ ] Verify graph relationships

### Phase 3 — Investigation Queries

- [ ] Card transaction window
- [ ] Card history profile
- [ ] Region history
- [ ] Device neighbors
- [ ] Region neighbors
- [ ] Recurring charge detection
- [ ] Similar historical cases

### Phase 4 — Policy

- [ ] Implement R1
- [ ] Implement R2
- [ ] Implement R3
- [ ] Implement R4
- [ ] Implement R5
- [ ] Implement R6
- [ ] Implement R7
- [ ] Implement R8
- [ ] Implement R9
- [ ] Implement R10
- [ ] Add unit tests
- [ ] Add action-route validation

### Phase 5 — Assessment

- [ ] Build interpretable scoring
- [ ] Add independent evidence counting
- [ ] Add uncertainty detection
- [ ] Add conflicting evidence detection
- [ ] Add stopping logic

### Phase 6 — Agent

- [ ] Implement LangGraph state
- [ ] Implement `open_case`
- [ ] Implement `gather`
- [ ] Implement `assess`
- [ ] Implement initial actions
- [ ] Implement evidence requests
- [ ] Implement response simulation
- [ ] Implement reassessment
- [ ] Implement final actions
- [ ] Implement explanation
- [ ] Implement graph memory

### Phase 7 — Output

- [ ] JSON writer
- [ ] JSON schema validation
- [ ] ID validation
- [ ] SAR validation
- [ ] Exposure validation
- [ ] Metrics instrumentation
- [ ] Generate all benchmark cases

### Phase 8 — Evaluation

- [ ] Historical replay
- [ ] Calibration evaluation
- [ ] Policy compliance tests
- [ ] Legitimate-case tests
- [ ] Over-blocking analysis
- [ ] Tune assessment

### Phase 9 — UI

- [ ] Case list
- [ ] Case timeline
- [ ] Initial vs final assessment
- [ ] Evidence panel
- [ ] Action/approval panel
- [ ] Graph visualization
- [ ] Metrics panel

---

## Important Constraints

### Do not commit benchmark data

Keep:

```text
data/*.csv
data/*.parquet
.env
```

out of Git.

### Do not recover hidden outcomes

The benchmark data should be used according to the supplied rules. Do not use external/public datasets to recover hidden benchmark outcomes.

### Do not let the LLM bypass policy

The LLM can explain an assessment, but deterministic policy code should control the action and approval route.

### Do not equate risk score with fraud

Risk score is one signal among multiple pieces of evidence.

### Do not treat a new device as proof of fraud

A new device is a signal that needs context.

### Do not block based on one weak signal

Weak evidence should generally lead to further verification or monitoring according to the applicable policy.

### Keep every decision auditable

Every important recommendation should be traceable to evidence and policy.

---

## Testing

Run the test suite with:

```bash
pytest -q
```

Recommended test categories:

```text
tests/
├── test_policy.py
├── test_scoring.py
└── test_validator.py
```

Every policy rule should have dedicated tests.

Worked examples from the benchmark README should also become regression tests.

---

## Development Philosophy

This project prioritizes:

1. **Correctness**
2. **Policy compliance**
3. **Calibration**
4. **Explainability**
5. **Auditability**
6. **Reproducibility**
7. **Then UI polish**

A simple investigation that follows the policy correctly is more valuable than a sophisticated interface that produces unreliable decisions.

---

## License

This project is currently intended for development and evaluation. Add the appropriate license before public distribution.
