# Timecell.AI Summer Internship 2025 - Technical Assessment

**Name:** Sri Sai Madhava Teja Yanduri 

**College ID:** 202301138

**College Name:** Dhirubhai Ambani University 

**Program:** B-Tech ICT, 3rd Year  

## Introduction: My Approach
When I started this assessment, my goal wasn't just to write scripts that output the correct numbers. I wanted to approach this with the mindset of an Engineer. In wealth management domain, codes that mostly works is dangerous. 

Throughout these four tasks, my thinking process centered on three core principles:
1.  **Defensive Engineering:** Financial data cannot rely on assumptions. I focused on eliminating hardcoded values (like FX rates), enforcing zero floors on asset drops, and catching network failures gracefully.
2.  **Architectural Separation:** Large Language Models are great at language but terrible at math. I made a strict rule early on: Python does 100% of the math, and the LLM only explains the verified results. 
3.  **Scalability over Quick Fixes:** Whether it was moving from `for` loops to vectorized `NumPy` arrays, or switching from synchronous `requests` to concurrent `httpx`, I wanted the architecture to hold up just as well for 1,000 assets as it does for 4. This repository documents not just the final code, but the iterations, mistakes, and architectural pivots I made along the way.

Throughout the 72-hours provided, I tused Perplexity to clarify my doubts in terms of both the financial domain along with coding, I am providing my chat so a rough idea of how my intution and development took place can be viewed

<https://www.perplexity.ai/search/9ef1b5b8-1390-4438-8a8f-88e51bedf8da>
---

## Table of Contents
1. **[Task 1: Portfolio Risk Calculator](#task-1-portfolio-risk-calculator)** 
   * *Moving from manual loops to NumPy vectorization for scale.*
2. **[Task 2: Live Market Data Fetch](#task-2-live-market-data-fetch)** 
   * *Handling network fragility with concurrency, retries, and strict typing.*
3. **[Task 3: AI-Powered Portfolio Explainer](#task-3-ai-powered-portfolio-explainer)** 
   * *Taming LLMs with native structured outputs and mitigating self-grading bias.*
4. **[Task 4: The Open Problem - "What-If" Stress Tester](#task-4-the-open-problem---what-if-stress-tester)** 
   * *Building a terminal-native, NLP-driven scenario engine.*

---
## Task 1: Portfolio Risk Calculator

### What We Are Doing
Building a quantitative engine to stress-test an investment portfolio. The script calculates post-crash values, estimates financial runway, and flags concentration risks based on different macroeconomic scenarios.

### Why We Are Doing It (The Thought Process)
When I first looked at this task, the easiest approach would have been a simple `for`-loop iterating over the asset dictionary. But in wealth management, portfolios don't have 4 assets; they have 400 or 4,000. 

I knew that a `for`-loop implementation would scale linearly (at best) and become a bottleneck. Therefore, my primary architectural decision was to **vectorize the math**. 

By utilizing `NumPy`, I transformed the asset dictionary into parallel arrays. This allows the CPU to perform operations on the entire portfolio simultaneously at C-level speeds, rather than processing asset by asset. 

### How We Are Doing It (Implementation Details)

#### 1. NumPy Vectorization
Instead of looping through assets to calculate their post-crash values, I extract the data into aligned arrays:
```python
allocations = np.array([a["allocation_pct"] for a in assets])
crashes = np.array([a["expected_crash_pct"] for a in assets])
```
This enables single-line, highly optimized calculations:
```python
asset_values = total_value * (allocations / 100)
post_crash_values = asset_values * (1 + crashes / 100)
```
*Performance note: My benchmarking showed that while NumPy has a slight initialization overhead for portfolios under 50 assets, it yields a **~19.5x speedup** for portfolios with 1,000+ assets.*

#### 2. Defensive Math (The Zero-Floor)
A critical edge case in financial modeling is extreme drops. If an asset is modeled to crash by 150%, basic math yields a negative value. Since these are unleveraged assets, they cannot drop below zero. I enforced a strict zero-floor:
```python
new_asset_value = max(0.0, current_asset_value * multiplier)
```

#### 3. Algorithmic Efficiency
To find the asset with the highest risk magnitude, I avoided manual tracking variables and instead used `np.argmax()`:
```python
risk_magnitudes = allocations * np.abs(crashes)
largest_risk_asset = names[int(np.argmax(risk_magnitudes))]
```

#### 4. Terminal-Native UI

| Feature / Aspect | `termgraph` (The Graphing Library) | `tabulate` (The Data Table Library) |
| :--- | :--- | :--- |
| **Primary Purpose** | Rendering actual charts and graphs directly in the terminal. | Creating beautifully aligned, dynamic text grids and data tables. |
| **Version Used** | `termgraph>=0.5.3` | `tabulate>=0.9.0` |
| **What it supports** | Horizontal bar charts, vertical bar charts, stacked charts, and histograms with titles and labels. | Dozens of table formats (grid, simple, HTML, GitHub markdown, rounded outlines). |
| **How it works** | Uses ASCII/Unicode block characters (like `▇` and `▏`) to draw proportional bars based on the numeric data you feed it. | Calculates the maximum width of your data in each column and automatically pads the text with spaces and border characters (`+`, `-`, `|`) so everything aligns perfectly. |
| **Project Use Case** | Used for **Asset allocation visualization** (e.g., showing what percentage of the portfolio is BTC vs. GOLD). | Used for the **Scenario Comparison** in Task 1 and the **Live Asset Prices** in Task 2. |
| **Why we chose it over alternatives** | It is the most feature-rich option for pure CLI charts without requiring an external GUI or pop-up window. | It is extremely lightweight (~50KB) and perfectly handles dynamic column widths, unlike manual string formatting (which breaks easily) or heavy libraries like `rich`. |

---
## Task 2: Live Market Data Fetch

### What We Are Doing
Building a real-time data pipeline that fetches live prices for diverse asset classes (Cryptocurrency, Indian Equities, and Gold) from multiple free public APIs (CoinGecko and Yahoo Finance) and renders them in a unified CLI table.

### Why We Are Doing It (The Thought Process)
Fetching data from an API is easy; building a financial data pipeline that survives production is hard. When I looked at this task, I identified several failure vectors that a naive implementation would struggle with:
1.  **Network Fragility:** APIs drop connections, rate-limit, or time out. A single failure shouldn't crash the entire pipeline.
2.  **The "Local Clock" Lie:** Using `datetime.now()` when the script runs tells you when the code executed, not when the market price was valid. 
3.  **Hardcoded FX Rates:** Hardcoding `$1 = ₹83` to convert Gold prices is a cardinal sin in wealth management. It introduces immediate, compounding errors.

I wanted to build an architecture that assumes the network is hostile, demands exact market timestamps, and executes concurrently to minimize latency.

### How We Are Doing It (Implementation Details)

#### 1. Concurrency (`httpx` + `asyncio`)
Instead of fetching assets sequentially (which wastes time waiting for network I/O) or using artificial `time.sleep()` delays, I migrated the pipeline to asynchronous execution. 
Using `httpx.AsyncClient` and `asyncio.gather()`, the script fires requests to Yahoo Finance and CoinGecko simultaneously. The entire fetch process now takes only as long as the single slowest API response (roughly ~800ms), making it 3x faster than sequential fetching.

#### 2. Resilience via Exponential Backoff (`tenacity`)
Instead of writing messy `try/except` loops, I used the `tenacity` library to wrap the fetchers in a declarative retry decorator.
```python
@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3)
)
```
If an API throws a transient 503 error or times out, the script automatically waits 1s, 2s, and 4s before giving up, ensuring transient hiccups don't break the user experience.

#### 3. Honest Timestamps & Eliminating FX Math
To maintain strict data integrity:
*   **No manual FX math:** For Gold, instead of calculating the INR price locally, I leverage CoinGecko's native multi-currency support (`vs_currencies=usd,inr`) to get the exact real-time exchange conversion directly from the source.
*   **Honest Timestamps:** I extract the Unix timestamps directly from the API payloads (e.g., `regularMarketTime` from Yahoo Finance). The CLI table shows exactly when the price was valid in the market, not when the user pressed "Enter".

#### 4. The Strategy Pattern for Scalability
If the founders ask to add Silver or Platinum tomorrow, the core logic shouldn't need to change. I separated the fetch logic from the asset configuration using an `AssetConfig` dataclass and isolated transform functions. 
```python
AssetConfig("GOLD", "coingecko", "pax-gold", "INR/g", _troy_oz_to_per_gram)
```
Combined with strict `Pydantic` models (`AssetPrice`) to replace raw, error-prone dictionaries, this ensures the system is type-safe and infinitely scalable to new assets.
---
## Task 3: AI-Powered Portfolio Explainer

### What We Are Doing
Building an AI-driven reporting engine using Google Gemini 2.5 Flash to translate raw quantitative risk metrics into plain-English financial advice. It features configurable tones (Beginner, Experienced, Expert) and a bonus "LLM-as-a-Judge" meta-review system.

### Why We Are Doing It (The Thought Process)
When integrating LLMs into financial tools, there are two massive pitfalls I wanted to avoid: **Hallucinations** and **Brittle Parsing**.
1.  **Hallucinations:** If you ask an LLM to calculate a portfolio's post-crash value, it will often invent a number that *looks* correct. In wealth management, this is catastrophic. 
2.  **Brittle Parsing:** Asking an LLM to "Return ONLY JSON" and then using regex to scrape the response out of the text is a 2023-era pattern. It fails frequently when models add preambles like "Here is your analysis:".

My approach was to use a **Deterministic Validation Layer**. Python does 100% of the math. The LLM is strictly a narrative engine.

### How We Are Doing It (Prompt Engineering & Architecture)

#### 1. The System / User Split
Most jailbreaks and instruction-following failures happen when developers stuff all instructions into the `user` prompt. I strictly separated the prompts:
*   **`system_instruction`:** Contains the persona ("You are a financial advisor"), the dynamic tone instructions, and the hard constraints (e.g., specific rules for what constitutes an "Aggressive" verdict). The API prioritizes this layer heavily.
*   **`user` message:** Contains *only* the raw portfolio data and the pre-calculated metrics injected as "ground truth".

#### 2. Native Structured Outputs (Pydantic + Gemini)
To eliminate regex, I defined the exact output schema using Pydantic `BaseModel` classes (e.g., `PortfolioAnalysis`).
```python
class PortfolioAnalysis(BaseModel):
    risk_summary: str = Field(...)
    verdict: Literal["Aggressive", "Balanced", "Conservative"] = Field(...)
```
By passing `PortfolioAnalysis.model_json_schema()` directly into Gemini's `response_json_schema` configuration, the model is physically constrained at the API level to only generate valid JSON that matches my exact types. I then use `model_validate_json` for type-safe parsing.

#### 3. Mitigating Bias in the LLM-as-a-Judge (Meta-Review)
For the bonus critique feature, I used a second Gemini API call to review the first call's analysis. However, LLMs have a known "self-serving bias" they almost always rate their own output a 10/10. To fix this:
*   **Different Persona:** The critique prompt explicitly casts the LLM as an "independent senior reviewer".
*   **Strict Grading Constraints:** I forced integer scoring (1-10) for accuracy and appropriateness, as research shows LLMs are more calibrated with integer scales than vague holistic grades.
*   **Temperature Control:** While the main analysis runs at `temperature=0.7` for narrative flow, the critique API call forces `temperature=0.1` to ensure analytical, deterministic grading.

---
## Task 4: The Open Problem - "What-If" Stress Tester

### What We Are Doing
Instead of building a generic, static rebalancing dashboard, I built a **Terminal-Native Natural Language Stress Tester**. A wealth manager can type a chaotic, real-world macroeconomic scenario directly into the CLI (e.g., *"What happens if Bitcoin gets banned and NIFTY drops by 15%?"*), and the system dynamically extracts the parameters, executes the math safely, and generates a verified survival brief.

### Why We Are Doing It (The Thought Process)
As the assignment explicitly mentioned that Timecell.AI runs inside a terminal, not a dashboard. I wanted to build a feature that felt native to that environment. Dashboards force users to click sliders, AI-native CLI tools should let users just ask questions.

However, building this required solving the core LLM hallucination problem. If you ask an LLM, "What happens to my ₹10M portfolio if Bitcoin crashes?", it will try to do the math and inevitably hallucinate. To solve this, I architected a strict **Three-Layer Pipeline** (Extract -> Execute -> Narrate) that completely separates language processing from mathematical computation.

### How We Are Doing It (Implementation Details)

#### Layer 1: Agentic NLP Extraction
I use Gemini 2.5 Flash strictly as a routing and extraction engine. I defined a Pydantic model (`ShockVector`) that expects negative integers for market drops.
```python
class ShockVector(BaseModel):
    BTC: int = Field(default=0, description="Percentage change for Bitcoin...")
    NIFTY50: int = Field(default=0, description="Percentage change for NIFTY50...")
```
By passing this schema into `response_json_schema` with `temperature=0.0`, Gemini translates the user's panicked English into a deterministic, strictly typed Python dictionary (e.g., `{"BTC": -100, "NIFTY50": -15}`). The LLM does absolutely zero financial math here.

#### Layer 2: Deterministic Quant Execution
The extracted `shock_vector` is passed to a pure Python function (`apply_shock`). This is where the actual risk calculation happens. 
I specifically engineered a **Zero Floor** constraint here:
```python
multiplier = 1 + (shock_pct / 100.0)
new_asset_value = max(0.0, current_asset_value * multiplier)
```
If a user asks, *"What if a bug causes Bitcoin to drop 150%?"*, standard math would result in a negative portfolio value. Unleveraged assets cannot drop below zero, so `max(0.0, ...)` guarantees financial reality is maintained.

#### Layer 3: Grounded Narration
Finally, the script makes a second call to Gemini 2.5 Flash. I inject the verified Python math (Post-Crash Value, Total Loss, and Runway) as "Ground Truth Metrics" into the prompt. The LLM is instructed to write a strict 3-sentence "Survival Brief" summarizing the damage using *only* the provided numbers, alongside one defensive recommendation.

### The Result
The output is an auditable, visually distinct terminal UI where the user can clearly see what the AI extracted (Layer 1), what Python calculated (Layer 2), and the final advisory brief (Layer 3). It turns the CLI into a conversational, yet mathematically bulletproof, risk engine.

---
## ⚙️ Getting Started (How to Run)

### Prerequisites
1. **Python 3.10+** installed on your system.
2. A valid **Google Gemini API Key** (Free tier is sufficient, but note rate limits).

### Step 1: Environment Setup
Clone the repository and set up your virtual environment to keep dependencies clean:
```bash
git clone <your-repo-url>
cd timecell-assessment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
Install all required packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys
Create a `.env` file in the root directory (do not commit this file). Add your Gemini API key:
```env
GEMINI_API_KEY="your_actual_api_key_here"
```

---

## 🚀 Execution Guide

Here is the rundown on how to run each individual task from your terminal.

### Task 1: Vectorized Portfolio Risk Calculator
Runs the pure Python quantitative engine against the baseline portfolio.
```bash
python task1_portfolio_risk_calculator.py
```

### Task 2: Live Market Data Fetcher
Executes the concurrent `httpx` pipeline to pull live prices from CoinGecko and Yahoo Finance.
```bash
python task2_live_market_data_fetch.py
```

### Task 3: AI-Powered Portfolio Explainer
Generates the plain-English risk narrative using Gemini. You can customize the behavior using CLI flags.

**Basic Run (Beginner Tone):**
```bash
python task3_ai_portfolio_explainer.py
```
**Advanced Run (Expert Tone + LLM-as-a-Judge Critique + Show Python Math):**
```bash
python task3_ai_portfolio_explainer.py --tone expert --critique --show-metrics
```

### Task 4: The "What-If" Stress Tester
Runs the terminal-native NLP engine. You **must** provide a natural language scenario using the `--query` flag.

**Example 1 (Crypto Focus):**
```bash
python task4_what_if.py --query "What if Bitcoin gets banned globally and NIFTY drops by 15%?"
```
**Example 2 (Macro Focus):**
```bash
python task4_what_if.py --query "Stagflation hits: stocks down 30%, gold acts as a safe haven and surges 20%"
```

