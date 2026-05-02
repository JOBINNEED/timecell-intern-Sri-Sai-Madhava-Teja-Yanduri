#!/usr/bin/env python3
"""
TASK 3: AI-Powered Portfolio Explainer
Timecell.AI Summer Internship 2025 - Technical Assessment

Uses Google Gemini API to generate plain-English portfolio risk explanations
and independent meta-review critique.

LLM Provider: Google Gemini 2.0 Flash
Why Gemini? Fast, cost-effective, excellent structured output support via
response_json_schema, and strong reasoning for financial analysis.

Two separate Gemini calls with different system prompts provide independent
evaluation — the critique call reviews the analysis objectively.

PRODUCTION PATTERNS APPLIED:
  1. Math stays in Python  — pre-calculated metrics are injected as facts; the LLM
                             explains numbers, it never computes them.
  2. System / user split  — persona, output schema, and hard constraints live in the
                             system_instruction; only the portfolio data goes in contents.
  3. Native structured output — response_json_schema enforces the JSON contract at
                             the API level, eliminating regex parsing entirely.
"""

import os
import json
import argparse
import numpy as np
import time
from typing import Dict, Literal, Optional, List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

# Load .env into the environment before anything reads os.environ
load_dotenv()


# ---------------------------------------------------------------------------
# Step 1: Python-side metric calculation (LLMs must never do the math)
# ---------------------------------------------------------------------------

def compute_risk_metrics(portfolio: Dict) -> Dict:
    """
    Calculate all quantitative risk metrics in Python before touching the LLM.

    Reuses the same vectorised logic as Task 1 so there is a single source of
    truth for every number that appears in the AI explanation.

    Returns a flat dict of pre-computed facts ready to be injected into the
    prompt as ground-truth context.
    """
    total_value      = portfolio["total_value_inr"]
    monthly_expenses = portfolio["monthly_expenses_inr"]
    assets           = portfolio["assets"]

    allocations = np.array([a["allocation_pct"]      for a in assets])
    crashes     = np.array([a["expected_crash_pct"]  for a in assets])
    names       =          [a["name"]                for a in assets]

    asset_values      = total_value * (allocations / 100)
    post_crash_values = asset_values * (1 + crashes / 100)
    post_crash_total  = float(np.sum(post_crash_values))

    runway_months = post_crash_total / monthly_expenses if monthly_expenses > 0 else float("inf")

    # Largest risk asset by (allocation × |crash magnitude|)
    risk_magnitudes  = allocations * np.abs(crashes)
    max_risk_idx     = int(np.argmax(risk_magnitudes))
    largest_risk_asset = names[max_risk_idx]

    # Concentration warning: any single asset > 40 %
    concentration_warning = bool(np.any(allocations > 40))
    concentrated_assets   = [names[i] for i, a in enumerate(allocations) if a > 40]

    # Per-asset breakdown for the prompt
    asset_breakdown = []
    for i, asset in enumerate(assets):
        asset_breakdown.append({
            "name":              asset["name"],
            "allocation_pct":    float(allocations[i]),
            "value_inr":         float(asset_values[i]),
            "crash_pct":         float(crashes[i]),
            "post_crash_value":  float(post_crash_values[i]),
            "loss_inr":          float(asset_values[i] - post_crash_values[i]),
            "risk_score":        float(risk_magnitudes[i]),   # allocation × |crash|
        })

    # Volatile assets = crypto + high-growth equities (crash > 30 %)
    volatile_alloc = float(sum(
        allocations[i] for i, a in enumerate(assets)
        if abs(crashes[i]) > 30
    ))

    return {
        "total_value_inr":        total_value,
        "monthly_expenses_inr":   monthly_expenses,
        "post_crash_value_inr":   round(post_crash_total, 2),
        "total_loss_inr":         round(total_value - post_crash_total, 2),
        "loss_pct":               round((1 - post_crash_total / total_value) * 100, 2),
        "runway_months":          round(runway_months, 1),
        "ruin_test":              "PASS" if runway_months > 12 else "FAIL",
        "largest_risk_asset":     largest_risk_asset,
        "concentration_warning":  concentration_warning,
        "concentrated_assets":    concentrated_assets,
        "volatile_allocation_pct": volatile_alloc,
        "asset_breakdown":        asset_breakdown,
    }


# ---------------------------------------------------------------------------
# Pydantic schemas for Gemini structured output
# ---------------------------------------------------------------------------

class PortfolioAnalysis(BaseModel):
    """
    Schema for the portfolio risk analysis.
    Passed to Gemini as response_json_schema so the output is enforced at the
    API level — no regex parsing or prompt begging needed.
    """
    risk_summary: str = Field(
        description=(
            "3-4 sentence plain-English explanation of the portfolio's overall "
            "risk level and what it means for the investor. Reference the "
            "pre-calculated numbers provided."
        )
    )
    doing_well: str = Field(
        description=(
            "One specific strength of this portfolio (e.g., good diversification, "
            "adequate cash buffer)."
        )
    )
    should_consider: str = Field(
        description=(
            "One specific, actionable improvement and the reason for it. "
            "Be constructive, not alarming."
        )
    )
    verdict: Literal["Aggressive", "Balanced", "Conservative"] = Field(
        description="Overall risk classification of the portfolio."
    )
    reasoning: str = Field(
        description="Brief explanation of why this verdict was chosen."
    )


class AnalysisCritique(BaseModel):
    """
    Schema for the meta-review critique.
    Passed to Gemini as response_json_schema for the second independent call.
    """
    overall_quality: Literal["Excellent", "Good", "Fair", "Poor"] = Field(
        description="Overall quality rating of the analysis."
    )
    accuracy_score: int = Field(
        ge=1, le=10,
        description="How accurately the analysis reflects the pre-calculated metrics (1-10).",
    )
    accuracy_notes: str = Field(
        description="Brief notes on factual / numerical accuracy."
    )
    appropriateness_score: int = Field(
        ge=1, le=10,
        description="How suitable the recommendations are for this investor (1-10).",
    )
    appropriateness_notes: str = Field(
        description="Brief notes on recommendation quality."
    )
    suggested_improvements: str = Field(
        description="What could be improved in the analysis."
    )
    verdict_validation: str = Field(
        description="Is the Aggressive/Balanced/Conservative verdict correct? Why?"
    )


# ---------------------------------------------------------------------------
# Tone-specific instructions (BONUS FEATURE)
# ---------------------------------------------------------------------------

TONE_INSTRUCTIONS: Dict[str, str] = {
    "beginner": (
        "Use simple language. Avoid jargon. Explain concepts as if talking to someone "
        "who just started learning about investing. Use analogies when helpful. "
        "Be encouraging but honest."
    ),
    "experienced": (
        "Use standard financial terminology. Assume familiarity with diversification, "
        "volatility, and asset allocation. Be direct and analytical. "
        "Focus on actionable insights."
    ),
    "expert": (
        "Use technical financial language. Reference concepts like drawdown, "
        "correlation, and tail risk where relevant. Be concise and data-driven. "
        "Assume deep market knowledge."
    ),
}


# ---------------------------------------------------------------------------
# Main advisor class
# ---------------------------------------------------------------------------

class PortfolioAdvisor:
    """AI-powered portfolio advisor using Google Gemini for both analysis and critique."""

    def __init__(self, api_key: Optional[str] = None, tone: str = "beginner", max_retries: int = 3):
        """
        Args:
            api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
            tone:    Communication tone — 'beginner', 'experienced', or 'expert'.
            max_retries: Maximum number of retry attempts for rate limit errors.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter.\n"
                "Get your key at: https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=self.api_key)
        self.tone   = tone
        self.model  = "models/gemini-2.5-flash"  # Using latest stable Gemini model
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_analysis_system_instruction(self) -> str:
        """
        System instruction for the portfolio analysis call.

        PROMPT ENGINEERING RATIONALE
        ─────────────────────────────
        Iteration 1 — everything in the user message.
          Problem: Gemini occasionally added extra commentary around the JSON,
                   requiring regex extraction.

        Iteration 2 — added "Return ONLY the JSON object" instruction.
          Problem: Still fragile; tone was inconsistent across calls.

        Iteration 3 — moved persona + constraints to system_instruction;
                       user message carries only data.
          Result: Instruction adherence improved significantly.

        Final — added response_json_schema to generation config.
          The output schema is now enforced at the API level; no parsing
          heuristics needed at all.
        """
        return (
            f"You are a friendly but honest financial advisor. "
            f"Your client is at the '{self.tone}' knowledge level.\n\n"
            f"TONE INSTRUCTIONS:\n{TONE_INSTRUCTIONS[self.tone]}\n\n"
            "HARD CONSTRAINTS:\n"
            "- Every number you cite must come from the PRE-CALCULATED METRICS "
            "  provided in the user message. Do NOT recompute or estimate figures.\n"
            "- Your response must be a valid JSON object matching the provided schema.\n\n"
            "VERDICT GUIDELINES:\n"
            "- Conservative : volatile_allocation_pct < 30 AND stable assets > 20 %\n"
            "- Balanced     : volatile_allocation_pct 30–60, diversified\n"
            "- Aggressive   : volatile_allocation_pct > 60, high concentration, "
            "  or extreme crash exposure\n"
        )

    def _build_analysis_user_message(self, portfolio: Dict, metrics: Dict) -> str:
        """
        User message for the analysis call: only the portfolio data + pre-calculated facts.
        """
        return (
            "Please analyse the following portfolio.\n\n"
            "PORTFOLIO CONFIGURATION:\n"
            f"{json.dumps(portfolio, indent=2)}\n\n"
            "PRE-CALCULATED METRICS (computed in Python — treat as ground truth):\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "Provide your complete assessment as a JSON object matching the schema."
        )

    def _build_critique_system_instruction(self) -> str:
        """System instruction for the meta-review call."""
        return (
            "You are a senior financial advisor reviewing another advisor's "
            "portfolio analysis for accuracy and quality. "
            "You are an independent reviewer — the analysis was produced by a "
            "different AI call, so evaluate it objectively.\n\n"
            "HARD CONSTRAINTS:\n"
            "- Verify all numbers against the PRE-CALCULATED METRICS provided.\n"
            "- Your response must be a valid JSON object matching the provided schema.\n"
        )

    def _build_critique_user_message(
        self, portfolio: Dict, metrics: Dict, analysis: Dict
    ) -> str:
        """User message for the meta-review call."""
        return (
            "ORIGINAL PORTFOLIO:\n"
            f"{json.dumps(portfolio, indent=2)}\n\n"
            "PRE-CALCULATED METRICS (ground truth, computed in Python):\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "FIRST ANALYSIS (what you are reviewing):\n"
            f"{json.dumps(analysis, indent=2)}\n\n"
            "Evaluate the analysis on accuracy, appropriateness, and completeness. "
            "Return your critique as a JSON object matching the provided schema."
        )

    # ------------------------------------------------------------------
    # API call helper with retry logic
    # ------------------------------------------------------------------

    def _call_gemini_with_retry(
        self, 
        user_message: str, 
        system_instruction: str, 
        schema: Dict,
        operation_name: str = "API call"
    ) -> str:
        """
        Call Gemini API with exponential backoff retry logic for rate limits.
        
        Args:
            user_message: The user message content
            system_instruction: The system instruction
            schema: The JSON schema for structured output
            operation_name: Name of the operation for logging
            
        Returns:
            The response text from Gemini
            
        Raises:
            Exception: If all retries are exhausted or non-retryable error occurs
        """
        for attempt in range(self.max_retries):
            try:
                print(f"Sending {operation_name} to Gemini API (attempt {attempt + 1}/{self.max_retries})...")
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_message,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_json_schema=schema,
                        temperature=0.7,  # Add some creativity while maintaining consistency
                    ),
                )
                
                print(f"✓ Received response from API\n")
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "quota" in error_str or "rate limit" in error_str or "resource exhausted" in error_str:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 2^attempt seconds (2s, 4s, 8s, ...)
                        wait_time = 2 ** (attempt + 1)
                        print(f"⚠️  Rate limit hit (429). Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\n✗ Rate limit error after {self.max_retries} attempts.")
                        print("\nPossible solutions:")
                        print("1. Wait a few minutes and try again")
                        print("2. Check your API quota at: https://aistudio.google.com/")
                        print("3. Upgrade to a paid plan if you're on the free tier")
                        print("4. Try using a different API key")
                        raise Exception(f"Rate limit exceeded after {self.max_retries} retries: {str(e)}")
                
                # Check if it's an invalid API key error
                elif "api key" in error_str or "authentication" in error_str or "401" in error_str:
                    print(f"\n✗ API Key Error: {str(e)}")
                    print("\nPlease check:")
                    print("1. Your API key is correct in the .env file")
                    print("2. The API key is enabled at: https://aistudio.google.com/apikey")
                    print("3. You're using the correct Google Cloud project")
                    raise
                
                # Check if it's a model not found error
                elif "404" in error_str or "not found" in error_str:
                    print(f"\n✗ Model Error: {str(e)}")
                    print(f"\nThe model '{self.model}' may not be available.")
                    print("Try updating the model name in the code or check available models.")
                    raise
                
                # For other errors, don't retry
                else:
                    print(f"\n✗ Error calling Gemini API: {str(e)}")
                    raise
        
        # Should never reach here, but just in case
        raise Exception(f"Failed after {self.max_retries} attempts")

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def analyze_portfolio(self, portfolio: Dict) -> Dict:
        """
        Analyse a portfolio and return a structured risk explanation.

        APPROACH:
          1. Compute all metrics in Python (no LLM math).
          2. Call Gemini with system_instruction and response_json_schema.
          3. Parse the response directly — Gemini guarantees valid JSON.

        Returns a dict with 'analysis', 'metrics', 'model', and 'tone'.
        """
        print("\n" + "=" * 70)
        print("  ANALYZING PORTFOLIO WITH AI")
        print("=" * 70)
        print(f"\nTone:     {self.tone.upper()}")
        print(f"Model:    {self.model}")
        print(f"Provider: Google Gemini\n")

        # Step 1: Python math — never delegate this to the LLM
        print("Computing risk metrics in Python...")
        metrics = compute_risk_metrics(portfolio)
        print(f"  Post-crash value : ₹{metrics['post_crash_value_inr']:,.0f}")
        print(f"  Runway           : {metrics['runway_months']} months")
        print(f"  Ruin test        : {metrics['ruin_test']}")
        print(f"  Volatile alloc   : {metrics['volatile_allocation_pct']}%\n")

        # Step 2: Build prompts
        system_instruction = self._build_analysis_system_instruction()
        user_message       = self._build_analysis_user_message(portfolio, metrics)

        try:
            # Step 3: Call API with retry logic
            response_text = self._call_gemini_with_retry(
                user_message=user_message,
                system_instruction=system_instruction,
                schema=PortfolioAnalysis.model_json_schema(),
                operation_name="portfolio analysis"
            )

            # Step 4: Parse structured output — Gemini guarantees valid JSON
            analysis = PortfolioAnalysis.model_validate_json(response_text)

            return {
                "analysis": analysis.model_dump(),
                "metrics":  metrics,
                "model":    self.model,
                "tone":     self.tone,
            }

        except Exception as e:
            print(f"✗ Failed to analyze portfolio: {str(e)}")
            raise

    def critique_analysis(self, portfolio: Dict, metrics: Dict, analysis: Dict) -> Dict:
        """
        Generate a meta-review critique using a second Gemini call (BONUS FEATURE).

        Uses a different system instruction to create an independent reviewer
        persona. This avoids self-serving bias — the second call evaluates the
        first objectively.

        Uses the same response_json_schema pattern for guaranteed structured output.
        """
        print("\n" + "=" * 70)
        print("  CRITIQUING ANALYSIS (META-REVIEW via 2nd Gemini call)")
        print("=" * 70 + "\n")

        system_instruction = self._build_critique_system_instruction()
        user_message       = self._build_critique_user_message(portfolio, metrics, analysis)

        print(f"Model:    {self.model}")
        print(f"Provider: Google Gemini\n")

        try:
            # Call API with retry logic
            response_text = self._call_gemini_with_retry(
                user_message=user_message,
                system_instruction=system_instruction,
                schema=AnalysisCritique.model_json_schema(),
                operation_name="analysis critique"
            )

            # response_text is guaranteed valid JSON matching AnalysisCritique
            critique = AnalysisCritique.model_validate_json(response_text)

            return {"critique": critique.model_dump()}

        except Exception as e:
            print(f"✗ Failed to critique analysis: {str(e)}")
            raise


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_analysis_results(result: Dict):
    """Pretty-print the analysis results."""
    analysis = result["analysis"]
    metrics  = result["metrics"]

    print("=" * 70)
    print("  PORTFOLIO RISK ANALYSIS")
    print("=" * 70)

    print(f"\n📊 RISK SUMMARY:")
    print(f"   {analysis['risk_summary']}")

    print(f"\n✅ DOING WELL:")
    print(f"   {analysis['doing_well']}")

    print(f"\n💡 SHOULD CONSIDER:")
    print(f"   {analysis['should_consider']}")

    print(f"\n⚖️  VERDICT: {analysis['verdict'].upper()}")
    if "reasoning" in analysis:
        print(f"   Reasoning: {analysis['reasoning']}")

    print(f"\n📐 VERIFIED METRICS (Python-calculated):")
    print(f"   Post-crash value : ₹{metrics['post_crash_value_inr']:,.0f}  "
          f"(loss: {metrics['loss_pct']}%)")
    print(f"   Runway           : {metrics['runway_months']} months  "
          f"[{metrics['ruin_test']}]")
    print(f"   Volatile alloc   : {metrics['volatile_allocation_pct']}%")
    if metrics["concentration_warning"]:
        print(f"   ⚠️  Concentration : {', '.join(metrics['concentrated_assets'])} > 40%")

    print("\n" + "=" * 70)


def print_critique_results(critique_result: Dict):
    """Pretty-print the critique results."""
    critique = critique_result["critique"]

    print("=" * 70)
    print("  META-REVIEW (CRITIQUE OF ANALYSIS)")
    print("=" * 70)

    print(f"\n📈 OVERALL QUALITY: {critique['overall_quality']}")

    print(f"\n🎯 ACCURACY: {critique['accuracy_score']}/10")
    print(f"   {critique['accuracy_notes']}")

    print(f"\n✨ APPROPRIATENESS: {critique['appropriateness_score']}/10")
    print(f"   {critique['appropriateness_notes']}")

    print(f"\n🔍 VERDICT VALIDATION:")
    print(f"   {critique['verdict_validation']}")

    print(f"\n💭 SUGGESTED IMPROVEMENTS:")
    print(f"   {critique['suggested_improvements']}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Portfolio Risk Advisor (Google Gemini)",
        epilog="""
Examples:
  python task3_ai_portfolio_explainer.py --tone beginner
  python task3_ai_portfolio_explainer.py --tone expert --critique
  python task3_ai_portfolio_explainer.py --custom portfolio.json

Set GEMINI_API_KEY environment variable before running.
        """,
    )

    parser.add_argument(
        "--tone",
        choices=["beginner", "experienced", "expert"],
        default="beginner",
        help="Communication tone level (default: beginner) [BONUS FEATURE]",
    )
    parser.add_argument(
        "--critique",
        action="store_true",
        help="Generate a critique of the analysis (BONUS FEATURE)",
    )
    parser.add_argument(
        "--custom",
        type=str,
        help="Path to custom portfolio JSON file",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show the full pre-calculated metrics dict",
    )

    args = parser.parse_args()

    # Default portfolio (from Task 1)
    default_portfolio = {
        "total_value_inr":      10_000_000,
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct":   0},
        ],
    }

    if args.custom:
        with open(args.custom, "r") as f:
            portfolio = json.load(f)
        print(f"\n✓ Loaded custom portfolio from {args.custom}")
    else:
        portfolio = default_portfolio

    # Print portfolio summary
    print("\n" + "=" * 70)
    print("  TASK 3: AI-POWERED PORTFOLIO EXPLAINER")
    print("=" * 70)
    print("\n  INPUT PORTFOLIO")
    print("  " + "-" * 66)
    print(f"  Total Value:      ₹{portfolio['total_value_inr']:,}")
    print(f"  Monthly Expenses: ₹{portfolio['monthly_expenses_inr']:,}")
    print("\n  Assets:")
    for asset in portfolio["assets"]:
        print(f"    • {asset['name']}: {asset['allocation_pct']}%  "
              f"(crash: {asset['expected_crash_pct']}%)")

    try:
        advisor = PortfolioAdvisor(tone=args.tone)
        result  = advisor.analyze_portfolio(portfolio)

        if args.show_metrics:
            print("\n" + "=" * 70)
            print("  PRE-CALCULATED METRICS (Python)")
            print("=" * 70)
            print(json.dumps(result["metrics"], indent=2))
            print("=" * 70)

        print_analysis_results(result)

        if args.critique:
            critique_result = advisor.critique_analysis(
                portfolio,
                result["metrics"],
                result["analysis"],
            )
            print_critique_results(critique_result)

        print("\n✓ Analysis complete\n")

    except ValueError as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nPlease set your GEMINI_API_KEY environment variable:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("\nGet your key at: https://aistudio.google.com/apikey")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
