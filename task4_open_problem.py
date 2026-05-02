#!/usr/bin/env python3
"""
TASK 4: Natural Language Portfolio Stress Tester
Timecell.AI Summer Internship 2025 - Technical Assessment

ARCHITECTURE:
1. Extraction Layer (Gemini + Pydantic) - Extract shock parameters via response_json_schema
2. Execution Layer (Python) - Pure Python calculates portfolio impact with zero-floor
3. Narration Layer (Gemini) - Generate survival brief from verified math

STRICT CONSTRAINTS:
- NO LLM MATH: All calculations in pure Python
- Native Structured Output: Pydantic response_json_schema (no legacy function calling)
- Consistent Models: gemini-2.5-flash for both extraction and narration
- Defensive Math: Zero-floor enforcement (assets cannot go negative)
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Dict, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================================
# STEP 1: BASELINE PORTFOLIO STATE (Hardcoded)
# ============================================================================

BASELINE_PORTFOLIO = {
    "total_value_inr": 10_000_000,
    "monthly_expenses_inr": 80_000,
    "assets": {
        "BTC": 0.30,      # 30%
        "NIFTY50": 0.40,  # 40%
        "GOLD": 0.20,     # 20%
        "CASH": 0.10      # 10%
    }
}


# ============================================================================
# STEP 2: NLP EXTRACTION LAYER (Gemini + Pydantic)
# ============================================================================

class ShockVector(BaseModel):
    """
    Pydantic model for portfolio shock parameters.
    All fields default to 0 (no change).
    Drops must be NEGATIVE integers (e.g., -80 for 80% crash).
    """
    BTC: int = Field(
        default=0,
        description="Percentage change for Bitcoin. Use NEGATIVE integers for drops (e.g., -100 for total ban, -80 for 80% crash). 0 if not mentioned."
    )
    NIFTY50: int = Field(
        default=0,
        description="Percentage change for NIFTY50 index. Use NEGATIVE integers for drops (e.g., -15 for 15% drop). 0 if not mentioned."
    )
    GOLD: int = Field(
        default=0,
        description="Percentage change for Gold. Use NEGATIVE integers for drops (e.g., -10 for 10% drop). 0 if not mentioned."
    )
    CASH: int = Field(
        default=0,
        description="Percentage change for Cash. Typically 0 unless currency crisis. Use NEGATIVE integers for devaluation."
    )


def extract_shock_vector(query: str, client: genai.Client) -> Dict[str, int]:
    """
    Use Gemini with Pydantic response_json_schema to extract shock parameters.
    
    Args:
        query: Natural language query (e.g., "What if Bitcoin crashes 80%?")
        client: Configured Gemini client
    
    Returns:
        Dictionary mapping asset names to percentage drops (negative integers)
        Example: {"BTC": -80, "NIFTY50": -15, "GOLD": 0, "CASH": 0}
    """
    # Construct the extraction prompt
    extraction_prompt = f"""You are a financial analyst extracting shock parameters from a stress test query.

Query: "{query}"

Analyze this query and extract the percentage changes for each asset class:
- BTC (Bitcoin)
- NIFTY50 (Indian stock market index)
- GOLD (Gold)
- CASH (Cash holdings)

CRITICAL RULES:
1. Use NEGATIVE integers for drops (e.g., -80 for "80% crash", -100 for "total ban")
2. Use POSITIVE integers for gains (e.g., 20 for "20% surge")
3. If an asset is not mentioned, use 0 (no change)
4. Be conservative: "market crash" typically means -30% to -40% for equities
5. "Bitcoin ban" or "crypto ban" means -100% for BTC
6. "Safe haven" for gold typically means +10% to +30%

Extract the shock vector now as a JSON object with BTC, NIFTY50, GOLD, and CASH fields."""

    # Call Gemini with response_json_schema
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=extraction_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ShockVector.model_json_schema(),
            temperature=0.0
        )
    )
    
    # Validate and return the structured output
    shock_vector = ShockVector.model_validate_json(response.text)
    return shock_vector.model_dump()


# ============================================================================
# STEP 3: QUANT EXECUTION LAYER (Pure Python with Zero-Floor)
# ============================================================================

def apply_shock(portfolio: Dict, shock_vector: Dict[str, int]) -> Tuple[float, float, float]:
    """
    Apply shock vector to portfolio and calculate impact using pure Python math.
    Enforces zero-floor: assets cannot drop below zero (no negative values).
    
    Args:
        portfolio: Baseline portfolio state
        shock_vector: Dictionary of asset -> percentage drop (negative integers)
    
    Returns:
        Tuple of (post_crash_value_inr, total_loss_inr, runway_months)
    """
    total_value = portfolio["total_value_inr"]
    monthly_expenses = portfolio["monthly_expenses_inr"]
    assets = portfolio["assets"]
    
    # Calculate post-crash value with zero-floor enforcement
    post_crash_value = 0.0
    
    for asset_name, allocation_pct in assets.items():
        # Current value of this asset
        current_asset_value = total_value * allocation_pct
        
        # Apply shock (convert percentage to multiplier)
        shock_pct = shock_vector.get(asset_name, 0)
        multiplier = 1 + (shock_pct / 100.0)
        
        # CRITICAL: Enforce zero-floor (assets cannot go negative)
        new_asset_value = max(0.0, current_asset_value * multiplier)
        
        post_crash_value += new_asset_value
    
    # Calculate total loss
    total_loss = total_value - post_crash_value
    
    # Calculate runway (how many months can expenses be covered)
    runway_months = post_crash_value / monthly_expenses if monthly_expenses > 0 else float('inf')
    
    return post_crash_value, total_loss, runway_months


# ============================================================================
# STEP 4: NARRATIVE LAYER (Gemini)
# ============================================================================

def generate_survival_brief(
    query: str,
    shock_vector: Dict[str, int],
    post_crash_value: float,
    total_loss: float,
    runway_months: float,
    client: genai.Client
) -> str:
    """
    Use Gemini to generate a 3-sentence survival brief based on verified Python math.
    
    Args:
        query: Original user query
        shock_vector: Extracted shock parameters
        post_crash_value: Calculated post-crash portfolio value
        total_loss: Calculated total loss
        runway_months: Calculated runway in months
        client: Configured Gemini client
    
    Returns:
        3-sentence survival brief as a string
    """
    # Construct the narration prompt with ground truth metrics
    narration_prompt = f"""You are a wealth advisor writing a survival brief for a high-net-worth client.

SCENARIO: "{query}"

GROUND TRUTH METRICS (calculated by verified Python math - DO NOT RECALCULATE):
- Post-Crash Portfolio Value: ₹{post_crash_value:,.0f}
- Total Loss: ₹{total_loss:,.0f}
- Runway: {runway_months:.1f} months

SHOCK VECTOR APPLIED:
{json.dumps(shock_vector, indent=2)}

Write a 3-sentence "Survival Brief":

Sentence 1: Summarize the financial damage using the EXACT numbers provided above (post-crash value, total loss).

Sentence 2: Assess the severity and runway implications. Is this survivable? How many years of runway remain?

Sentence 3: Suggest ONE specific defensive portfolio reallocation to improve resilience (e.g., "shift 10% from X to Y").

CRITICAL RULES:
- Use the EXACT numbers from Ground Truth Metrics (do NOT recalculate or modify them)
- Be direct and professional
- Focus on actionable insights
- Keep it to EXACTLY 3 sentences
- Use Indian Rupee symbol ₹ for currency"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=narration_prompt,
        config=types.GenerateContentConfig(temperature=0.7)
    )
    
    return response.text.strip()


# ============================================================================
# STEP 5: TERMINAL UI
# ============================================================================

def print_terminal_output(
    query: str,
    shock_vector: Dict[str, int],
    post_crash_value: float,
    total_loss: float,
    runway_months: float,
    survival_brief: str
):
    """
    Print beautiful terminal output showing all three layers.
    """
    print("\n" + "=" * 80)
    print("  NATURAL LANGUAGE PORTFOLIO STRESS TESTER")
    print("=" * 80)
    
    print("\n📝 USER QUERY:")
    print(f"   \"{query}\"")
    
    print("\n" + "-" * 80)
    print("  LAYER 1: EXTRACTED SHOCK VECTOR (Gemini + Pydantic)")
    print("-" * 80)
    
    for asset, shock_pct in shock_vector.items():
        if shock_pct < 0:
            symbol = "🔻"
            color = "DROP"
        elif shock_pct > 0:
            symbol = "🔺"
            color = "GAIN"
        else:
            symbol = "➡️"
            color = "FLAT"
        
        print(f"   {symbol} {asset:<10} {shock_pct:>+4}%  ({color})")
    
    print("\n" + "-" * 80)
    print("  LAYER 2: VERIFIED PYTHON MATH (Zero-Floor Enforced)")
    print("-" * 80)
    
    print(f"   💰 Baseline Portfolio Value:  ₹{BASELINE_PORTFOLIO['total_value_inr']:>12,}")
    print(f"   📉 Post-Crash Value:          ₹{post_crash_value:>12,.0f}")
    print(f"   💸 Total Loss:                ₹{total_loss:>12,.0f}")
    print(f"   ⏱️  Financial Runway:          {runway_months:>12.1f} months")
    
    loss_pct = (total_loss / BASELINE_PORTFOLIO['total_value_inr']) * 100
    print(f"   📊 Loss Percentage:           {loss_pct:>12.1f}%")
    
    print("\n" + "-" * 80)
    print("  LAYER 3: AI SURVIVAL BRIEF (Gemini Narration)")
    print("-" * 80)
    
    # Wrap text nicely for readability
    wrapped_brief = textwrap.fill(
        survival_brief, 
        width=76, 
        initial_indent="   ", 
        subsequent_indent="   "
    )
    print(wrapped_brief)
    
    print("\n" + "=" * 80)
    
    # Risk assessment
    if runway_months < 12:
        print("   ⚠️  CRITICAL: Less than 1 year runway remaining")
    elif runway_months < 24:
        print("   ⚠️  WARNING: Less than 2 years runway remaining")
    elif runway_months < 60:
        print("   ⚡ CAUTION: Less than 5 years runway remaining")
    else:
        print("   ✅ STABLE: Sufficient runway for long-term recovery")
    
    print("=" * 80 + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline: Extract -> Execute -> Narrate
    """
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Natural Language Portfolio Stress Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python task4_what_if.py --query "What if Bitcoin gets banned and NIFTY drops 15%?"
  python task4_what_if.py --query "Market crash: stocks down 40%, gold safe haven up 10%"
  python task4_what_if.py --query "Total crypto wipeout but equities stable"
  python task4_what_if.py --query "Currency crisis: cash devalues 30%, flight to gold"
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language stress test query"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in environment variables")
        print("   Please set it in your .env file or export it:")
        print("   export GEMINI_API_KEY='your-key-here'")
        print("   Get your key at: https://aistudio.google.com/apikey")
        sys.exit(1)
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    try:
        # STEP 2: Extract shock vector using Gemini + Pydantic
        print("\n🔍 Extracting shock parameters from query...")
        shock_vector = extract_shock_vector(args.query, client)
        
        # STEP 3: Apply shock using pure Python math with zero-floor
        print("🧮 Calculating portfolio impact (zero-floor enforced)...")
        post_crash_value, total_loss, runway_months = apply_shock(
            BASELINE_PORTFOLIO,
            shock_vector
        )
        
        # STEP 4: Generate survival brief using Gemini
        print("📝 Generating AI survival brief...")
        survival_brief = generate_survival_brief(
            args.query,
            shock_vector,
            post_crash_value,
            total_loss,
            runway_months,
            client
        )
        
        # STEP 5: Display results
        print_terminal_output(
            args.query,
            shock_vector,
            post_crash_value,
            total_loss,
            runway_months,
            survival_brief
        )
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
