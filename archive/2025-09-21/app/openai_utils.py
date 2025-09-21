def analyze_prompt_with_gpt(prompt: str) -> str:
    # Cost-free, built-in analysis placeholder.
    # This keeps the endpoint stable even with no API keys.
    # You can replace with a real model later; the interface stays the same.
    return (
        "Correlation: Prefer a QB with 1–2 pass-catchers from the same team; consider an opposing bring-back "
        "if the game total is high.\n"
        "Leverage: Mix 1–3 low-owned ceiling plays to differentiate from chalky core pieces.\n"
        "Risk/Ceiling: Rushing QBs and target hog WRs drive most of the variance; aim for at least two high-ceiling anchors.\n"
        "Strengths: Salary efficiency and concentration of team volume.\n"
        "Weaknesses: If a single game under-performs, the correlated pieces can sink the lineup.\n"
        "Suggested Swaps: Swap a chalk RB for a lower-owned WR with strong air yards; or pivot TE into the FLEX when pricing is tight."
    )
