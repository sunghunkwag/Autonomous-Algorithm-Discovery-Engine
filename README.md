# Autonomous-Algorithm-Discovery-Engine

Experimental framework for taskless, open-ended algorithm discovery via autonomous goal generation in program synthesis.

## Motivation 

Most program synthesis and algorithm discovery systems assume human-defined tasks or rewards.
This project explores whether a minimal evolutionary system can autonomously generate its own goals and curricula,
without external task specification.

## Main Engine

**[OMEGA_FORGE_V13_STAGED_EVOLVED.py](OMEGA_FORGE_V13_STAGED_EVOLVED.py)** — Strict Structural Evolution Engine (Current SOTA).

- **Strict Validation**: Enforces Loops, Recursion, and SCCs. Rejects linear cheats.
- **High Efficiency**: 104 Verified Innovations in 85 Generations.
- **Autonomous Gating**: Separates 'Worker' (generator) and 'Judge' (strict validator).

```bash
# Run validation (evidence collection)
python OMEGA_FORGE_V13_STAGED_EVOLVED.py evidence_run --target 100
```

## Status

⚠️ **Research stage** — Does not yet discover practical algorithms. Intended for experimental analysis of structural evolution.

## Archive

Older versions (`V11`, `V12`, `V13` variants) and execution logs are preserved in `/archive`.
`V13_STAGED_EVOLVED` supersedes all previous versions.

## License

MIT
