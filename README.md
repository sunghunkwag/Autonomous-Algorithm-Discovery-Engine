# Autonomous-Algorithm-Discovery-Engine

Experimental framework for taskless, open-ended algorithm discovery via autonomous goal generation in program synthesis.

## Main Engine

**[OMEGA_FORGE_V11.py](OMEGA_FORGE_V11.py)** — Autonomous goal discovery engine.

- No human-defined tasks required
- Self-generates goals from capability gaps
- Goals evolve alongside solvers

```bash
python OMEGA_FORGE_V11.py --selftest
python OMEGA_FORGE_V11.py --run --generations 1000
```

## Status

⚠️ **Research stage** — Does not yet discover practical algorithms.

## Archive

V9 and V10 are preserved in `/archive` to document the incremental evolution.  
V11 fully subsumes their functionality and represents the final autonomous engine.

## License

MIT
