# Blaque Baux Bespoke

**Residual (idiosyncratic) momentum — does momentum on factor-residuals beat raw price momentum?**

> **Not investment advice.** Educational/research software. See [DISCLAIMER](DISCLAIMER.md) and [LICENSE](LICENSE).

```bash
python3 research/bespoke_1.py   # needs Alpaca data keys in the environment
```

Blitz-Huij-Martens: strip each stock's market exposure, run momentum on the *residuals* — the idiosyncratic ("bespoke") part — on the theory it has a higher information ratio and a milder crash than raw momentum. Really a construction-v2 of `backwards`/`beyond`.

## Finding (null-ish, this window)
| book | Sharpe | maxDD | skew | β/SPY |
|---|---|---|---|---|
| RAW momentum | +0.31 | −36% | −1.12 | +0.02 |
| RESIDUAL momentum | +0.35 | −37% | −1.36 | −0.09 |

Residualizing gives ~the same Sharpe (+0.35 vs +0.31), a slightly *worse* skew, and a marginally cleaner beta — no clear improvement on large caps 2016-26 (where momentum itself is weak, see `bulwark`).

## Verdict
**NULL-ish (this window).** On a large-cap universe with market-only residualization, residual momentum doesn't clearly beat raw. The documented residual-momentum edge is stronger on **broad/small-cap** universes with full **multi-factor** residualization and residual-vol scaling = `bespoke_2`.

## Status
**Research.** Honest null — a thinner white-space leg built out as its own sleeve; proxy limits named, `bespoke_2` path noted. Fat-tail scorecard, causal, Alpaca SIP daily total return. No live capital.
