# Labelling rubric

`rubric_version: 1`

The single source of truth for what each class means. The weak-labeller prompt
and the Label Studio annotation instructions both quote this file verbatim.
Changing it is a versioned PR: bump `rubric_version`, and re-label the frozen
gold anchor (it was labelled under the old definition).

## The task

Classify a financial-news **headline** by its **investor polarity, judged from
the text alone** — would this read as good, bad, or neutral news for an investor
holding the mentioned company? Judge the text, not your own market view, and not
what the stock later did.

## Classes

- **bullish** — the headline implies the company's value goes up / is good news
  for a holder.
- **bearish** — the headline implies the company's value goes down / is bad news
  for a holder.
- **neutral** — about a tradable company or market, but states a fact with no
  clear directional implication (factual, procedural, or balanced).
- **irrelevant** — not about a tradable company or market at all (general news,
  sport, lifestyle, etc.).

## Tie-breakers

- Direction unclear or mixed → **neutral** (not a coin-flip between bull/bear).
- Financial topic but no company/market subject (macro chatter, listicles,
  passing mention with no stance) → **neutral**, not irrelevant. `irrelevant` is
  for genuinely off-topic text.
- Headline is ambiguous only because it lacks context you'd need the body for →
  label on the headline as written; do not infer.

## Examples

| Headline | Label |
|---|---|
| Operating profit rose to EUR 1.2 mn from EUR 0.8 mn a year earlier | bullish |
| Company X cuts full-year guidance after weak quarter | bearish |
| Company X to hold its annual general meeting on 14 May | neutral |
| Shares of Company X were unchanged in early trading | neutral |
| Local football club wins the regional cup final | irrelevant |
