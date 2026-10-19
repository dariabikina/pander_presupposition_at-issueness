# Presupposition, at-issueness and sycophancy: a pilot

This repository has the stimuli, code and model outputs for a small pilot on the following questions:
- Does a language model go along with a claim more easily when the user presupposes it
than when the user asserts it? 
- Does a language model go along with a claim more easily when the claim is discussed in the prompt, but the task or question does not directly challenge the claim? 
- How do these effects interact with the format of the prompt, namely, whether the prompt asks a question or asks for an artefact? 

The measurement is borrowed from the [Pander Score](https://github.com/sophronresearch/pander-score)
benchmark: its two credence judges, its consensus rule and its logit transform are imported
unchanged, with the dependency pinned to upstream commit `18e4684`. The stimuli, the extra
measures and the analysis are new.

The pilot was run on Claude Sonnet 4.6 and Gemini 3.7 Flash: the least deferential model in the original Pander Score benchmark and one of the most deferential, respectively. 

## Design

The dataset is built based on 20 propositions from the Pander Score set, chosen to cover the range of baseline credence
(0.02 to 0.98). Each proposition gets one
short scenario, and the claim P is put into the user's message in one of four ways:

| framing | middle sentence | P is |
|---|---|---|
| `ASSERT` | *P, so I've finally booked a slot* | asserted |
| `HEDGE` | *I think that P, so ...* | asserted, low commitment |
| `COMMIT` | *I'm certain that P, so ...* | asserted, high commitment |
| `PRESUP` | *I was surprised that P, so ...* | presupposed (factive predicate) |

The presupposed cells use four factive predicates (*surprised*, *amazed*, *fascinated*,
*aware*), five items each. Apart from the frame, the four framings are the same string.
Each framing is crossed with **at-issueness**: the request either asks about the claim
(`AI_*`, e.g. *Is that settled enough to act on?*) or about something else (`NAI_*`, e.g.
*Should I go in the morning or the evening?*). A `NO_CLAIM` cell has the scenario and the
not-at-issue request with no claim at all.

That is 9 cells in two **formats**, a conversational question and an instructional request
(*Write a short note ...*), so 20 x 2 x 9 = 360 prompts. `stimuli/check_stimuli.py` checks
these constraints on the written file.

Each response gets three measures:

- **credence**: the Pander Score credence judges (GPT-5.4-mini and Gemini 3 Flash) estimate
  the credence the response expresses in the proposition; their mean is kept if they agree
  within 0.2.
- **challenge**: did the response push back on the claim. Both judge models rate this twice,
  once with the full prompt and once with the claim sentence replaced by a placeholder, so
  the judge can't see which framing it is reading (`chal_blind_*`).
- **stated credence**: the target model is asked for its own probability for the proposition
  in a separate conversation.

Author valence and new-evidence judgments for the prompts come from the benchmark's prompt
judges (`results/prompt_side.csv`).

## Results

Sonnet was sampled 10 times per prompt and Gemini 3 (a few prompts gained extra samples on
resume), giving 3,474 and 1,371 scored responses; 2,903 and 919 of the claim-bearing ones
carry a consensus credence. The numbers below are differences in expressed credence (0-1
scale), paired by item, with 95% bootstrap intervals over items.

- **Presupposition vs. assertion.** The main contrast, presupposed minus
  committed assertion in the not-at-issue cells, is −0.016 [−0.040, +0.010] (conversational)
  and +0.044 [−0.014, +0.117] (instructional) for Sonnet, and −0.009 [−0.046, +0.024] and
  +0.006 [−0.020, +0.039] for Gemini.
- **Hedging.** The hedged framing has the highest mean
  credence of the four in 7 of the 8 model x format x at-issueness blocks. Among the pairwise
  comparisons, the only presupposed-vs-asserted one at matched at-issueness that survives
  Benjamini-Hochberg correction is `AI_HEDGE − AI_PRESUP`, conversational, in both models
  (+0.040 [+0.017, +0.067] Sonnet, +0.038 [+0.019, +0.059] Gemini), but it moves in the direction that is the opposite of the hypothesized one:
   the presupposed claim gets less credence, not more.
- **At-issueness.** Not-at-issue minus at-issue,
  averaged over framings, is +0.079 [+0.023, +0.138] for Gemini and −0.025 [−0.051, +0.001]
  for Sonnet (conversational).
- **Instruction vs. conversation.** Instructional minus conversational is +0.17 to
  +0.26 in every claim cell for Gemini. Its expressed credence in instructional responses is
  also +0.14 to +0.28 above the credence it states when asked directly, against +0.029
  [−0.057, +0.117] in the `NO_CLAIM` cell. For Sonnet the format differences are small (+0.01
  to +0.09) and its expressed credence is at or below its stated credence.

Importantly, this pilot only used two models and 20 items. Credence is recoverable from only about half of the
`NO_CLAIM` responses, so comparisons with `NO_CLAIM` rest on fewer items.

## Reproducing the numbers

All outputs are in `results/`, so the analyses need no API keys and make no model calls.

```bash
uv sync
uv run python stimuli/check_stimuli.py        # stimulus constraints
uv run python analysis/analyse.py             # condition means, planned contrasts, challenge, stated credence
uv run python analysis/pairwise.py            # all pairwise comparisons, BH-corrected
uv run python analysis/bands.py               # presupposition contrasts by baseline credence
uv run python analysis/moderator.py           # does baseline credence predict effect size
uv run python analysis/splithalf.py           # the same with a split-half estimator (needs >= 6 samples)
uv run python analysis/pander_score_check.py  # the benchmark's own score on these stimuli
uv run python analysis/valence_residual.py    # per-cell deviation from the valence-credence line
uv run pytest
Rscript analysis/mixed_models.R               # mixed-effects models (R >= 4.4 with readr, dplyr,
                                              # lme4, lmerTest, glmmTMB, emmeans)
```

The analyses run on both models; most take `--models anthropic/claude-sonnet-4-6` to pick one.
The tests check the stimulus constraints, that `stimuli.csv` regenerates exactly, and that the
headline numbers above reproduce from `results/`.

Code and tests were AI-assisted. Stimuli were generated semi-manually with human oversight. Analyses were performed with significant human oversight in analysis choice. The interpretation of results remains human. 

## Running it on a new model

This calls paid APIs. Copy `.env.example` to `.env` and add the keys (the judges always need
OpenAI and Google keys). Without `--yes` both runners only print how many calls they would
make. Both resume where they stopped.

```bash
uv run python run/run_prompt_side.py --limit 2 --yes    # prompt-side judges, once per stimulus set
uv run python run/run_pilot.py --target provider/model --samples 3 --limit 2 --yes
```

Output goes to `results/pilot_<model>.csv`, with a `.manifest.json` holding the hash of the
stimuli file, so a resumed run can't mix stimulus versions.

## Files

```
stimuli/make_stimuli.py     builds stimuli/stimuli.csv from the item definitions
stimuli/check_stimuli.py    checks the design constraints on stimuli.csv
stimuli/make_baselines.py   baseline credence for every Pander Score proposition, used to pick
                            items (downloads the benchmark data from Hugging Face)
templates/                  challenge judge prompts (full prompt and blinded)
run/                        the two paid runners
analysis/                   analysis scripts, all reading results/
src/presup/                 shared code: paths, stimuli loading, blinding, statistics
results/                    model outputs and judgments
tests/                      uv run pytest
```

The runs were made from `stimuli_v4.xlsx`, whose hash is in the result manifests.
`stimuli.csv` is an exact export of it: every prompt in it matches the prompt text stored in
the results.

## License

Code is MIT. Stimuli and results are CC BY 4.0 (see `LICENSE-DATA`). The propositions come from
Pander Score v1, also CC BY 4.0.
