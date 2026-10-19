# Mixed-effects models for the pilot's credence and challenge measures.
#
# Reads results/pilot_*.csv (no API calls) and fits, per target model:
#
#   (1) A Gaussian LMM on item x cell mean credence - the primary model, whose
#       coefficients are the ones quoted in the write-up. One observation per
#       item x format x condition; random intercept and random at-issueness
#       slope by item, i.e. each proposition keeps its own overall credence
#       level and its own at-issueness effect. This is the model-based analogue
#       of the item-paired bootstrap in analysis/pairwise.py.
#   (2) A beta mixed model on individual responses (logit link) as a robustness
#       check with a likelihood that respects the bounded outcome. Random
#       intercepts for item and for prompt (= item x format x condition): the
#       prompt term stops the k repeated samples of one prompt string from
#       counting as independent evidence. A random at-issueness slope variant
#       is fitted alongside and compared by AIC.
#       (The judges clip credence to [0.01, 0.995], so there is no mass at the
#       endpoints and zero/one-inflated variants have nothing extra to fit.)
#   (3) Tukey-adjusted pairwise contrasts between the four framings, from (1):
#       presupposed vs certain / bare / hedged, and so on.
#   (4) For Gemini's blinded challenge measure: a binomial GLMM on fully-
#       challenged counts aggregated per prompt, testing the format x
#       at-issueness interaction by likelihood ratio. (Sonnet is omitted here:
#       it challenges 83-95% of responses in every claim cell, and the
#       near-separation makes the logistic fit uninformative.)
#
# Factor coding: at_issue reference "at-issue" (coefficient = not-at-issue
# minus at-issue), format reference "conv" (coefficient = instructional minus
# conversational), encoding reference "certain", so `encodingpresup` is the
# preregistered presupposed-minus-certain contrast.
#
# Run from the repository root:
#     Rscript analysis/mixed_models.R
#
# Requires R >= 4.4 with readr, dplyr, lme4, lmerTest, glmmTMB, emmeans.

suppressMessages({
  library(readr); library(dplyr)
  library(lme4); library(lmerTest)
  library(glmmTMB); library(emmeans)
})
options(width = 110)

MODELS <- c(
  "Gemini 3.7 Flash"  = "results/pilot_google-gemini-3.7-flash.csv",
  "Claude Sonnet 4.6" = "results/pilot_anthropic-claude-sonnet-4-6.csv"
)

# Only the columns the models use; read as character first because several
# columns (premise, the chal_* flags) mix strings that readr would otherwise
# guess inconsistently.
read_pilot <- function(path) {
  read_csv(path, show_col_types = FALSE,
           col_types = cols(.default = col_character())) |>
    transmute(
      item_id, format, condition, at_issue, encoding,
      consensus_credence = as.numeric(consensus_credence),
      chal_blind_score   = as.numeric(chal_blind_score)
    )
}

prep_claim_cells <- function(d) {
  d |>
    filter(condition != "NO_CLAIM", !is.na(consensus_credence)) |>
    mutate(
      at_issue = relevel(factor(at_issue), ref = "at-issue"),
      format   = relevel(factor(format),   ref = "conv"),
      encoding = factor(encoding,
        levels = c("asserted, high commitment", "asserted",
                   "asserted, low commitment", "presupposed"),
        labels = c("certain", "bare", "hedge", "presup")),
      prompt   = interaction(item_id, format, condition, drop = TRUE)
    )
}

coef_table <- function(fit) {
  s <- if (inherits(fit, "glmmTMB")) summary(fit)$coefficients$cond
       else summary(fit)$coefficients
  print(round(s, 3))
}

for (name in names(MODELS)) {
  cat("\n=========================", name, "=========================\n")
  d <- prep_claim_cells(read_pilot(MODELS[[name]]))

  cat("\n--- (1) Gaussian LMM on item x cell means",
      " [RE: (1 + at_issue | item_id)] ---\n")
  means <- d |>
    summarise(cm = mean(consensus_credence),
              .by = c(item_id, format, condition, at_issue, encoding))
  m1 <- lmer(cm ~ at_issue * format + encoding + (1 + at_issue | item_id),
             data = means)
  if (isSingular(m1)) cat("note: singular fit\n")
  coef_table(m1)
  cat("95% Wald CI:\n")
  print(round(confint(m1, method = "Wald")[-(1:4), ], 3))

  cat("\n--- (2) Beta mixed model, response level",
      " [RE: (1 | item_id) + (1 | prompt)] ---\n")
  cat(sprintf("n = %d responses, %d items, %d prompts\n",
              nrow(d), n_distinct(d$item_id), n_distinct(d$prompt)))
  m2 <- glmmTMB(consensus_credence ~ at_issue * format + encoding +
                  (1 | item_id) + (1 | prompt),
                family = beta_family(), data = d)
  coef_table(m2)
  m2s <- glmmTMB(consensus_credence ~ at_issue * format + encoding +
                   (1 + at_issue | item_id) + (1 | prompt),
                 family = beta_family(), data = d)
  cat(sprintf("with random at-issueness slope: at_issue b = %+.3f (p = %.3f); ",
              fixef(m2s)$cond["at_issuenot-at-issue"],
              summary(m2s)$coefficients$cond["at_issuenot-at-issue", 4]))
  cat(sprintf("AIC %.1f vs %.1f (lower wins)\n", AIC(m2), AIC(m2s)))

  cat("\n--- (3) Pairwise framing contrasts from (1), Tukey-adjusted ---\n")
  print(summary(contrast(emmeans(m1, ~ encoding), method = "pairwise"),
                infer = TRUE)[, c("contrast", "estimate", "lower.CL",
                                  "upper.CL", "p.value")])
}

cat("\n========================= Gemini challenge measure =========================\n")
cat("--- (4) Binomial GLMM on blinded challenge",
    " [RE: (1 + at_issue | item_id)] ---\n")
push <- read_pilot(MODELS[["Gemini 3.7 Flash"]]) |>
  filter(condition != "NO_CLAIM", !is.na(chal_blind_score)) |>
  mutate(at_issue = relevel(factor(at_issue), ref = "at-issue"),
         format   = relevel(factor(format),   ref = "conv"),
         challenged = as.integer(chal_blind_score == 1)) |>
  summarise(y = sum(challenged), n = n(), .by = c(item_id, format, at_issue))
p1 <- glmer(cbind(y, n - y) ~ at_issue * format + (1 + at_issue | item_id),
            family = binomial, data = push)
p0 <- update(p1, . ~ . - at_issue:format)
coef_table(p1)
cat("format x at-issueness interaction, likelihood ratio test:\n")
print(anova(p0, p1))
