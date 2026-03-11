# ================================================================
# NBA Steals Model — Backtesting & Calibration
# Adam Wickwire - @WalrusQuant
# 03/11/2026
# ================================================================
#
# OVERVIEW
# ----------------------------------------------------------------
# This script evaluates the steals model by generating predictions
# for games that have ALREADY HAPPENED, then comparing predicted
# probabilities to actual outcomes.
#
# This is NOT the same as the posterior predictive check in the
# fitting script. That checks in-sample fit. This script does
# proper out-of-sample evaluation:
#
#   1. Fit the model on games BEFORE a cutoff date
#   2. Predict steals for games AFTER the cutoff
#   3. Compare predictions to actual results
#
# WHAT WE MEASURE
# ----------------------------------------------------------------
#   CALIBRATION: When we say 30% chance of 2+ steals, does it
#     actually happen ~30% of the time? Plotted as a calibration
#     curve — perfect model sits on the diagonal.
#
#   ACCURACY: Mean absolute error between predicted and actual
#     steals. Broken down by player usage tier.
#
#   SHARPNESS: Are our probabilities meaningfully different from
#     base rates, or are we just predicting the average for everyone?
#     A sharp model separates high-steal players from low-steal ones.
#
#   LOG LOSS: Proper scoring rule that penalizes confident wrong
#     predictions more than uncertain ones.
#
# USAGE
# ----------------------------------------------------------------
#   Requires a fitted model. Run steals_model_fit.R first.
#
#   Option A: Use the existing fit and held-out recent games
#   Option B: Specify a train/test split date
#
# OUTPUTS
# ----------------------------------------------------------------
#   - Calibration plot (predicted prob vs observed frequency)
#   - MAE by player tier
#   - Log loss summary
#   - data/backtest_results.csv
# ================================================================

library(tidyverse)
library(rstan)

# ============================================================
# 1. LOAD MODEL AND DATA
# ============================================================

cat("Loading model and data...\n")

fit          <- readRDS("model/steals_model_fit.rds")
lambda_draws <- readRDS("model/lambda_draws.rds")
player_lookup <- read_csv("data/player_lookup.csv", show_col_types = FALSE)

std_constants <- read_csv("data/standardization_constants.csv", show_col_types = FALSE)
opp_tov_mean  <- std_constants$opp_tov_mean
opp_tov_sd    <- std_constants$opp_tov_sd

beta_opp_draws <- as.vector(rstan::extract(fit, "beta_opp")$beta_opp)
phi_draws      <- as.vector(rstan::extract(fit, "phi")$phi)
n_sims         <- length(beta_opp_draws)

# Load raw data for the test set
player_box <- read_csv("data/player_box.csv", show_col_types = FALSE)
team_box   <- read_csv("data/team_box.csv", show_col_types = FALSE)

# ============================================================
# 2. DEFINE TEST SET
# ============================================================
# Use the last N games as the test set.
# The model was fit on all data, so this is technically in-sample
# for the steal rates — but the minutes simulation and opponent
# matchup are still forward-looking predictions.
#
# For a true out-of-sample test, you'd refit the model on a
# training window and predict a held-out window. That's expensive
# (refitting Stan each time), so we do the simpler version first.
#
# Set TEST_DAYS to control how many days of games to evaluate.

TEST_DAYS <- 14  # last 2 weeks of available data

all_dates <- player_box %>%
  filter(season_type == 2, !is.na(minutes), minutes > 0) %>%
  distinct(game_date) %>%
  arrange(game_date) %>%
  pull(game_date)

cutoff_date <- max(all_dates) - TEST_DAYS
test_dates  <- all_dates[all_dates > cutoff_date]

cat(sprintf("Test set: %d days, from %s to %s\n",
            length(test_dates), min(test_dates), max(test_dates)))

# ============================================================
# 3. BUILD OPPONENT CONTEXT FOR TEST GAMES
# ============================================================
# Recompute season-long team tov rates using all data
# (in a true backtest you'd use only data up to each game date)

asg_teams <- c("STARS", "STRIPES", "WORLD", "RISING", "USA")

team_season_tov <- team_box %>%
  filter(
    season_type == 2,
    team_id > 0,
    !(team_abbreviation %in% asg_teams)
  ) %>%
  mutate(
    poss_est = field_goals_attempted - offensive_rebounds +
      total_turnovers + 0.44 * free_throws_attempted
  ) %>%
  group_by(team_id, team_abbreviation) %>%
  summarise(
    season_tov_rate = (sum(total_turnovers) / sum(poss_est)) * 100,
    season_avg_poss = mean(poss_est),
    .groups = "drop"
  )

# ============================================================
# 4. BUILD PLAYER MINUTES DISTRIBUTIONS
# ============================================================
# For each test game, ideally we'd use only games BEFORE that date.
# Simplified: use last 15 games from the full dataset.

player_clean <- player_box %>%
  filter(
    !is.na(minutes), minutes > 0,
    did_not_play == FALSE,
    season_type == 2,
    team_id > 0,
    !(team_abbreviation %in% asg_teams)
  )

player_minutes_dist <- player_clean %>%
  group_by(athlete_id) %>%
  arrange(desc(game_date)) %>%
  slice_head(n = 15) %>%
  summarise(
    min_mean   = mean(minutes),
    min_sd     = pmax(sd(minutes), 2),
    n_recent   = n(),
    .groups    = "drop"
  )

# ============================================================
# 5. GET ACTUAL OUTCOMES FOR TEST GAMES
# ============================================================

test_actuals <- player_clean %>%
  filter(game_date %in% test_dates) %>%
  select(
    game_id, game_date, athlete_id, athlete_display_name,
    team_id, team_abbreviation, opponent_team_id,
    minutes, steals
  )

cat(sprintf("Test set: %d player-game observations\n", nrow(test_actuals)))

# ============================================================
# 6. PREDICT FUNCTION (SAME AS PREDICTION SCRIPT)
# ============================================================

predict_steals <- function(player_id, opponent_team_id) {

  p_idx <- player_lookup %>%
    filter(athlete_id == player_id) %>%
    pull(player_idx)
  if (length(p_idx) == 0) return(NULL)

  lambda_p <- lambda_draws[, p_idx]

  opp_tov <- team_season_tov %>%
    filter(team_id == opponent_team_id) %>%
    pull(season_tov_rate)
  if (length(opp_tov) == 0) return(NULL)
  opp_z <- (opp_tov - opp_tov_mean) / opp_tov_sd

  min_d <- player_minutes_dist %>% filter(athlete_id == player_id)
  if (nrow(min_d) == 0) return(NULL)

  log_mu    <- log(min_d$min_mean^2 / sqrt(min_d$min_sd^2 + min_d$min_mean^2))
  log_sigma <- sqrt(log(1 + min_d$min_sd^2 / min_d$min_mean^2))
  sim_min   <- pmin(rlnorm(n_sims, log_mu, log_sigma), 48)

  opp_pace <- team_season_tov %>%
    filter(team_id == opponent_team_id) %>%
    pull(season_avg_poss)
  if (length(opp_pace) == 0) return(NULL)
  sim_poss <- (sim_min / 48) * opp_pace

  log_rate <- log(lambda_p) + beta_opp_draws * opp_z + log(sim_poss / 100)
  sim_steals <- rnbinom(n_sims, size = phi_draws, mu = exp(log_rate))
  return(sim_steals)
}

# ============================================================
# 7. GENERATE PREDICTIONS FOR TEST SET
# ============================================================

cat("Generating predictions for test set...\n")

results <- test_actuals %>%
  rowwise() %>%
  mutate(
    sims        = list(predict_steals(athlete_id, opponent_team_id)),
    pred_mean   = if (!is.null(sims)) mean(sims) else NA_real_,
    pred_median = if (!is.null(sims)) median(sims) else NA_real_,
    prob_0      = if (!is.null(sims)) mean(sims == 0) else NA_real_,
    prob_1plus  = if (!is.null(sims)) mean(sims >= 1) else NA_real_,
    prob_2plus  = if (!is.null(sims)) mean(sims >= 2) else NA_real_,
    prob_3plus  = if (!is.null(sims)) mean(sims >= 3) else NA_real_
  ) %>%
  ungroup() %>%
  filter(!is.na(pred_mean)) %>%
  select(-sims)

cat(sprintf("Generated predictions for %d player-games\n", nrow(results)))

# ============================================================
# 8. CALIBRATION ANALYSIS
# ============================================================
# For each probability threshold (P(1+), P(2+), P(3+)):
# Bin predictions into buckets, compute observed frequency,
# and compare to predicted probability.

calibration_check <- function(predicted_prob, actual_outcome, n_bins = 10) {
  # predicted_prob: vector of predicted probabilities
  # actual_outcome: binary vector (did the event happen?)

  df <- tibble(pred = predicted_prob, actual = actual_outcome) %>%
    filter(!is.na(pred))

  # Create bins
  df <- df %>%
    mutate(bin = cut(pred, breaks = seq(0, 1, length.out = n_bins + 1),
                     include.lowest = TRUE, labels = FALSE)) %>%
    group_by(bin) %>%
    summarise(
      n           = n(),
      pred_mean   = mean(pred),
      obs_freq    = mean(actual),
      .groups     = "drop"
    )

  return(df)
}

# Calibration for each threshold
cal_1plus <- calibration_check(results$prob_1plus, results$steals >= 1) %>%
  mutate(threshold = "1+ steals")
cal_2plus <- calibration_check(results$prob_2plus, results$steals >= 2) %>%
  mutate(threshold = "2+ steals")
cal_3plus <- calibration_check(results$prob_3plus, results$steals >= 3, n_bins = 8) %>%
  mutate(threshold = "3+ steals")

cal_all <- bind_rows(cal_1plus, cal_2plus, cal_3plus)

# Calibration plot
cal_plot <- cal_all %>%
  ggplot(aes(x = pred_mean, y = obs_freq, color = threshold, size = n)) +
  geom_abline(slope = 1, intercept = 0, lty = 2, color = "gray50") +
  geom_point(alpha = 0.8) +
  scale_color_manual(values = c("1+ steals" = "#3b82f6",
                                 "2+ steals" = "#f59e0b",
                                 "3+ steals" = "#ef4444")) +
  scale_size_continuous(range = c(2, 8), guide = "none") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Calibration Plot",
    subtitle = "Predicted probability vs observed frequency",
    x = "Predicted Probability",
    y = "Observed Frequency",
    color = "Threshold"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

print(cal_plot)

# ============================================================
# 9. ACCURACY METRICS
# ============================================================

# Overall MAE
mae_overall <- mean(abs(results$steals - results$pred_mean))
cat(sprintf("\n=== Accuracy Metrics ===\n"))
cat(sprintf("Overall MAE: %.3f steals\n", mae_overall))

# MAE by predicted steal tier
mae_by_tier <- results %>%
  mutate(tier = case_when(
    pred_mean >= 1.5 ~ "High (1.5+)",
    pred_mean >= 0.75 ~ "Mid (0.75-1.5)",
    TRUE ~ "Low (<0.75)"
  )) %>%
  group_by(tier) %>%
  summarise(
    n    = n(),
    mae  = mean(abs(steals - pred_mean)),
    rmse = sqrt(mean((steals - pred_mean)^2)),
    mean_predicted = mean(pred_mean),
    mean_actual    = mean(steals),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_predicted))

cat("\nMAE by prediction tier:\n")
print(mae_by_tier)

# MAE by position
mae_by_pos <- results %>%
  left_join(
    player_lookup %>% select(athlete_id, athlete_position_abbreviation),
    by = "athlete_id"
  ) %>%
  mutate(pos_group = case_when(
    athlete_position_abbreviation %in% c("G", "PG", "SG") ~ "Guard",
    athlete_position_abbreviation %in% c("F", "SF", "PF") ~ "Forward",
    athlete_position_abbreviation == "C" ~ "Center",
    TRUE ~ "Other"
  )) %>%
  group_by(pos_group) %>%
  summarise(
    n    = n(),
    mae  = mean(abs(steals - pred_mean)),
    mean_predicted = mean(pred_mean),
    mean_actual    = mean(steals),
    .groups = "drop"
  )

cat("\nMAE by position:\n")
print(mae_by_pos)

# ============================================================
# 10. LOG LOSS (PROPER SCORING RULE)
# ============================================================
# Log loss penalizes confident wrong predictions harshly.
# Lower is better. Random guessing at base rate is the baseline.

log_loss <- function(predicted_prob, actual_outcome) {
  # Clip probabilities to avoid log(0)
  eps <- 1e-6
  p <- pmax(pmin(predicted_prob, 1 - eps), eps)
  -mean(actual_outcome * log(p) + (1 - actual_outcome) * log(1 - p), na.rm = TRUE)
}

# Base rates for comparison
base_rate_1plus <- mean(results$steals >= 1)
base_rate_2plus <- mean(results$steals >= 2)
base_rate_3plus <- mean(results$steals >= 3)

cat(sprintf("\n=== Log Loss (lower is better) ===\n"))
cat(sprintf("                Model    Base Rate\n"))
cat(sprintf("P(1+ steals):   %.4f   %.4f\n",
            log_loss(results$prob_1plus, results$steals >= 1),
            log_loss(rep(base_rate_1plus, nrow(results)), results$steals >= 1)))
cat(sprintf("P(2+ steals):   %.4f   %.4f\n",
            log_loss(results$prob_2plus, results$steals >= 2),
            log_loss(rep(base_rate_2plus, nrow(results)), results$steals >= 2)))
cat(sprintf("P(3+ steals):   %.4f   %.4f\n",
            log_loss(results$prob_3plus, results$steals >= 3),
            log_loss(rep(base_rate_3plus, nrow(results)), results$steals >= 3)))

# ============================================================
# 11. SHARPNESS
# ============================================================
# How spread out are our probability predictions?
# A model that predicts 0.50 for everyone is useless even if calibrated.
# We want wide spread between high and low predictions.

cat(sprintf("\n=== Sharpness (spread of predictions) ===\n"))
cat(sprintf("P(1+) — sd: %.3f, range: [%.3f, %.3f]\n",
            sd(results$prob_1plus), min(results$prob_1plus), max(results$prob_1plus)))
cat(sprintf("P(2+) — sd: %.3f, range: [%.3f, %.3f]\n",
            sd(results$prob_2plus), min(results$prob_2plus), max(results$prob_2plus)))
cat(sprintf("P(3+) — sd: %.3f, range: [%.3f, %.3f]\n",
            sd(results$prob_3plus), min(results$prob_3plus), max(results$prob_3plus)))

# ============================================================
# 12. DISTRIBUTION OF ERRORS
# ============================================================

error_plot <- results %>%
  mutate(error = steals - pred_mean) %>%
  ggplot(aes(x = error)) +
  geom_histogram(binwidth = 0.25, fill = "#3b82f6", alpha = 0.7, color = "#1e3a5f") +
  geom_vline(xintercept = 0, lty = 2, color = "red") +
  labs(
    title = "Prediction Error Distribution",
    subtitle = "Actual steals minus predicted mean",
    x = "Error (actual - predicted)",
    y = "Count"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(error_plot)

# ============================================================
# 13. PREDICTED vs ACTUAL BY PLAYER (TOP PLAYERS)
# ============================================================

# For players with enough test games, compare predicted vs actual
player_accuracy <- results %>%
  group_by(athlete_id, athlete_display_name, team_abbreviation) %>%
  filter(n() >= 5) %>%
  summarise(
    games       = n(),
    actual_mean = mean(steals),
    pred_mean   = mean(pred_mean),
    mae         = mean(abs(steals - pred_mean)),
    .groups     = "drop"
  ) %>%
  arrange(desc(actual_mean))

cat("\n=== Player-Level Accuracy (min 5 test games) ===\n")
player_accuracy %>%
  head(20) %>%
  print(n = 20)

# Scatter: predicted vs actual player averages
player_scatter <- player_accuracy %>%
  ggplot(aes(x = pred_mean, y = actual_mean)) +
  geom_abline(slope = 1, intercept = 0, lty = 2, color = "red") +
  geom_point(aes(size = games), alpha = 0.6, color = "#3b82f6") +
  geom_text(
    data = player_accuracy %>% filter(actual_mean >= 1.2 | pred_mean >= 1.2),
    aes(label = athlete_display_name),
    size = 2.5, nudge_y = 0.05, color = "gray40"
  ) +
  scale_size_continuous(range = c(1, 5)) +
  coord_equal() +
  labs(
    title = "Predicted vs Actual Steal Averages",
    subtitle = "Each point is a player (min 5 test games)",
    x = "Predicted Mean Steals",
    y = "Actual Mean Steals",
    size = "Games"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(player_scatter)

# ============================================================
# 14. SAVE RESULTS
# ============================================================

write_csv(results, "data/backtest_results.csv")
write_csv(cal_all, "data/calibration_results.csv")
write_csv(player_accuracy, "data/player_backtest_accuracy.csv")

cat("\nBacktest complete. Results saved to data/\n")
