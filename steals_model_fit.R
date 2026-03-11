# ================================================================
# NBA Steals Prediction Model
# Adam Wickwire - @WalrusQuant
# 03/11/2026
# ================================================================
#
# OVERVIEW
# ----------------------------------------------------------------
# This script builds a hierarchical Bayesian model to predict NBA
# player steals on a per-game basis. The approach treats steals as
# a generative process:
#
#   1. Each player has a latent "true" steal rate per 100 possessions
#   2. That rate is partially pooled within position groups (G, F, C)
#   3. Each game, the rate is adjusted by the opponent's turnover tendency
#   4. Minutes are uncertain, so we model their distribution too
#   5. Steals are drawn from a Negative Binomial (handles overdispersion)
#
# The model is fit in Stan using rstan. After fitting on historical
# game logs, we generate full predictive distributions for upcoming
# games — giving us P(0 steals), P(1+ steals), P(2+ steals), etc.
#
# DATA SOURCES
# ----------------------------------------------------------------
# All data comes from ESPN via the hoopR package:
#   - player_box: player-level box scores (minutes, steals, etc.)
#   - team_box:   team-level box scores (turnovers, FGA, etc.)
#   - schedule:   game schedule with status (scheduled/completed)
#
# PIPELINE
# ----------------------------------------------------------------
#   Section 1:  Load data from hoopR
#   Section 2:  Remove All-Star game data
#   Section 3:  Clean player box scores
#   Section 4:  Build opponent turnover rates (rolling, no leakage)
#   Section 5:  Join opponent context onto player data
#   Section 6:  Create Stan index variables and standardize
#   Section 7:  Compute player minutes distributions for predictions
#   Section 8:  Assemble Stan data and fit the model
#   Section 9:  Diagnostics (divergences, traceplots, Rhat)
#   Section 10: Extract and summarize player steal rate posteriors
#   Section 11: Posterior predictive check (model validation)
#   Section 12: Predict steals for upcoming scheduled games
#   Section 13: Output and save results
#
# KEY OUTPUTS
# ----------------------------------------------------------------
#   - player_steal_rates.rds: posterior steal rates per 100 poss
#   - steals_predictions.rds: upcoming game predictions with probs
#   - steals_model_fit.rds:   full Stan fit object for further analysis
# ================================================================

library(tidyverse)
library(hoopR)
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ============================================================
# 1. LOAD DATA
# ============================================================
# hoopR pulls from ESPN API. Season 2026 = the 2025-26 NBA season.
# player_box: one row per player per game (~25k rows)
# team_box:   one row per team per game (~2k rows)
# schedule:   one row per game (~1200 rows, includes future games)

player_box <- load_nba_player_box(2026)
team_box   <- load_nba_team_box(2026)
schedule   <- load_nba_schedule(2026)

# ============================================================
# 2. REMOVE ALL-STAR GAME DATA
# ============================================================
# The All-Star game uses fake team names (STARS, STRIPES, etc.)
# and mixes players from different teams. These games would
# pollute both the player steal rate estimates and the team-level
# turnover calculations, so we remove them from ALL tables.

asg_teams <- c("STARS", "STRIPES", "WORLD", "RISING", "USA")

# Identify All-Star game IDs from team abbreviations
asg_game_ids <- player_box %>%
  filter(team_abbreviation %in% asg_teams) %>%
  distinct(game_id) %>%
  pull(game_id)

# Also check the schedule table for All-Star type flags
if ("type_abbreviation" %in% names(schedule)) {
  asg_schedule_ids <- schedule %>%
    filter(type_abbreviation == "ALLSTAR") %>%
    pull(id)
  asg_game_ids <- unique(c(asg_game_ids, asg_schedule_ids))
}

cat("Removing", length(asg_game_ids), "All-Star game IDs\n")

player_box <- player_box %>% filter(!(game_id %in% asg_game_ids))
team_box   <- team_box   %>% filter(!(game_id %in% asg_game_ids))
schedule   <- schedule    %>% filter(!(id %in% asg_game_ids))

# ============================================================
# 3. CLEAN PLAYER BOX SCORES
# ============================================================
# Keep only regular season games (season_type == 2) where the
# player actually played (has minutes, didn't DNP).
# Filter out any remaining non-NBA team IDs (just in case).

player_clean <- player_box %>%
  filter(
    !is.na(minutes),
    minutes > 0,
    did_not_play == FALSE,
    season_type == 2
  ) %>%
  filter(team_id > 0) %>%
  select(
    game_id, game_date, season,
    athlete_id, athlete_display_name,
    athlete_position_abbreviation,
    team_id, team_abbreviation,
    opponent_team_id, opponent_team_abbreviation,
    minutes, steals, starter
  )

# ============================================================
# 4. BUILD OPPONENT TURNOVER RATES
# ============================================================
# We need to know how turnover-prone each opponent is GOING INTO
# each game. This is critical for two reasons:
#   a) It's a real predictor — sloppy teams create more steal chances
#   b) We must avoid data leakage — can't use the current game's
#      turnovers to predict the current game's steals
#
# Approach: compute a cumulative turnover rate (turnovers per 100
# estimated possessions) for each team, updated after each game.
# Each game gets the rate calculated from ALL PRIOR games only.
#
# Possessions are estimated using the standard formula:
#   Poss ≈ FGA - OREB + TOV + 0.44 * FTA

team_tov <- team_box %>%
  filter(season_type == 2, team_id > 0) %>%
  select(game_id, game_date, team_id, team_abbreviation,
         turnovers, total_turnovers, steals,
         field_goals_attempted, offensive_rebounds,
         free_throws_attempted) %>%
  arrange(team_id, game_date)

# Estimate possessions and compute game-level turnover rate
team_tov <- team_tov %>%
  mutate(
    poss_est = field_goals_attempted - offensive_rebounds +
      total_turnovers + 0.44 * free_throws_attempted,
    tov_rate = (total_turnovers / poss_est) * 100
  )

# Rolling cumulative turnover rate — excludes current game
# First game of the season gets NA (no prior data available)
team_tov <- team_tov %>%
  group_by(team_id) %>%
  arrange(game_date, .by_group = TRUE) %>%
  mutate(
    cum_tov      = cumsum(total_turnovers) - total_turnovers,
    cum_poss     = cumsum(poss_est) - poss_est,
    games_played = row_number() - 1,
    opp_tov_rate_prior = if_else(
      games_played > 0,
      (cum_tov / cum_poss) * 100,
      NA_real_
    )
  ) %>%
  ungroup() %>%
  select(game_id, game_date, team_id, team_abbreviation,
         tov_rate, opp_tov_rate_prior, poss_est)

# Season-long turnover rate per team (for PREDICTION of future games).
# Unlike the rolling rate above, this uses all games to date —
# appropriate because we're predicting forward, not fitting.
team_season_tov <- team_box %>%
  filter(season_type == 2, team_id > 0) %>%
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
# 5. JOIN OPPONENT CONTEXT ONTO PLAYER DATA
# ============================================================
# Each player-game row needs two pieces of opponent info:
#   a) The opponent's rolling turnover rate (how sloppy they are)
#   b) The player's own team's pace that game (to estimate possessions)
#
# We also estimate the player's possessions played that game:
#   player_poss ≈ (minutes / 48) * team_total_possessions

# Join opponent's rolling tov rate
player_model <- player_clean %>%
  left_join(
    team_tov %>% select(game_id, team_id, opp_tov_rate_prior, poss_est),
    by = c("game_id" = "game_id", "opponent_team_id" = "team_id")
  )

# Join player's own team pace (possessions that game)
player_model <- player_model %>%
  left_join(
    team_tov %>% select(game_id, team_id, poss_est) %>%
      rename(team_poss = poss_est),
    by = c("game_id" = "game_id", "team_id" = "team_id")
  )

# Estimate player-level possessions from their share of team minutes
# A team plays 48 minutes total (5 players * 48 = 240 player-minutes)
player_model <- player_model %>%
  mutate(player_poss_est = (minutes / 48) * team_poss)

# Drop rows missing opponent data (first game of season) or possessions
player_model <- player_model %>%
  filter(!is.na(opp_tov_rate_prior), !is.na(player_poss_est))

# ============================================================
# 6. CREATE INDEX VARIABLES FOR STAN
# ============================================================
# Stan needs integer indices for players and positions.
# We also standardize the opponent turnover rate (z-score) so that
# the beta_opp coefficient is on a comparable scale.

# Player lookup: unique mapping from athlete_id → integer index
# For traded players, we use their most recent team
player_lookup <- player_model %>%
  distinct(athlete_id, athlete_display_name,
           athlete_position_abbreviation, team_id, team_abbreviation) %>%
  group_by(athlete_id) %>%
  slice_tail(n = 1) %>%
  ungroup() %>%
  arrange(athlete_id) %>%
  mutate(player_idx = row_number())

# Position mapping: ESPN uses both broad (G, F, C) and specific
# (PG, SG, SF, PF) abbreviations. We collapse to 3 groups:
#   1 = Guards (G, PG, SG)
#   2 = Forwards (F, SF, PF)
#   3 = Centers (C)
pos_lookup <- tibble(
  athlete_position_abbreviation = c("G", "PG", "SG", "F", "SF", "PF", "C"),
  pos_idx =                        c( 1,    1,    1,   2,    2,    2,   3)
)

# Join indices onto the model data
player_model <- player_model %>%
  left_join(player_lookup %>% select(athlete_id, player_idx), by = "athlete_id") %>%
  left_join(pos_lookup, by = "athlete_position_abbreviation") %>%
  filter(!is.na(pos_idx))

# Standardize opponent turnover rate (z-score)
# Save the mean and sd — we need these to standardize future opponents too
opp_tov_mean <- mean(player_model$opp_tov_rate_prior)
opp_tov_sd   <- sd(player_model$opp_tov_rate_prior)

player_model <- player_model %>%
  mutate(opp_tov_z = (opp_tov_rate_prior - opp_tov_mean) / opp_tov_sd)

# Player-level position vector (length = N_players, not N_observations)
# This tells Stan which position group each player belongs to
player_pos <- player_lookup %>%
  left_join(pos_lookup, by = "athlete_position_abbreviation") %>%
  arrange(player_idx)

# ============================================================
# 7. PLAYER MINUTES DISTRIBUTION (FOR PREDICTIONS)
# ============================================================
# For upcoming games, we don't know how many minutes a player will
# get. Instead of using a point estimate, we model the distribution
# of their recent minutes and simulate from it.
#
# We use the last 15 games to capture their CURRENT role — a player
# who was recently promoted to starter should reflect starter minutes,
# not their season-long average that includes bench time.
#
# The sd is floored at 2 minutes to avoid degenerate distributions
# for players with very consistent minutes.

player_minutes_dist <- player_model %>%
  group_by(athlete_id) %>%
  arrange(desc(game_date)) %>%
  slice_head(n = 15) %>%
  summarise(
    min_mean   = mean(minutes),
    min_sd     = pmax(sd(minutes), 2),
    min_median = median(minutes),
    n_recent   = n(),
    .groups    = "drop"
  )

# ============================================================
# 8. ASSEMBLE STAN DATA & FIT THE MODEL
# ============================================================
# The data list passed to Stan contains:
#   - N, N_players, N_positions: dimensions
#   - player: which player each obs belongs to (integer index)
#   - position: which position each player belongs to (integer index)
#   - steals: observed steals per game (response)
#   - log_poss: log(possessions/100) — the exposure offset
#   - opp_tov_z: standardized opponent turnover rate (covariate)
#
# Sampling settings:
#   - 4 chains, 1000 warmup + 1000 sampling each = 4000 posterior draws
#   - adapt_delta = 0.95 (cautious step size for hierarchical model)
#   - max_treedepth = 12 (allow deeper trees if needed)

stan_data <- list(
  N           = nrow(player_model),
  N_players   = max(player_model$player_idx),
  N_positions = 3,
  player      = player_model$player_idx,
  position    = player_pos$pos_idx,
  steals      = player_model$steals,
  log_poss    = log(player_model$player_poss_est / 100),
  opp_tov_z   = player_model$opp_tov_z
)

fit <- stan(
  file    = "steals_model.stan",
  data    = stan_data,
  chains  = 4,
  iter    = 2000,
  warmup  = 1000,
  seed    = 42,
  control = list(adapt_delta = 0.95, max_treedepth = 12)
)

# ============================================================
# 9. DIAGNOSTICS
# ============================================================
# Key things to check:
#   - Rhat should be ~1.00 for all parameters (chains agree)
#   - n_eff should be >100 at minimum, ideally >400
#   - 0 divergent transitions (model geometry is well-behaved)
#   - No saturated tree depth (sampler isn't struggling)
#   - Traceplots should look like "fuzzy caterpillars" (good mixing)

print(fit, pars = c("mu_pos", "sigma_pos", "beta_opp", "phi"))
check_hmc_diagnostics(fit)
traceplot(fit, pars = c("mu_pos", "sigma_pos", "beta_opp", "phi"))

# ============================================================
# 10. EXTRACT PLAYER STEAL RATE POSTERIORS
# ============================================================
# lambda[j] = player j's steal rate per 100 possessions
# This is the core output: each player's "true" steal ability
# with full uncertainty quantification.
#
# Players with many games get tight posteriors (we're confident).
# Players with few games get wide posteriors and are shrunk toward
# their position group mean (partial pooling doing its job).

lambda_draws <- as.matrix(fit, pars = "lambda")  # [4000 draws x N_players]

lambda_summary <- tibble(
  player_idx = 1:ncol(lambda_draws),
  mean  = colMeans(lambda_draws),
  sd    = apply(lambda_draws, 2, sd),
  q05   = apply(lambda_draws, 2, quantile, 0.05),
  q25   = apply(lambda_draws, 2, quantile, 0.25),
  q50   = apply(lambda_draws, 2, quantile, 0.50),
  q75   = apply(lambda_draws, 2, quantile, 0.75),
  q95   = apply(lambda_draws, 2, quantile, 0.95)
) %>%
  left_join(player_lookup, by = "player_idx") %>%
  arrange(desc(mean))

cat("\n=== Top 20 Steal Rates (per 100 possessions) ===\n")
lambda_summary %>%
  select(athlete_display_name, team_abbreviation,
         athlete_position_abbreviation, mean, q05, q50, q95) %>%
  head(20) %>%
  print(n = 20)

# ============================================================
# 11. POSTERIOR PREDICTIVE CHECK
# ============================================================
# Compare the model's predictions on TRAINING data to actual outcomes.
# For each observation in the data, Stan drew a "replicated" steal
# count from the fitted model. If the model is well-calibrated,
# the distribution of replicated steals should match the observed.
#
# This plot shows posterior mean prediction vs observed steals.
# Expect shrinkage: predictions compress toward 0-2 because
# the model won't chase rare 5+ steal outlier games.

steals_rep <- as.matrix(fit, pars = "steals_rep")
pred_mean  <- colMeans(steals_rep)

tibble(observed = player_model$steals, predicted = pred_mean) %>%
  ggplot(aes(x = observed, y = predicted)) +
  geom_abline(slope = 1, intercept = 0, lty = 2, color = "red") +
  geom_point(alpha = 0.05) +
  labs(title = "Posterior Predictive Check",
       x = "Observed Steals", y = "Predicted (Posterior Mean)") +
  theme_minimal()

# ============================================================
# 12. PREDICT UPCOMING GAMES
# ============================================================
# For each player in each upcoming game, we simulate steals by:
#
#   a) Drawing from their posterior steal rate (lambda)
#   b) Simulating minutes from a log-normal fit to their last 15 games
#      (log-normal keeps minutes positive and right-skewed, capped at 48)
#   c) Converting minutes → possessions using the opponent's season pace
#   d) Adjusting for opponent turnover tendency (season-long rate, z-scored)
#   e) Drawing steals from NegBinomial with posterior phi
#
# This produces 4000 simulated steal counts per player per game,
# from which we compute:
#   - pred_mean:  expected steals
#   - pred_median: median steals
#   - prob_Nplus: probability of N or more steals (for props/DFS)
#
# ACTIVE ROSTER: We define "active" as any player who appeared in a
# game within the last 14 days. This filters out injured/inactive players
# without needing an external injury report.

# Get upcoming scheduled regular season games
upcoming <- schedule %>%
  filter(status_type_name == "STATUS_SCHEDULED", season_type == 2) %>%
  select(game_id = id, game_date,
         home_id, home_abbreviation, home_name,
         away_id, away_abbreviation, away_name)

# Build active roster from recent appearances
recent_cutoff <- max(player_model$game_date) - 14
active_roster <- player_model %>%
  filter(game_date >= recent_cutoff) %>%
  distinct(athlete_id, team_id) %>%
  left_join(
    player_lookup %>% select(athlete_id, player_idx, athlete_display_name,
                             athlete_position_abbreviation, team_abbreviation),
    by = "athlete_id"
  )

# Extract posterior draws needed for prediction
beta_opp_draws <- as.vector(rstan::extract(fit, "beta_opp")$beta_opp)
phi_draws      <- as.vector(rstan::extract(fit, "phi")$phi)
n_sims         <- length(beta_opp_draws)  # 4000

# ---- Prediction function ----
# Takes a player ID and opponent team ID, returns a vector of
# 4000 simulated steal counts. Returns NULL if player/opponent
# data is missing (e.g., player not in model, opponent not found).
predict_steals <- function(player_id, opponent_team_id) {

  # Look up player index in the model
  p_idx <- player_lookup %>%
    filter(athlete_id == player_id) %>%
    pull(player_idx)
  if (length(p_idx) == 0) return(NULL)

  # Get 4000 posterior draws of this player's steal rate
  lambda_p <- lambda_draws[, p_idx]

  # Look up opponent's season turnover rate and standardize
  opp_tov <- team_season_tov %>%
    filter(team_id == opponent_team_id) %>%
    pull(season_tov_rate)
  if (length(opp_tov) == 0) return(NULL)
  opp_z <- (opp_tov - opp_tov_mean) / opp_tov_sd

  # Get this player's minutes distribution parameters
  min_d <- player_minutes_dist %>% filter(athlete_id == player_id)
  if (nrow(min_d) == 0) return(NULL)

  # Simulate minutes from log-normal distribution
  # Convert mean/sd to log-normal parameters:
  #   log_mu = log(mean^2 / sqrt(sd^2 + mean^2))
  #   log_sigma = sqrt(log(1 + sd^2 / mean^2))
  log_mu    <- log(min_d$min_mean^2 / sqrt(min_d$min_sd^2 + min_d$min_mean^2))
  log_sigma <- sqrt(log(1 + min_d$min_sd^2 / min_d$min_mean^2))
  sim_min   <- pmin(rlnorm(n_sims, log_mu, log_sigma), 48)  # cap at 48 min

  # Convert simulated minutes → estimated possessions
  # Uses the opponent's season average pace
  opp_pace <- team_season_tov %>%
    filter(team_id == opponent_team_id) %>%
    pull(season_avg_poss)
  sim_poss <- (sim_min / 48) * opp_pace

  # Compute log-rate for each simulation draw:
  #   log(steals) = log(lambda) + beta_opp * opp_z + log(poss/100)
  log_rate <- log(lambda_p) + beta_opp_draws * opp_z + log(sim_poss / 100)

  # Draw steals from negative binomial
  sim_steals <- rnbinom(n_sims, size = phi_draws, mu = exp(log_rate))
  return(sim_steals)
}

# ---- Loop through upcoming games ----
cat("\n=== Generating Predictions for Upcoming Games ===\n")
predictions <- list()

for (i in seq_len(nrow(upcoming))) {
  game <- upcoming[i, ]

  # Get active players for each team
  home_players <- active_roster %>% filter(team_id == game$home_id)
  away_players <- active_roster %>% filter(team_id == game$away_id)

  # Helper: run predictions for a roster and summarize
  summarise_sims <- function(roster, opp_id, side) {
    roster %>%
      rowwise() %>%
      mutate(
        sims        = list(predict_steals(athlete_id, opp_id)),
        pred_mean   = if (!is.null(sims)) mean(sims) else NA_real_,
        pred_median = if (!is.null(sims)) median(sims) else NA_real_,
        prob_0      = if (!is.null(sims)) mean(sims == 0) else NA_real_,
        prob_1plus  = if (!is.null(sims)) mean(sims >= 1) else NA_real_,
        prob_2plus  = if (!is.null(sims)) mean(sims >= 2) else NA_real_,
        prob_3plus  = if (!is.null(sims)) mean(sims >= 3) else NA_real_
      ) %>%
      ungroup() %>%
      mutate(home_away = side, opponent_id = opp_id) %>%
      select(-sims)
  }

  # Predict for both sides
  game_preds <- bind_rows(
    summarise_sims(home_players, game$away_id, "home"),
    summarise_sims(away_players, game$home_id, "away")
  ) %>%
    mutate(game_id = game$game_id, game_date = game$game_date)

  predictions[[i]] <- game_preds
}

all_predictions <- bind_rows(predictions)

# ============================================================
# 13. OUTPUT & SAVE
# ============================================================

cat("\n=== Steal Predictions for Upcoming Games ===\n")
all_predictions %>%
  filter(!is.na(pred_mean)) %>%
  arrange(game_date, game_id, desc(pred_mean)) %>%
  select(game_date, athlete_display_name, team_abbreviation,
         athlete_position_abbreviation, home_away,
         pred_mean, pred_median, prob_1plus, prob_2plus, prob_3plus) %>%
  print(n = 50)

# ---- Save model fit and supporting objects ----
# The Stan fit object must be RDS (complex S4 object).
# Lambda draws is a large matrix (4000 x N_players), also RDS.
# Everything else goes to CSV for portability.

# Model objects (needed to re-run predictions without refitting)
saveRDS(fit, "model/steals_model_fit.rds")
saveRDS(lambda_draws, "model/lambda_draws.rds")

# Lookup tables and supporting data (CSV for portability)
write_csv(player_lookup, "data/player_lookup.csv")
write_csv(player_minutes_dist, "data/player_minutes_dist.csv")
write_csv(team_season_tov, "data/team_season_tov.csv")

# Standardization constants (needed to z-score future opponents)
tibble(opp_tov_mean = opp_tov_mean, opp_tov_sd = opp_tov_sd) %>%
  write_csv("data/standardization_constants.csv")

# Results
write_csv(all_predictions, "data/steals_predictions.csv")
write_csv(lambda_summary, "data/player_steal_rates.csv")
