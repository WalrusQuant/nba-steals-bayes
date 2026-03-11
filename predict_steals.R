# ================================================================
# NBA Steals — Daily Prediction Script
# Adam Wickwire - @WalrusQuant
# 03/11/2026
# ================================================================
#
# OVERVIEW
# ----------------------------------------------------------------
# This script generates steal predictions for upcoming games using
# a PREVIOUSLY FITTED model. It does NOT refit the Stan model.
#
# Use this for daily predictions between model refits.
# Run the full fitting script (steals_model_fit.R) to update the
# model when you want to incorporate new game data — typically
# once a week or after significant roster changes.
#
# WHAT THIS SCRIPT DOES
# ----------------------------------------------------------------
#   1. Loads the saved model objects (posterior draws, lookups)
#   2. Pulls fresh schedule data from hoopR
#   3. Updates player minutes distributions from recent games
#   4. Updates team turnover rates from recent games
#   5. Generates steal predictions for all upcoming games
#   6. Saves predictions to CSV
#
# REQUIRED FILES (from the fitting script)
# ----------------------------------------------------------------
#   model/lambda_draws.rds          — posterior steal rate draws
#   model/steals_model_fit.rds      — full Stan fit (for beta_opp, phi)
#   data/player_lookup.csv          — player ID → index mapping
#   data/standardization_constants.csv — opp_tov mean/sd for z-scoring
#
# WHEN TO REFIT vs REPREDICT
# ----------------------------------------------------------------
#   REFIT (run steals_model_fit.R) when:
#     - A week+ of new games have been played
#     - Major trades or roster shakeups
#     - You want to update player steal rate estimates
#
#   REPREDICT (run this script) when:
#     - You just want fresh predictions for tonight's games
#     - Schedule has been updated (postponements, etc.)
#     - You want to update minutes distributions with recent games
# ================================================================

library(tidyverse)
library(hoopR)
library(rstan)

# ============================================================
# 1. LOAD SAVED MODEL OBJECTS
# ============================================================
# These were saved by the fitting script after Stan finished sampling.

cat("Loading saved model objects...\n")

fit          <- readRDS("model/steals_model_fit.rds")
lambda_draws <- readRDS("model/lambda_draws.rds")
player_lookup <- read_csv("data/player_lookup.csv", show_col_types = FALSE)

std_constants <- read_csv("data/standardization_constants.csv", show_col_types = FALSE)
opp_tov_mean  <- std_constants$opp_tov_mean
opp_tov_sd    <- std_constants$opp_tov_sd

# Extract posterior draws for beta_opp and phi
beta_opp_draws <- as.vector(rstan::extract(fit, "beta_opp")$beta_opp)
phi_draws      <- as.vector(rstan::extract(fit, "phi")$phi)
n_sims         <- length(beta_opp_draws)

cat(sprintf("Loaded model with %d players and %d posterior draws\n",
            ncol(lambda_draws), n_sims))

# ============================================================
# 2. PULL FRESH DATA FROM hoopR
# ============================================================
# We pull fresh data each run to get:
#   - Updated schedule (new games, postponements)
#   - Recent box scores (for minutes distributions)
#   - Updated team turnover rates

cat("Pulling fresh data from hoopR...\n")

player_box <- load_nba_player_box(2026)
team_box   <- load_nba_team_box(2026)
schedule   <- load_nba_schedule(2026)

# Remove All-Star game data
asg_teams <- c("STARS", "STRIPES", "WORLD", "RISING", "USA")
asg_game_ids <- player_box %>%
  filter(team_abbreviation %in% asg_teams) %>%
  distinct(game_id) %>%
  pull(game_id)

player_box <- player_box %>% filter(!(game_id %in% asg_game_ids))
team_box   <- team_box   %>% filter(!(game_id %in% asg_game_ids))
schedule   <- schedule    %>% filter(!(id %in% asg_game_ids))

# ============================================================
# 3. UPDATE TEAM TURNOVER RATES
# ============================================================
# Recompute season-long turnover rates using all games to date.
# This captures any recent changes in team tendencies.

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
# 4. UPDATE PLAYER MINUTES DISTRIBUTIONS
# ============================================================
# Recompute from the last 15 games so minutes reflect current role.
# This is important — a player who just got promoted to starter
# should show starter-level minutes, not their season average.

player_clean <- player_box %>%
  filter(
    !is.na(minutes),
    minutes > 0,
    did_not_play == FALSE,
    season_type == 2,
    team_id > 0
  )

player_minutes_dist <- player_clean %>%
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
# 5. BUILD ACTIVE ROSTER
# ============================================================
# Players who appeared in a game in the last 14 days.
# This naturally filters out injured/inactive players.

max_date <- max(player_clean$game_date, na.rm = TRUE)
recent_cutoff <- max_date - 14

active_roster <- player_clean %>%
  filter(game_date >= recent_cutoff) %>%
  distinct(athlete_id, team_id) %>%
  left_join(
    player_lookup %>% select(athlete_id, player_idx, athlete_display_name,
                             athlete_position_abbreviation, team_abbreviation),
    by = "athlete_id"
  ) %>%
  # Only keep players who exist in the fitted model
  filter(!is.na(player_idx))

cat(sprintf("Active roster: %d players across %d teams\n",
            nrow(active_roster), n_distinct(active_roster$team_id)))

# ============================================================
# 6. GET UPCOMING GAMES
# ============================================================

upcoming <- schedule %>%
  filter(status_type_name == "STATUS_SCHEDULED", season_type == 2) %>%
  select(game_id = id, game_date,
         home_id, home_abbreviation, home_name,
         away_id, away_abbreviation, away_name)

cat(sprintf("Found %d upcoming regular season games\n", nrow(upcoming)))

# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================
# For a given player and opponent, simulate 4000 steal outcomes:
#   1. Draw steal rate from posterior (lambda)
#   2. Simulate minutes from log-normal
#   3. Convert to possessions using opponent pace
#   4. Adjust for opponent turnover tendency
#   5. Draw steals from negative binomial

predict_steals <- function(player_id, opponent_team_id) {

  p_idx <- player_lookup %>%
    filter(athlete_id == player_id) %>%
    pull(player_idx)
  if (length(p_idx) == 0) return(NULL)

  # Posterior steal rate draws
  lambda_p <- lambda_draws[, p_idx]

  # Opponent turnover rate → standardize using training data constants
  opp_tov <- team_season_tov %>%
    filter(team_id == opponent_team_id) %>%
    pull(season_tov_rate)
  if (length(opp_tov) == 0) return(NULL)
  opp_z <- (opp_tov - opp_tov_mean) / opp_tov_sd

  # Player minutes distribution → log-normal simulation
  min_d <- player_minutes_dist %>% filter(athlete_id == player_id)
  if (nrow(min_d) == 0) return(NULL)

  log_mu    <- log(min_d$min_mean^2 / sqrt(min_d$min_sd^2 + min_d$min_mean^2))
  log_sigma <- sqrt(log(1 + min_d$min_sd^2 / min_d$min_mean^2))
  sim_min   <- pmin(rlnorm(n_sims, log_mu, log_sigma), 48)

  # Minutes → possessions using opponent's pace
  opp_pace <- team_season_tov %>%
    filter(team_id == opponent_team_id) %>%
    pull(season_avg_poss)
  sim_poss <- (sim_min / 48) * opp_pace

  # Log-rate = log(lambda) + beta_opp * opp_z + log(poss/100)
  log_rate <- log(lambda_p) + beta_opp_draws * opp_z + log(sim_poss / 100)

  # Draw steals from negative binomial
  sim_steals <- rnbinom(n_sims, size = phi_draws, mu = exp(log_rate))
  return(sim_steals)
}

# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

cat("\n=== Generating Predictions ===\n")
predictions <- list()

for (i in seq_len(nrow(upcoming))) {
  game <- upcoming[i, ]

  home_players <- active_roster %>% filter(team_id == game$home_id)
  away_players <- active_roster %>% filter(team_id == game$away_id)

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

  game_preds <- bind_rows(
    summarise_sims(home_players, game$away_id, "home"),
    summarise_sims(away_players, game$home_id, "away")
  ) %>%
    mutate(game_id = game$game_id, game_date = game$game_date)

  predictions[[i]] <- game_preds
}

all_predictions <- bind_rows(predictions)

# ============================================================
# 9. OUTPUT & SAVE
# ============================================================

cat("\n=== Steal Predictions for Upcoming Games ===\n")
all_predictions %>%
  filter(!is.na(pred_mean)) %>%
  arrange(game_date, game_id, desc(pred_mean)) %>%
  select(game_date, athlete_display_name, team_abbreviation,
         athlete_position_abbreviation, home_away,
         pred_mean, pred_median, prob_1plus, prob_2plus, prob_3plus) %>%
  print(n = 50)

# Save predictions (timestamped so you can track changes over time)
write_csv(all_predictions, "data/steals_predictions.csv")
write_csv(
  all_predictions,
  sprintf("data/steals_predictions_%s.csv", format(Sys.Date(), "%Y%m%d"))
)

cat("\nDone! Predictions saved to data/steals_predictions.csv\n")

today_predictions <- all_predictions %>% 
  filter(game_date == Sys.Date())

write_csv(today_predictions, "data/today_predictions.csv")