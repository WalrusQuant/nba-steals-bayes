// ================================================================
// steals_model.stan
// Hierarchical Bayesian Model for NBA Player Steals
// Adam Wickwire - @WalrusQuant
// 03/11/2026
// ================================================================
//
// MODEL OVERVIEW
// ----------------------------------------------------------------
// This model estimates each NBA player's "true" steal rate per 100
// possessions, then uses that rate to predict steals in future games.
//
// The key insight is that observed steals are noisy — a player who
// averages 1.5 steals/game might get 0 one night and 5 the next.
// Rather than treating raw averages as truth, we estimate a posterior
// distribution over each player's latent steal rate, borrowing
// strength across players within the same position group.
//
// GENERATIVE STORY
// ----------------------------------------------------------------
// 1. Each POSITION GROUP (G, F, C) has a population-level distribution
//    of steal rates on the log scale: Normal(mu_pos, sigma_pos).
//    Guards steal more than forwards, forwards more than centers.
//
// 2. Each PLAYER draws a baseline steal rate (lambda) from their
//    position group. Players with few games get pulled ("shrunk")
//    toward their position average — this is partial pooling.
//
// 3. Each game, the player's effective steal rate is ADJUSTED by
//    the opponent's turnover tendency. Facing a careless team
//    increases steal opportunities; facing a disciplined team
//    decreases them. This enters multiplicatively (additively on
//    the log scale).
//
// 4. STEALS are drawn from a Negative Binomial distribution:
//      steals ~ NegBinomial(effective_rate * exposure, phi)
//    where exposure = estimated possessions / 100 (from minutes
//    and team pace). The Negative Binomial handles overdispersion —
//    steals are "burstier" than a Poisson would predict.
//
// WHAT WE GET OUT
// ----------------------------------------------------------------
// - lambda[j]: posterior distribution of player j's steal rate
//              per 100 possessions (used for predictions)
// - beta_opp:  how much opponent turnover tendency matters
// - phi:       overdispersion (higher = more Poisson-like)
// - steals_rep: posterior predictive draws for model checking
//
// FOR PREDICTIONS (done in R, not Stan):
//   1. Simulate minutes from each player's recent distribution
//   2. Convert minutes → possessions using opponent pace
//   3. Apply opponent turnover adjustment
//   4. Draw steals from NegBinomial using posterior lambda + phi
//   → gives full predictive distribution (prob of 0, 1, 2, 3+...)
// ================================================================

data {
  int<lower=1> N;                         // total observations (player-games)
  int<lower=1> N_players;                 // number of unique players
  int<lower=1> N_positions;               // number of positions (G=1, F=2, C=3)

  array[N] int<lower=1, upper=N_players> player;     // which player for each obs
  array[N_players] int<lower=1, upper=N_positions> position;  // which position for each player

  array[N] int<lower=0> steals;           // observed steals (response variable)
  vector[N] log_poss;                      // log(player_possessions / 100) — exposure offset
  vector[N] opp_tov_z;                     // standardized opponent turnover rate per 100 poss
}

parameters {
  // ---- Position-level hyperparameters (log-steal-rate scale) ----
  // These define the "population" of steal rates for each position.
  // mu_pos[1] should be highest (guards), mu_pos[3] lowest (centers).
  vector[N_positions] mu_pos;              // mean log-steal-rate per position
  vector<lower=0>[N_positions] sigma_pos;  // spread of steal rates within position

  // ---- Player-level parameters (non-centered) ----
  // Raw z-scores that get scaled by position sigma.
  // Non-centered parameterization: log_lambda[j] = mu_pos[pos] + sigma_pos[pos] * raw[j]
  // This helps Stan sample efficiently when there are many players.
  vector[N_players] log_lambda_raw;

  // ---- Opponent effect ----
  // How much does facing a turnover-prone team boost steal rate?
  // Positive = more opponent turnovers → more steals (expected).
  real beta_opp;

  // ---- Overdispersion ----
  // Controls how "bursty" steals are beyond Poisson variance.
  // Higher phi = less overdispersion (closer to Poisson).
  // Lower phi = more variance (steal counts are lumpier).
  real<lower=0> phi;
}

transformed parameters {
  // ---- Construct player-level log steal rates ----
  // Non-centered: each player's log-rate = position mean + position sd * z-score
  vector[N_players] log_lambda;
  for (j in 1:N_players) {
    log_lambda[j] = mu_pos[position[j]] + sigma_pos[position[j]] * log_lambda_raw[j];
  }

  // ---- Effective log-rate per observation ----
  // Combines: player ability + opponent context + exposure (possessions)
  //   log(E[steals]) = log(lambda) + beta_opp * opp_tov_z + log(poss/100)
  //
  // On the natural scale this is multiplicative:
  //   E[steals] = lambda * exp(beta_opp * opp_tov_z) * (poss/100)
  vector[N] log_mu;
  for (i in 1:N) {
    log_mu[i] = log_lambda[player[i]] + beta_opp * opp_tov_z[i] + log_poss[i];
  }
}

model {
  // =============== PRIORS ===============

  // Position-level: where should steal rates live on the log scale?
  // log(1.5 steals/100poss) ≈ 0.4, log(0.5) ≈ -0.7, log(3.0) ≈ 1.1
  // So Normal(0, 0.5) covers the plausible range well.
  mu_pos ~ normal(0, 0.5);

  // How spread out are players within a position?
  // Exponential(2) → mean 0.5, keeps it from blowing up.
  sigma_pos ~ exponential(2);

  // Player z-scores: standard normal by construction
  log_lambda_raw ~ std_normal();

  // Opponent effect: weakly informative, centered at 0
  // Normal(0, 0.3) allows moderate effects but prevents wild swings
  beta_opp ~ normal(0, 0.3);

  // Overdispersion: exponential(0.5) → mean 2, allows wide range
  // Observed phi ≈ 20-30 means steals aren't hugely overdispersed
  phi ~ exponential(0.5);

  // =============== LIKELIHOOD ===============
  // Negative Binomial (log parameterization):
  //   E[steals] = exp(log_mu)
  //   Var[steals] = E[steals] + E[steals]^2 / phi
  // When phi → ∞, this converges to Poisson.
  steals ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // ---- Player steal rates on natural scale (for reporting) ----
  // lambda[j] = steals per 100 possessions for player j
  // This is the "headline number" — comparable across players
  // regardless of minutes or team pace.
  vector[N_players] lambda;
  lambda = exp(log_lambda);

  // ---- Posterior predictive draws (for model checking) ----
  // Simulate "fake" steals for every observation in the training data.
  // Compare these to actual steals to assess model fit.
  array[N] int steals_rep;
  for (i in 1:N) {
    real mu_clamped = fmin(log_mu[i], 5.0);  // prevent numerical overflow
    steals_rep[i] = neg_binomial_2_log_rng(mu_clamped, phi);
  }
}
