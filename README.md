# NBA Steals Prediction Model

**Author:** Adam Wickwire — [\@WalrusQuant](https://twitter.com/WalrusQuant)\
**Date:** March 11, 2026\
**Language:** R + Stan

A hierarchical Bayesian model that estimates each NBA player's true steal rate per 100 possessions and generates full predictive distributions for upcoming games. Built for DFS, props markets, and analytical research.

------------------------------------------------------------------------

## Why This Model Exists

Raw steal averages are noisy. A player who averages 1.5 steals per game might get 0 one night and 5 the next. Counting stats don't separate skill from opportunity — a player on a fast-paced team who plays 36 minutes has far more chances to record a steal than a bench player on a slow team who plays 12.

This model solves both problems:

-   **Separates rate from volume.** We estimate steals per 100 possessions, not per game. This gives a "pace-and-minutes-neutral" view of each player's steal ability.
-   **Quantifies uncertainty.** Instead of a single number, every player gets a full posterior distribution. Players with many games get tight estimates; players with few games get wider intervals and are pulled toward their position group average (partial pooling).
-   **Predicts distributions, not points.** For any upcoming game, the model outputs the probability of 0, 1, 2, 3+ steals — not just an expected value. This is directly useful for over/under props.

------------------------------------------------------------------------

## Model Overview

### The Generative Story

The model treats each observed steal count as the outcome of a layered generative process:

1.  **Position group.** Each position (Guard, Forward, Center) has a population-level distribution of steal rates. Guards steal more than forwards, forwards more than centers. This is learned from the data, not hardcoded.

2.  **Player ability.** Each player draws a baseline steal rate (λ) from their position group. This is the core parameter — a player's "true" steal talent per 100 possessions. Players with limited data get shrunk toward the position mean (partial pooling), which prevents overfitting to small samples.

3.  **Opponent adjustment.** Each game, the steal rate is adjusted by how turnover-prone the opponent is. Facing a careless team nudges the rate up; facing a disciplined team nudges it down. This enters multiplicatively on the log scale.

4.  **Minutes uncertainty.** For future games, we don't know how many minutes a player will get. We model their recent minutes as a log-normal distribution (positive, right-skewed, capped at 48) and simulate from it.

5.  **Steal count.** Given the effective rate and estimated possessions, steals are drawn from a Negative Binomial distribution. This handles overdispersion — the tendency for steals to be "burstier" than a simple Poisson model would predict.

### Mathematical Specification

```         
Position level:
  mu_pos[k] ~ Normal(0, 0.5)           — mean log-steal-rate for position k
  sigma_pos[k] ~ Exponential(2)         — spread within position k

Player level (non-centered):
  log_lambda[j] = mu_pos[pos[j]] + sigma_pos[pos[j]] * z[j]
  z[j] ~ Normal(0, 1)

Opponent adjustment:
  beta_opp ~ Normal(0, 0.3)

Observation level:
  log_mu[i] = log_lambda[player[i]] + beta_opp * opp_tov_z[i] + log(poss[i]/100)
  steals[i] ~ NegBinomial2(exp(log_mu[i]), phi)

Overdispersion:
  phi ~ Exponential(0.5)
```

### Key Parameters

| Parameter       | Meaning                                   | Typical Value |
|-----------------|---------------------------------------|-----------------|
| `mu_pos[1]` (G) | Average guard steal rate (log scale)      | \~0.53        |
| `mu_pos[2]` (F) | Average forward steal rate (log scale)    | \~0.42        |
| `mu_pos[3]` (C) | Average center steal rate (log scale)     | \~0.25        |
| `sigma_pos`     | Within-position spread                    | 0.24–0.32     |
| `beta_opp`      | Opponent turnover effect                  | \~0.05        |
| `phi`           | Overdispersion (higher = less bursty)     | \~22          |
| `lambda[j]`     | Player j's steal rate per 100 possessions | 0.3–4.0       |

------------------------------------------------------------------------

## Project Structure

```         
nba-steals-model/
│
├── steals_model_fit.R        # Full pipeline: data → fit → predictions
├── predict_steals.R          # Daily predictions using saved model
├── steals_model.stan         # Stan model specification
├── README.md
│
├── model/                    # Saved model objects (created by fitting script)
│   ├── steals_model_fit.rds  # Full Stan fit object (~large file)
│   └── lambda_draws.rds      # Posterior steal rate matrix [4000 x N_players]
│
└── data/                     # Input data and output predictions
    ├── player_box.csv        # Raw player box scores from hoopR
    ├── team_box.csv          # Raw team box scores from hoopR
    ├── schedule.csv          # Season schedule from hoopR
    ├── player_lookup.csv     # Player ID → model index mapping
    ├── player_minutes_dist.csv
    ├── team_season_tov.csv
    ├── standardization_constants.csv
    ├── player_steal_rates.csv    # Posterior steal rate summaries
    └── steals_predictions.csv    # Upcoming game predictions
```

------------------------------------------------------------------------

## Setup

### Requirements

-   **R** (4.x recommended)
-   **rstan** — R interface to Stan
-   **tidyverse** — data manipulation and plotting
-   **hoopR** — ESPN API wrapper for NBA data

### Installation

``` r
install.packages(c("tidyverse", "rstan", "hoopR"))
```

If `rstan` gives you trouble (it often does), consult the [RStan Getting Started Guide](https://mc-stan.org/rstan/articles/rstan.html). The most common issue is C++ toolchain configuration.

### Directory Setup

Create the output directories before your first run:

``` r
dir.create("model", showWarnings = FALSE)
dir.create("data", showWarnings = FALSE)
```

------------------------------------------------------------------------

## Usage

### First Run: Fit the Model

Run the full fitting script to pull data, fit the Stan model, and generate initial predictions:

``` r
source("steals_model_fit.R")
```

This will:

1.  Pull current season data from ESPN via hoopR
2.  Clean the data and remove All-Star game entries
3.  Build rolling opponent turnover rates (no data leakage)
4.  Fit the hierarchical model in Stan (\~5–15 minutes depending on hardware)
5.  Run diagnostics and posterior predictive checks
6.  Generate predictions for all upcoming scheduled games
7.  Save everything to `model/` and `data/`

**Check the diagnostics before trusting the results.** You want: - 0 divergent transitions - All Rhat values ≈ 1.00 - Effective sample sizes (n_eff) \> 400 - Traceplots that look like "fuzzy caterpillars" (good mixing)

### Daily Predictions

Once the model is fitted, run predictions daily without refitting:

``` r
source("predict_steals.R")
```

This loads the saved model, pulls fresh schedule and box score data, updates minutes distributions and team turnover rates, and generates new predictions. Runs in seconds.

A timestamped copy is saved automatically (e.g., `steals_predictions_20260311.csv`) so you can track how predictions shift over time.

### When to Refit

| Scenario                                | Action               |
|-----------------------------------------|----------------------|
| Daily predictions for tonight's games   | `predict_steals.R`   |
| A week of new games have been played    | `steals_model_fit.R` |
| Trade deadline / major roster moves     | `steals_model_fit.R` |
| Player returns from long injury absence | `steals_model_fit.R` |
| Schedule changes or postponements       | `predict_steals.R`   |

------------------------------------------------------------------------

## Understanding the Output

### Player Steal Rates (`player_steal_rates.csv`)

Each row is a player with their posterior steal rate distribution:

| Column | Description |
|----|----|
| `athlete_display_name` | Player name |
| `team_abbreviation` | Current team |
| `athlete_position_abbreviation` | Position |
| `mean` | Posterior mean steal rate per 100 possessions |
| `q05`, `q25`, `q50`, `q75`, `q95` | Posterior quantiles |
| `sd` | Posterior standard deviation |

Players with high `sd` relative to their `mean` have more uncertain estimates — typically low-minute or recently acquired players.

### Game Predictions (`steals_predictions.csv`)

Each row is a player-game prediction:

| Column                 | Description                               |
|------------------------|-------------------------------------------|
| `game_date`            | Date of the upcoming game                 |
| `athlete_display_name` | Player name                               |
| `team_abbreviation`    | Player's team                             |
| `home_away`            | Whether the player's team is home or away |
| `pred_mean`            | Expected steals for this game             |
| `pred_median`          | Median predicted steals                   |
| `prob_0`               | Probability of 0 steals                   |
| `prob_1plus`           | Probability of 1 or more steals           |
| `prob_2plus`           | Probability of 2 or more steals           |
| `prob_3plus`           | Probability of 3 or more steals           |

The `prob_Nplus` columns are directly comparable to prop lines. If the model gives a player `prob_2plus = 0.35` and the book is offering Over 1.5 Steals at -110 (implied \~52%), that's a pass. If they're offering it at +200 (implied \~33%), there might be value.

------------------------------------------------------------------------

## Data Pipeline Details

### Data Source

All data comes from ESPN via the [hoopR](https://hoopr.sportsdataverse.org/) package. This pulls:

-   **Player box scores** — one row per player per game with minutes, steals, and all standard stats (\~25k rows for a full season)
-   **Team box scores** — one row per team per game with turnovers, FGA, FTA, etc. (\~2k rows)
-   **Schedule** — all games including future scheduled games with status flags

### Data Cleaning

-   **All-Star games** are removed by detecting fake team abbreviations (STARS, STRIPES, RISING, etc.) and dropping those `game_id` values from all tables.
-   **DNP rows** (did not play, NA minutes, 0 minutes) are filtered out.
-   **Regular season only** — `season_type == 2` filters out preseason and playoffs.
-   **Position standardization** — ESPN uses both broad (G, F, C) and specific (PG, SG, SF, PF) abbreviations. All are mapped to three groups: Guard (1), Forward (2), Center (3).

### Possessions Estimation

Player possessions are not directly available in box score data. We estimate them using the standard formula:

```         
Team possessions ≈ FGA - OREB + TOV + 0.44 * FTA
Player possessions ≈ (player_minutes / 48) * team_possessions
```

The 0.44 FTA coefficient accounts for and-ones, technical free throws, and three-shot fouls that don't end possessions.

### Opponent Turnover Rate

For model **fitting**, we use a rolling cumulative turnover rate — each game gets the opponent's rate computed from all their prior games only. This prevents data leakage (using the current game's turnovers to predict the current game's steals).

For **prediction**, we use the opponent's full season-long turnover rate, since we're forecasting forward and all games to date are fair game.

The rate is standardized (z-scored) before entering the model so that `beta_opp` is interpretable: a 1 SD increase in opponent turnover tendency corresponds to a \~5% multiplicative increase in steal rate.

### Active Roster Detection

Rather than maintaining an external injury report, we define "active" as any player who appeared in a game within the last 14 days. This naturally excludes injured, G-League, and inactive players.

The tradeoff is that a player returning from a 15+ day absence won't appear in predictions until they play a game. For known returns from injury, you could manually add them to the `active_roster` dataframe.

------------------------------------------------------------------------

## Limitations and Future Work

### Current Limitations

-   **Single-season data.** The model currently fits on one season only. Multi-season data would give better estimates for established players and let us model year-over-year changes.
-   **No home/away effect.** Some players may steal at different rates at home vs away. This could be added as another covariate.
-   **No matchup granularity.** We use team-level opponent turnover rates. Position-specific opponent turnover rates (e.g., "turnovers allowed to opposing guards") would be more precise.
-   **No rest/back-to-back adjustment.** Minutes distributions don't account for fatigue or rest days.
-   **Static steal rate within season.** A player's λ is estimated as a single value for the season. A time-varying model could capture mid-season improvements or declines.

### Possible Extensions

-   **Add position-specific opponent adjustment** — opponent steals allowed to G/F/C
-   **Time-varying steal rates** — Gaussian process or random walk on λ
-   **Home/away and rest covariates**
-   **Multi-season hierarchical structure** — player-level priors informed by prior seasons
-   **Calibration analysis** — bin predicted probabilities and compare to observed frequencies
-   **Extend to other counting stats** — blocks, assists, rebounds using the same framework

------------------------------------------------------------------------

## License

This project is for personal and educational use.
