# ================================================================
# NBA Steals Predictions — Shiny Dashboard
# Adam Wickwire - @WalrusQuant
# 03/11/2026
# ================================================================
#
# USAGE
# ----------------------------------------------------------------
# 1. Place this file (app.R) in its own directory
# 2. Put today_predictions.csv in the same directory (or data/)
# 3. Run: shiny::runApp()
#
# REQUIREMENTS
# ----------------------------------------------------------------
# install.packages(c("shiny", "DT", "tidyverse"))
# ================================================================

library(shiny)
library(DT)
library(tidyverse)

# ============================================================
# LOAD & PREP DATA
# ============================================================

# Try a few paths for the predictions CSV
pred_path <- if (file.exists("today_predictions.csv")) {
  "today_predictions.csv"
} else if (file.exists("data/today_predictions.csv")) {
  "data/today_predictions.csv"
} else {
  stop("Cannot find today_predictions.csv — place it in the app directory or data/")
}

preds_raw <- read_csv(pred_path, show_col_types = FALSE)

# Clean: drop NAs, select display columns, format
preds <- preds_raw %>%
  filter(!is.na(pred_mean)) %>%
  transmute(
    Game = game_date,
    Player = athlete_display_name,
    Pos = athlete_position_abbreviation,
    Team = team_abbreviation,
    Side = toupper(home_away),
    `E[STL]` = round(pred_mean, 2),
    Median = as.integer(pred_median),
    `P(1+)` = round(prob_1plus, 3),
    `P(2+)` = round(prob_2plus, 3),
    `P(3+)` = round(prob_3plus, 3),
    game_id = game_id,
    opponent_id = opponent_id
  )

# Build game labels for the filter dropdown
game_labels <- preds %>%
  group_by(game_id, Game) %>%
  summarise(
    away = first(Team[Side == "AWAY"]),
    home = first(Team[Side == "HOME"]),
    .groups = "drop"
  ) %>%
  mutate(label = paste0(away, " @ ", home, " (", Game, ")")) %>%
  arrange(Game)

# ============================================================
# UI
# ============================================================

ui <- fluidPage(

  # Custom CSS for dark theme
  tags$head(tags$style(HTML("
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

    body {
      background-color: #0b1120;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
    }

    .container-fluid { max-width: 1100px; }

    h2 {
      color: #f1f5f9;
      font-weight: 700;
      font-size: 22px;
      letter-spacing: -0.02em;
      margin-bottom: 2px;
    }

    .subtitle {
      font-size: 11px;
      color: #475569;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 20px;
    }

    .well {
      background-color: #0f172a !important;
      border: 1px solid #1e293b !important;
      border-radius: 8px !important;
      color: #e2e8f0 !important;
    }

    label { color: #94a3b8 !important; font-size: 12px !important; }

    .selectize-input, .selectize-dropdown {
      background: #1e293b !important;
      color: #e2e8f0 !important;
      border-color: #334155 !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 12px !important;
    }
    .selectize-dropdown-content .option {
      color: #e2e8f0 !important;
    }
    .selectize-dropdown-content .active {
      background: #1e3a5f !important;
      color: #93c5fd !important;
    }

    .form-control {
      background: #1e293b !important;
      color: #e2e8f0 !important;
      border-color: #334155 !important;
      font-family: 'JetBrains Mono', monospace !important;
    }

    .irs-bar, .irs-bar-edge { background: #3b82f6 !important; border-color: #3b82f6 !important; }
    .irs-single, .irs-from, .irs-to { background: #3b82f6 !important; }
    .irs-line { background: #1e293b !important; }
    .irs-grid-text { color: #475569 !important; }
    .irs-min, .irs-max { color: #475569 !important; background: transparent !important; }
    .irs-handle { border-color: #3b82f6 !important; }

    /* DT table styling */
    table.dataTable {
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 13px !important;
      border-collapse: collapse !important;
    }
    table.dataTable thead th {
      background: #0f172a !important;
      color: #64748b !important;
      border-bottom: 2px solid #1e293b !important;
      font-size: 11px !important;
      font-weight: 600 !important;
      letter-spacing: 0.05em !important;
      text-transform: uppercase !important;
      padding: 10px 8px !important;
    }
    table.dataTable thead th.sorting_asc,
    table.dataTable thead th.sorting_desc {
      color: #f1f5f9 !important;
      border-bottom-color: #3b82f6 !important;
    }
    table.dataTable tbody td {
      background: #0b1120 !important;
      color: #e2e8f0 !important;
      border-bottom: 1px solid #131d35 !important;
      padding: 8px !important;
    }
    table.dataTable tbody tr:nth-child(even) td {
      background: #0f1629 !important;
    }
    table.dataTable tbody tr:hover td {
      background: #131d35 !important;
    }
    table.dataTable.no-footer {
      border-bottom: 1px solid #1e293b !important;
    }

    .dataTables_wrapper .dataTables_length,
    .dataTables_wrapper .dataTables_filter,
    .dataTables_wrapper .dataTables_info,
    .dataTables_wrapper .dataTables_paginate {
      color: #64748b !important;
      font-size: 11px !important;
    }
    .dataTables_wrapper .dataTables_filter input {
      background: #1e293b !important;
      color: #e2e8f0 !important;
      border: 1px solid #334155 !important;
      border-radius: 4px !important;
      padding: 4px 8px !important;
    }
    .dataTables_wrapper .dataTables_length select {
      background: #1e293b !important;
      color: #e2e8f0 !important;
      border: 1px solid #334155 !important;
    }
    .dataTables_wrapper .dataTables_paginate .paginate_button {
      color: #64748b !important;
      background: transparent !important;
      border: 1px solid #1e293b !important;
    }
    .dataTables_wrapper .dataTables_paginate .paginate_button.current {
      color: #93c5fd !important;
      background: #1e3a5f !important;
      border-color: #3b82f6 !important;
    }
    .dataTables_wrapper .dataTables_paginate .paginate_button:hover {
      color: #e2e8f0 !important;
      background: #1e293b !important;
      border-color: #334155 !important;
    }

    .footer-text {
      font-size: 10px;
      color: #334155;
      margin-top: 16px;
      display: flex;
      justify-content: space-between;
    }
  "))),

  # Header
  div(style = "margin-top: 24px;",
    h2("Steal Predictions"),
    div(class = "subtitle",
        "HIERARCHICAL BAYESIAN MODEL \u2014 NEG. BINOMIAL \u2014 4,000 POSTERIOR DRAWS")
  ),

  # Controls
  fluidRow(
    column(4,
      selectInput("game_filter", "Game",
        choices = c("All Games" = "all", setNames(game_labels$game_id, game_labels$label)),
        selected = "all"
      )
    ),
    column(3,
      selectInput("pos_filter", "Position",
        choices = c("All" = "all", "Guards" = "G", "Forwards" = "F", "Centers" = "C"),
        selected = "all"
      )
    ),
    column(3,
      sliderInput("min_mean", "Min E[STL]",
        min = 0, max = 2, value = 0.5, step = 0.1
      )
    )
  ),

  # Table
  DTOutput("predictions_table"),

  # Footer
  div(class = "footer-text",
    span("@WalrusQuant"),
    span("Hierarchical NegBin / rstan / hoopR")
  )
)

# ============================================================
# SERVER
# ============================================================

server <- function(input, output, session) {

  filtered_data <- reactive({
    df <- preds

    # Game filter
    if (input$game_filter != "all") {
      df <- df %>% filter(game_id == input$game_filter)
    }

    # Position filter (handle PG/SG/PF/SF mapping)
    if (input$pos_filter != "all") {
      guard_pos <- c("G", "PG", "SG")
      fwd_pos   <- c("F", "SF", "PF")
      center_pos <- c("C")

      if (input$pos_filter == "G") df <- df %>% filter(Pos %in% guard_pos)
      if (input$pos_filter == "F") df <- df %>% filter(Pos %in% fwd_pos)
      if (input$pos_filter == "C") df <- df %>% filter(Pos %in% center_pos)
    }

    # Min mean filter
    df <- df %>% filter(`E[STL]` >= input$min_mean)

    # Drop internal columns
    df %>% select(-game_id, -opponent_id)
  })

  output$predictions_table <- renderDT({
    datatable(
      filtered_data(),
      rownames = FALSE,
      options = list(
        pageLength = 25,
        order = list(list(5, "desc")),  # Sort by E[STL] descending
        dom = "ftip",                    # filter, table, info, pagination
        scrollX = TRUE,
        columnDefs = list(
          list(className = "dt-center", targets = c(2, 3, 4, 6)),
          list(className = "dt-right", targets = c(5, 7, 8, 9))
        )
      )
    ) %>%
      # Color the E[STL] column
      formatStyle(
        "E[STL]",
        color = styleInterval(
          c(0.75, 1.25),
          c("#64748b", "#e2e8f0", "#fbbf24")
        ),
        fontWeight = "bold"
      ) %>%
      # Color probability columns with bars
      formatStyle(
        "P(1+)",
        background = styleColorBar(c(0, 1), "#1e3a5f"),
        backgroundSize = "98% 60%",
        backgroundRepeat = "no-repeat",
        backgroundPosition = "left center"
      ) %>%
      formatStyle(
        "P(2+)",
        background = styleColorBar(c(0, 1), "#422006"),
        backgroundSize = "98% 60%",
        backgroundRepeat = "no-repeat",
        backgroundPosition = "left center"
      ) %>%
      formatStyle(
        "P(3+)",
        background = styleColorBar(c(0, 1), "#450a0a"),
        backgroundSize = "98% 60%",
        backgroundRepeat = "no-repeat",
        backgroundPosition = "left center"
      ) %>%
      # Position colors
      formatStyle(
        "Pos",
        color = styleEqual(
          c("G", "PG", "SG", "F", "SF", "PF", "C"),
          c("#3b82f6", "#3b82f6", "#3b82f6", "#f59e0b", "#f59e0b", "#f59e0b", "#ef4444")
        ),
        fontWeight = "bold",
        fontSize = "11px"
      ) %>%
      # Format percentages
      formatPercentage(c("P(1+)", "P(2+)", "P(3+)"), digits = 1)
  })
}

# ============================================================
# RUN
# ============================================================

shinyApp(ui = ui, server = server)
