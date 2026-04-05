// MLB Oracle v4.0 — Core Type Definitions

export interface Venue {
  id: number;
  name: string;
  city: string;
  latitude?: number;
  longitude?: number;
  cf_heading?: number; // compass degrees to center field
}

export interface PitcherStats {
  playerId: number;
  playerName: string;
  teamId: number;
  era: number;
  fip: number;
  xfip: number;      // fallback to FIP if unavailable
  siera: number;     // fallback to ERA-based estimate if unavailable
  whip: number;
  kPct: number;      // K%
  bbPct: number;     // BB%
  kBBPct: number;    // K-BB%
  cswRate: number;   // Called Strikes + Whiffs rate (fallback 0.28)
  rollingGameScore: number; // rolling 5-game score (0-100)
  inningsPitched: number;
  gamesStarted: number;
  era_minus?: number;
  handedness: 'L' | 'R' | 'S';
}

export interface BatterStats {
  playerId: number;
  playerName: string;
  teamId: number;
  woba: number;
  wrcPlus: number;
  avg: number;
  obp: number;
  slg: number;
  ops: number;
  xba?: number;
  barrelRate?: number;
  hardHitRate?: number;
  exitVelocity?: number;
  gbRate?: number;
}

export interface TeamStats {
  teamId: number;
  teamAbbr: string;
  teamName: string;
  // Hitting
  runsPerGame: number;
  woba: number;
  wrcPlus: number;
  obp: number;
  slg: number;
  avg: number;
  // Pitching
  era: number;
  fip: number;
  whip: number;
  kPct: number;
  bbPct: number;
  // Statcast
  xba?: number;
  barrelRate?: number;
  hardHitRate?: number;
  exitVelocity?: number;
  gbRate?: number;
  // Rolling (last 10 games)
  rolling10dWoba?: number;
  rolling10dFip?: number;
  rolling10dRunsPerGame?: number;
}

export interface ProbablePitcher {
  id: number;
  fullName: string;
  link: string;
}

export interface GameTeam {
  id: number;
  name: string;
  abbreviation?: string;
  score?: number;
  probablePitcher?: ProbablePitcher;
  battingOrder?: number[];
  lineup?: BatterStats[];
}

export interface Game {
  gamePk: number;
  gameDate: string;     // YYYY-MM-DD
  gameTime: string;     // ISO datetime string
  status: string;
  venue: Venue;
  homeTeam: GameTeam;
  awayTeam: GameTeam;
  weather?: WeatherData;
  umpireId?: number;
  umpireName?: string;
}

export interface WeatherData {
  temperature: number;      // Fahrenheit
  windSpeed: number;        // mph
  windDirection: number;    // degrees (0=N, 90=E, 180=S, 270=W)
  condition: string;
}

export interface ParkFactor {
  name: string;
  team: string;
  teamId?: number;
  run_factor: number;      // >1.0 hitter friendly, <1.0 pitcher friendly
  cf_heading: number;      // compass degrees to center field
  latitude: number;
  longitude: number;
  roof: 'open' | 'retractable' | 'dome';
  wind_factor_out?: number; // run multiplier when wind blowing out to CF
  wind_factor_in?: number;  // run multiplier when wind blowing in from CF
}

export interface UmpireFactors {
  run_factor: number;   // runs/game above/below average (>1 = more runs)
  k_modifier: number;  // K rate multiplier (>1 = more strikeouts)
}

export interface DrsTeam {
  drs: number;   // Defensive Runs Saved for current season
  year: number;
}

export interface ParkFactors {
  [stadiumName: string]: ParkFactor;
}

export interface WindAdjustment {
  wind_out_cf: number;   // positive = wind blowing out to CF
  wind_in_cf: number;    // positive = wind blowing in from CF
  net_wind_factor: number;
}

export interface EloRating {
  teamAbbr: string;
  rating: number;
  updatedAt: string;
}

export interface FeatureVector {
  // Pitcher differentials (home - away)
  elo_diff: number;
  sp_xfip_diff: number;
  sp_kbb_diff: number;
  sp_siera_diff: number;
  sp_csw_diff: number;
  sp_rolling_gs_diff: number;

  // Bullpen & lineup
  bullpen_strength_diff: number;
  lineup_woba_diff: number;
  lineup_wrc_plus_diff: number;

  // Rolling team form (last 10 days)
  team_10d_woba_diff: number;
  team_10d_fip_diff: number;

  // Win probability indicators
  pythagorean_diff: number;
  log5_prob: number;

  // Defense & game factors
  drs_diff: number;
  catcher_framing_diff: number;

  // Park & weather
  park_factor: number;
  wind_out_cf: number;
  wind_in_cf: number;
  temperature: number;
  umpire_run_factor: number;

  // Situational
  rest_days_diff: number;
  travel_tz_shift: number;
  day_after_night: number;
  is_home: number;         // always 1 (home team perspective)

  // Statcast
  statcast_xba_diff: number;
  statcast_barrel_diff: number;
  statcast_hardhit_diff: number;
  statcast_ev_diff: number;

  // Advanced
  gb_rate_diff: number;
  sci_adjusted_diff: number;

  // Vegas closing moneyline (normalized, vig-removed)
  // 0.0 when odds not available (treated as prior ~0.5)
  vegas_home_prob: number;
}

export interface SPScore {
  score: number;
  suppression_multiplier: number;
  tier: 'elite' | 'average' | 'bad';
}

export interface LambdaEstimate {
  lambda_home: number;
  lambda_away: number;
  home_offense_strength: number;
  away_offense_strength: number;
  home_pitcher_suppression: number;
  away_pitcher_suppression: number;
}

export interface MonteCarloResult {
  win_probability: number;       // home team win probability
  away_win_probability: number;
  run_line: number;              // home team expected run line spread
  total_runs: number;            // expected total runs
  most_likely_score: [number, number]; // [home, away]
  upset_probability: number;    // lower-elo team wins
  blowout_probability: number;  // margin >= 5
  home_lambda: number;
  away_lambda: number;
  simulations: number;
  sci_home: number;
  sci_away: number;
}

export interface Prediction {
  game_date: string;
  game_pk: number;
  home_team: string;
  away_team: string;
  venue: string;
  feature_vector: FeatureVector;
  mc_win_pct: number;            // raw Monte Carlo
  calibrated_prob: number;       // after calibration (Phase 3)
  vegas_prob?: number;           // implied from odds
  edge?: number;                 // calibrated_prob - vegas_prob
  model_version: string;
  home_lambda: number;
  away_lambda: number;
  total_runs: number;
  run_line: number;
  most_likely_score: string;
  upset_probability: number;
  blowout_probability: number;
  actual_winner?: string;
  correct?: boolean;
  created_at: string;
}

export interface AccuracyLog {
  date: string;
  brier_score: number;
  log_loss: number;
  accuracy: number;
  high_conv_accuracy: number;
  vs_vegas_brier: number;
}

export interface GameResult {
  game_id: number;
  date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  home_sp: string;
  away_sp: string;
  venue: string;
  umpire: string;
  lineups: string; // JSON
}

export interface BullpenUsage {
  player_id: number;
  player_name: string;
  team: string;
  date: string;
  pitches: number;
  innings: number;
  leverage_index: number;
  days_rest: number;
}

export interface CalibrationLog {
  date: string;
  game_id: number;
  model_prob: number;
  vegas_prob: number;
  edge: number;
  outcome: number; // 1 = home win, 0 = away win
}

export interface ModelRegistry {
  version: string;
  weights_hash: string;
  train_dates: string;
  test_brier: number;
  test_accuracy: number;
}

export interface ScheduleResponse {
  dates: Array<{
    date: string;
    games: Array<RawGame>;
  }>;
}

export interface RawGame {
  gamePk: number;
  gameDate: string;
  status: { abstractGameState: string; detailedState: string };
  teams: {
    home: {
      team: { id: number; name: string };
      score?: number;
      probablePitcher?: { id: number; fullName: string; link: string };
      battingOrder?: number[];
    };
    away: {
      team: { id: number; name: string };
      score?: number;
      probablePitcher?: { id: number; fullName: string; link: string };
      battingOrder?: number[];
    };
  };
  venue: { id: number; name: string };
  weather?: {
    condition: string;
    temp: string;
    wind: string;
  };
}

export interface PipelineOptions {
  date?: string;       // YYYY-MM-DD, defaults to today
  forceRefresh?: boolean;
  verbose?: boolean;
}

export interface TeamAbbreviationMap {
  [teamId: number]: string;
}

// ─── Phase 3: Market Edge ──────────────────────────────────────────────────────

/** Edge category tiers based on absolute probability difference */
export type EdgeCategory =
  | 'none'        // |edge| < 2%  — noise, not actionable
  | 'small'       // 2–5%  — model slightly disagrees with market
  | 'meaningful'  // 5–10% — warrants attention
  | 'large'       // 10–15% — market may be mispriced
  | 'extreme';    // ≥15%  — strong signal or data error; verify manually

/** Full edge computation result for one game */
export interface EdgeResult {
  modelProb: number;          // Calibrated model win probability (home team)
  vegasProb: number;          // Vig-removed implied probability (home team)
  rawHomeImplied: number;     // Raw home implied prob (includes vig)
  rawAwayImplied: number;     // Raw away implied prob (includes vig)
  vigPct: number;             // Bookmaker vig as a fraction (e.g., 0.046)
  edge: number;               // modelProb − vegasProb
  edgeCategory: EdgeCategory;
  homeFavorite: boolean;
  impliedHomeML: number;      // Vig-removed probability back-converted to ML
}
