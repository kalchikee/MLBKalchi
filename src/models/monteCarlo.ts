// MLB Oracle v4.0 — Monte Carlo Simulation Engine
// 10,000 Poisson simulations per game
// Lambda estimation with all adjustment factors
// Extra innings: lambda × 0.4 per half-inning

import type {
  FeatureVector, MonteCarloResult, LambdaEstimate, TeamStats, PitcherStats
} from '../types.js';

const LEAGUE_AVG_RPG = 4.5;
const N_SIMULATIONS = 10_000;
const EXTRA_INNINGS_LAMBDA_FACTOR = 0.4;

// ─── Poisson random number generator ─────────────────────────────────────────
// Knuth algorithm for lambda < 30; Normal approximation for larger values.

function poissonRandom(lambda: number): number {
  if (lambda <= 0) return 0;

  if (lambda > 30) {
    // Normal approximation: mean=lambda, std=sqrt(lambda)
    const u1 = Math.random();
    const u2 = Math.random();
    // Box-Muller transform
    const z = Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
    return Math.max(0, Math.round(lambda + z * Math.sqrt(lambda)));
  }

  // Knuth algorithm
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1.0;
  do {
    k++;
    p *= Math.random();
  } while (p > L);
  return k - 1;
}

// ─── Lambda estimation ────────────────────────────────────────────────────────
//
// lambda = league_avg_rpg × offense_strength × opp_pitcher_suppression
//          × park_factor × weather_adj × umpire_adj × lineup_adj
//
// offense_strength = team RPG / league_avg_rpg
// opp_pitcher_suppression = SP tier multiplier (0.82 elite, 1.00 avg, 1.18 bad)
// park_factor = from park_factors.json (with wind adjustment)
// weather_adj = temperature/rain adjustment
// umpire_adj = umpire run factor
// lineup_adj = lineup wOBA relative to league average

export function estimateLambda(
  features: FeatureVector,
  isHome: boolean
): LambdaEstimate {
  // Offense strength from wOBA differential
  // wOBA baseline 0.320, scale: each 0.010 wOBA ≈ 3% change in run scoring
  const homeWobaAdj = 1.0 + (features.lineup_woba_diff) / 0.320 * 1.5;
  const awayWobaAdj = 1.0 - (features.lineup_woba_diff) / 0.320 * 1.5;

  const homeOffenseStrength = Math.max(0.5, Math.min(2.0, homeWobaAdj));
  const awayOffenseStrength = Math.max(0.5, Math.min(2.0, awayWobaAdj));

  // Pitcher suppression from SP score difference
  // Negative sp_xfip_diff = home pitcher has lower xFIP = better = suppresses more
  // We invert: better away pitcher suppresses home offense
  const homePitcherSuppressionRaw = computePitcherSuppression(
    features.sp_xfip_diff,
    features.sp_kbb_diff,
    features.sp_rolling_gs_diff
  );
  const awayPitcherSuppressionRaw = computePitcherSuppression(
    -features.sp_xfip_diff,
    -features.sp_kbb_diff,
    -features.sp_rolling_gs_diff
  );

  // Park factor
  const parkFactor = features.park_factor;

  // Wind adjustment
  const windAdj = 1.0 + features.wind_out_cf * 0.06 - features.wind_in_cf * 0.06;

  // Temperature adjustment (centered around 72°F)
  const temp = features.temperature;
  const tempAdj = temp < 50
    ? 1.0 - (50 - temp) * 0.004
    : temp > 85
    ? 1.0 - (temp - 85) * 0.002
    : 1.0;

  // Umpire adjustment
  const umpireAdj = features.umpire_run_factor;

  // Lineup adjustment from rolling 10-day wOBA
  const homeLineupAdj = 1.0 + features.team_10d_woba_diff * 3;
  const awayLineupAdj = 1.0 - features.team_10d_woba_diff * 3;

  // Home field advantage in lambda (research: ~+0.15 RPG for home team)
  const homeFieldAdj = isHome ? 1.033 : 1.0;

  // Compute lambdas
  // Home team scores against AWAY pitcher suppression
  const lambda_home = LEAGUE_AVG_RPG
    * homeOffenseStrength
    * awayPitcherSuppressionRaw   // away SP suppresses home offense
    * parkFactor
    * windAdj
    * tempAdj
    * umpireAdj
    * Math.max(0.7, Math.min(1.3, homeLineupAdj))
    * homeFieldAdj;

  // Away team scores against HOME pitcher suppression
  const lambda_away = LEAGUE_AVG_RPG
    * awayOffenseStrength
    * homePitcherSuppressionRaw   // home SP suppresses away offense
    * parkFactor
    * windAdj
    * tempAdj
    * umpireAdj
    * Math.max(0.7, Math.min(1.3, awayLineupAdj));

  return {
    lambda_home: Math.max(1.5, Math.min(12.0, lambda_home)),
    lambda_away: Math.max(1.5, Math.min(12.0, lambda_away)),
    home_offense_strength: homeOffenseStrength,
    away_offense_strength: awayOffenseStrength,
    home_pitcher_suppression: homePitcherSuppressionRaw,
    away_pitcher_suppression: awayPitcherSuppressionRaw,
  };
}

function computePitcherSuppression(
  xfipDiff: number,
  kbbDiff: number,
  gsDiff: number
): number {
  // Negative xfip_diff = pitcher is better (lower xFIP vs opponent)
  // We want suppression multiplier: elite pitcher → 0.82, bad → 1.18
  // Score: combine xFIP, K-BB%, and game score

  // Normalize: xFIP diff typical range ±2.0 → map to [-1, 1]
  const xfipScore = Math.max(-1, Math.min(1, -xfipDiff / 2.0));
  // kBB diff: positive = this pitcher better; range ±0.20
  const kbbScore = Math.max(-1, Math.min(1, kbbDiff / 0.20));
  // rolling game score diff: range ±60 points
  const gsScore = Math.max(-1, Math.min(1, gsDiff / 60));

  // Weighted average
  const compositeScore = 0.40 * xfipScore + 0.35 * kbbScore + 0.25 * gsScore;

  // Map to suppression multiplier [0.82, 1.18]
  // compositeScore = 1 → elite (0.82), -1 → bad (1.18), 0 → average (1.00)
  return 1.00 - compositeScore * 0.18;
}

// ─── Main Monte Carlo simulation ──────────────────────────────────────────────

export function simulate(
  lambda_home: number,
  lambda_away: number,
  n: number = N_SIMULATIONS
): MonteCarloResult {
  let homeWins = 0;
  let totalHomeRuns = 0;
  let totalAwayRuns = 0;
  let blowouts = 0;
  let upsets = 0;

  // Score frequency map for most likely score
  const scoreFreq = new Map<string, number>();

  // Determine which team is the "underdog" based on lambda
  const homeFavored = lambda_home >= lambda_away;

  for (let i = 0; i < n; i++) {
    const result = simulateSingleGame(lambda_home, lambda_away);

    totalHomeRuns += result.homeScore;
    totalAwayRuns += result.awayScore;

    const key = `${result.homeScore}-${result.awayScore}`;
    scoreFreq.set(key, (scoreFreq.get(key) ?? 0) + 1);

    if (result.homeScore > result.awayScore) {
      homeWins++;
      if (!homeFavored) upsets++;
    } else {
      if (homeFavored) upsets++;
    }

    const margin = Math.abs(result.homeScore - result.awayScore);
    if (margin >= 5) blowouts++;
  }

  const win_probability = homeWins / n;
  const avg_home = totalHomeRuns / n;
  const avg_away = totalAwayRuns / n;

  // Most likely score
  let maxFreq = 0;
  let mostLikelyKey = '4-3';
  for (const [key, freq] of scoreFreq.entries()) {
    if (freq > maxFreq) {
      maxFreq = freq;
      mostLikelyKey = key;
    }
  }
  const [mlHome, mlAway] = mostLikelyKey.split('-').map(Number);

  // SCI computation (using simulated variance)
  const homeVariance = computeSimVariance(lambda_home, n);
  const awayVariance = computeSimVariance(lambda_away, n);
  const sci_home = (1 / (1 + Math.sqrt(homeVariance) / Math.max(0.1, avg_home))) *
                   Math.sqrt(avg_home / LEAGUE_AVG_RPG);
  const sci_away = (1 / (1 + Math.sqrt(awayVariance) / Math.max(0.1, avg_away))) *
                   Math.sqrt(avg_away / LEAGUE_AVG_RPG);

  return {
    win_probability,
    away_win_probability: 1 - win_probability,
    run_line: avg_home - avg_away,
    total_runs: avg_home + avg_away,
    most_likely_score: [mlHome ?? 4, mlAway ?? 3],
    upset_probability: upsets / n,
    blowout_probability: blowouts / n,
    home_lambda: lambda_home,
    away_lambda: lambda_away,
    simulations: n,
    sci_home,
    sci_away,
  };
}

function simulateSingleGame(
  lambdaHome: number,
  lambdaAway: number
): { homeScore: number; awayScore: number } {
  // Simulate 9 innings
  let homeScore = 0;
  let awayScore = 0;

  // Per-inning lambdas (runs per inning = RPG / 9)
  const lambdaPerInningHome = lambdaHome / 9;
  const lambdaPerInningAway = lambdaAway / 9;

  for (let inning = 0; inning < 9; inning++) {
    homeScore += poissonRandom(lambdaPerInningHome);
    awayScore += poissonRandom(lambdaPerInningAway);
  }

  // Extra innings (tied after 9)
  if (homeScore === awayScore) {
    const extraLambdaHome = lambdaPerInningHome * EXTRA_INNINGS_LAMBDA_FACTOR;
    const extraLambdaAway = lambdaPerInningAway * EXTRA_INNINGS_LAMBDA_FACTOR;
    let extraInning = 0;

    while (homeScore === awayScore && extraInning < 12) {
      // In extra innings, ghost runner rule: slightly higher scoring
      const inningHomeRuns = poissonRandom(extraLambdaHome + 0.15);
      const inningAwayRuns = poissonRandom(extraLambdaAway + 0.15);
      homeScore += inningHomeRuns;
      awayScore += inningAwayRuns;
      extraInning++;
    }

    // Force a result after 12 extra innings
    if (homeScore === awayScore) {
      if (Math.random() < 0.5) homeScore++;
      else awayScore++;
    }
  }

  return { homeScore, awayScore };
}

// ─── Variance estimation for SCI ─────────────────────────────────────────────
// For Poisson(lambda): variance = lambda
// With game-to-game variability (pitcher quality variance, etc.), inflate by ~20%

function computeSimVariance(lambda: number, _n: number): number {
  return lambda * 1.2; // Poisson variance = lambda, inflated for real-world variance
}

// ─── Convenience wrapper ──────────────────────────────────────────────────────

export function runFullMonteCarlo(features: FeatureVector): {
  result: MonteCarloResult;
  lambdas: LambdaEstimate;
} {
  const lambdas = estimateLambda(features, true);
  const result = simulate(lambdas.lambda_home, lambdas.lambda_away, N_SIMULATIONS);

  return { result, lambdas };
}
