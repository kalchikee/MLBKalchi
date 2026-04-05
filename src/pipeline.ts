// MLB Oracle v4.0 — Daily Pipeline
// Orchestrates: Fetch → Features → Monte Carlo → ML Model → Edge → Store → Print
// Phase 3: ML meta-model + isotonic calibration + market edge detection
// Phase 4: Statcast preload, bullpen model integration

import { logger } from './logger.js';
import { fetchSchedule, TEAM_ID_TO_ABBR } from './api/mlbClient.js';
import { computeFeatures } from './features/featureEngine.js';
import { runFullMonteCarlo } from './models/monteCarlo.js';
import { upsertPrediction, initDb } from './db/database.js';
import { preloadStatcastData } from './api/statcastClient.js';
import { loadModel, predict as mlPredict, isModelLoaded, getModelInfo } from './models/metaModel.js';
import { computeEdgeFromFile, formatEdge, loadOddsApiLines } from './features/marketEdge.js';
import { getOddsForGame, hasAnyOdds } from './api/oddsClient.js';
import type { Game, Prediction, PipelineOptions } from './types.js';

const MODEL_VERSION = '4.0.0';

// ─── Main pipeline ────────────────────────────────────────────────────────────

export async function runPipeline(options: PipelineOptions = {}): Promise<Prediction[]> {
  const today = new Date().toISOString().split('T')[0];
  const gameDate = options.date ?? today;
  const year = new Date(gameDate).getFullYear();

  logger.info({ gameDate, version: MODEL_VERSION }, '=== MLB Oracle v4.0 Pipeline Start ===');

  // 1. Initialize database
  await initDb();

  // 2. Attempt to load ML meta-model (Phase 3).
  //    loadModel() is idempotent — safe to call every run.
  //    Returns true if model loaded, false if falling back to MC.
  const modelLoaded = loadModel();
  if (modelLoaded) {
    const info = getModelInfo();
    logger.info(
      { version: info?.version, avgBrier: info?.avgBrier, trainDates: info?.trainDates },
      'Using ML model for calibrated predictions'
    );
  } else {
    logger.info('ML model not found — using Monte Carlo win probability as calibrated_prob');
  }

  // 3. Check if Vegas lines are available (Phase 3)
  const vegasAvailable = hasAnyOdds();
  if (vegasAvailable) {
    logger.info('Vegas lines available — will compute market edge for each game');
  } else {
    logger.debug('No Vegas lines found (data/vegas_lines.json not present)');
  }

  // 4. Preload Statcast data for the season (Phase 4)
  //    This warms the cache so individual lookups are instant.
  try {
    logger.info({ year }, 'Preloading Statcast data');
    await preloadStatcastData(year);
  } catch (err) {
    logger.warn({ err }, 'Statcast preload failed — continuing without Statcast enrichment');
  }

  // 5. Fetch today's schedule
  const games = await fetchSchedule(gameDate);
  if (games.length === 0) {
    logger.warn({ gameDate }, 'No games found for date');
    return [];
  }

  logger.info({ gameDate, games: games.length }, 'Games fetched');

  // 6. Fetch live moneylines from The Odds API (30-min cached)
  let oddsApiMap: Map<string, { homeML: number; awayML: number }> = new Map();
  try {
    oddsApiMap = await loadOddsApiLines(gameDate);
    if (oddsApiMap.size > 0) {
      logger.info({ games: oddsApiMap.size }, 'The Odds API lines loaded');
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to load Odds API lines — continuing without live odds');
  }

  // 7. Process each game
  const predictions: Prediction[] = [];
  let processed = 0;
  let failed = 0;

  for (const game of games) {
    try {
      const pred = await processGame(game, gameDate, modelLoaded, oddsApiMap);
      if (pred) {
        predictions.push(pred);
        processed++;
      }
    } catch (err) {
      failed++;
      logger.error({
        err,
        gamePk: game.gamePk,
        home: game.homeTeam.abbreviation,
        away: game.awayTeam.abbreviation,
      }, 'Failed to process game');
    }
  }

  logger.info({ processed, failed, total: games.length }, 'Pipeline complete');

  // 8. Print formatted predictions
  if (options.verbose !== false) {
    printPredictions(predictions, gameDate, modelLoaded);
  }

  return predictions;
}

// ─── Single game processing ───────────────────────────────────────────────────

async function processGame(
  game: Game,
  gameDate: string,
  modelLoaded: boolean,
  oddsApiMap: Map<string, { homeML: number; awayML: number }> = new Map(),
): Promise<Prediction | null> {
  const homeAbbr = game.homeTeam.abbreviation ?? TEAM_ID_TO_ABBR[game.homeTeam.id] ?? 'UNK';
  const awayAbbr = game.awayTeam.abbreviation ?? TEAM_ID_TO_ABBR[game.awayTeam.id] ?? 'UNK';

  logger.info({ gamePk: game.gamePk, matchup: `${awayAbbr} @ ${homeAbbr}` }, 'Processing game');

  // Skip games not in a playable state
  const skipStatuses = ['Postponed', 'Cancelled', 'Suspended'];
  if (skipStatuses.some(s => game.status.includes(s))) {
    logger.info({ status: game.status }, 'Skipping non-playable game');
    return null;
  }

  // ── Step A: Compute features (Phase 4: includes bullpen, umpire, DRS, etc.) ──
  const features = await computeFeatures(game, gameDate);

  // ── Step B: Monte Carlo ────────────────────────────────────────────────────
  const { result: mc } = runFullMonteCarlo(features);

  // ── Step C: ML calibration (Phase 3) ──────────────────────────────────────
  // If meta-model is loaded: scale features → logit → sigmoid → isotonic
  // Otherwise: use MC win probability directly (Phase 1 behavior)
  let calibrated_prob: number;

  if (modelLoaded && isModelLoaded()) {
    calibrated_prob = mlPredict(features, mc.win_probability);
    logger.debug(
      {
        gamePk: game.gamePk,
        mc_prob: mc.win_probability.toFixed(3),
        ml_prob: calibrated_prob.toFixed(3),
        delta: (calibrated_prob - mc.win_probability).toFixed(3),
      },
      'ML model applied'
    );
  } else {
    calibrated_prob = mc.win_probability;
    logger.debug(
      { gamePk: game.gamePk, win_pct: mc.win_probability.toFixed(3) },
      'Monte Carlo fallback (no ML model)'
    );
  }

  // ── Step D: Market edge computation (Phase 3 + Odds API) ─────────────────
  // Priority: manual lines / vegas_lines.json → The Odds API live feed
  let vegas_prob: number | undefined;
  let edge: number | undefined;

  const gameOdds = getOddsForGame(game.gamePk);
  if (gameOdds) {
    vegas_prob = gameOdds.homeImpliedProb;
    edge = calibrated_prob - vegas_prob;

    const edgeResult = computeEdgeFromFile(game.gamePk, calibrated_prob);
    if (edgeResult) {
      logger.info(
        { gamePk: game.gamePk, matchup: `${awayAbbr} @ ${homeAbbr}` },
        formatEdge(edgeResult)
      );
    }
  } else {
    // Fall back to The Odds API live lines
    const oddsKey = `${awayAbbr}@${homeAbbr}`;
    const apiLine = oddsApiMap.get(oddsKey);
    if (apiLine) {
      try {
        const { computeEdge } = await import('./features/marketEdge.js');
        const edgeResult = computeEdge(calibrated_prob, apiLine.homeML, apiLine.awayML);
        vegas_prob = edgeResult.vegasProb;
        edge = edgeResult.edge;
        logger.info(
          { gamePk: game.gamePk, matchup: oddsKey, source: 'OddsAPI' },
          formatEdge(edgeResult)
        );
      } catch (err) {
        logger.warn({ err, gamePk: game.gamePk }, 'Failed to compute edge from Odds API line');
      }
    }
  }

  // ── Step E: Build prediction record ───────────────────────────────────────
  const prediction: Prediction = {
    game_date: gameDate,
    game_pk: game.gamePk,
    home_team: homeAbbr,
    away_team: awayAbbr,
    venue: game.venue.name,
    feature_vector: features,
    mc_win_pct: mc.win_probability,
    calibrated_prob,
    vegas_prob,
    edge,
    model_version: MODEL_VERSION,
    home_lambda: mc.home_lambda,
    away_lambda: mc.away_lambda,
    total_runs: mc.total_runs,
    run_line: mc.run_line,
    most_likely_score: `${mc.most_likely_score[0]}-${mc.most_likely_score[1]}`,
    upset_probability: mc.upset_probability,
    blowout_probability: mc.blowout_probability,
    created_at: new Date().toISOString(),
  };

  // ── Step F: Store in DB ────────────────────────────────────────────────────
  upsertPrediction(prediction);
  logger.debug(
    {
      gamePk: game.gamePk,
      mc_prob: mc.win_probability.toFixed(3),
      cal_prob: calibrated_prob.toFixed(3),
      edge: edge?.toFixed(3) ?? 'n/a',
    },
    'Prediction stored'
  );

  return prediction;
}

// ─── Console output ───────────────────────────────────────────────────────────

function printPredictions(
  predictions: Prediction[],
  gameDate: string,
  mlModelActive: boolean = false,
): void {
  if (predictions.length === 0) {
    console.log(`\nNo predictions for ${gameDate}\n`);
    return;
  }

  const modelLabel = mlModelActive ? 'ML+Isotonic' : 'Monte Carlo';
  const hasEdge = predictions.some(p => p.edge !== undefined);
  const totalWidth = hasEdge ? 105 : 95;

  console.log('\n' + '═'.repeat(totalWidth));
  console.log(
    `  MLB ORACLE v4.0  ·  Predictions for ${gameDate}  ·  ${predictions.length} games  ·  [${modelLabel}]`
  );
  console.log('═'.repeat(totalWidth));

  // Build header with optional EDGE column
  const headerCols = [
    pad('MATCHUP', 22),
    pad('VENUE', 20),
    pad('CAL WIN%', 10),
    pad('MC WIN%', 9),
    pad('λ HOME', 8),
    pad('λ AWAY', 8),
    pad('TOTAL', 7),
    pad('PROJ SCORE', 11),
  ];
  if (hasEdge) headerCols.push(pad('EDGE', 9));

  console.log('\n' + headerCols.join('  '));
  console.log('─'.repeat(totalWidth));

  // Sort by calibrated probability confidence (most confident first)
  const sorted = [...predictions].sort((a, b) =>
    Math.abs(b.calibrated_prob - 0.5) - Math.abs(a.calibrated_prob - 0.5)
  );

  for (const p of sorted) {
    const calPct  = (p.calibrated_prob * 100).toFixed(1) + '%';
    const mcPct   = (p.mc_win_pct * 100).toFixed(1) + '%';
    const matchup = `${p.away_team} @ ${p.home_team}`;

    const confidence = Math.abs(p.calibrated_prob - 0.5);
    const marker = confidence >= 0.15 ? ' ★' : confidence >= 0.10 ? ' ·' : '  ';

    const rowCols = [
      pad(matchup, 22),
      pad(p.venue.slice(0, 19), 20),
      pad(`${calPct} (H)`, 10),
      pad(mcPct, 9),
      pad(p.home_lambda.toFixed(2), 8),
      pad(p.away_lambda.toFixed(2), 8),
      pad(p.total_runs.toFixed(1), 7),
      pad(p.most_likely_score, 11),
    ];

    if (hasEdge) {
      if (p.edge !== undefined) {
        const sign = p.edge >= 0 ? '+' : '';
        rowCols.push(pad(`${sign}${(p.edge * 100).toFixed(1)}%`, 9));
      } else {
        rowCols.push(pad('—', 9));
      }
    }

    console.log(rowCols.join('  ') + marker);
  }

  console.log('─'.repeat(totalWidth));
  console.log('\nLegend: ★ = high confidence (>65% cal prob)  · = medium confidence (>60%)');
  if (hasEdge) console.log('EDGE: model vs vig-removed Vegas probability (+ = model favors home)');

  // Summary stats
  const calProbs  = predictions.map(p => p.calibrated_prob);
  const avgCalWin = calProbs.reduce((s, v) => s + v, 0) / calProbs.length;
  const avgMCWin  = predictions.reduce((s, p) => s + p.mc_win_pct, 0) / predictions.length;
  const avgTotal  = predictions.reduce((s, p) => s + p.total_runs, 0) / predictions.length;
  const highConf  = predictions.filter(p => Math.abs(p.calibrated_prob - 0.5) >= 0.15).length;
  const edgePicks = predictions.filter(p => p.edge !== undefined && Math.abs(p.edge) >= 0.05);

  let summary = (
    `\nSummary: avg cal win% = ${(avgCalWin * 100).toFixed(1)}%` +
    `  |  avg MC win% = ${(avgMCWin * 100).toFixed(1)}%` +
    `  |  avg total = ${avgTotal.toFixed(1)}` +
    `  |  ${highConf} high-confidence picks`
  );
  if (edgePicks.length > 0) summary += `  |  ${edgePicks.length} edge picks (≥5%)`;
  console.log(summary);
  console.log('═'.repeat(totalWidth) + '\n');

  // Edge picks table
  if (edgePicks.length > 0) {
    console.log('─'.repeat(72));
    console.log('  EDGE PICKS (model vs Vegas disagreement ≥ 5%)');
    console.log('─'.repeat(72));
    const edgeSorted = [...edgePicks].sort((a, b) => Math.abs(b.edge!) - Math.abs(a.edge!));
    for (const p of edgeSorted) {
      if (p.edge === undefined || p.vegas_prob === undefined) continue;
      const matchup = `${p.away_team} @ ${p.home_team}`;
      const side = p.edge >= 0 ? p.home_team : p.away_team;
      const sign = p.edge >= 0 ? '+' : '';
      const tier = Math.abs(p.edge) >= 0.15 ? 'EXTREME' : Math.abs(p.edge) >= 0.10 ? 'LARGE' : 'MEANINGFUL';
      console.log(
        `  ${pad(matchup, 22)}  Model: ${(p.calibrated_prob * 100).toFixed(1)}%` +
        `  Vegas: ${(p.vegas_prob * 100).toFixed(1)}%` +
        `  Edge: ${sign}${(p.edge * 100).toFixed(1)}%  Lean: ${side}  [${tier}]`
      );
    }
    console.log('─'.repeat(72) + '\n');
  }
}

function pad(str: string, width: number): string {
  if (str.length >= width) return str.slice(0, width);
  return str + ' '.repeat(width - str.length);
}

export { processGame };
