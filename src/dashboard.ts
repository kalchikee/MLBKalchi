#!/usr/bin/env node
// MLB Oracle v4.0 — CLI Accuracy Dashboard
// Usage:
//   npm run dashboard                        → season summary
//   npm run dashboard -- --date 2026-04-04  → specific day

import { initDb, getDb } from './db/database.js';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync, readFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ─── CLI argument parsing ──────────────────────────────────────────────────

const args = process.argv.slice(2);
let targetDate: string | null = null;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--date' && args[i + 1]) {
    targetDate = args[i + 1];
    break;
  }
}

const today = new Date().toISOString().split('T')[0];
const displayDate = targetDate ?? today;

// ─── DB query helpers ──────────────────────────────────────────────────────

function queryAll<T = Record<string, unknown>>(
  sql: string,
  params: (string | number | null)[] = []
): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject() as T);
  }
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(
  sql: string,
  params: (string | number | null)[] = []
): T | undefined {
  return queryAll<T>(sql, params)[0];
}

// ─── ASCII Sparkline ───────────────────────────────────────────────────────

function sparkline(values: number[]): string {
  if (values.length === 0) return '(no data)';

  const blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;

  if (range === 0) return blocks[3].repeat(values.length);

  return values.map(v => {
    const normalized = (v - min) / range;
    const idx = Math.min(7, Math.floor(normalized * 8));
    return blocks[idx];
  }).join('');
}

// ─── Formatting helpers ────────────────────────────────────────────────────

function pad(str: string, width: number, align: 'left' | 'right' = 'left'): string {
  if (str.length >= width) return str.slice(0, width);
  const padding = ' '.repeat(width - str.length);
  return align === 'left' ? str + padding : padding + str;
}

function pct(n: number | null | undefined): string {
  if (n === null || n === undefined || isNaN(Number(n))) return 'N/A';
  return (Number(n) * 100).toFixed(1) + '%';
}

function num(n: number | null | undefined, decimals: number = 4): string {
  if (n === null || n === undefined || isNaN(Number(n))) return 'N/A';
  return Number(n).toFixed(decimals);
}

// ─── Section renderers ─────────────────────────────────────────────────────

function renderSeasonRecord(): void {
  console.log('\n' + '─'.repeat(70));
  console.log('  1. SEASON RECORD');
  console.log('─'.repeat(70));

  // Get all predictions with outcomes
  const rows = queryAll<{
    correct: number | null;
    calibrated_prob: number;
    game_date: string;
  }>(
    `SELECT correct, calibrated_prob, game_date
     FROM predictions
     WHERE actual_winner IS NOT NULL
     ORDER BY game_date`
  );

  if (rows.length === 0) {
    console.log('  No resolved predictions yet.\n');
    return;
  }

  const wins = rows.filter(r => r.correct === 1).length;
  const losses = rows.length - wins;
  const acc = wins / rows.length;

  // Brier score
  const brier = rows.reduce((sum, r) => {
    const outcome = r.correct === 1 ? 1 : 0;
    const p = Number(r.calibrated_prob);
    return sum + Math.pow(p - outcome, 2);
  }, 0) / rows.length;

  // Log loss
  const eps = 1e-10;
  const logLoss = -rows.reduce((sum, r) => {
    const outcome = r.correct === 1 ? 1 : 0;
    const p = Math.max(eps, Math.min(1 - eps, Number(r.calibrated_prob)));
    return sum + outcome * Math.log(p) + (1 - outcome) * Math.log(1 - p);
  }, 0) / rows.length;

  console.log(`  Record:          ${wins}-${losses}  (${pct(acc)} accuracy)`);
  console.log(`  Games evaluated: ${rows.length}`);
  console.log(`  Brier score:     ${num(brier, 4)}  (lower = better; 0.25 = random)`);
  console.log(`  Log loss:        ${num(logLoss, 4)}`);
}

function renderBrierTrend(): void {
  console.log('\n' + '─'.repeat(70));
  console.log('  2. BRIER TREND (Last 14 Days)');
  console.log('─'.repeat(70));

  const rows = queryAll<{ date: string; brier_score: number }>(
    `SELECT date, brier_score
     FROM accuracy_log
     ORDER BY date DESC
     LIMIT 14`
  ).reverse();

  if (rows.length === 0) {
    // Try computing from predictions directly
    const preds = queryAll<{ game_date: string; calibrated_prob: number; correct: number | null }>(
      `SELECT game_date, calibrated_prob, correct
       FROM predictions
       WHERE actual_winner IS NOT NULL
       ORDER BY game_date DESC
       LIMIT 100`
    );

    if (preds.length === 0) {
      console.log('  No accuracy data yet.\n');
      return;
    }

    // Group by date
    const byDate = new Map<string, { probs: number[]; outcomes: number[] }>();
    for (const p of preds) {
      if (!byDate.has(p.game_date)) byDate.set(p.game_date, { probs: [], outcomes: [] });
      const entry = byDate.get(p.game_date)!;
      entry.probs.push(Number(p.calibrated_prob));
      entry.outcomes.push(p.correct === 1 ? 1 : 0);
    }

    const dateEntries = [...byDate.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .slice(-14);

    const brierByDay = dateEntries.map(([date, { probs, outcomes }]) => ({
      date,
      brier: probs.reduce((s, p, i) => s + Math.pow(p - outcomes[i], 2), 0) / probs.length,
    }));

    console.log(`  Daily Brier: ${sparkline(brierByDay.map(d => d.brier))}`);
    console.log();
    for (const d of brierByDay) {
      const bar = '█'.repeat(Math.round(d.brier * 20));
      console.log(`  ${d.date}  ${num(d.brier, 4)}  ${bar}`);
    }
    return;
  }

  const brierValues = rows.map(r => Number(r.brier_score));
  console.log(`\n  Sparkline: ${sparkline(brierValues)}\n`);

  for (const row of rows) {
    const b = Number(row.brier_score);
    const bar = '█'.repeat(Math.round(b * 20));
    const trend = b < 0.22 ? ' ✓' : b > 0.27 ? ' ✗' : '';
    console.log(`  ${row.date}  ${num(b, 4)}  ${bar}${trend}`);
  }
}

function renderCalibration(): void {
  console.log('\n' + '─'.repeat(70));
  console.log('  3. CALIBRATION CHECK (Bucket Analysis)');
  console.log('─'.repeat(70));

  const rows = queryAll<{ calibrated_prob: number; correct: number | null }>(
    `SELECT calibrated_prob, correct
     FROM predictions
     WHERE actual_winner IS NOT NULL`
  );

  if (rows.length === 0) {
    console.log('  No calibration data yet.\n');
    return;
  }

  interface Bucket {
    probs: number[];
    outcomes: number[];
  }
  const buckets = new Map<string, Bucket>();

  const bucketRanges = [
    [0.50, 0.55], [0.55, 0.60], [0.60, 0.65],
    [0.65, 0.70], [0.70, 0.75], [0.75, 1.00],
  ];

  for (const [lo, hi] of bucketRanges) {
    buckets.set(`${(lo * 100).toFixed(0)}-${(hi * 100).toFixed(0)}%`, { probs: [], outcomes: [] });
  }

  for (const row of rows) {
    const p = Number(row.calibrated_prob);
    const outcome = row.correct === 1 ? 1 : 0;

    // Use max(p, 1-p) to bucket both sides
    const normalizedP = p >= 0.5 ? p : 1 - p;

    for (const [lo, hi] of bucketRanges) {
      if (normalizedP >= lo && normalizedP < hi) {
        const key = `${(lo * 100).toFixed(0)}-${(hi * 100).toFixed(0)}%`;
        const bucket = buckets.get(key)!;
        // Outcome should match the higher-prob team
        const adjustedOutcome = p >= 0.5 ? outcome : 1 - outcome;
        bucket.probs.push(normalizedP);
        bucket.outcomes.push(adjustedOutcome);
        break;
      }
    }
  }

  console.log(`\n  ${pad('Bucket', 10)} ${pad('Predicted', 10, 'right')} ${pad('Actual', 10, 'right')} ${pad('N', 6, 'right')} ${pad('Gap', 8, 'right')}`);
  console.log('  ' + '─'.repeat(48));

  for (const [label, bucket] of buckets) {
    if (bucket.probs.length === 0) {
      console.log(`  ${pad(label, 10)} ${pad('N/A', 10, 'right')} ${pad('N/A', 10, 'right')} ${pad('0', 6, 'right')}`);
      continue;
    }
    const avgPred = bucket.probs.reduce((s, v) => s + v, 0) / bucket.probs.length;
    const actualRate = bucket.outcomes.reduce((s, v) => s + v, 0) / bucket.outcomes.length;
    const gap = actualRate - avgPred;
    const gapStr = (gap >= 0 ? '+' : '') + (gap * 100).toFixed(1) + '%';
    const indicator = Math.abs(gap) > 0.05 ? ' !' : '';

    console.log(
      `  ${pad(label, 10)} ${pad(pct(avgPred), 10, 'right')} ${pad(pct(actualRate), 10, 'right')} ${pad(String(bucket.probs.length), 6, 'right')} ${pad(gapStr, 8, 'right')}${indicator}`
    );
  }
}

function renderHighConviction(): void {
  console.log('\n' + '─'.repeat(70));
  console.log('  4. HIGH CONVICTION PERFORMANCE (>=65% Picks)');
  console.log('─'.repeat(70));

  const rows = queryAll<{ calibrated_prob: number; correct: number | null; game_date: string; home_team: string; away_team: string }>(
    `SELECT calibrated_prob, correct, game_date, home_team, away_team
     FROM predictions
     WHERE actual_winner IS NOT NULL
       AND (calibrated_prob >= 0.65 OR calibrated_prob <= 0.35)
     ORDER BY game_date DESC
     LIMIT 50`
  );

  if (rows.length === 0) {
    console.log('  No high-conviction picks resolved yet.\n');
    return;
  }

  const wins = rows.filter(r => r.correct === 1).length;
  const acc = wins / rows.length;

  const brier = rows.reduce((sum, r) => {
    const outcome = r.correct === 1 ? 1 : 0;
    const p = Number(r.calibrated_prob);
    return sum + Math.pow(p - outcome, 2);
  }, 0) / rows.length;

  console.log(`\n  High-conviction picks: ${rows.length}`);
  console.log(`  Record:                ${wins}-${rows.length - wins}  (${pct(acc)})`);
  console.log(`  Brier score:           ${num(brier, 4)}`);
  console.log(`\n  Recent high-conviction picks:`);
  console.log(`  ${pad('Date', 12)} ${pad('Matchup', 18)} ${pad('Prob', 7)} ${pad('Result', 8)}`);
  console.log('  ' + '─'.repeat(50));

  for (const row of rows.slice(0, 10)) {
    const p = Number(row.calibrated_prob);
    const matchup = p >= 0.65
      ? `${row.away_team} @ ${row.home_team}`
      : `${row.home_team} @ ${row.away_team}`;
    const result = row.correct === 1 ? 'WIN' : 'LOSS';
    const probStr = pct(p >= 0.5 ? p : 1 - p);
    console.log(
      `  ${pad(row.game_date, 12)} ${pad(matchup, 18)} ${pad(probStr, 7)} ${result}`
    );
  }
}

function renderVsVegas(): void {
  console.log('\n' + '─'.repeat(70));
  console.log('  5. VS VEGAS (Model Brier vs Vegas Brier)');
  console.log('─'.repeat(70));

  const rows = queryAll<{
    model_prob: number;
    vegas_prob: number;
    outcome: number | null;
  }>(
    `SELECT model_prob, vegas_prob, outcome
     FROM calibration_log
     WHERE outcome IS NOT NULL AND vegas_prob > 0`
  );

  if (rows.length === 0) {
    console.log('  No Vegas calibration data yet.\n');
    return;
  }

  const modelBrier = rows.reduce((sum, r) => {
    return sum + Math.pow(Number(r.model_prob) - Number(r.outcome ?? 0), 2);
  }, 0) / rows.length;

  const vegasBrier = rows.reduce((sum, r) => {
    return sum + Math.pow(Number(r.vegas_prob) - Number(r.outcome ?? 0), 2);
  }, 0) / rows.length;

  const edgeGames = rows.filter(r => {
    const edge = Math.abs(Number(r.model_prob) - Number(r.vegas_prob));
    return edge >= 0.05;
  });
  const edgeWins = edgeGames.filter(r => {
    const modelFavorsHome = Number(r.model_prob) > Number(r.vegas_prob);
    return (modelFavorsHome && r.outcome === 1) || (!modelFavorsHome && r.outcome === 0);
  }).length;

  console.log(`\n  Games with Vegas data: ${rows.length}`);
  console.log(`  Model Brier:    ${num(modelBrier, 4)}`);
  console.log(`  Vegas Brier:    ${num(vegasBrier, 4)}`);
  const diff = modelBrier - vegasBrier;
  const diffStr = diff >= 0 ? `+${num(diff, 4)}` : num(diff, 4);
  console.log(`  Difference:     ${diffStr} (negative = model beats Vegas)`);
  if (edgeGames.length > 0) {
    console.log(`\n  Edge games (|edge| >= 5%): ${edgeGames.length}`);
    console.log(`  Edge game record: ${edgeWins}-${edgeGames.length - edgeWins}  (${pct(edgeWins / edgeGames.length)})`);
  }
}

function renderFeatureImportance(): void {
  console.log('\n' + '─'.repeat(70));
  console.log('  6. FEATURE IMPORTANCE (Model Coefficients)');
  console.log('─'.repeat(70));

  // Check for trained model weights file
  const modelPath = resolve(__dirname, '../data/model_weights.json');

  if (!existsSync(modelPath)) {
    console.log('  No trained model found. Run training first: python python/train_model.py');

    // Show feature correlations from prediction data instead
    const preds = queryAll<{ feature_vector: string; correct: number | null }>(
      `SELECT feature_vector, correct
       FROM predictions
       WHERE actual_winner IS NOT NULL
       LIMIT 500`
    );

    if (preds.length < 20) {
      console.log('  (Not enough predictions to compute correlations)\n');
      return;
    }

    // Parse feature vectors and compute correlations with outcome
    const featureNames = [
      'elo_diff', 'sp_xfip_diff', 'sp_kbb_diff', 'sp_siera_diff',
      'bullpen_strength_diff', 'lineup_woba_diff', 'team_10d_woba_diff',
      'pythagorean_diff', 'log5_prob', 'drs_diff', 'park_factor',
      'wind_out_cf', 'temperature', 'umpire_run_factor',
    ];

    const correlations: Array<{ feature: string; correlation: number }> = [];

    for (const feat of featureNames) {
      const pairs: Array<[number, number]> = [];
      for (const p of preds) {
        try {
          const fv = JSON.parse(p.feature_vector as string);
          const val = fv[feat];
          if (typeof val === 'number' && p.correct !== null) {
            pairs.push([val, p.correct]);
          }
        } catch {
          // skip
        }
      }

      if (pairs.length < 10) continue;

      // Pearson correlation
      const n = pairs.length;
      const meanX = pairs.reduce((s, [x]) => s + x, 0) / n;
      const meanY = pairs.reduce((s, [, y]) => s + y, 0) / n;
      const numCov = pairs.reduce((s, [x, y]) => s + (x - meanX) * (y - meanY), 0);
      const denomX = Math.sqrt(pairs.reduce((s, [x]) => s + Math.pow(x - meanX, 2), 0));
      const denomY = Math.sqrt(pairs.reduce((s, [, y]) => s + Math.pow(y - meanY, 2), 0));
      const corr = (denomX * denomY) > 0 ? numCov / (denomX * denomY) : 0;

      correlations.push({ feature: feat, correlation: corr });
    }

    correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

    console.log('\n  Feature correlations with win outcome (from predictions):');
    console.log(`  ${pad('Feature', 28)} ${pad('Correlation', 12, 'right')} ${pad('Direction', 12)}`);
    console.log('  ' + '─'.repeat(55));

    for (const { feature, correlation } of correlations.slice(0, 10)) {
      const dir = correlation > 0 ? '+ (home favored)' : '- (away favored)';
      console.log(`  ${pad(feature, 28)} ${pad(num(correlation, 4), 12, 'right')} ${dir}`);
    }
    return;
  }

  try {
    const weights = JSON.parse(readFileSync(modelPath, 'utf-8'));
    const coefs: Array<{ feature: string; coef: number }> = Object.entries(weights.coefficients ?? {})
      .map(([feature, coef]) => ({ feature, coef: Number(coef) }))
      .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));

    console.log(`\n  Model version: ${weights.version ?? 'unknown'}`);
    console.log(`  Trained on: ${weights.train_dates ?? 'unknown'}`);
    console.log(`  Test Brier: ${num(weights.test_brier, 4)}\n`);
    console.log(`  ${pad('Feature', 28)} ${pad('Coefficient', 12, 'right')}`);
    console.log('  ' + '─'.repeat(42));

    for (const { feature, coef } of coefs.slice(0, 10)) {
      const marker = coef > 0 ? '+' : '';
      console.log(`  ${pad(feature, 28)} ${pad(marker + num(coef, 4), 12, 'right')}`);
    }
  } catch {
    console.log('  Could not load model weights.\n');
  }
}

function renderRecentPredictions(date: string): void {
  console.log('\n' + '─'.repeat(70));
  console.log(`  7. RECENT PREDICTIONS (Last 7 Days from ${date})`);
  console.log('─'.repeat(70));

  const cutoff = new Date(date);
  cutoff.setDate(cutoff.getDate() - 7);
  const cutoffStr = cutoff.toISOString().split('T')[0];

  const rows = queryAll<{
    game_date: string;
    home_team: string;
    away_team: string;
    mc_win_pct: number;
    calibrated_prob: number;
    total_runs: number;
    most_likely_score: string;
    actual_winner: string | null;
    correct: number | null;
  }>(
    `SELECT game_date, home_team, away_team, mc_win_pct, calibrated_prob,
            total_runs, most_likely_score, actual_winner, correct
     FROM predictions
     WHERE game_date >= ? AND game_date <= ?
     ORDER BY game_date DESC, mc_win_pct DESC`,
    [cutoffStr, date]
  );

  if (rows.length === 0) {
    console.log('  No predictions in last 7 days.\n');
    return;
  }

  console.log(`\n  ${pad('Date', 12)} ${pad('Matchup', 18)} ${pad('Prob', 7)} ${pad('Total', 6)} ${pad('Score', 8)} ${pad('Result', 8)}`);
  console.log('  ' + '─'.repeat(65));

  for (const row of rows) {
    const p = Number(row.calibrated_prob);
    const matchup = `${row.away_team} @ ${row.home_team}`;

    let resultStr = '';
    if (row.correct === 1) resultStr = '✓ WIN';
    else if (row.correct === 0) resultStr = '✗ LOSS';
    else resultStr = 'Pending';

    const confidence = Math.abs(p - 0.5);
    const marker = confidence >= 0.15 ? ' ★' : '';

    console.log(
      `  ${pad(row.game_date, 12)} ${pad(matchup, 18)} ${pad(pct(p), 7)} ${pad(Number(row.total_runs).toFixed(1), 6)} ${pad(String(row.most_likely_score), 8)} ${resultStr}${marker}`
    );
  }

  // Day totals
  const dayGroups = new Map<string, { correct: number; total: number }>();
  for (const row of rows) {
    if (!dayGroups.has(row.game_date)) dayGroups.set(row.game_date, { correct: 0, total: 0 });
    const day = dayGroups.get(row.game_date)!;
    if (row.correct !== null) {
      day.total++;
      if (row.correct === 1) day.correct++;
    }
  }

  const daysWithResults = [...dayGroups.entries()].filter(([, d]) => d.total > 0);
  if (daysWithResults.length > 0) {
    console.log('\n  Daily accuracy:');
    for (const [date, { correct, total }] of daysWithResults.sort()) {
      console.log(`    ${date}:  ${correct}/${total}  (${pct(correct / total)})`);
    }
  }
}

// ─── Today's games view ────────────────────────────────────────────────────

function renderTodayGames(date: string): void {
  console.log('\n' + '─'.repeat(70));
  console.log(`  TODAY'S PREDICTIONS (${date})`);
  console.log('─'.repeat(70));

  const rows = queryAll<{
    home_team: string;
    away_team: string;
    calibrated_prob: number;
    home_lambda: number;
    away_lambda: number;
    total_runs: number;
    run_line: number;
    most_likely_score: string;
    venue: string;
  }>(
    `SELECT home_team, away_team, calibrated_prob, home_lambda, away_lambda,
            total_runs, run_line, most_likely_score, venue
     FROM predictions
     WHERE game_date = ?
     ORDER BY ABS(calibrated_prob - 0.5) DESC`,
    [date]
  );

  if (rows.length === 0) {
    console.log(`  No predictions for ${date}.\n`);
    return;
  }

  console.log(`\n  ${pad('Matchup', 20)} ${pad('Home%', 7)} ${pad('λH', 5)} ${pad('λA', 5)} ${pad('Total', 6)} ${pad('Score', 8)}`);
  console.log('  ' + '─'.repeat(58));

  for (const row of rows) {
    const p = Number(row.calibrated_prob);
    const matchup = `${row.away_team}@${row.home_team}`;
    const confidence = Math.abs(p - 0.5);
    const marker = confidence >= 0.15 ? ' ★' : confidence >= 0.10 ? ' ·' : '';

    console.log(
      `  ${pad(matchup, 20)} ${pad(pct(p), 7)} ${pad(Number(row.home_lambda).toFixed(2), 5)} ${pad(Number(row.away_lambda).toFixed(2), 5)} ${pad(Number(row.total_runs).toFixed(1), 6)} ${pad(String(row.most_likely_score), 8)}${marker}`
    );
  }

  console.log('\n  ★ = high confidence (>65%)  · = medium confidence (>60%)\n');
}

// ─── Main dashboard ────────────────────────────────────────────────────────

async function main(): Promise<void> {
  await initDb();

  const width = 70;
  console.log('\n' + '═'.repeat(width));
  console.log('  MLB ORACLE v4.0 — ACCURACY DASHBOARD');
  console.log(`  ${displayDate}`);
  console.log('═'.repeat(width));

  if (targetDate) {
    // Specific date mode: show predictions for that date + recent context
    renderTodayGames(displayDate);
    renderRecentPredictions(displayDate);
    renderSeasonRecord();
    renderCalibration();
  } else {
    // Full dashboard mode
    renderSeasonRecord();
    renderBrierTrend();
    renderCalibration();
    renderHighConviction();
    renderVsVegas();
    renderFeatureImportance();
    renderTodayGames(displayDate);
    renderRecentPredictions(displayDate);
  }

  console.log('\n' + '═'.repeat(width) + '\n');
}

main().catch(err => {
  console.error('Dashboard error:', err);
  process.exit(1);
});
