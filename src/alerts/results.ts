// MLB Oracle v4.0 — Results Fetcher & Recap Metrics Calculator
// Fetches final scores from MLB API, updates predictions table, and computes accuracy metrics.

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import {
  initDb,
  getDb,
  getPredictionsByDate,
  updatePredictionResult,
  upsertGameResult,
  upsertAccuracyLog,
  upsertCalibration,
} from '../db/database.js';
import { TEAM_ID_TO_ABBR } from '../api/mlbClient.js';
import type { Prediction } from '../types.js';
import type { RecapGame, RecapMetrics, SeasonStats } from './discord.js';

const MLB_BASE = 'https://statsapi.mlb.com/api/v1';

// ─── MLB API: fetch final scores for a date ───────────────────────────────────

interface LiveGameData {
  gamePk: number;
  gameDate: string;
  status: { abstractGameState: string; detailedState: string };
  teams: {
    home: {
      team: { id: number; name: string };
      score?: number;
      probablePitcher?: { fullName: string };
    };
    away: {
      team: { id: number; name: string };
      score?: number;
      probablePitcher?: { fullName: string };
    };
  };
  venue: { id: number; name: string };
}

interface ScheduleResponse {
  dates: Array<{
    date: string;
    games: LiveGameData[];
  }>;
}

export interface FinalScore {
  gamePk: number;
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  homeSpName: string;
  awaySpName: string;
  venue: string;
  isFinal: boolean;
}

export async function fetchFinalScores(date: string): Promise<FinalScore[]> {
  const url = `${MLB_BASE}/schedule?sportId=1&date=${date}&hydrate=linescore,probablePitcher,venue`;

  let data: ScheduleResponse;
  try {
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'MLBOracle/4.0 (educational)' },
      signal: AbortSignal.timeout(15000),
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    data = (await resp.json()) as ScheduleResponse;
  } catch (err) {
    logger.error({ err, date }, 'Failed to fetch final scores from MLB API');
    return [];
  }

  const scores: FinalScore[] = [];

  for (const dateEntry of data.dates ?? []) {
    for (const game of dateEntry.games ?? []) {
      const isFinal = game.status.abstractGameState === 'Final';
      const homeId = game.teams.home.team.id;
      const awayId = game.teams.away.team.id;

      scores.push({
        gamePk: game.gamePk,
        homeTeam: TEAM_ID_TO_ABBR[homeId] ?? game.teams.home.team.name.slice(0, 3).toUpperCase(),
        awayTeam: TEAM_ID_TO_ABBR[awayId] ?? game.teams.away.team.name.slice(0, 3).toUpperCase(),
        homeScore: game.teams.home.score ?? 0,
        awayScore: game.teams.away.score ?? 0,
        homeSpName: game.teams.home.probablePitcher?.fullName ?? '',
        awaySpName: game.teams.away.probablePitcher?.fullName ?? '',
        venue: game.venue?.name ?? '',
        isFinal,
      });
    }
  }

  logger.info({ date, games: scores.length, final: scores.filter(s => s.isFinal).length }, 'Scores fetched');
  return scores;
}

// ─── Brier score & log loss helpers ──────────────────────────────────────────

function brierScore(predictions: Array<{ prob: number; outcome: number }>): number {
  if (predictions.length === 0) return 0;
  const sum = predictions.reduce((s, p) => s + Math.pow(p.prob - p.outcome, 2), 0);
  return sum / predictions.length;
}

function logLoss(predictions: Array<{ prob: number; outcome: number }>): number {
  if (predictions.length === 0) return 0;
  const eps = 1e-7;
  const sum = predictions.reduce((s, p) => {
    const clipped = Math.max(eps, Math.min(1 - eps, p.prob));
    return s + (p.outcome * Math.log(clipped) + (1 - p.outcome) * Math.log(1 - clipped));
  }, 0);
  return -sum / predictions.length;
}

// ─── Main: process results for a date ─────────────────────────────────────────

export async function processResults(date: string): Promise<{
  games: RecapGame[];
  metrics: RecapMetrics;
}> {
  await initDb();

  const predictions = getPredictionsByDate(date);
  if (predictions.length === 0) {
    logger.warn({ date }, 'No predictions found for recap');
    return {
      games: [],
      metrics: {
        dailyAccuracy: 0,
        brierScore: 0,
        logLoss: 0,
      },
    };
  }

  const finalScores = await fetchFinalScores(date);
  const scoreMap = new Map<number, FinalScore>();
  for (const s of finalScores) {
    scoreMap.set(s.gamePk, s);
  }

  const recapGames: RecapGame[] = [];
  const brierInputs: Array<{ prob: number; outcome: number }> = [];

  for (const pred of predictions) {
    const score = scoreMap.get(pred.game_pk);
    if (!score || !score.isFinal) continue;

    const homeWon = score.homeScore > score.awayScore;
    const actualWinner = homeWon ? pred.home_team : pred.away_team;
    const predictedHomeWin = pred.calibrated_prob >= 0.5;
    const correct = homeWon === predictedHomeWin;
    const outcome = homeWon ? 1 : 0;

    // Update DB
    updatePredictionResult(pred.game_pk, actualWinner, correct);

    // Store game result
    upsertGameResult({
      game_id: pred.game_pk,
      date,
      home_team: pred.home_team,
      away_team: pred.away_team,
      home_score: score.homeScore,
      away_score: score.awayScore,
      home_sp: score.homeSpName,
      away_sp: score.awaySpName,
      venue: score.venue,
      umpire: '',
      lineups: '{}',
    });

    // Calibration log entry
    upsertCalibration({
      date,
      game_id: pred.game_pk,
      model_prob: pred.calibrated_prob,
      vegas_prob: pred.vegas_prob ?? 0,
      edge: pred.edge ?? 0,
      outcome,
    });

    brierInputs.push({ prob: pred.calibrated_prob, outcome });

    recapGames.push({
      prediction: pred,
      homeScore: score.homeScore,
      awayScore: score.awayScore,
      correct,
      actualWinner,
    });
  }

  const correct = recapGames.filter(g => g.correct).length;
  const dailyAccuracy = recapGames.length > 0 ? correct / recapGames.length : 0;
  const brier = brierScore(brierInputs);
  const loss = logLoss(brierInputs);

  // Vegas Brier (only where we have vegas_prob)
  const vegasInputs = recapGames
    .filter(g => g.prediction.vegas_prob !== undefined && g.prediction.vegas_prob !== null)
    .map(g => ({
      prob: g.prediction.vegas_prob!,
      outcome: g.homeScore > g.awayScore ? 1 : 0,
    }));

  const vegasBrier = vegasInputs.length > 0 ? brierScore(vegasInputs) : undefined;

  // High conviction accuracy
  const highConvGames = recapGames.filter(
    g => g.prediction.calibrated_prob >= 0.65 || (1 - g.prediction.calibrated_prob) >= 0.65
  );
  const highConvAccuracy = highConvGames.length > 0
    ? highConvGames.filter(g => g.correct).length / highConvGames.length
    : 0;

  // Persist accuracy log for the day
  upsertAccuracyLog({
    date,
    brier_score: brier,
    log_loss: loss,
    accuracy: dailyAccuracy,
    high_conv_accuracy: highConvAccuracy,
    vs_vegas_brier: vegasBrier ?? 0,
  });

  // Season stats: query all predictions with actual_winner set
  const seasonStats = computeSeasonStats();

  const metrics: RecapMetrics = {
    dailyAccuracy,
    brierScore: brier,
    logLoss: loss,
    vegasBrierScore: vegasBrier,
    seasonStats,
  };

  logger.info(
    { date, games: recapGames.length, correct, brier: brier.toFixed(4) },
    'Recap metrics computed'
  );

  return { games: recapGames, metrics };
}

// ─── Season stats helper ──────────────────────────────────────────────────────

function computeSeasonStats(): SeasonStats {
  try {
    const db = getDb();
    const stmt = db.prepare(
      `SELECT COUNT(*) as total,
              SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count
       FROM predictions
       WHERE actual_winner IS NOT NULL`
    );
    stmt.step();
    const row = stmt.getAsObject() as { total: number; correct_count: number };
    stmt.free();

    const total = Number(row.total ?? 0);
    const correctCount = Number(row.correct_count ?? 0);

    return {
      totalGames: total,
      correctPredictions: correctCount,
      accuracy: total > 0 ? correctCount / total : 0,
    };
  } catch {
    return { totalGames: 0, correctPredictions: 0, accuracy: 0 };
  }
}
