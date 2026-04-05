// MLB Oracle v4.0 — Market Edge Detection
// Converts Vegas moneylines to implied probabilities, removes vig,
// and computes edge between the model's calibrated probability and the market.

import { existsSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const VEGAS_LINES_PATH = resolve(__dirname, '../../data/vegas_lines.json');

// ─── Types ────────────────────────────────────────────────────────────────────

export interface EdgeResult {
  modelProb: number;          // Calibrated model win probability (home team)
  vegasProb: number;          // Vig-removed implied probability (home team)
  rawHomeImplied: number;     // Raw implied prob from home moneyline (with vig)
  rawAwayImplied: number;     // Raw implied prob from away moneyline (with vig)
  vigPct: number;             // Total vig as a percentage (e.g., 0.045 = 4.5%)
  edge: number;               // modelProb − vegasProb (positive = model favors home)
  edgeCategory: EdgeCategory; // Bucketed edge label
  homeFavorite: boolean;      // Whether Vegas has home team as favorite
  impliedHomeML: number;      // Reverse-converted moneyline from vegasProb
}

export type EdgeCategory =
  | 'none'        // |edge| < 2%
  | 'small'       // 2% ≤ |edge| < 5%
  | 'meaningful'  // 5% ≤ |edge| < 10%
  | 'large'       // 10% ≤ |edge| < 15%  (historically ~74–76% accuracy tier)
  | 'extreme';    // |edge| ≥ 15%

export interface VegasLine {
  home_ml: number;  // American moneyline for home team (e.g., -130 or +115)
  away_ml: number;  // American moneyline for away team
}

export interface VegasLinesFile {
  [gameId: string]: VegasLine;
}

// ─── Moneyline math ───────────────────────────────────────────────────────────

/**
 * Convert American moneyline to raw implied probability.
 * This includes the bookmaker's vig (overround).
 *
 * Examples:
 *   -130 → 0.5652   (130 / (130+100))
 *   +110 → 0.4762   (100 / (110+100))
 *   -200 → 0.6667
 *   +200 → 0.3333
 */
export function mlToImpliedProb(ml: number): number {
  if (ml < 0) {
    // Favorite: -ml / (-ml + 100)
    return (-ml) / (-ml + 100);
  } else if (ml > 0) {
    // Underdog: 100 / (ml + 100)
    return 100 / (ml + 100);
  } else {
    // Even money (+0 is rare but handle it)
    return 0.5;
  }
}

/**
 * Convert a true probability back to American moneyline.
 * Inverse of mlToImpliedProb (no vig applied — clean probability input).
 *
 * Examples:
 *   0.6 → -150
 *   0.4 → +150
 *   0.5 → +100
 */
export function probToML(prob: number): number {
  // Clamp to avoid division by zero / infinite lines
  const p = Math.max(0.01, Math.min(0.99, prob));
  if (p >= 0.5) {
    // Favorite
    return Math.round(-(p / (1 - p)) * 100);
  } else {
    // Underdog
    return Math.round(((1 - p) / p) * 100);
  }
}

/**
 * Remove vig using the additive (proportional) method.
 *
 * The bookmaker's line includes extra probability (vig/juice) on both sides
 * so the total implied probabilities sum to > 1.0 (e.g., 1.045 = 4.5% vig).
 *
 * Additive removal: divide each side by the total implied probability.
 *   p_home_fair = p_home_raw / (p_home_raw + p_away_raw)
 *   p_away_fair = p_away_raw / (p_home_raw + p_away_raw)
 *
 * After removal, home_prob + away_prob = 1.0 exactly.
 *
 * Returns { homeProb, awayProb, vigPct }
 */
export function removeVig(
  homeML: number,
  awayML: number
): { homeProb: number; awayProb: number; vigPct: number } {
  const rawHome = mlToImpliedProb(homeML);
  const rawAway = mlToImpliedProb(awayML);
  const total = rawHome + rawAway;

  const homeProb = rawHome / total;
  const awayProb = rawAway / total;
  const vigPct = total - 1.0;  // e.g., 1.046 → 0.046 = 4.6% vig

  return { homeProb, awayProb, vigPct };
}

// ─── Edge categorization ──────────────────────────────────────────────────────

/**
 * Categorize a model edge into human-readable tiers.
 * Thresholds reflect the historical accuracy jump at each tier:
 *   - <2%: noise, not actionable
 *   - 2–5%: small edge, model is slightly disagreeing with market
 *   - 5–10%: meaningful edge, warrants attention
 *   - 10–15%: large edge, market may be mispriced
 *   - ≥15%: extreme — either a very strong signal or a data error; verify manually
 */
export function categorizeEdge(edgeMagnitude: number): EdgeCategory {
  const abs = Math.abs(edgeMagnitude);
  if (abs < 0.02) return 'none';
  if (abs < 0.05) return 'small';
  if (abs < 0.10) return 'meaningful';
  if (abs < 0.15) return 'large';
  return 'extreme';
}

// ─── Primary function ─────────────────────────────────────────────────────────

/**
 * Compute market edge for a single game.
 *
 * @param modelProb     Calibrated model win probability for the home team (0–1)
 * @param homeMoneyline American moneyline for home team (e.g., -130 or +115)
 * @param awayMoneyline American moneyline for away team
 * @returns EdgeResult with full breakdown
 *
 * Example:
 *   computeEdge(0.62, -140, +120)
 *   // Vegas has home at -140 (~58.3% implied, 56.0% after vig removal)
 *   // Model says 62% → edge = +6% (meaningful, lean home)
 */
export function computeEdge(
  modelProb: number,
  homeMoneyline: number,
  awayMoneyline: number
): EdgeResult {
  // Validate inputs
  if (typeof modelProb !== 'number' || modelProb < 0 || modelProb > 1) {
    throw new Error(`Invalid modelProb: ${modelProb} (must be 0–1)`);
  }
  if (homeMoneyline === 0 || awayMoneyline === 0) {
    throw new Error('Moneylines cannot be 0');
  }

  const rawHomeImplied = mlToImpliedProb(homeMoneyline);
  const rawAwayImplied = mlToImpliedProb(awayMoneyline);
  const { homeProb: vegasProb, vigPct } = removeVig(homeMoneyline, awayMoneyline);

  const edge = modelProb - vegasProb;
  const edgeCategory = categorizeEdge(edge);
  const homeFavorite = homeMoneyline < 0;
  const impliedHomeML = probToML(vegasProb);

  return {
    modelProb,
    vegasProb,
    rawHomeImplied,
    rawAwayImplied,
    vigPct,
    edge,
    edgeCategory,
    homeFavorite,
    impliedHomeML,
  };
}

// ─── Vegas lines loader ───────────────────────────────────────────────────────

/** Cache to avoid re-reading from disk on every call */
let _vegasLinesCache: VegasLinesFile | null = null;
let _vegasLinesCacheTime = 0;
const CACHE_TTL_MS = 60_000; // Re-read from disk every 60 seconds

/**
 * Load Vegas lines from data/vegas_lines.json.
 * Returns null if the file does not exist.
 *
 * The file format is:
 *   { "717123": { "home_ml": -130, "away_ml": +110 }, ... }
 * Keys are gamePk integers as strings.
 *
 * You can populate this file manually before running the pipeline:
 *   {
 *     "717123": { "home_ml": -155, "away_ml": +130 },
 *     "717124": { "home_ml": +110, "away_ml": -130 }
 *   }
 */
export function loadVegasLines(filePath = VEGAS_LINES_PATH): VegasLinesFile | null {
  const now = Date.now();
  if (_vegasLinesCache && now - _vegasLinesCacheTime < CACHE_TTL_MS) {
    return _vegasLinesCache;
  }

  if (!existsSync(filePath)) {
    return null;
  }

  try {
    const raw = readFileSync(filePath, 'utf-8');
    _vegasLinesCache = JSON.parse(raw) as VegasLinesFile;
    _vegasLinesCacheTime = now;
    logger.debug({ file: filePath, games: Object.keys(_vegasLinesCache).length }, 'Vegas lines loaded');
    return _vegasLinesCache;
  } catch (err) {
    logger.warn({ err, file: filePath }, 'Failed to parse vegas_lines.json');
    return null;
  }
}

/**
 * Get Vegas line for a specific game, if available.
 *
 * @param gamePk  MLB game primary key (integer)
 * @returns VegasLine or null if not found
 */
export function getVegasLineForGame(gamePk: number): VegasLine | null {
  const lines = loadVegasLines();
  if (!lines) return null;
  return lines[String(gamePk)] ?? null;
}

/**
 * Compute market edge for a game, using the Vegas lines file.
 * Returns null if no Vegas line is available for this game.
 *
 * @param gamePk     MLB game primary key
 * @param modelProb  Calibrated model win probability for home team
 */
export function computeEdgeFromFile(
  gamePk: number,
  modelProb: number
): EdgeResult | null {
  const line = getVegasLineForGame(gamePk);
  if (!line) return null;

  try {
    return computeEdge(modelProb, line.home_ml, line.away_ml);
  } catch (err) {
    logger.warn({ err, gamePk }, 'Failed to compute edge from file');
    return null;
  }
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

/**
 * Format an EdgeResult as a compact human-readable string for logging.
 *
 * Example: "Model: 62.0% | Vegas: 56.0% | Edge: +6.0% (meaningful)"
 */
export function formatEdge(result: EdgeResult): string {
  const modelPct = (result.modelProb * 100).toFixed(1);
  const vegasPct = (result.vegasProb * 100).toFixed(1);
  const edgePct = (result.edge * 100).toFixed(1);
  const sign = result.edge >= 0 ? '+' : '';
  const vigPct = (result.vigPct * 100).toFixed(1);
  const mlStr = result.homeFavorite
    ? `${result.impliedHomeML}`
    : `+${result.impliedHomeML}`;

  return (
    `Model: ${modelPct}% | Vegas: ${vegasPct}% (${mlStr}) | ` +
    `Edge: ${sign}${edgePct}% (${result.edgeCategory}) | Vig: ${vigPct}%`
  );
}

/**
 * Determine if the model has a meaningful bet recommendation.
 * Returns null if no edge, otherwise returns direction ('home' | 'away').
 */
export function getBetRecommendation(
  result: EdgeResult
): { side: 'home' | 'away'; category: EdgeCategory } | null {
  if (result.edgeCategory === 'none') return null;
  const side = result.edge > 0 ? 'home' : 'away';
  return { side, category: result.edgeCategory };
}
