// MLB Oracle v4.0 — Vegas Odds Client
// Accepts Vegas moneylines as manual input or loads from data/vegas_lines.json.
// Returns per-game implied probabilities (vig-removed) for edge computation.

import { existsSync, readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import { mlToImpliedProb, removeVig } from '../features/marketEdge.js';
import type { VegasLine, VegasLinesFile } from '../features/marketEdge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const VEGAS_LINES_PATH = resolve(__dirname, '../../data/vegas_lines.json');

// ─── Types ────────────────────────────────────────────────────────────────────

export interface GameOdds {
  gamePk: number;
  homeMl: number;
  awayMl: number;
  homeImpliedProb: number;   // vig-removed implied probability for home team
  awayImpliedProb: number;   // vig-removed implied probability for away team
  vigPct: number;            // bookmaker's vig percentage
  source: 'file' | 'manual' | 'default';
}

// In-memory store for manually provided lines this session
const _manualLines: Map<number, VegasLine> = new Map();

// ─── Manual line input ────────────────────────────────────────────────────────

/**
 * Set Vegas lines manually (e.g., from a CLI input or external data source).
 * Overwrites any existing lines for the given game IDs.
 *
 * Usage:
 *   setManualLines({ 717123: { home_ml: -155, away_ml: +130 } })
 *
 * @param lines  Object mapping gamePk (number) → VegasLine
 */
export function setManualLines(lines: Record<number, VegasLine>): void {
  for (const [gamePk, line] of Object.entries(lines)) {
    _manualLines.set(Number(gamePk), line as VegasLine);
  }
  logger.info({ count: Object.keys(lines).length }, 'Manual Vegas lines set');
}

/**
 * Clear all manually set lines (e.g., at the start of a new day's pipeline run).
 */
export function clearManualLines(): void {
  _manualLines.clear();
}

// ─── File-based line loading ──────────────────────────────────────────────────

/**
 * Load all lines from data/vegas_lines.json.
 * Returns empty object if file doesn't exist or is invalid JSON.
 */
function loadLinesFromFile(): VegasLinesFile {
  if (!existsSync(VEGAS_LINES_PATH)) {
    return {};
  }
  try {
    const raw = readFileSync(VEGAS_LINES_PATH, 'utf-8');
    return JSON.parse(raw) as VegasLinesFile;
  } catch (err) {
    logger.warn({ err, path: VEGAS_LINES_PATH }, 'Failed to load vegas_lines.json');
    return {};
  }
}

/**
 * Persist the current in-memory manual lines to data/vegas_lines.json.
 * Merges with any existing lines in the file (new entries overwrite duplicates).
 *
 * Call this if you want manual lines to survive a pipeline restart.
 */
export function saveManualLinesToFile(): void {
  const existing = loadLinesFromFile();
  const merged: VegasLinesFile = { ...existing };

  for (const [gamePk, line] of _manualLines.entries()) {
    merged[String(gamePk)] = line;
  }

  writeFileSync(VEGAS_LINES_PATH, JSON.stringify(merged, null, 2), 'utf-8');
  logger.info({ path: VEGAS_LINES_PATH, total: Object.keys(merged).length }, 'Vegas lines saved to file');
}

// ─── Primary lookup ───────────────────────────────────────────────────────────

/**
 * Look up odds for a single game.
 * Priority order:
 *   1. Manually set lines (_manualLines map, set via setManualLines())
 *   2. Lines from data/vegas_lines.json
 *   3. Returns null if no data available
 *
 * @param gamePk  MLB game primary key
 * @returns GameOdds with vig-removed probabilities, or null if unavailable
 */
export function getOddsForGame(gamePk: number): GameOdds | null {
  // 1. Check manual lines first
  const manualLine = _manualLines.get(gamePk);
  if (manualLine) {
    return buildGameOdds(gamePk, manualLine, 'manual');
  }

  // 2. Check file-based lines
  const fileLines = loadLinesFromFile();
  const fileLine = fileLines[String(gamePk)];
  if (fileLine) {
    return buildGameOdds(gamePk, fileLine, 'file');
  }

  // 3. No data available
  return null;
}

/**
 * Get odds for multiple games at once.
 * Returns a map of gamePk → GameOdds for games that have odds available.
 * Games without odds are omitted from the result.
 *
 * @param gamePks  Array of MLB game primary keys
 */
export function getOddsForGames(gamePks: number[]): Map<number, GameOdds> {
  const result = new Map<number, GameOdds>();

  // Load file lines once (avoid repeated disk reads)
  const fileLines = loadLinesFromFile();

  for (const gamePk of gamePks) {
    // Manual lines take priority
    const manualLine = _manualLines.get(gamePk);
    if (manualLine) {
      result.set(gamePk, buildGameOdds(gamePk, manualLine, 'manual'));
      continue;
    }

    // File lines
    const fileLine = fileLines[String(gamePk)];
    if (fileLine) {
      result.set(gamePk, buildGameOdds(gamePk, fileLine, 'file'));
    }
  }

  if (result.size > 0) {
    logger.debug(
      { found: result.size, requested: gamePks.length },
      'Vegas odds loaded'
    );
  } else if (gamePks.length > 0) {
    logger.debug('No Vegas lines available — edge computation skipped');
  }

  return result;
}

/**
 * Check whether any Vegas lines are currently available
 * (either manual or from the file).
 */
export function hasAnyOdds(): boolean {
  if (_manualLines.size > 0) return true;
  if (!existsSync(VEGAS_LINES_PATH)) return false;
  const lines = loadLinesFromFile();
  return Object.keys(lines).length > 0;
}

/**
 * Get all currently loaded game IDs that have odds.
 */
export function getAvailableGameIds(): number[] {
  const ids = new Set<number>(_manualLines.keys());
  const fileLines = loadLinesFromFile();
  for (const key of Object.keys(fileLines)) {
    ids.add(Number(key));
  }
  return Array.from(ids);
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

function buildGameOdds(
  gamePk: number,
  line: VegasLine,
  source: 'file' | 'manual' | 'default'
): GameOdds {
  const { homeProb, awayProb, vigPct } = removeVig(line.home_ml, line.away_ml);

  return {
    gamePk,
    homeMl: line.home_ml,
    awayMl: line.away_ml,
    homeImpliedProb: homeProb,
    awayImpliedProb: awayProb,
    vigPct,
    source,
  };
}

// ─── Convenience: parse lines from a simple string format ──────────────────

/**
 * Parse a quick-entry string of Vegas lines for a single game.
 * Format: "HOME_ML/AWAY_ML" or "HOME_ML,AWAY_ML"
 *
 * Examples:
 *   parseLineString("-155/+130")  → { home_ml: -155, away_ml: 130 }
 *   parseLineString("-155,+130")  → { home_ml: -155, away_ml: 130 }
 *   parseLineString("+110,-130")  → { home_ml: 110, away_ml: -130 }
 *
 * @throws if the string cannot be parsed
 */
export function parseLineString(input: string): VegasLine {
  const parts = input.trim().split(/[,/]/);
  if (parts.length !== 2) {
    throw new Error(`Cannot parse Vegas line string: "${input}". Expected "HOME_ML/AWAY_ML"`);
  }
  const home_ml = parseInt(parts[0].trim(), 10);
  const away_ml = parseInt(parts[1].trim(), 10);
  if (isNaN(home_ml) || isNaN(away_ml)) {
    throw new Error(`Invalid moneylines in: "${input}"`);
  }
  if (home_ml === 0 || away_ml === 0) {
    throw new Error('Moneylines cannot be 0');
  }
  return { home_ml, away_ml };
}

/**
 * Convenience: Set a single game's lines from a quick-entry string.
 *
 * @param gamePk   MLB game primary key
 * @param lineStr  String like "-155/+130"
 */
export function setLineFromString(gamePk: number, lineStr: string): void {
  const line = parseLineString(lineStr);
  _manualLines.set(gamePk, line);
  logger.debug({ gamePk, home_ml: line.home_ml, away_ml: line.away_ml }, 'Line set from string');
}

// ─── Logging helper ────────────────────────────────────────────────────────────

/**
 * Print a summary of all currently available odds to the console.
 */
export function printOddsSummary(): void {
  const ids = getAvailableGameIds();
  if (ids.length === 0) {
    console.log('No Vegas lines loaded. Add lines to data/vegas_lines.json or call setManualLines().');
    return;
  }

  const fileLines = loadLinesFromFile();

  console.log(`\nVegas Lines (${ids.length} games):`);
  console.log('─'.repeat(55));
  console.log(
    `${'GamePk'.padEnd(10)} ${'Home ML'.padEnd(10)} ${'Away ML'.padEnd(10)} ${'Home Prob'.padEnd(12)} ${'Vig'.padEnd(7)} Source`
  );
  console.log('─'.repeat(55));

  for (const gamePk of ids.sort()) {
    const source = _manualLines.has(gamePk) ? 'manual' : 'file';
    const line = _manualLines.get(gamePk) ?? fileLines[String(gamePk)];
    if (!line) continue;

    const { homeProb, vigPct } = removeVig(line.home_ml, line.away_ml);
    const homeMLStr = line.home_ml > 0 ? `+${line.home_ml}` : String(line.home_ml);
    const awayMLStr = line.away_ml > 0 ? `+${line.away_ml}` : String(line.away_ml);

    console.log(
      `${String(gamePk).padEnd(10)} ${homeMLStr.padEnd(10)} ${awayMLStr.padEnd(10)} ` +
      `${(homeProb * 100).toFixed(1).padEnd(12)}% ${(vigPct * 100).toFixed(1).padEnd(7)}% ${source}`
    );
  }
  console.log('─'.repeat(55));
}
