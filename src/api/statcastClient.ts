// MLB Oracle v4.0 — Statcast Client
// Downloads CSV data from Baseball Savant leaderboard endpoints.
// Caches locally in data/statcast/ with weekly refresh.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const STATCAST_DIR = resolve(__dirname, '../../data/statcast');
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

mkdirSync(STATCAST_DIR, { recursive: true });

// ─── Statcast Data Interfaces ─────────────────────────────────────────────────

export interface PitcherStatcast {
  player_id: number;
  player_name: string;
  p_game: number;
  k_percent: number;        // e.g. 27.5
  bb_percent: number;       // e.g. 8.2
  xfip: number;
  xera: number;
  barrel_batted_rate: number;
  hard_hit_percent: number;
  whiff_percent: number;
  csw_rate: number;
}

export interface BatterStatcast {
  player_id: number;
  player_name: string;
  pa: number;
  xba: number;
  xslg: number;
  xwoba: number;
  barrel_batted_rate: number;
  hard_hit_percent: number;
  exit_velocity_avg: number;
}

export interface CatcherFraming {
  player_id: number;
  player_name: string;
  runs_extra_strikes: number;
  strike_rate: number;
  called_strike_rate: number;
}

// ─── URL builders ─────────────────────────────────────────────────────────────

function pitchingUrl(year: number): string {
  return (
    `https://baseballsavant.mlb.com/leaderboard/custom` +
    `?year=${year}&type=pitcher&filter=&sort=p_game&sortDir=desc&min=10` +
    `&selections=p_game,k_percent,bb_percent,xfip,xera,barrel_batted_rate,hard_hit_percent,whiff_percent,csw_rate` +
    `&csv=true`
  );
}

function battingUrl(year: number): string {
  return (
    `https://baseballsavant.mlb.com/leaderboard/custom` +
    `?year=${year}&type=batter&filter=&sort=pa&sortDir=desc&min=50` +
    `&selections=pa,xba,xslg,xwoba,barrel_batted_rate,hard_hit_percent,exit_velocity_avg` +
    `&csv=true`
  );
}

function catcherFramingUrl(year: number): string {
  return (
    `https://baseballsavant.mlb.com/leaderboard/custom` +
    `?year=${year}&type=catcher_framing&filter=&sort=runs_extra_strikes&sortDir=desc&min=100` +
    `&selections=runs_extra_strikes,strike_rate,called_strike_rate` +
    `&csv=true`
  );
}

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cacheFilename(type: string, year: number): string {
  // Weekly stamp: use ISO week (YYYY-Www)
  const now = new Date();
  const dayOfYear = Math.floor(
    (now.getTime() - new Date(now.getFullYear(), 0, 0).getTime()) / 86400000
  );
  const week = Math.ceil(dayOfYear / 7);
  const stamp = `${now.getFullYear()}-W${String(week).padStart(2, '0')}`;
  return resolve(STATCAST_DIR, `${type}_${year}_${stamp}.csv`);
}

function readCachedCsv(filepath: string): string | null {
  if (!existsSync(filepath)) return null;
  const stat = statSync(filepath);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try {
    return readFileSync(filepath, 'utf-8');
  } catch {
    return null;
  }
}

async function downloadCsv(url: string, filepath: string): Promise<string | null> {
  try {
    logger.debug({ url }, 'Downloading Statcast CSV');
    const resp = await fetch(url, {
      headers: {
        'User-Agent': 'MLBOracle/4.0 (educational)',
        'Accept': 'text/csv,text/plain,*/*',
      },
      signal: AbortSignal.timeout(30000),
    });

    if (!resp.ok) {
      logger.warn({ url, status: resp.status }, 'Statcast CSV download failed');
      return null;
    }

    const text = await resp.text();
    if (!text || text.trim().length < 10) {
      logger.warn({ url }, 'Statcast CSV empty or too short');
      return null;
    }

    writeFileSync(filepath, text, 'utf-8');
    logger.info({ filepath }, 'Statcast CSV cached');
    return text;
  } catch (err) {
    logger.warn({ url, err }, 'Statcast CSV fetch error');
    return null;
  }
}

// ─── Simple CSV parser ────────────────────────────────────────────────────────

function parseCsv(csv: string): Record<string, string>[] {
  const lines = csv.trim().split('\n').filter(l => l.trim().length > 0);
  if (lines.length < 2) return [];

  // Parse headers — handle quoted fields
  const headers = splitCsvLine(lines[0]);

  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = splitCsvLine(lines[i]);
    const row: Record<string, string> = {};
    for (let j = 0; j < headers.length; j++) {
      row[headers[j].trim()] = (values[j] ?? '').trim();
    }
    rows.push(row);
  }
  return rows;
}

/** Split a CSV line respecting quoted fields */
function splitCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

function safeFloat(val: string | undefined, fallback: number): number {
  if (!val || val.trim() === '' || val.trim() === 'null') return fallback;
  const n = parseFloat(val.replace('%', ''));
  return isNaN(n) ? fallback : n;
}

function safeInt(val: string | undefined, fallback: number): number {
  if (!val || val.trim() === '') return fallback;
  const n = parseInt(val, 10);
  return isNaN(n) ? fallback : n;
}

// ─── Data loaders ─────────────────────────────────────────────────────────────

// In-memory caches (year → records)
const _pitcherCache: Map<number, PitcherStatcast[]> = new Map();
const _batterCache: Map<number, BatterStatcast[]> = new Map();
const _framingCache: Map<number, CatcherFraming[]> = new Map();

async function loadPitcherData(year: number): Promise<PitcherStatcast[]> {
  if (_pitcherCache.has(year)) return _pitcherCache.get(year)!;

  const filepath = cacheFilename('pitching', year);
  let csv = readCachedCsv(filepath);

  if (!csv) {
    csv = await downloadCsv(pitchingUrl(year), filepath);
  }

  if (!csv) {
    _pitcherCache.set(year, []);
    return [];
  }

  const rows = parseCsv(csv);
  const data: PitcherStatcast[] = rows.map(r => ({
    player_id: safeInt(r['player_id'] ?? r['mlb_id'], 0),
    player_name: r['player_name'] ?? r['last_name, first_name'] ?? '',
    p_game: safeInt(r['p_game'] ?? r['games'], 0),
    k_percent: safeFloat(r['k_percent'], 22.0),
    bb_percent: safeFloat(r['bb_percent'], 8.0),
    xfip: safeFloat(r['xfip'], 4.20),
    xera: safeFloat(r['xera'], 4.50),
    barrel_batted_rate: safeFloat(r['barrel_batted_rate'], 7.0),
    hard_hit_percent: safeFloat(r['hard_hit_percent'], 37.0),
    whiff_percent: safeFloat(r['whiff_percent'], 24.0),
    csw_rate: safeFloat(r['csw_rate'], 0.28),
  })).filter(d => d.player_id > 0);

  _pitcherCache.set(year, data);
  logger.info({ year, count: data.length }, 'Pitcher Statcast data loaded');
  return data;
}

async function loadBatterData(year: number): Promise<BatterStatcast[]> {
  if (_batterCache.has(year)) return _batterCache.get(year)!;

  const filepath = cacheFilename('batting', year);
  let csv = readCachedCsv(filepath);

  if (!csv) {
    csv = await downloadCsv(battingUrl(year), filepath);
  }

  if (!csv) {
    _batterCache.set(year, []);
    return [];
  }

  const rows = parseCsv(csv);
  const data: BatterStatcast[] = rows.map(r => ({
    player_id: safeInt(r['player_id'] ?? r['mlb_id'], 0),
    player_name: r['player_name'] ?? r['last_name, first_name'] ?? '',
    pa: safeInt(r['pa'], 0),
    xba: safeFloat(r['xba'], 0.248),
    xslg: safeFloat(r['xslg'], 0.400),
    xwoba: safeFloat(r['xwoba'], 0.320),
    barrel_batted_rate: safeFloat(r['barrel_batted_rate'], 7.0),
    hard_hit_percent: safeFloat(r['hard_hit_percent'], 37.0),
    exit_velocity_avg: safeFloat(r['exit_velocity_avg'], 88.5),
  })).filter(d => d.player_id > 0);

  _batterCache.set(year, data);
  logger.info({ year, count: data.length }, 'Batter Statcast data loaded');
  return data;
}

async function loadFramingData(year: number): Promise<CatcherFraming[]> {
  if (_framingCache.has(year)) return _framingCache.get(year)!;

  const filepath = cacheFilename('framing', year);
  let csv = readCachedCsv(filepath);

  if (!csv) {
    csv = await downloadCsv(catcherFramingUrl(year), filepath);
  }

  if (!csv) {
    _framingCache.set(year, []);
    return [];
  }

  const rows = parseCsv(csv);
  const data: CatcherFraming[] = rows.map(r => ({
    player_id: safeInt(r['player_id'] ?? r['mlb_id'], 0),
    player_name: r['player_name'] ?? r['last_name, first_name'] ?? '',
    runs_extra_strikes: safeFloat(r['runs_extra_strikes'], 0),
    strike_rate: safeFloat(r['strike_rate'], 0),
    called_strike_rate: safeFloat(r['called_strike_rate'], 0),
  })).filter(d => d.player_id > 0);

  _framingCache.set(year, data);
  logger.info({ year, count: data.length }, 'Catcher framing data loaded');
  return data;
}

// ─── Public API ───────────────────────────────────────────────────────────────

export async function getPitcherStatcast(
  playerId: number,
  year: number = new Date().getFullYear()
): Promise<PitcherStatcast | null> {
  try {
    const data = await loadPitcherData(year);
    const found = data.find(d => d.player_id === playerId);
    if (!found && year > 2020) {
      // Fallback to prior year
      const prior = await loadPitcherData(year - 1);
      return prior.find(d => d.player_id === playerId) ?? null;
    }
    return found ?? null;
  } catch {
    return null;
  }
}

export async function getBatterStatcast(
  playerId: number,
  year: number = new Date().getFullYear()
): Promise<BatterStatcast | null> {
  try {
    const data = await loadBatterData(year);
    const found = data.find(d => d.player_id === playerId);
    if (!found && year > 2020) {
      const prior = await loadBatterData(year - 1);
      return prior.find(d => d.player_id === playerId) ?? null;
    }
    return found ?? null;
  } catch {
    return null;
  }
}

export async function getCatcherFraming(
  playerId: number,
  year: number = new Date().getFullYear()
): Promise<CatcherFraming | null> {
  try {
    const data = await loadFramingData(year);
    const found = data.find(d => d.player_id === playerId);
    if (!found && year > 2020) {
      const prior = await loadFramingData(year - 1);
      return prior.find(d => d.player_id === playerId) ?? null;
    }
    return found ?? null;
  } catch {
    return null;
  }
}

/** Preload all Statcast data for a given year (call once at pipeline start) */
export async function preloadStatcastData(year: number = new Date().getFullYear()): Promise<void> {
  await Promise.all([
    loadPitcherData(year),
    loadBatterData(year),
    loadFramingData(year),
  ]);
}

/** Get all pitchers for a given year (for bulk team-level lookups) */
export async function getAllPitcherStatcast(
  year: number = new Date().getFullYear()
): Promise<PitcherStatcast[]> {
  try {
    return await loadPitcherData(year);
  } catch {
    return [];
  }
}

/** Get all batters for a given year */
export async function getAllBatterStatcast(
  year: number = new Date().getFullYear()
): Promise<BatterStatcast[]> {
  try {
    return await loadBatterData(year);
  } catch {
    return [];
  }
}
