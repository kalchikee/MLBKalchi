// MLB Oracle v4.0 — MLB Stats API Client
// Free API, no key required. Includes: JSON caching (1hr TTL) + exponential backoff retry.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type {
  Game, PitcherStats, TeamStats, GameTeam, RawGame, ScheduleResponse, WeatherData
} from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 1)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

const MLB_BASE = 'https://statsapi.mlb.com/api/v1';

// ─── Team abbreviation mapping (MLB API id → abbr) ────────────────────────────
export const TEAM_ID_TO_ABBR: Record<number, string> = {
  108: 'LAA', 109: 'ARI', 110: 'BAL', 111: 'BOS', 112: 'CHC',
  113: 'CIN', 114: 'CLE', 115: 'COL', 116: 'DET', 117: 'HOU',
  118: 'KC',  119: 'LAD', 120: 'WSH', 121: 'NYM', 133: 'OAK',
  134: 'PIT', 135: 'SD',  136: 'SEA', 137: 'SF',  138: 'STL',
  139: 'TB',  140: 'TEX', 141: 'TOR', 142: 'MIN', 143: 'PHI',
  144: 'ATL', 145: 'CWS', 146: 'MIA', 147: 'NYY', 158: 'MIL',
};

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cacheKey(url: string): string {
  return url.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 200) + '.json';
}

function readCache<T>(key: string): T | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as T;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: unknown): void {
  const path = resolve(CACHE_DIR, key);
  try {
    writeFileSync(path, JSON.stringify(data), 'utf-8');
  } catch (err) {
    logger.warn({ err }, 'Failed to write cache');
  }
}

// ─── Exponential backoff fetch ────────────────────────────────────────────────

async function fetchWithRetry<T>(url: string, attempts: number = 3): Promise<T> {
  const key = cacheKey(url);
  const cached = readCache<T>(key);
  if (cached !== null) {
    logger.debug({ url }, 'Cache HIT');
    return cached;
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      logger.debug({ url, attempt }, 'Fetching');
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'MLBOracle/4.0 (educational)' },
        signal: AbortSignal.timeout(15000),
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} for ${url}`);
      }

      const data = (await resp.json()) as T;
      writeCache(key, data);
      return data;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < attempts - 1) {
        const delay = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s
        logger.warn({ url, attempt, delay, err: lastError.message }, 'Retrying after delay');
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }

  throw lastError ?? new Error(`Failed to fetch ${url}`);
}

// ─── Schedule fetcher ─────────────────────────────────────────────────────────

export async function fetchSchedule(date: string): Promise<Game[]> {
  const url = `${MLB_BASE}/schedule?sportId=1&date=${date}&hydrate=lineups,probablePitcher,venue,weather,linescore`;
  logger.info({ date }, 'Fetching schedule');

  let schedule: ScheduleResponse;
  try {
    schedule = await fetchWithRetry<ScheduleResponse>(url);
  } catch (err) {
    logger.error({ err, date }, 'Failed to fetch schedule');
    return [];
  }

  const games: Game[] = [];
  for (const dateEntry of schedule.dates ?? []) {
    for (const rawGame of dateEntry.games ?? []) {
      const game = parseRawGame(rawGame);
      if (game) games.push(game);
    }
  }

  logger.info({ date, count: games.length }, 'Schedule fetched');
  return games;
}

function parseRawGame(raw: RawGame): Game | null {
  if (!raw?.gamePk || !raw?.teams) return null;

  const homeTeamId = raw.teams.home?.team?.id;
  const awayTeamId = raw.teams.away?.team?.id;

  const homeTeam: GameTeam = {
    id: homeTeamId,
    name: raw.teams.home.team.name,
    abbreviation: TEAM_ID_TO_ABBR[homeTeamId] ?? raw.teams.home.team.name.slice(0, 3).toUpperCase(),
    score: raw.teams.home.score,
    probablePitcher: raw.teams.home.probablePitcher ? {
      id: raw.teams.home.probablePitcher.id,
      fullName: raw.teams.home.probablePitcher.fullName,
      link: raw.teams.home.probablePitcher.link,
    } : undefined,
    battingOrder: raw.teams.home.battingOrder,
  };

  const awayTeam: GameTeam = {
    id: awayTeamId,
    name: raw.teams.away.team.name,
    abbreviation: TEAM_ID_TO_ABBR[awayTeamId] ?? raw.teams.away.team.name.slice(0, 3).toUpperCase(),
    score: raw.teams.away.score,
    probablePitcher: raw.teams.away.probablePitcher ? {
      id: raw.teams.away.probablePitcher.id,
      fullName: raw.teams.away.probablePitcher.fullName,
      link: raw.teams.away.probablePitcher.link,
    } : undefined,
    battingOrder: raw.teams.away.battingOrder,
  };

  // Parse embedded weather from schedule
  let weather: WeatherData | undefined;
  if (raw.weather) {
    const tempF = parseFloat(raw.weather.temp ?? '70');
    const windStr = raw.weather.wind ?? '0 mph, None';
    const windMatch = windStr.match(/(\d+)\s*mph/i);
    const windSpeed = windMatch ? parseInt(windMatch[1]) : 0;
    weather = {
      temperature: isNaN(tempF) ? 70 : tempF,
      windSpeed,
      windDirection: 0, // direction from text is complex; weather client will override
      condition: raw.weather.condition ?? 'Clear',
    };
  }

  return {
    gamePk: raw.gamePk,
    gameDate: raw.gameDate.split('T')[0],
    gameTime: raw.gameDate,
    status: raw.status?.detailedState ?? 'Scheduled',
    venue: {
      id: raw.venue?.id ?? 0,
      name: raw.venue?.name ?? 'Unknown',
      city: '',
    },
    homeTeam,
    awayTeam,
    weather,
  };
}

// ─── Pitcher stats fetcher ────────────────────────────────────────────────────

export async function fetchPitcherStats(playerId: number, season: number = new Date().getFullYear()): Promise<PitcherStats | null> {
  if (!playerId || playerId <= 0) return null;

  const url = `${MLB_BASE}/people/${playerId}/stats?stats=season&season=${season}&group=pitching`;
  let data: Record<string, unknown>;
  try {
    data = await fetchWithRetry<Record<string, unknown>>(url);
  } catch {
    logger.warn({ playerId }, 'Could not fetch pitcher stats');
    return null;
  }

  const stats = extractFirstStat(data);
  if (!stats) {
    // Try prior season as fallback
    if (season > 2024) {
      return fetchPitcherStats(playerId, season - 1);
    }
    return null;
  }

  // Fetch player metadata for handedness
  const personUrl = `${MLB_BASE}/people/${playerId}`;
  let handedness: 'L' | 'R' | 'S' = 'R';
  try {
    const personData = await fetchWithRetry<Record<string, unknown>>(personUrl);
    const people = (personData as any)?.people?.[0];
    handedness = (people?.pitchHand?.code ?? 'R') as 'L' | 'R' | 'S';
  } catch {
    // ignore
  }

  const era = safeNum(stats.era, 4.50);
  const fip = safeNum(stats.fip, era);
  const whip = safeNum(stats.whip, 1.30);
  const strikeouts = safeNum(stats.strikeOuts, 0);
  const walks = safeNum(stats.baseOnBalls, 0);
  const battersF = safeNum(stats.battersFaced, 1);
  const ip = safeNum(stats.inningsPitched, 0);
  const gamesStarted = safeNum(stats.gamesStarted, 0);

  const kPct = battersF > 0 ? strikeouts / battersF : 0.22;
  const bbPct = battersF > 0 ? walks / battersF : 0.08;
  const kBBPct = kPct - bbPct;

  // Estimate xFIP from FIP (xFIP ≈ FIP - 0.1 to +0.2 range)
  const xfip = fip + (Math.random() * 0.3 - 0.1); // placeholder until Statcast data
  // SIERA estimate: correlated with FIP but slightly different weighting
  const siera = 0.6 * fip + 0.4 * era;

  // Rolling game score: estimate from K rate, walk rate, era
  // game_score = 50 + K*2 - BB*4 - H*2 ... simplified to stat-based estimate
  const rollingGameScore = Math.max(20, Math.min(80,
    50 + (kPct - 0.22) * 200 - (bbPct - 0.08) * 300 - (era - 4.5) * 5
  ));

  return {
    playerId,
    playerName: String(stats.playerName ?? ''),
    teamId: safeNum(stats.teamId, 0),
    era,
    fip,
    xfip,
    siera,
    whip,
    kPct,
    bbPct,
    kBBPct,
    cswRate: 0.28, // fallback; Statcast metric not in free API
    rollingGameScore,
    inningsPitched: ip,
    gamesStarted,
    handedness,
  };
}

// ─── Team stats fetcher ───────────────────────────────────────────────────────

export async function fetchTeamStats(teamId: number, season: number = new Date().getFullYear()): Promise<TeamStats | null> {
  const abbr = TEAM_ID_TO_ABBR[teamId] ?? 'UNK';

  // Fetch hitting stats
  const hitUrl = `${MLB_BASE}/teams/${teamId}/stats?stats=season&season=${season}&group=hitting`;
  // Fetch pitching stats
  const pitUrl = `${MLB_BASE}/teams/${teamId}/stats?stats=season&season=${season}&group=pitching`;

  let hitData: Record<string, unknown>;
  let pitData: Record<string, unknown>;

  try {
    [hitData, pitData] = await Promise.all([
      fetchWithRetry<Record<string, unknown>>(hitUrl),
      fetchWithRetry<Record<string, unknown>>(pitUrl),
    ]);
  } catch (err) {
    logger.warn({ teamId, err }, 'Failed to fetch team stats');
    return getDefaultTeamStats(teamId, abbr);
  }

  const hitting = extractFirstStat(hitData);
  const pitching = extractFirstStat(pitData);

  if (!hitting && !pitching) {
    return getDefaultTeamStats(teamId, abbr);
  }

  const gamesPlayed = Math.max(safeNum(hitting?.gamesPlayed, 1), 1);
  const runsScored = safeNum(hitting?.runs, gamesPlayed * 4.5);
  const runsPerGame = runsScored / gamesPlayed;

  // Estimate wOBA from OBP + SLG (wOBA ≈ 0.9 × OBP + 0.11 × ISO)
  const obp = safeNum(hitting?.obp, 0.320);
  const slg = safeNum(hitting?.slg, 0.400);
  const avg = safeNum(hitting?.avg, 0.250);
  const iso = slg - avg;
  const woba = Math.max(0.250, Math.min(0.400, 0.9 * obp + 0.11 * iso + 0.05));
  const wrcPlus = Math.round((woba / 0.320) * 100); // rough estimate

  const era = safeNum(pitching?.era, 4.50);
  const fip = safeNum(pitching?.fip, era);
  const strikeouts = safeNum(pitching?.strikeOuts, 0);
  const walks = safeNum(pitching?.baseOnBalls, 0);
  const battersF = Math.max(safeNum(pitching?.battersFaced, 1), 1);
  const kPct = strikeouts / battersF;
  const bbPct = walks / battersF;

  return {
    teamId,
    teamAbbr: abbr,
    teamName: '',
    runsPerGame,
    woba,
    wrcPlus,
    obp,
    slg,
    avg,
    era,
    fip,
    whip: safeNum(pitching?.whip, 1.30),
    kPct,
    bbPct,
    xba: woba * 0.95,
    barrelRate: 0.07,
    hardHitRate: 0.37,
    exitVelocity: 88.5,
    gbRate: 0.43,
  };
}

// ─── Games played helper ──────────────────────────────────────────────────────

/**
 * Get the number of games a team has played this season.
 * Reads from cached team stats to avoid extra API calls.
 *
 * @param teamAbbr  Three-letter team abbreviation (e.g. "DET")
 * @returns         Games played this season, or 0 if stats unavailable
 */
export async function getTeamGamesPlayed(teamAbbr: string): Promise<number> {
  // Reverse-lookup teamId from abbreviation
  const teamId = Object.entries(TEAM_ID_TO_ABBR).find(
    ([, abbr]) => abbr === teamAbbr
  )?.[0];

  if (!teamId) {
    logger.warn({ teamAbbr }, 'getTeamGamesPlayed: unknown team abbreviation');
    return 0;
  }

  const season = new Date().getFullYear();
  const hitUrl = `${MLB_BASE}/teams/${teamId}/stats?stats=season&season=${season}&group=hitting`;

  let data: Record<string, unknown>;
  try {
    data = await fetchWithRetry<Record<string, unknown>>(hitUrl);
  } catch {
    logger.warn({ teamAbbr }, 'getTeamGamesPlayed: failed to fetch team stats');
    return 0;
  }

  const stats = (data as any)?.stats;
  if (!Array.isArray(stats) || stats.length === 0) return 0;
  const splits = stats[0]?.splits;
  if (!Array.isArray(splits) || splits.length === 0) return 0;
  const gp = parseFloat(String(splits[0]?.stat?.gamesPlayed ?? '0'));
  return isNaN(gp) ? 0 : Math.floor(gp);
}

// ─── Recent game log fetcher ──────────────────────────────────────────────────

export async function fetchRecentGameLog(
  teamId: number,
  startDate: string,
  endDate: string
): Promise<Array<{ date: string; runsScored: number; runsAllowed: number }>> {
  const url = `${MLB_BASE}/schedule?sportId=1&teamId=${teamId}&startDate=${startDate}&endDate=${endDate}&hydrate=linescore`;

  let data: ScheduleResponse;
  try {
    data = await fetchWithRetry<ScheduleResponse>(url);
  } catch {
    return [];
  }

  const results: Array<{ date: string; runsScored: number; runsAllowed: number }> = [];

  for (const dateEntry of data.dates ?? []) {
    for (const game of dateEntry.games ?? []) {
      const isHome = game.teams.home.team.id === teamId;
      const myScore = isHome ? (game.teams.home.score ?? 0) : (game.teams.away.score ?? 0);
      const oppScore = isHome ? (game.teams.away.score ?? 0) : (game.teams.home.score ?? 0);
      if ((game.teams.home.score !== undefined) || (game.teams.away.score !== undefined)) {
        results.push({
          date: dateEntry.date,
          runsScored: myScore,
          runsAllowed: oppScore,
        });
      }
    }
  }

  return results;
}

// ─── Pitcher actual recent form (last N starts game log) ─────────────────────

/**
 * Fetch a pitcher's actual last-N-starts average game score from the game log.
 * Falls back to the estimated value if no game log entries are found.
 *
 * Game score formula (simplified): 50 + (K*2) - (BB*4) - (H*2) - (ER*6)
 * MLB API doesn't return game score directly, so we compute it from the log splits.
 */
export async function fetchPitcherRecentForm(
  playerId: number,
  season: number = new Date().getFullYear(),
  nStarts: number = 5,
): Promise<number | null> {
  if (!playerId || playerId <= 0) return null;

  const url = `${MLB_BASE}/people/${playerId}/stats?stats=gameLog&season=${season}&group=pitching`;
  let data: Record<string, unknown>;
  try {
    data = await fetchWithRetry<Record<string, unknown>>(url);
  } catch {
    return null;
  }

  const stats = (data as any)?.stats;
  if (!Array.isArray(stats) || stats.length === 0) return null;

  // Game log splits are newest-first
  const splits: Record<string, unknown>[] = (stats[0]?.splits ?? []) as Record<string, unknown>[];
  if (splits.length === 0) return null;

  // Filter to starts only (gamesStarted > 0 or inningsPitched > 1)
  const startSplits = splits.filter((s: any) =>
    safeNum(s?.stat?.gamesStarted, 0) > 0 || safeNum(s?.stat?.inningsPitched, 0) >= 1
  ).slice(0, nStarts);

  if (startSplits.length === 0) return null;

  const scores = startSplits.map((s: any) => {
    const stat = s?.stat ?? {};
    const k  = safeNum(stat.strikeOuts, 0);
    const bb = safeNum(stat.baseOnBalls, 0);
    const h  = safeNum(stat.hits, 0);
    const er = safeNum(stat.earnedRuns, 0);
    const ip = safeNum(stat.inningsPitched, 1);
    // Scale to roughly 6-inning outing (fair comparison across starts)
    const scale = Math.min(1.5, ip / 6);
    const raw = 50 + (k * 2) - (bb * 4) - (h * 2) - (er * 6);
    return Math.max(10, Math.min(90, raw * scale));
  });

  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  return Math.max(10, Math.min(90, avg));
}

// ─── IL/Injury roster fetch ───────────────────────────────────────────────────

export interface ILPlayer {
  playerId: number;
  playerName: string;
  position: string;   // 'P' (pitcher), 'C', '1B', '2B', '3B', 'SS', 'OF', 'DH', etc.
  ilType: '10-Day' | '15-Day' | '60-Day' | 'Other';
}

/**
 * Fetch the current 10-day + 60-day IL roster for a team.
 * Returns an empty array if the API call fails.
 */
export async function fetchTeamILPlayers(teamId: number): Promise<ILPlayer[]> {
  const url = `${MLB_BASE}/teams/${teamId}/roster?rosterType=40Man`;
  let data: Record<string, unknown>;
  try {
    data = await fetchWithRetry<Record<string, unknown>>(url);
  } catch {
    return [];
  }

  const roster: unknown[] = (data as any)?.roster ?? [];
  const ilPlayers: ILPlayer[] = [];

  for (const entry of roster) {
    const e = entry as any;
    const status = (e?.status?.description ?? '').toLowerCase();
    if (!status.includes('il') && !status.includes('injured')) continue;

    const posCode: string = e?.position?.abbreviation ?? e?.position?.code ?? 'UNK';
    let ilType: ILPlayer['ilType'] = 'Other';
    if (status.includes('60')) ilType = '60-Day';
    else if (status.includes('15')) ilType = '15-Day';
    else if (status.includes('10')) ilType = '10-Day';

    ilPlayers.push({
      playerId: e?.person?.id ?? 0,
      playerName: e?.person?.fullName ?? '',
      position: posCode,
      ilType,
    });
  }

  return ilPlayers;
}

// ─── Rolling stats computation ────────────────────────────────────────────────

export async function computeRollingStats(
  teamId: number,
  asOfDate: string,
  days: number = 10
): Promise<{ rpg: number; wobaEst: number; fipEst: number }> {
  const endDate = asOfDate;
  const startD = new Date(asOfDate);
  startD.setDate(startD.getDate() - days);
  const startDate = startD.toISOString().split('T')[0];

  const logs = await fetchRecentGameLog(teamId, startDate, endDate);

  if (logs.length === 0) {
    return { rpg: 4.5, wobaEst: 0.320, fipEst: 4.20 };
  }

  const avgRPG = logs.reduce((s, g) => s + g.runsScored, 0) / logs.length;
  const avgAllowed = logs.reduce((s, g) => s + g.runsAllowed, 0) / logs.length;

  // Rough wOBA from RPG (regression line: wOBA = 0.04 × RPG + 0.14)
  const wobaEst = Math.max(0.260, Math.min(0.380, 0.04 * avgRPG + 0.14));
  // FIP estimate from runs allowed
  const fipEst = Math.max(2.50, Math.min(6.00, avgAllowed * 0.9 + 0.45));

  return { rpg: avgRPG, wobaEst, fipEst };
}

// ─── Utility ──────────────────────────────────────────────────────────────────

function extractFirstStat(data: Record<string, unknown>): Record<string, unknown> | null {
  const stats = (data as any)?.stats;
  if (!Array.isArray(stats) || stats.length === 0) return null;
  const splits = stats[0]?.splits;
  if (!Array.isArray(splits) || splits.length === 0) return null;
  return splits[0]?.stat ?? null;
}

function safeNum(val: unknown, fallback: number): number {
  const n = parseFloat(String(val ?? ''));
  return isNaN(n) ? fallback : n;
}

function getDefaultTeamStats(teamId: number, abbr: string): TeamStats {
  return {
    teamId,
    teamAbbr: abbr,
    teamName: '',
    runsPerGame: 4.5,
    woba: 0.320,
    wrcPlus: 100,
    obp: 0.320,
    slg: 0.410,
    avg: 0.250,
    era: 4.50,
    fip: 4.20,
    whip: 1.30,
    kPct: 0.22,
    bbPct: 0.08,
    xba: 0.248,
    barrelRate: 0.07,
    hardHitRate: 0.37,
    exitVelocity: 88.5,
    gbRate: 0.43,
  };
}

export { logger } from '../logger.js';
