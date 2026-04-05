// MLB Oracle v4.0 — Bullpen Availability & Fatigue Model
// Fetches reliever usage from MLB Stats API game logs (last 3 days),
// computes fatigue adjustments, and returns a bullpen strength score.

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getDb } from '../db/database.js';
import type { BullpenUsage } from '../types.js';

const MLB_BASE = 'https://statsapi.mlb.com/api/v1';

// ─── Types ────────────────────────────────────────────────────────────────────

interface RelieverAppearance {
  playerId: number;
  playerName: string;
  teamId: number;
  teamAbbr: string;
  date: string;           // YYYY-MM-DD
  pitches: number;
  innings: number;        // decimal (0.333 = 1 out)
  leverageIndex: number;
  era: number;            // season ERA for quality rating
  role: 'closer' | 'setup' | 'other';
}

export interface BullpenScore {
  teamAbbr: string;
  bullpenStrengthScore: number;   // 0.7 – 1.3 multiplier (1.0 = league avg)
  closerAvailable: boolean;
  relieverCount: number;
  fatigueDetail: string;
}

// ─── Fatigue rules ────────────────────────────────────────────────────────────

/**
 * Returns effectiveness multiplier for a reliever based on recent usage.
 * 1.0 = full effectiveness, <1.0 = fatigued, >1.0 = well-rested.
 */
function computeFatigueMultiplier(
  appearances: RelieverAppearance[],
  asOfDate: string
): number {
  const asOf = new Date(asOfDate).getTime();
  const dayMs = 86400000;

  // Last 4 days of appearances sorted recent-first
  const recent = appearances
    .filter(a => {
      const diff = (asOf - new Date(a.date).getTime()) / dayMs;
      return diff >= 1 && diff <= 4; // 1–4 days ago (not today)
    })
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  if (recent.length === 0) {
    // No appearances in last 4 days — check for 4+ days rest
    const last = appearances
      .filter(a => (asOf - new Date(a.date).getTime()) / dayMs > 4)
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())[0];
    if (!last) return 1.05; // fresh/unknown = slight bonus
    return 1.05; // 4+ days rest
  }

  const pitchedYesterday = recent.some(a => {
    const diff = (asOf - new Date(a.date).getTime()) / dayMs;
    return diff >= 1 && diff < 2;
  });

  const pitchedHighVolYesterday = recent.some(a => {
    const diff = (asOf - new Date(a.date).getTime()) / dayMs;
    return diff >= 1 && diff < 2 && a.pitches > 25;
  });

  // Count days with appearances in last 4
  const daysWithAppearances = new Set(recent.map(a => a.date)).size;
  const pitchedThreeOfLast4 = daysWithAppearances >= 3;

  if (pitchedThreeOfLast4) return 0.85;       // −15%
  if (pitchedHighVolYesterday) return 0.81;    // −7% + −12% stacked
  if (pitchedYesterday) return 0.93;           // −7%

  return 1.0; // normal rest
}

// ─── Quality tier from ERA ────────────────────────────────────────────────────

function eraToQuality(era: number): number {
  if (era < 2.50) return 1.0;   // elite
  if (era < 3.50) return 0.80;  // good
  if (era < 4.50) return 0.55;  // average
  return 0.30;                   // bad
}

// ─── Bullpen quality formula ──────────────────────────────────────────────────
// expected_bullpen_quality = 0.4 × closer + 0.3 × setup + 0.3 × rest_of_pen

function computeBullpenQuality(
  relievers: Array<{ appearance: RelieverAppearance; fatigue: number }>,
  closerAvailable: boolean
): number {
  const closers = relievers.filter(r => r.appearance.role === 'closer');
  const setups = relievers.filter(r => r.appearance.role === 'setup');
  const others = relievers.filter(r => r.appearance.role === 'other');

  function avgQuality(group: typeof relievers): number {
    if (group.length === 0) return 0.55; // league average fallback
    const vals = group.map(r => eraToQuality(r.appearance.era) * r.fatigue);
    return vals.reduce((s, v) => s + v, 0) / vals.length;
  }

  let closerQ = avgQuality(closers);

  // If closer unavailable, best available reliever at 85% effectiveness
  if (!closerAvailable && closers.length > 0) {
    closerQ = closerQ * 0.85;
  }

  const setupQ = avgQuality(setups);
  const restQ = avgQuality(others);

  return 0.4 * closerQ + 0.3 * setupQ + 0.3 * restQ;
}

// ─── MLB API: fetch boxscores for date range ──────────────────────────────────

async function fetchBullpenUsageForTeam(
  teamId: number,
  teamAbbr: string,
  asOfDate: string
): Promise<RelieverAppearance[]> {
  const asOf = new Date(asOfDate);
  const start = new Date(asOf);
  start.setDate(start.getDate() - 3);
  const startDate = start.toISOString().split('T')[0];
  const endDate = asOf.toISOString().split('T')[0];

  const url =
    `${MLB_BASE}/schedule?sportId=1&teamId=${teamId}` +
    `&startDate=${startDate}&endDate=${endDate}&hydrate=boxscore`;

  let data: any;
  try {
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'MLBOracle/4.0 (educational)' },
      signal: AbortSignal.timeout(15000),
    });
    if (!resp.ok) return [];
    data = await resp.json();
  } catch (err) {
    logger.warn({ teamId, err }, 'Failed to fetch bullpen usage');
    return [];
  }

  const appearances: RelieverAppearance[] = [];

  for (const dateEntry of data?.dates ?? []) {
    const gameDate: string = dateEntry.date;
    for (const game of dateEntry.games ?? []) {
      const boxscore = game.boxscore;
      if (!boxscore) continue;

      // Determine which side this team played
      const homeId = game.teams?.home?.team?.id;
      const side = homeId === teamId ? 'home' : 'away';
      const teamPitchers = boxscore?.teams?.[side]?.pitchers ?? [];
      const pitcherInfo = boxscore?.teams?.[side]?.players ?? {};

      for (const pitcherId of teamPitchers) {
        const key = `ID${pitcherId}`;
        const player = pitcherInfo[key];
        if (!player) continue;

        const stats = player?.stats?.pitching;
        if (!stats) continue;

        // Only include relievers (not the starter — no gamesStarted)
        const isStarter = (stats.gamesStarted ?? 0) > 0;
        if (isStarter) continue;

        const pitches = stats.numberOfPitches ?? 0;
        const inningsPitched = parseFloat(stats.inningsPitched ?? '0') || 0;

        // Determine role heuristically from order in pitchers array
        const pitcherIndex = teamPitchers.indexOf(pitcherId);
        const totalPitchers = teamPitchers.length;
        let role: 'closer' | 'setup' | 'other' = 'other';
        if (pitcherIndex === totalPitchers - 1 && totalPitchers > 1) {
          role = 'closer';
        } else if (pitcherIndex >= totalPitchers - 2 && totalPitchers > 2) {
          role = 'setup';
        }

        const seasonEra = parseFloat(
          player?.seasonStats?.pitching?.era ?? player?.stats?.pitching?.era ?? '4.50'
        );

        appearances.push({
          playerId: pitcherId,
          playerName: player.person?.fullName ?? `Player ${pitcherId}`,
          teamId,
          teamAbbr,
          date: gameDate,
          pitches,
          innings: inningsPitched,
          leverageIndex: 1.0, // leverage index not always in free API
          era: isNaN(seasonEra) ? 4.50 : seasonEra,
          role,
        });
      }
    }
  }

  return appearances;
}

// ─── Save reliever workload to DB ─────────────────────────────────────────────

function saveBullpenUsage(appearances: RelieverAppearance[]): void {
  try {
    const db = getDb();
    for (const a of appearances) {
      db.run(
        `INSERT OR IGNORE INTO bullpen_usage
         (player_id, player_name, team, date, pitches, innings, leverage_index, days_rest)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [a.playerId, a.playerName, a.teamAbbr, a.date, a.pitches, a.innings, a.leverageIndex, 1]
      );
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to save bullpen usage to DB');
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Compute bullpen strength score for a team as of a given date.
 * Returns a lambda multiplier: 0.7 (weak) – 1.3 (strong), 1.0 = average.
 */
export async function getBullpenScore(
  teamId: number,
  teamAbbr: string,
  asOfDate: string
): Promise<BullpenScore> {
  const defaultScore: BullpenScore = {
    teamAbbr,
    bullpenStrengthScore: 1.0,
    closerAvailable: true,
    relieverCount: 0,
    fatigueDetail: 'no data',
  };

  try {
    const appearances = await fetchBullpenUsageForTeam(teamId, teamAbbr, asOfDate);

    if (appearances.length === 0) {
      return defaultScore;
    }

    // Save to DB
    saveBullpenUsage(appearances);

    // Get unique relievers
    const relieverMap = new Map<number, RelieverAppearance[]>();
    for (const a of appearances) {
      if (!relieverMap.has(a.playerId)) relieverMap.set(a.playerId, []);
      relieverMap.get(a.playerId)!.push(a);
    }

    // Get most recent appearance per reliever (for role/era)
    const uniqueRelievers: Array<{ appearance: RelieverAppearance; fatigue: number }> = [];
    const fatigueDetails: string[] = [];
    let closerAvailable = true;

    for (const [, apps] of relieverMap) {
      const latestApp = apps.sort((a, b) =>
        new Date(b.date).getTime() - new Date(a.date).getTime()
      )[0];

      const fatigue = computeFatigueMultiplier(apps, asOfDate);
      uniqueRelievers.push({ appearance: latestApp, fatigue });

      if (latestApp.role === 'closer' && fatigue < 0.90) {
        closerAvailable = false;
        fatigueDetails.push(`closer(${(fatigue * 100).toFixed(0)}%)`);
      } else if (fatigue < 0.90) {
        fatigueDetails.push(`${latestApp.playerName.split(',')[0]}(${(fatigue * 100).toFixed(0)}%)`);
      }
    }

    const quality = computeBullpenQuality(uniqueRelievers, closerAvailable);

    // Map quality (0–1 scale) to lambda multiplier (0.7–1.3)
    // quality 0.55 (league avg) → 1.0; elite (0.8+) → ~1.2; bad (0.3) → ~0.8
    const leagueAvgQuality = 0.55;
    const rawScore = quality / leagueAvgQuality;
    const bullpenStrengthScore = Math.max(0.70, Math.min(1.30, rawScore));

    return {
      teamAbbr,
      bullpenStrengthScore,
      closerAvailable,
      relieverCount: uniqueRelievers.length,
      fatigueDetail: fatigueDetails.length > 0
        ? fatigueDetails.join(', ')
        : 'all fresh',
    };
  } catch (err) {
    logger.warn({ teamId, teamAbbr, err }, 'getBullpenScore failed, using default');
    return defaultScore;
  }
}

/**
 * Get bullpen scores for both teams in a game.
 * Returns home and away multipliers.
 */
export async function getGameBullpenScores(
  homeTeamId: number,
  homeAbbr: string,
  awayTeamId: number,
  awayAbbr: string,
  gameDate: string
): Promise<{ home: BullpenScore; away: BullpenScore }> {
  const [home, away] = await Promise.all([
    getBullpenScore(homeTeamId, homeAbbr, gameDate),
    getBullpenScore(awayTeamId, awayAbbr, gameDate),
  ]);
  return { home, away };
}
