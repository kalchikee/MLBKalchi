// MLB Oracle v4.0 — Elo Rating System
// K-factor = 4, margin of victory multiplier = log(1 + margin)
// Offseason regression: new_elo = 0.65 × prev + 0.35 × 1500

import { getElo, upsertElo, getAllElos } from '../db/database.js';
import { logger } from '../logger.js';

const BASE_ELO = 1500;
const K_FACTOR = 4;
const OFFSEASON_REGRESSION = 0.65; // retain this fraction of deviation from 1500

// ─── Core Elo calculation ─────────────────────────────────────────────────────

/**
 * Expected win probability for teamA given ratings.
 */
export function expectedWinProb(eloA: number, eloB: number): number {
  return 1 / (1 + Math.pow(10, (eloB - eloA) / 400));
}

/**
 * Update Elo ratings after a game result.
 * @param winnerAbbr - abbreviation of winning team
 * @param loserAbbr - abbreviation of losing team
 * @param margin - run margin (winner score - loser score)
 * @param homeTeam - abbreviation of the home team
 */
export async function updateElo(
  winnerAbbr: string,
  loserAbbr: string,
  margin: number,
  homeTeam: string
): Promise<{ winnerNew: number; loserNew: number }> {
  const winnerElo = getElo(winnerAbbr);
  const loserElo = getElo(loserAbbr);

  // Home team gets small advantage in expected calculation
  const homeAdj = 25; // typical home field advantage in Elo points
  const winnerIsHome = winnerAbbr === homeTeam;
  const adjWinnerElo = winnerElo + (winnerIsHome ? homeAdj : 0);
  const adjLoserElo = loserElo + (winnerIsHome ? 0 : homeAdj);

  const expectedWinner = expectedWinProb(adjWinnerElo, adjLoserElo);
  const expectedLoser = 1 - expectedWinner;

  // Margin of victory multiplier
  const movMultiplier = Math.log(1 + Math.abs(margin));

  const change = K_FACTOR * movMultiplier * (1 - expectedWinner);

  const winnerNew = winnerElo + change;
  const loserNew = loserElo - change;

  const now = new Date().toISOString();
  upsertElo({ teamAbbr: winnerAbbr, rating: winnerNew, updatedAt: now });
  upsertElo({ teamAbbr: loserAbbr, rating: loserNew, updatedAt: now });

  logger.debug({
    winner: winnerAbbr, loser: loserAbbr, margin,
    change: change.toFixed(2), winnerNew: winnerNew.toFixed(1), loserNew: loserNew.toFixed(1),
  }, 'Elo updated');

  return { winnerNew, loserNew };
}

/**
 * Apply offseason regression to all teams.
 * Call this once at start of each new season.
 */
export function applyOffseasonRegression(season: number): void {
  const allElos = getAllElos();
  const now = new Date().toISOString();

  for (const elo of allElos) {
    const regressed = OFFSEASON_REGRESSION * elo.rating + (1 - OFFSEASON_REGRESSION) * BASE_ELO;
    upsertElo({ teamAbbr: elo.teamAbbr, rating: regressed, updatedAt: now });
  }

  logger.info({ season, teams: allElos.length }, 'Offseason Elo regression applied');
}

/**
 * Initialize all 30 teams to 1500 if not already seeded.
 */
export function seedElos(teamAbbrs: string[]): void {
  const now = new Date().toISOString();
  for (const abbr of teamAbbrs) {
    const existing = getElo(abbr);
    if (existing === 1500) {
      // Only seed if at default (meaning it was never set)
      upsertElo({ teamAbbr: abbr, rating: BASE_ELO, updatedAt: now });
    }
  }
}

/**
 * Get win probability from Elo differential, with home field advantage.
 * Returns probability for the home team.
 */
export function eloWinProbability(homeAbbr: string, awayAbbr: string): number {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);
  const HOME_ADVANTAGE = 25;
  return expectedWinProb(homeElo + HOME_ADVANTAGE, awayElo);
}

/**
 * Get the Elo differential (home - away), clamped to ±400.
 */
export function getEloDiff(homeAbbr: string, awayAbbr: string): number {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);
  return Math.max(-400, Math.min(400, homeElo - awayElo));
}

// All 30 MLB team abbreviations for seeding
export const ALL_TEAM_ABBRS = [
  'LAA', 'ARI', 'BAL', 'BOS', 'CHC', 'CIN', 'CLE', 'COL', 'DET',
  'HOU', 'KC',  'LAD', 'WSH', 'NYM', 'OAK', 'PIT', 'SD',  'SEA',
  'SF',  'STL', 'TB',  'TEX', 'TOR', 'MIN', 'PHI', 'ATL', 'CWS',
  'MIA', 'NYY', 'MIL',
];
