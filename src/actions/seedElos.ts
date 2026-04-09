// MLB Oracle v4.0 — Historical Elo Seeder
// Replays a prior season's game results through the Elo engine, then applies
// offseason regression so the new season starts with meaningful team ratings.
//
// Can be called as a module (from morning.ts) or run standalone:
//   node --loader ts-node/esm src/actions/seedElos.ts [season]

import 'dotenv/config';
import { initDb, closeDb, getAllElos } from '../db/database.js';
import { updateElo, applyOffseasonRegression, ALL_TEAM_ABBRS, seedElos } from '../features/elo.js';
import { TEAM_ID_TO_ABBR } from '../api/mlbClient.js';
import { logger } from '../logger.js';

const MLB_BASE = 'https://statsapi.mlb.com/api/v1';

// ─── Core seeding logic (callable from morning.ts without DB lifecycle) ───────

export async function runEloSeed(season: number): Promise<void> {
  logger.info({ season }, '[SeedElos] Seeding Elo ratings from prior season');

  // Ensure all teams start at 1500 before replaying
  seedElos(ALL_TEAM_ABBRS);

  // Fetch full season schedule with linescore
  const schedUrl =
    `${MLB_BASE}/schedule?sportId=1&season=${season}&gameType=R` +
    `&hydrate=linescore,teams&startDate=${season}-03-01&endDate=${season}-11-30`;

  const schedResp = await fetch(schedUrl, {
    headers: { 'User-Agent': 'MLBOracle/4.0 (educational)' },
    signal: AbortSignal.timeout(60000),
  });

  if (!schedResp.ok) {
    throw new Error(`Schedule fetch failed: HTTP ${schedResp.status}`);
  }

  const schedData = (await schedResp.json()) as Record<string, unknown>;
  const dates = (schedData.dates as { date: string; games: unknown[] }[]) ?? [];

  let processed = 0;
  let skipped = 0;

  for (const dateObj of dates) {
    const games = (dateObj.games ?? []) as Record<string, unknown>[];

    for (const game of games) {
      const status = (game.status as Record<string, string>)?.abstractGameState;
      if (status !== 'Final') { skipped++; continue; }

      const teams = game.teams as Record<string, Record<string, unknown>>;
      const homeTeamId = Number((teams?.home?.team as Record<string, unknown>)?.id ?? 0);
      const awayTeamId = Number((teams?.away?.team as Record<string, unknown>)?.id ?? 0);

      const homeAbbr = TEAM_ID_TO_ABBR[homeTeamId];
      const awayAbbr = TEAM_ID_TO_ABBR[awayTeamId];
      if (!homeAbbr || !awayAbbr) { skipped++; continue; }

      // Scores from linescore
      const linescore = game.linescore as Record<string, unknown> | undefined;
      const homeScore = Number(
        (linescore?.teams as Record<string, Record<string, unknown>>)?.home?.runs ?? -1
      );
      const awayScore = Number(
        (linescore?.teams as Record<string, Record<string, unknown>>)?.away?.runs ?? -1
      );

      if (homeScore < 0 || awayScore < 0 || homeScore === awayScore) { skipped++; continue; }

      const homeWon = homeScore > awayScore;
      const winner = homeWon ? homeAbbr : awayAbbr;
      const loser  = homeWon ? awayAbbr : homeAbbr;
      const margin = Math.abs(homeScore - awayScore);

      await updateElo(winner, loser, margin, homeAbbr);
      processed++;
    }
  }

  logger.info({ processed, skipped }, '[SeedElos] Season replay complete');

  // Print end-of-season rankings
  const eosRatings = getAllElos().sort((a, b) => b.rating - a.rating);
  logger.info('[SeedElos] End-of-season Elo rankings:');
  for (const r of eosRatings) {
    const dev = r.rating - 1500;
    logger.info(`  ${r.teamAbbr.padEnd(4)} ${r.rating.toFixed(0)} (${dev >= 0 ? '+' : ''}${dev.toFixed(0)})`);
  }

  // Apply offseason regression: 65% carry-over + 35% regression to 1500
  applyOffseasonRegression(season);

  const openingRatings = getAllElos().sort((a, b) => b.rating - a.rating);
  console.log(`\n${season + 1} opening Elo ratings (post-regression):`);
  for (const r of openingRatings) {
    const dev = r.rating - 1500;
    console.log(`  ${r.teamAbbr.padEnd(4)} ${r.rating.toFixed(0).padStart(4)}  (${dev >= 0 ? '+' : ''}${dev.toFixed(0)})`);
  }

  logger.info({ season }, '[SeedElos] Elo seed complete — database updated');
}

// ─── Standalone entry point ───────────────────────────────────────────────────

const isMain = process.argv[1]?.includes('seedElos');
if (isMain) {
  const season = parseInt(process.argv[2] ?? String(new Date().getFullYear() - 1), 10);
  await initDb();
  await runEloSeed(season);
  closeDb();
}
