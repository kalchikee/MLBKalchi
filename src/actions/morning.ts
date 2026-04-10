// GitHub Actions entry point — Morning (6 AM ET)
// 1. Run prediction pipeline
// 2. Run bet engine (paper trades)
// 3. Send ONE Discord message with all predictions + today's bets

import 'dotenv/config';
import { runPipeline } from '../pipeline.js';
import { initDb, closeDb, getPredictionsByDate, getAllElos } from '../db/database.js';
import { runBetEngine, ensureKalshiBetsTable } from '../kalshi/betEngine.js';
import { sendMorningBriefing } from '../alerts/discord.js';
import { runEloSeed } from './seedElos.js';
import { logger } from '../logger.js';

const date = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
const currentYear = new Date().getFullYear();

logger.info({ date }, '[Morning Action] Starting');

try {
  await initDb();
  ensureKalshiBetsTable();

  // ── Auto-seed Elos from prior season if all teams are at default 1500 ──────
  // This happens at the start of each season when the DB is freshly restored
  // from cache (GitHub Actions) or first run of the year.
  const allElos = getAllElos();
  const allAtDefault = allElos.length === 0 || allElos.every(e => Math.abs(e.rating - 1500) < 1);
  if (allAtDefault) {
    logger.info({ season: currentYear - 1 }, '[Morning Action] All Elos at default — seeding from prior season');
    await runEloSeed(currentYear - 1);
  }

  // 1. Run predictions
  await runPipeline({ date, verbose: false });

  // 2. Paper bet engine — finds high-conviction games on Kalshi, records 1 contract each
  const bets = await runBetEngine(date);

  // 3. Single Discord message
  await sendMorningBriefing(date, bets);

  logger.info({ date, bets: bets.length }, '[Morning Action] Complete');
} catch (err) {
  logger.error({ err }, '[Morning Action] Failed');
  process.exit(1);
} finally {
  closeDb();
}
