// GitHub Actions entry point — Morning (6 AM ET)
// 1. Run prediction pipeline
// 2. Write predictions/<date>.json for kalshi-safety to consume
// 3. Send morning Discord briefing with picks
//
// Bet placement has moved to the kalshi-safety service.

import 'dotenv/config';
import { runPipeline } from '../pipeline.js';
import { initDb, closeDb, getPredictionsByDate, getAllElos } from '../db/database.js';
import { sendMorningBriefing } from '../alerts/discord.js';
import { writePredictionsFile } from '../kalshi/predictionsFile.js';
import { runEloSeed } from './seedElos.js';
import { logger } from '../logger.js';

const date = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
const currentYear = new Date().getFullYear();

logger.info({ date }, '[Morning Action] Starting');

try {
  await initDb();

  // Auto-seed Elos from prior season if all teams are at default 1500
  const allElos = getAllElos();
  const allAtDefault = allElos.length === 0 || allElos.every(e => Math.abs(e.rating - 1500) < 1);
  if (allAtDefault) {
    logger.info({ season: currentYear - 1 }, '[Morning Action] All Elos at default — seeding from prior season');
    await runEloSeed(currentYear - 1);
  }

  await runPipeline({ date, verbose: false });

  const predictions = getPredictionsByDate(date);
  const path = writePredictionsFile(date, predictions);
  logger.info({ path, picks: predictions.length }, '[Morning Action] Wrote predictions JSON');

  await sendMorningBriefing(date);

  logger.info({ date }, '[Morning Action] Complete');
} catch (err) {
  logger.error({ err }, '[Morning Action] Failed');
  process.exit(1);
} finally {
  closeDb();
}
