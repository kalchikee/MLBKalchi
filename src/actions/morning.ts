// GitHub Actions entry point — Morning (6 AM ET)
// 1. Run prediction pipeline
// 2. Run bet engine (paper trades)
// 3. Send ONE Discord message with all predictions + today's bets

import 'dotenv/config';
import { runPipeline } from '../pipeline.js';
import { initDb, closeDb } from '../db/database.js';
import { runBetEngine, ensureKalshiBetsTable } from '../kalshi/betEngine.js';
import { sendMorningBriefing } from '../alerts/discord.js';
import { logger } from '../logger.js';

const date = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });

logger.info({ date }, '[Morning Action] Starting');

try {
  await initDb();
  ensureKalshiBetsTable();

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
