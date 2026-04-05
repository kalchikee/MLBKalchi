// MLB Oracle v4.0 — Cron Scheduler
// Runs the full daily pipeline, morning briefing, per-game alerts, and evening recap.
//
// Usage: npm run scheduler  (long-running process)

import 'dotenv/config';
import cron from 'node-cron';
import { logger } from '../logger.js';
import { runPipeline } from '../pipeline.js';
import { initDb, getPredictionsByDate, closeDb } from '../db/database.js';
import { sendMorningBriefing } from '../alerts/discord.js';
import { sendEODSummaryAlert } from '../alerts/discord.js';
import { runBetEngine, getAllBetsForDate, ensureKalshiBetsTable } from '../kalshi/betEngine.js';
import { pollPositions, sendEndOfDaySummary } from '../kalshi/positionMonitor.js';
import { PAPER_TRADING } from '../kalshi/kalshiClient.js';
import type { Prediction } from '../types.js';

// ─── Utility ──────────────────────────────────────────────────────────────────

function todayStr(): string {
  return new Date().toISOString().split('T')[0];
}

function msUntil(hour: number, minute: number = 0): number {
  const now = new Date();
  const target = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
    hour,
    minute,
    0,
    0
  );
  let ms = target.getTime() - now.getTime();
  if (ms < 0) ms += 24 * 60 * 60 * 1000; // already passed today, schedule tomorrow
  return ms;
}

/** Parse an ISO game time string and return ms until that point minus offsetMinutes */
function msUntilGame(gameTimeIso: string, offsetMinutes: number): number {
  const gameMs = new Date(gameTimeIso).getTime();
  const targetMs = gameMs - offsetMinutes * 60 * 1000;
  const now = Date.now();
  return Math.max(0, targetMs - now);
}

// ─── Morning Pipeline + Briefing (10 AM) ─────────────────────────────────────
// One Discord message: all predictions + paper trades being placed today

async function runMorningRoutine(): Promise<void> {
  const date = todayStr();
  logger.info({ date }, '[Scheduler] Morning routine starting');

  try {
    // 1. Run full prediction pipeline
    await runPipeline({ date, verbose: false });

    // 2. Run bet engine — decides which games to paper-bet and saves to DB
    ensureKalshiBetsTable();
    const bets = await runBetEngine(date);

    // 3. Send ONE Discord message with predictions + today's paper trades
    await sendMorningBriefing(date, bets);

    logger.info({ date, bets: bets.length }, '[Scheduler] Morning routine complete');
  } catch (err) {
    logger.error({ err, date }, '[Scheduler] Morning routine failed');
  }
}

// ─── Position monitor (every 60s during game hours) ──────────────────────────
// Checks for 20% stop-loss on open bets. Runs silently — no per-trade Discord msgs.

let monitorIntervalHandle: ReturnType<typeof setInterval> | null = null;

function startDayMonitor(): void {
  if (monitorIntervalHandle) return;
  const POLL_MS = parseInt(process.env.KALSHI_MONITOR_INTERVAL_MS ?? '60000', 10);
  monitorIntervalHandle = setInterval(() => {
    void pollPositions();
  }, POLL_MS);
  logger.info({ pollMs: POLL_MS }, '[Scheduler] Position monitor started');
}

function stopDayMonitor(): void {
  if (monitorIntervalHandle) {
    clearInterval(monitorIntervalHandle);
    monitorIntervalHandle = null;
    logger.info('[Scheduler] Position monitor stopped');
  }
}

// ─── EOD Summary (11 PM) ─────────────────────────────────────────────────────
// One Discord message: all bets + stop-losses + net P&L

async function runEODRoutine(): Promise<void> {
  const date = todayStr();
  logger.info({ date }, '[Scheduler] EOD routine starting');
  stopDayMonitor();

  try {
    await sendEndOfDaySummary(date);
    logger.info({ date }, '[Scheduler] EOD routine complete');
  } catch (err) {
    logger.error({ err, date }, '[Scheduler] EOD routine failed');
  }
}

// ─── Register cron jobs ───────────────────────────────────────────────────────

async function startScheduler(): Promise<void> {
  logger.info('[Scheduler] MLB Oracle v4.0 Scheduler starting…');

  await initDb();

  // 6 AM ET daily — pipeline + bet engine + ONE morning Discord message
  cron.schedule('0 6 * * *', () => {
    logger.info('[Scheduler] 6 AM cron fired — morning routine');
    void runMorningRoutine().then(() => startDayMonitor());
  }, { timezone: 'America/New_York' });

  // 11 PM ET daily — EOD P&L summary (ONE Discord message)
  cron.schedule('0 23 * * *', () => {
    logger.info('[Scheduler] 11 PM cron fired — EOD summary');
    void runEODRoutine();
  }, { timezone: 'America/New_York' });

  logger.info('[Scheduler] Cron registered: 10 AM morning + 11 PM EOD. Process running…');

  // If already past 10 AM with no predictions yet — run immediately
  const now = new Date();
  const hour = now.getHours();
  const date = todayStr();
  const existingPreds = getPredictionsByDate(date);

  if (hour >= 6 && hour < 23 && existingPreds.length === 0) {
    logger.info('[Scheduler] No predictions yet today — running morning routine immediately');
    void runMorningRoutine().then(() => startDayMonitor());
  } else if (existingPreds.length > 0) {
    logger.info({ count: existingPreds.length }, '[Scheduler] Predictions exist — starting position monitor');
    startDayMonitor();
  }
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────

process.on('SIGINT', () => {
  logger.info('[Scheduler] SIGINT received — shutting down');
  closeDb();
  process.exit(0);
});

process.on('SIGTERM', () => {
  logger.info('[Scheduler] SIGTERM received — shutting down');
  closeDb();
  process.exit(0);
});

process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, '[Scheduler] Unhandled rejection');
});

process.on('uncaughtException', (err) => {
  logger.error({ err }, '[Scheduler] Uncaught exception');
  closeDb();
  process.exit(1);
});

// Run
startScheduler();
