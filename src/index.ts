// MLB Oracle v4.0 — CLI Entry Point
// Usage:
//   npm start                              → predictions for today
//   npm start -- --date 2026-04-04        → predictions for specific date
//   npm start -- --alert morning          → send morning briefing for today
//   npm start -- --alert recap            → send evening recap for today
//   npm run kalshi:bet                     → run bet engine for today (paper mode by default)
//   npm run kalshi:monitor                 → start position monitor (60s polling, stop-loss)
//   npm run kalshi:eod                     → send end-of-day P&L summary to Discord
//   npm start -- --help                   → show help

import 'dotenv/config';
import { logger } from './logger.js';
import { runPipeline } from './pipeline.js';
import { closeDb, initDb, getPredictionsByDate } from './db/database.js';
import type { PipelineOptions } from './types.js';

// ─── CLI argument parsing ─────────────────────────────────────────────────────

type AlertMode = 'morning' | 'recap' | 'kalshi-bet' | 'kalshi-monitor' | 'kalshi-eod' | null;

function parseArgs(): PipelineOptions & { help: boolean; alertMode: AlertMode } {
  const args = process.argv.slice(2);
  const opts: PipelineOptions & { help: boolean; alertMode: AlertMode } = {
    help: false,
    verbose: true,
    forceRefresh: false,
    alertMode: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--help':
      case '-h':
        opts.help = true;
        break;
      case '--date':
      case '-d':
        opts.date = args[++i];
        break;
      case '--force-refresh':
      case '-f':
        opts.forceRefresh = true;
        break;
      case '--quiet':
      case '-q':
        opts.verbose = false;
        break;
      case '--alert':
      case '-a': {
        const mode = args[++i];
        if (mode === 'morning' || mode === 'recap' || mode === 'kalshi-bet' || mode === 'kalshi-monitor' || mode === 'kalshi-eod') {
          opts.alertMode = mode as AlertMode;
        } else {
          console.error(`Unknown alert mode: "${mode}". Use "morning", "recap", "kalshi-bet", "kalshi-monitor", or "kalshi-eod".`);
          process.exit(1);
        }
        break;
      }
      default:
        // Allow positional date argument
        if (/^\d{4}-\d{2}-\d{2}$/.test(arg)) {
          opts.date = arg;
        }
    }
  }

  return opts;
}

function printHelp(): void {
  console.log(`
MLB Oracle v4.0 — Poisson Monte Carlo Prediction Engine
========================================================

USAGE:
  npm start [options]
  node --loader ts-node/esm src/index.ts [options]

OPTIONS:
  --date, -d YYYY-MM-DD        Run predictions for a specific date (default: today)
  --force-refresh, -f          Bypass cache and re-fetch all data
  --quiet, -q                  Suppress prediction table output
  --alert, -a <morning|recap>  Send a Discord/email alert for today (or --date)
  --help, -h                   Show this help message

EXAMPLES:
  npm start                              # Today's predictions
  npm start -- --date 2026-04-04        # Specific date
  npm start -- -d 2026-04-04 -f         # Specific date, force fresh data
  npm run alerts:morning                 # Send morning briefing
  npm run alerts:recap                   # Send evening recap
  npm run scheduler                      # Start the long-running cron scheduler

OUTPUT:
  Predictions are stored in ./data/mlb_oracle.db (SQLite)
  Cache files are stored in ./cache/
  Logs are written to ./logs/

ENVIRONMENT:
  DISCORD_WEBHOOK_URL    Discord webhook (optional — alerts skipped if unset)
  RESEND_API_KEY         Resend API key (optional — email skipped if unset)
  RESEND_FROM            From address for email
  RESEND_TO              Recipient address for email

ARCHITECTURE:
  MLB API → Feature Engineering → Lambda Estimation → Monte Carlo (10k sims) → SQLite
`);
}

// ─── Alert handlers ───────────────────────────────────────────────────────────

async function runMorningAlert(date: string): Promise<void> {
  const { sendMorningBriefing } = await import('./alerts/discord.js');
  const { sendMorningBriefingEmail } = await import('./alerts/email.js');

  await initDb();
  const predictions = getPredictionsByDate(date);

  if (predictions.length === 0) {
    logger.warn({ date }, 'No predictions in DB for morning alert — running pipeline first');
    const preds = await runPipeline({ date, verbose: false });
    await sendMorningBriefing(date);
    await sendMorningBriefingEmail(date, preds);
  } else {
    await sendMorningBriefing(date);
    await sendMorningBriefingEmail(date, predictions);
  }
}

async function runRecapAlert(date: string): Promise<void> {
  await initDb();  // Defensive: same init-missing bug hit NFL+EPL recaps
  const { sendEveningRecap } = await import('./alerts/discord.js');
  const { sendEveningRecapEmail } = await import('./alerts/email.js');
  const { processResults } = await import('./alerts/results.js');

  const { games, metrics } = await processResults(date);
  await sendEveningRecap(date, games, metrics);
  await sendEveningRecapEmail(date, games, metrics);
}

// ─── Entry point ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  // Validate date format if provided
  if (opts.date && !/^\d{4}-\d{2}-\d{2}$/.test(opts.date)) {
    logger.error({ date: opts.date }, 'Invalid date format. Use YYYY-MM-DD');
    process.exit(1);
  }

  const date = opts.date ?? new Date().toISOString().split('T')[0];

  logger.info({
    date,
    version: '4.0.0',
    pid: process.pid,
    alertMode: opts.alertMode ?? 'pipeline',
  }, 'MLB Oracle starting');

  try {
    // ── Alert-only modes ──────────────────────────────────────────────────────
    if (opts.alertMode === 'morning') {
      await runMorningAlert(date);
      return;
    }

    if (opts.alertMode === 'recap') {
      await runRecapAlert(date);
      return;
    }

    // Kalshi bet/monitor/eod modes were removed — the kalshi-safety service
    // now owns bet placement, stop-loss, and recap across all sports.

    // ── Full pipeline mode ────────────────────────────────────────────────────

    // If force refresh, clear relevant cache files
    if (opts.forceRefresh) {
      logger.info('Force refresh: clearing cache');
      const { readdirSync, unlinkSync } = await import('fs');
      const cacheDir = process.env.CACHE_DIR ?? './cache';
      try {
        const files = readdirSync(cacheDir);
        for (const file of files) {
          if (file.endsWith('.json')) {
            unlinkSync(`${cacheDir}/${file}`);
          }
        }
        logger.info({ cleared: files.length }, 'Cache cleared');
      } catch {
        // Cache dir may not exist yet
      }
    }

    const predictions = await runPipeline(opts);

    if (predictions.length === 0) {
      console.log(`\nNo games scheduled for ${date}. Check back later or try a different date.\n`);
      process.exit(0);
    }

    logger.info({ predictions: predictions.length }, 'Pipeline completed successfully');

  } catch (err) {
    logger.error({ err }, 'Fatal error');
    process.exit(1);
  } finally {
    closeDb();
  }
}

// ─── Handle unhandled rejections ──────────────────────────────────────────────

process.on('unhandledRejection', (reason, promise) => {
  logger.error({ reason, promise }, 'Unhandled promise rejection');
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  logger.error({ err }, 'Uncaught exception');
  closeDb();
  process.exit(1);
});

// Run
main();
