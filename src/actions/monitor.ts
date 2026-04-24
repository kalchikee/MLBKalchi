// GitHub Actions entry point — Position Monitor (1 PM – 11 PM ET)
// Loops every 2 minutes checking all open bets for 20% stop-loss.
// Exits cleanly at 11 PM ET so the EOD job can send the final summary.
// No Discord notifications during the loop — silent stop-losses only.

import 'dotenv/config';
import { initDb, closeDb } from '../db/database.js';
import { ensureKalshiBetsTable, getOpenBets, updateBetClosed } from '../kalshi/betEngine.js';
import { getMarket, PAPER_TRADING, CASHOUT_LOSS_PCT } from '../kalshi/kalshiClient.js';
import { logger } from '../logger.js';
// Also scan paper bets for stop-loss. Paper bets are in safety-state/,
// NOT in the Kalshi bets DB — they need their own scan so mark-to-market
// stops still fire in dry-run mode.
import { scanForPaperStopLosses, loadPaperState } from 'kalshi-safety';

const POLL_MS = parseInt(process.env.KALSHI_MONITOR_INTERVAL_MS ?? '120000', 10);

/** Returns current ET hour (0–23) */
function etHour(): number {
  return parseInt(
    new Date().toLocaleString('en-US', { timeZone: 'America/New_York', hour: 'numeric', hour12: false }),
    10,
  );
}

/** Returns today's date in ET as YYYY-MM-DD */
function etDate(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

async function checkPaperPositions(): Promise<void> {
  // Build a ticker → current YES price map by fetching markets for any open
  // paper bet. In paper mode there's no Kalshi order but we can still mark
  // to market using the public market data.
  const paperState = loadPaperState('MLB');
  const openPaper = paperState.bets.filter(b => !b.settledAt);
  if (openPaper.length === 0) return;

  const priceMap: Record<string, number> = {};
  for (const bet of openPaper) {
    try {
      const market = await getMarket(bet.ticker);
      if (market?.yes_bid && market.yes_bid > 0) {
        priceMap[bet.ticker] = market.yes_bid;
      }
    } catch {
      // skip tickers we can't fetch
    }
  }
  const triggered = await scanForPaperStopLosses('MLB', priceMap);
  if (triggered.length > 0) {
    logger.info({ count: triggered.length, tickers: triggered.map(b => b.ticker) },
                '[Monitor] Paper stop-losses triggered');
  }
}

async function checkPositions(date: string): Promise<void> {
  // Always scan paper positions first (these exist even in dry-run)
  await checkPaperPositions();

  const openBets = getOpenBets(date);
  if (openBets.length === 0) return;

  logger.info({ count: openBets.length, time: new Date().toISOString() }, '[Monitor] Checking positions');

  for (const bet of openBets) {
    try {
      const market = await getMarket(bet.ticker);
      if (!market) continue;

      // Settled market — record final outcome
      if (market.status === 'finalized' || market.result) {
        const won = (market.result === 'yes' && bet.side === 'yes') ||
                    (market.result === 'no'  && bet.side === 'no');
        const pnl = won ? (1 - bet.cost_basis) : -bet.cost_basis;
        updateBetClosed(bet.id!, won ? 100 : 0, 'settled', pnl);
        logger.info({ ticker: bet.ticker, won, pnl: pnl.toFixed(2) }, '[Monitor] Bet settled');
        continue;
      }

      // Live market — check 20% stop-loss
      const currentBid = bet.side === 'yes' ? market.yes_bid : market.no_bid;
      if (!currentBid || currentBid <= 0) continue;

      const currentValue = (bet.contracts * currentBid) / 100;
      const pctChange = (currentValue - bet.cost_basis) / bet.cost_basis;

      logger.debug({
        ticker: bet.ticker,
        side: bet.side,
        entry: `${bet.entry_price}¢`,
        current: `${currentBid}¢`,
        pctChange: `${(pctChange * 100).toFixed(1)}%`,
      }, '[Monitor] Position');

      if (pctChange <= -CASHOUT_LOSS_PCT) {
        const pnl = currentValue - bet.cost_basis;
        updateBetClosed(bet.id!, currentBid, 'stop_loss_20pct', pnl);
        logger.warn({
          ticker: bet.ticker,
          loss: `${(pctChange * 100).toFixed(1)}%`,
          pnl: `$${pnl.toFixed(2)}`,
          paper: PAPER_TRADING,
        }, '[Monitor] STOP-LOSS triggered — position closed');
        // No Discord alert here — EOD summary will show it
      }
    } catch (err) {
      logger.error({ err, ticker: bet.ticker }, '[Monitor] Position check error');
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Main loop ────────────────────────────────────────────────────────────────

logger.info({ pollMs: POLL_MS, paper: PAPER_TRADING }, '[Monitor] Position monitor starting');

await initDb();
ensureKalshiBetsTable();

let polls = 0;

// GitHub Actions has a hard 6h max per job — we configure 2 shifts (1-6 PM
// and 6-11 PM ET) that each exit at MONITOR_EXIT_HOUR (ET). Default 23 = 11 PM.
const EXIT_HOUR = parseInt(process.env.MONITOR_EXIT_HOUR ?? '23', 10);

while (true) {
  const hour = etHour();
  const date = etDate();

  // Stop at configured ET hour — either EOD or the next shift takes over
  if (hour >= EXIT_HOUR) {
    logger.info({ exitHour: EXIT_HOUR }, '[Monitor] Exit hour reached — handing off');
    break;
  }

  polls++;
  logger.info({ poll: polls, etHour: hour, date }, '[Monitor] Poll');

  await checkPositions(date);
  await sleep(POLL_MS);
}

closeDb();
logger.info({ polls }, '[Monitor] Done');
