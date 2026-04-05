// GitHub Actions entry point — Position Monitor (1 PM – 11 PM ET)
// Loops every 2 minutes checking all open bets for 20% stop-loss.
// Exits cleanly at 11 PM ET so the EOD job can send the final summary.
// No Discord notifications during the loop — silent stop-losses only.

import 'dotenv/config';
import { initDb, closeDb } from '../db/database.js';
import { ensureKalshiBetsTable, getOpenBets, updateBetClosed } from '../kalshi/betEngine.js';
import { getMarket, PAPER_TRADING, CASHOUT_LOSS_PCT } from '../kalshi/kalshiClient.js';
import { logger } from '../logger.js';

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

async function checkPositions(date: string): Promise<void> {
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

while (true) {
  const hour = etHour();
  const date = etDate();

  // Stop at 11 PM ET — EOD job takes over
  if (hour >= 23) {
    logger.info('[Monitor] 11 PM ET reached — exiting for EOD job');
    break;
  }

  polls++;
  logger.info({ poll: polls, etHour: hour, date }, '[Monitor] Poll');

  await checkPositions(date);
  await sleep(POLL_MS);
}

closeDb();
logger.info({ polls }, '[Monitor] Done');
