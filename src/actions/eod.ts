// GitHub Actions entry point — EOD Summary (11 PM ET)
// 1. Check open positions for 20% stop-losses (single scan, not continuous)
// 2. Send ONE Discord message with all bet results + net P&L

import 'dotenv/config';
import { initDb, closeDb } from '../db/database.js';
import { ensureKalshiBetsTable, getOpenBets, getAllBetsForDate, updateBetClosed } from '../kalshi/betEngine.js';
import { getMarket, PAPER_TRADING, CASHOUT_LOSS_PCT } from '../kalshi/kalshiClient.js';
import { sendEODSummaryAlert } from '../alerts/discord.js';
import { logger } from '../logger.js';

const date = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });

logger.info({ date, paper: PAPER_TRADING }, '[EOD Action] Starting');

try {
  await initDb();
  ensureKalshiBetsTable();

  // 1. Single stop-loss scan on all open bets
  const openBets = getOpenBets(date);
  logger.info({ count: openBets.length }, '[EOD Action] Checking open positions');

  for (const bet of openBets) {
    try {
      const market = await getMarket(bet.ticker);
      if (!market) continue;

      // If market settled, record outcome
      if (market.status === 'finalized' || market.result) {
        const won = (market.result === 'yes' && bet.side === 'yes') ||
                    (market.result === 'no'  && bet.side === 'no');
        const pnl = won ? (1 - bet.cost_basis) : -bet.cost_basis;
        updateBetClosed(bet.id!, won ? 100 : 0, 'settled', pnl);
        logger.info({ ticker: bet.ticker, won, pnl: pnl.toFixed(2) }, '[EOD] Bet settled');
        continue;
      }

      // Check 20% stop-loss
      const currentBid = bet.side === 'yes' ? market.yes_bid : market.no_bid;
      if (!currentBid) continue;

      const currentValue = (bet.contracts * currentBid) / 100;
      const pctChange = (currentValue - bet.cost_basis) / bet.cost_basis;

      if (pctChange <= -CASHOUT_LOSS_PCT) {
        const pnl = currentValue - bet.cost_basis;
        updateBetClosed(bet.id!, currentBid, 'stop_loss_20pct', pnl);
        logger.warn(
          { ticker: bet.ticker, loss: `${(pctChange * 100).toFixed(1)}%`, pnl: pnl.toFixed(2) },
          '[EOD] Stop-loss triggered',
        );
      }
    } catch (err) {
      logger.error({ err, ticker: bet.ticker }, '[EOD] Position check failed');
    }
  }

  // 2. Send EOD Discord summary
  const allBets = getAllBetsForDate(date);
  const totalCost  = allBets.reduce((s, b) => s + b.cost_basis, 0);
  const totalPnl   = allBets.reduce((s, b) => s + (b.pnl ?? 0), 0);
  const wins       = allBets.filter(b => (b.pnl ?? 0) > 0).length;
  const losses     = allBets.filter(b => b.status !== 'open' && (b.pnl ?? 0) <= 0).length;
  const open       = allBets.filter(b => b.status === 'open').length;

  await sendEODSummaryAlert(date, allBets, { totalCost, totalPnl, wins, losses, open }, PAPER_TRADING);

  logger.info({ date, bets: allBets.length, pnl: totalPnl.toFixed(2) }, '[EOD Action] Complete');
} catch (err) {
  logger.error({ err }, '[EOD Action] Failed');
  process.exit(1);
} finally {
  closeDb();
}
