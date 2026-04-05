// MLB Oracle v4.0 — Kalshi Position Monitor
// Polls every 60 seconds, auto-cashes out if position loses 20% of value

import {
  getPositions,
  getMarket,
  sellPosition,
  PAPER_TRADING,
  CASHOUT_LOSS_PCT,
  type KalshiPosition,
} from './kalshiClient.js';
import {
  getOpenBets,
  updateBetClosed,
  getAllBetsForDate,
  type KalshiBetRecord,
} from './betEngine.js';
import { sendCashoutAlert, sendEODSummaryAlert } from '../alerts/discord.js';
import { logger } from '../logger.js';

const POLL_INTERVAL_MS = parseInt(process.env.KALSHI_MONITOR_INTERVAL_MS ?? '60000', 10);

let monitorInterval: ReturnType<typeof setInterval> | null = null;
let isRunning = false;

// ─── Single poll cycle ────────────────────────────────────────────────────────

export async function pollPositions(): Promise<void> {
  if (isRunning) return; // prevent overlapping polls
  isRunning = true;

  try {
    const today = new Date().toISOString().split('T')[0];
    const openBets = getOpenBets(today);

    if (openBets.length === 0) {
      logger.debug('No open bets to monitor');
      return;
    }

    // Fetch live positions from Kalshi
    let livePositions: KalshiPosition[] = [];
    try {
      livePositions = await getPositions();
    } catch (err) {
      if (!PAPER_TRADING) {
        logger.error({ err }, 'Failed to fetch positions');
        return;
      }
      // In paper mode, simulate no position change
      logger.debug('Paper mode — skipping live position fetch');
      return;
    }

    const positionMap = new Map<string, KalshiPosition>(
      livePositions.filter(p => parseFloat(p.position_fp) > 0).map(p => [p.ticker, p]),
    );

    for (const bet of openBets) {
      await checkAndHandleBet(bet, positionMap);
    }
  } catch (err) {
    logger.error({ err }, 'Position monitor poll error');
  } finally {
    isRunning = false;
  }
}

async function checkAndHandleBet(
  bet: KalshiBetRecord,
  positionMap: Map<string, KalshiPosition>,
): Promise<void> {
  // Get current market price
  const market = await getMarket(bet.ticker);
  if (!market) {
    logger.debug({ ticker: bet.ticker }, 'Market not found — may be settled');
    return;
  }

  // Current bid price for our side (what we could sell for now)
  const currentBidCents = bet.side === 'yes' ? market.yes_bid : market.no_bid;
  if (!currentBidCents || currentBidCents <= 0) return;

  // Cost basis per contract
  const entryPerContract = bet.entry_price; // cents
  const currentPerContract = currentBidCents;

  // Current value vs what we paid
  const currentValue = (bet.contracts * currentPerContract) / 100;
  const costBasis = bet.cost_basis;
  const pctChange = (currentValue - costBasis) / costBasis;

  logger.debug(
    {
      ticker: bet.ticker,
      side: bet.side,
      entryPrice: `${entryPerContract}¢`,
      currentBid: `${currentBidCents}¢`,
      costBasis: `$${costBasis.toFixed(2)}`,
      currentValue: `$${currentValue.toFixed(2)}`,
      pctChange: `${(pctChange * 100).toFixed(1)}%`,
    },
    'Position check',
  );

  // Check 20% stop-loss
  if (pctChange <= -CASHOUT_LOSS_PCT) {
    logger.warn(
      {
        ticker: bet.ticker,
        loss: `${(pctChange * 100).toFixed(1)}%`,
        costBasis: `$${costBasis.toFixed(2)}`,
        currentValue: `$${currentValue.toFixed(2)}`,
      },
      'STOP-LOSS triggered — cashing out',
    );

    try {
      await sellPosition(bet.ticker, bet.side, bet.contracts, currentBidCents);

      const pnl = currentValue - costBasis;
      updateBetClosed(bet.id!, currentBidCents, 'stop_loss_20pct', pnl);

      await sendCashoutAlert(bet, currentBidCents, pnl, pctChange, PAPER_TRADING);
    } catch (err) {
      logger.error({ err, ticker: bet.ticker }, 'Failed to execute stop-loss sell');
    }
    return;
  }

  // Check if market settled (status = 'settled')
  if (market.status === 'finalized' || market.result) {
    const settledValue = resolvedValue(market.result ?? '', bet.side, bet.contracts);
    const pnl = settledValue - costBasis;

    updateBetClosed(bet.id!, settledValue > 0 ? 100 : 0, 'settled', pnl);
    logger.info(
      { ticker: bet.ticker, result: market.result, pnl: `$${pnl.toFixed(2)}` },
      'Bet settled',
    );
  }
}

/** Calculate settled value: if YES wins and we hold YES, we get $1/contract */
function resolvedValue(result: string, side: 'yes' | 'no', contracts: number): number {
  const r = result.toLowerCase();
  if ((r === 'yes' && side === 'yes') || (r === 'no' && side === 'no')) {
    return contracts * 1.0; // $1 per winning contract
  }
  return 0;
}

// ─── Start / stop monitor ─────────────────────────────────────────────────────

export function startPositionMonitor(): void {
  if (monitorInterval) return;

  const mode = PAPER_TRADING ? '[PAPER]' : '[LIVE]';
  logger.info({ intervalMs: POLL_INTERVAL_MS, mode }, 'Position monitor started');

  // Run immediately, then every interval
  pollPositions().catch(err => logger.error({ err }, 'Initial poll failed'));
  monitorInterval = setInterval(() => {
    pollPositions().catch(err => logger.error({ err }, 'Poll failed'));
  }, POLL_INTERVAL_MS);
}

export function stopPositionMonitor(): void {
  if (monitorInterval) {
    clearInterval(monitorInterval);
    monitorInterval = null;
    logger.info('Position monitor stopped');
  }
}

// ─── End-of-day summary ───────────────────────────────────────────────────────

export async function sendEndOfDaySummary(date: string): Promise<void> {
  const allBets = getAllBetsForDate(date);

  if (allBets.length === 0) {
    logger.info({ date }, 'No bets placed today — no EOD summary needed');
    return;
  }

  const totalCost = allBets.reduce((s, b) => s + b.cost_basis, 0);
  const totalPnl = allBets.reduce((s, b) => s + (b.pnl ?? 0), 0);
  const wins = allBets.filter(b => (b.pnl ?? 0) > 0).length;
  const losses = allBets.filter(b => (b.pnl ?? 0) < 0).length;
  const open = allBets.filter(b => b.status === 'open').length;

  logger.info(
    {
      date,
      bets: allBets.length,
      wins,
      losses,
      open,
      totalCost: `$${totalCost.toFixed(2)}`,
      totalPnl: `$${totalPnl.toFixed(2)}`,
    },
    'EOD summary',
  );

  await sendEODSummaryAlert(date, allBets, { totalCost, totalPnl, wins, losses, open }, PAPER_TRADING);
}
