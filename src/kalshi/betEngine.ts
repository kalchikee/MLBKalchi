// MLB Oracle v4.0 — Kalshi Bet Engine
// Evaluates predictions, finds value, places bets (or logs in paper mode)

import { initDb, getDb, persistDb, getPredictionsByDate } from '../db/database.js';
import {
  scanTodayMLBMarkets,
  placeOrder,
  getBalance,
  PAPER_TRADING,
  BET_SIZE_DOLLARS,
  MIN_MODEL_PROB,
  MIN_EDGE,
} from './kalshiClient.js';
import { matchPredictionsToMarkets } from './marketMatcher.js';
import { sendNoBetsAlert } from '../alerts/discord.js';
import { getTeamGamesPlayed, fetchSchedule } from '../api/mlbClient.js';
import { logger } from '../logger.js';

// ─── Thresholds (all configurable via env) ────────────────────────────────────

const KALSHI_MIN_TEAM_GAMES = parseInt(process.env.KALSHI_MIN_TEAM_GAMES ?? '10', 10);

// Minimum absolute pythagorean win% gap between teams.
// Below this the game is a toss-up — model accuracy drops to ~55% (skip it).
// Segment analysis: |pyth_diff| < 0.03 → 55.1% accuracy vs 71.9% for strong mismatches.
const KALSHI_MIN_PYTH_DIFF = parseFloat(process.env.KALSHI_MIN_PYTH_DIFF ?? '0.03');

// Minimum absolute Elo gap between teams.
// Below this there's no meaningful quality separation.
// Segment analysis: |elo_diff| < 25 → 56.3% accuracy vs 68.1% for clear favourites.
const KALSHI_MIN_ELO_DIFF = parseFloat(process.env.KALSHI_MIN_ELO_DIFF ?? '25');

// Maximum dollars to risk on any single bet (hard cap on Kelly sizing).
const KALSHI_MAX_BET_DOLLARS = parseFloat(process.env.KALSHI_MAX_BET_DOLLARS ?? '1');

export interface KalshiBetRecord {
  id?: number;
  date: string;
  game_pk: number;
  home_team: string;
  away_team: string;
  ticker: string;
  side: 'yes' | 'no';
  contracts: number;
  entry_price: number;       // cents per contract
  cost_basis: number;        // total dollars paid
  model_prob: number;
  edge: number;
  order_id: string;
  status: 'open' | 'sold' | 'settled' | 'cancelled';
  exit_price?: number;       // cents per contract
  exit_reason?: string;
  pnl?: number;              // dollars
  created_at: string;
  closed_at?: string;
}

// ─── DB Schema (added lazily) ─────────────────────────────────────────────────

export function ensureKalshiBetsTable(): void {
  const db = getDb();
  db.run(`
    CREATE TABLE IF NOT EXISTS kalshi_bets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      date TEXT NOT NULL,
      game_pk INTEGER NOT NULL,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      ticker TEXT NOT NULL,
      side TEXT NOT NULL,
      contracts INTEGER NOT NULL,
      entry_price INTEGER NOT NULL,
      cost_basis REAL NOT NULL,
      model_prob REAL NOT NULL,
      edge REAL NOT NULL DEFAULT 0,
      order_id TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'open',
      exit_price INTEGER,
      exit_reason TEXT,
      pnl REAL,
      created_at TEXT NOT NULL,
      closed_at TEXT
    );
  `);
  persistDb();
}

export function saveBet(bet: KalshiBetRecord): number {
  const db = getDb();
  db.run(
    `INSERT INTO kalshi_bets
     (date, game_pk, home_team, away_team, ticker, side, contracts,
      entry_price, cost_basis, model_prob, edge, order_id, status, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      bet.date, bet.game_pk, bet.home_team, bet.away_team,
      bet.ticker, bet.side, bet.contracts,
      bet.entry_price, bet.cost_basis, bet.model_prob, bet.edge,
      bet.order_id, bet.status, bet.created_at,
    ],
  );
  persistDb();
  const row = db.exec('SELECT last_insert_rowid() as id')[0];
  return (row?.values?.[0]?.[0] as number) ?? 0;
}

export function updateBetClosed(
  id: number,
  exitPrice: number,
  exitReason: string,
  pnl: number,
): void {
  const db = getDb();
  db.run(
    `UPDATE kalshi_bets
     SET status = 'sold', exit_price = ?, exit_reason = ?, pnl = ?, closed_at = ?
     WHERE id = ?`,
    [exitPrice, exitReason, pnl, new Date().toISOString(), id],
  );
  persistDb();
}

export function getOpenBets(date?: string): KalshiBetRecord[] {
  const db = getDb();
  const sql = date
    ? `SELECT * FROM kalshi_bets WHERE status = 'open' AND date = ? ORDER BY created_at`
    : `SELECT * FROM kalshi_bets WHERE status = 'open' ORDER BY created_at`;
  const params = date ? [date] : [];
  const stmt = db.prepare(sql);
  if (params.length) stmt.bind(params);
  const rows: KalshiBetRecord[] = [];
  while (stmt.step()) rows.push(stmt.getAsObject() as unknown as KalshiBetRecord);
  stmt.free();
  return rows;
}

export function getAllBetsForDate(date: string): KalshiBetRecord[] {
  const db = getDb();
  const stmt = db.prepare(`SELECT * FROM kalshi_bets WHERE date = ? ORDER BY created_at`);
  stmt.bind([date]);
  const rows: KalshiBetRecord[] = [];
  while (stmt.step()) rows.push(stmt.getAsObject() as unknown as KalshiBetRecord);
  stmt.free();
  return rows;
}

// ─── Main Bet Engine ──────────────────────────────────────────────────────────

export async function runBetEngine(date: string): Promise<KalshiBetRecord[]> {
  await initDb();
  ensureKalshiBetsTable();

  const mode = PAPER_TRADING ? '[PAPER TRADING]' : '[LIVE]';
  logger.info({ date, mode, betSize: BET_SIZE_DOLLARS, minProb: MIN_MODEL_PROB }, 'Bet engine starting');

  // 1. Load today's predictions from DB
  const predictions = getPredictionsByDate(date);
  if (predictions.length === 0) {
    logger.warn({ date }, 'No predictions found — run pipeline first');
    return [];
  }

  // 2. Get balance (skip in paper mode if API creds not fully configured)
  let balanceDollars = 0;
  try {
    const bal = await getBalance();
    balanceDollars = bal.balance / 100;
    logger.info({ balance: `$${balanceDollars.toFixed(2)}` }, 'Kalshi balance');
  } catch (err) {
    if (!PAPER_TRADING) throw err;
    logger.warn('Could not fetch balance (paper mode — continuing)');
    balanceDollars = 9999; // unlimited in paper mode
  }

  // 3. Scan today's Kalshi MLB markets
  let markets: Map<string, import('./kalshiClient.js').KalshiMarket>;
  try {
    markets = await scanTodayMLBMarkets(date);
  } catch (err) {
    logger.error({ err }, 'Failed to scan Kalshi markets');
    return [];
  }

  if (markets.size === 0) {
    logger.warn('No MLB markets found on Kalshi for today');
    await sendNoBetsAlert(date, 'No MLB markets found on Kalshi');
    return [];
  }

  // 4. Match predictions to markets
  const candidates = matchPredictionsToMarkets(predictions, markets, MIN_MODEL_PROB);

  if (candidates.length === 0) {
    logger.info({ date }, 'No high-conviction bets found today');
    await sendNoBetsAlert(date, `No games matched a Kalshi market at ${(MIN_MODEL_PROB * 100).toFixed(0)}%+ model confidence`);
    return [];
  }

  // 5. Fetch live schedule for day-of SP change detection
  let liveGames = new Map<number, import('../types.js').Game>();
  try {
    const gameList = await fetchSchedule(date);
    for (const g of gameList) liveGames.set(g.gamePk, g);
    logger.info({ games: liveGames.size }, 'Live schedule loaded for SP verification');
  } catch (err) {
    logger.warn({ err }, 'Could not fetch live schedule — SP change check disabled');
  }

  // 6. Place bets
  const placed: KalshiBetRecord[] = [];

  for (const candidate of candidates) {
    // Skip if we don't have enough balance
    if (balanceDollars < BET_SIZE_DOLLARS) {
      logger.warn({ balance: balanceDollars }, 'Insufficient balance — stopping');
      break;
    }

    // ── Early-season data reliability check ───────────────────────────────
    const homeTeam = candidate.prediction.home_team;
    const awayTeam = candidate.prediction.away_team;
    const matchup = `${awayTeam} @ ${homeTeam}`;

    try {
      const [homeGP, awayGP] = await Promise.all([
        getTeamGamesPlayed(homeTeam),
        getTeamGamesPlayed(awayTeam),
      ]);

      if (homeGP < KALSHI_MIN_TEAM_GAMES || awayGP < KALSHI_MIN_TEAM_GAMES) {
        const minGP = Math.min(homeGP, awayGP);
        logger.info(
          { matchup, homeGP, awayGP, minRequired: KALSHI_MIN_TEAM_GAMES },
          `Skipping ${matchup} — early season (${minGP} games played)`
        );
        continue;
      }
    } catch (err) {
      // If we can't verify games played, log and continue (don't block bets)
      logger.warn({ err, matchup }, 'Could not verify games played — skipping early-season check');
    }

    // ── Structural quality filters (from segment analysis) ────────────────
    // These are the stable, non-overfitted signals. Per-team/stadium filters
    // are intentionally excluded — accuracy there fluctuates year to year.
    const fv = candidate.prediction.feature_vector;

    // 1. Skip toss-up games — pythagorean gap too small
    const pythDiff = Math.abs(fv?.pythagorean_diff ?? 0);
    if (pythDiff < KALSHI_MIN_PYTH_DIFF) {
      logger.info(
        { matchup, pythDiff: pythDiff.toFixed(3), threshold: KALSHI_MIN_PYTH_DIFF },
        `Skipping — toss-up (pythagorean gap ${pythDiff.toFixed(3)} < ${KALSHI_MIN_PYTH_DIFF})`
      );
      continue;
    }

    // 2. Skip games with no Elo separation
    const eloDiff = Math.abs(fv?.elo_diff ?? 0);
    if (eloDiff < KALSHI_MIN_ELO_DIFF) {
      logger.info(
        { matchup, eloDiff: eloDiff.toFixed(0), threshold: KALSHI_MIN_ELO_DIFF },
        `Skipping — Elo gap too small (${eloDiff.toFixed(0)} pts < ${KALSHI_MIN_ELO_DIFF})`
      );
      continue;
    }

    // 3. Skip October — model accuracy drops ~2%, small sample, playoff-adjacent chaos
    const gameMonth = new Date(date).getMonth() + 1; // 1=Jan, 10=Oct
    if (gameMonth === 10) {
      logger.info({ matchup }, 'Skipping — October games less predictable');
      continue;
    }

    // 4. SP change detection — skip if either pitcher is now TBD
    const liveGame = liveGames.get(candidate.prediction.game_pk);
    if (liveGame) {
      const homeHasSP = !!liveGame.homeTeam.probablePitcher;
      const awayHasSP = !!liveGame.awayTeam.probablePitcher;
      if (!homeHasSP || !awayHasSP) {
        const missing = [!homeHasSP && 'home', !awayHasSP && 'away'].filter(Boolean).join('+');
        logger.info(
          { matchup, missing },
          `Skipping — ${missing} SP is now TBD (probable pitcher changed day-of)`
        );
        continue;
      }
    }

    // Check edge if Vegas lines are loaded (optional guard)
    const edge = candidate.prediction.edge ?? 0;
    if (candidate.prediction.vegas_prob !== undefined && candidate.prediction.vegas_prob > 0) {
      if (Math.abs(edge) < MIN_EDGE) {
        logger.debug({ ticker: candidate.ticker, edge }, 'Edge below threshold — skipping');
        continue;
      }
    }

    // Kelly Criterion sizing (quarter Kelly, capped at KALSHI_MAX_BET_DOLLARS)
    // b = (100 - price) / price  (payoff ratio per dollar risked)
    // f* = (p*b - (1-p)) / b     (full Kelly fraction of bankroll)
    const priceCents = candidate.entryPriceCents;
    const p = candidate.modelProb;
    const b = (100 - priceCents) / priceCents;
    const kellyFraction = Math.max(0, (p * b - (1 - p)) / b);

    // Skip if no positive Kelly edge — previously we fell through to minimum
    // BET_SIZE_DOLLARS even with 0 or negative edge when Vegas data was
    // unavailable for edge-based rejection.
    if (kellyFraction <= 0) {
      logger.debug({ ticker: candidate.ticker, p, priceCents }, 'No positive edge (model prob ≤ market price) — skipping');
      continue;
    }

    const quarterKelly = kellyFraction * 0.25;
    const betDollars = Math.min(
      KALSHI_MAX_BET_DOLLARS,
      Math.max(BET_SIZE_DOLLARS, quarterKelly * balanceDollars),
    );
    const contracts = Math.max(1, Math.round(betDollars / (priceCents / 100)));

    try {
      const result = await placeOrder(
        candidate.ticker,
        candidate.side,
        candidate.entryPriceCents,
        contracts,
      );

      const costBasis = (contracts * candidate.entryPriceCents) / 100;
      balanceDollars -= costBasis;

      const bet: KalshiBetRecord = {
        date,
        game_pk: candidate.prediction.game_pk,
        home_team: candidate.prediction.home_team,
        away_team: candidate.prediction.away_team,
        ticker: candidate.ticker,
        side: candidate.side,
        contracts,
        entry_price: candidate.entryPriceCents,
        cost_basis: costBasis,
        model_prob: candidate.modelProb,
        edge,
        order_id: result.order.order_id,
        status: 'open',
        created_at: new Date().toISOString(),
      };

      const betId = saveBet(bet);
      bet.id = betId;
      placed.push(bet);

      // No per-bet Discord alert — morning briefing and EOD summary only
      logger.info(
        {
          mode,
          ticker: candidate.ticker,
          side: candidate.side,
          entryPrice: `${candidate.entryPriceCents}¢`,
          contracts,
          costBasis: `$${costBasis.toFixed(2)}`,
          modelProb: `${(candidate.modelProb * 100).toFixed(1)}%`,
          matchup: `${candidate.prediction.away_team} @ ${candidate.prediction.home_team}`,
        },
        'Bet placed',
      );
    } catch (err) {
      logger.error({ err, ticker: candidate.ticker }, 'Failed to place order');
    }
  }

  logger.info({ placed: placed.length, mode }, 'Bet engine complete');
  return placed;
}
