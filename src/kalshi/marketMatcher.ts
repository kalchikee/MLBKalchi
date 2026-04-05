// MLB Oracle v4.0 — Kalshi Market Matcher
// Matches MLB Oracle predictions to live Kalshi MLB game markets

import type { KalshiMarket } from './kalshiClient.js';
import type { Prediction } from '../types.js';
import { logger } from '../logger.js';

// ─── Team abbreviation mapping ────────────────────────────────────────────────
// Maps MLB Stats API abbreviations → Kalshi ticker codes (usually the same)
const MLB_TO_KALSHI: Record<string, string> = {
  'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS',
  'CHC': 'CHC', 'CWS': 'CWS', 'CIN': 'CIN', 'CLE': 'CLE',
  'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU', 'KC':  'KC',
  'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
  'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'OAK': 'OAK',
  'PHI': 'PHI', 'PIT': 'PIT', 'SD':  'SD',  'SF':  'SF',
  'SEA': 'SEA', 'STL': 'STL', 'TB':  'TB',  'TEX': 'TEX',
  'TOR': 'TOR', 'WSH': 'WSH',
  // Alternates
  'SFG': 'SF', 'SDP': 'SD', 'KCR': 'KC',
  'TBR': 'TB', 'WSN': 'WSH', 'CHW': 'CWS',
};

// Reverse map: Kalshi code → MLB Stats API abbr
const KALSHI_TO_MLB: Record<string, string> = {};
for (const [mlb, kalshi] of Object.entries(MLB_TO_KALSHI)) {
  KALSHI_TO_MLB[kalshi] = mlb;
}

export interface MatchedBet {
  prediction: Prediction;
  market: KalshiMarket;
  ticker: string;
  /** What YES means: which team wins if YES resolves */
  yesTeam: string;         // MLB abbreviation
  noTeam: string;          // MLB abbreviation
  /** Which side to buy to back our model's prediction */
  side: 'yes' | 'no';
  /** The market price for our chosen side in cents */
  entryPriceCents: number;
  /** Our model's win probability for the favored team (0–1) */
  modelProb: number;
  /** Model prob converted to cents */
  modelProbCents: number;
}

/**
 * Given today's predictions and open Kalshi markets,
 * return matched bets where we have a strong enough edge.
 */
export function matchPredictionsToMarkets(
  predictions: Prediction[],
  markets: Map<string, KalshiMarket>,
  minProb: number,
): MatchedBet[] {
  const results: MatchedBet[] = [];

  for (const pred of predictions) {
    // Only bet on high-conviction predictions
    if (pred.calibrated_prob < minProb && (1 - pred.calibrated_prob) < minProb) continue;

    // Find matching Kalshi market for this game
    const match = findMarketForGame(pred, markets);
    if (!match) {
      logger.debug(
        { home: pred.home_team, away: pred.away_team },
        'No Kalshi market found for game',
      );
      continue;
    }

    const { market, yesTeam, noTeam } = match;

    // Determine which side to buy
    const homeWinProb = pred.calibrated_prob;
    const awayWinProb = 1 - pred.calibrated_prob;

    let side: 'yes' | 'no';
    let modelProb: number;

    if (yesTeam === pred.home_team || yesTeam === toKalshi(pred.home_team)) {
      // YES = home team wins
      side = homeWinProb >= minProb ? 'yes' : 'no';
      modelProb = side === 'yes' ? homeWinProb : awayWinProb;
    } else {
      // YES = away team wins
      side = awayWinProb >= minProb ? 'yes' : 'no';
      modelProb = side === 'yes' ? awayWinProb : homeWinProb;
    }

    if (modelProb < minProb) continue;

    const entryPriceCents = side === 'yes' ? market.yes_ask : market.no_ask;
    if (!entryPriceCents || entryPriceCents <= 0) continue;

    // Basic value check: only bet if market is at least somewhat cheaper than our model
    const modelProbCents = Math.round(modelProb * 100);
    if (entryPriceCents >= modelProbCents + 2) {
      // Market already priced at or above our estimate — skip
      logger.debug(
        { ticker: market.ticker, modelProbCents, entryPriceCents },
        'Market priced above model estimate — skipping',
      );
      continue;
    }

    results.push({
      prediction: pred,
      market,
      ticker: market.ticker,
      yesTeam,
      noTeam,
      side,
      entryPriceCents,
      modelProb,
      modelProbCents,
    });
  }

  return results;
}

/**
 * Find the Kalshi market matching a given game prediction.
 * Tries multiple ticker formats and fuzzy title matching.
 */
function findMarketForGame(
  pred: Prediction,
  markets: Map<string, KalshiMarket>,
): { market: KalshiMarket; yesTeam: string; noTeam: string } | null {
  const homeKalshi = toKalshi(pred.home_team);
  const awayKalshi = toKalshi(pred.away_team);

  // Try all markets and find one that mentions both teams
  for (const [ticker, market] of markets) {
    const upperTicker = ticker.toUpperCase();
    const title = (market.title ?? '').toUpperCase();
    const yesSub = (market.yes_sub_title ?? '').toUpperCase();

    // Check if ticker mentions one of the teams
    const tickerMentionsHome = upperTicker.includes(homeKalshi) || upperTicker.includes(pred.home_team.toUpperCase());
    const tickerMentionsAway = upperTicker.includes(awayKalshi) || upperTicker.includes(pred.away_team.toUpperCase());
    const titleMentionsBoth =
      (title.includes(homeKalshi) || title.includes(pred.home_team.toUpperCase())) &&
      (title.includes(awayKalshi) || title.includes(pred.away_team.toUpperCase()));

    if (!tickerMentionsHome && !tickerMentionsAway && !titleMentionsBoth) continue;

    // Determine what YES means from the title/subtitle
    const yesTeam = parseYesTeam(market, pred.home_team, pred.away_team);
    if (!yesTeam) continue;

    const noTeam = yesTeam === pred.home_team ? pred.away_team : pred.home_team;

    logger.debug(
      { ticker, home: pred.home_team, away: pred.away_team, yesTeam },
      'Matched market to game',
    );
    return { market, yesTeam, noTeam };
  }

  return null;
}

/**
 * Parse which team YES corresponds to from market metadata.
 * Returns MLB abbreviation or null if indeterminate.
 */
function parseYesTeam(
  market: KalshiMarket,
  homeAbbr: string,
  awayAbbr: string,
): string | null {
  const homeKalshi = toKalshi(homeAbbr).toUpperCase();
  const awayKalshi = toKalshi(awayAbbr).toUpperCase();

  // Try yes_sub_title first (most reliable)
  const yesSub = (market.yes_sub_title ?? '').toUpperCase();
  if (yesSub) {
    if (yesSub.includes(homeKalshi) || yesSub.includes(homeAbbr.toUpperCase())) return homeAbbr;
    if (yesSub.includes(awayKalshi) || yesSub.includes(awayAbbr.toUpperCase())) return awayAbbr;
  }

  // Try title: "Will [AWAY] beat [HOME]?" → YES = away team
  const title = (market.title ?? '').toUpperCase();
  if (title.includes('WILL')) {
    if (title.includes(awayKalshi) || title.includes(awayAbbr.toUpperCase())) {
      // "Will AWAY beat HOME?" — YES = away
      if (title.includes(homeKalshi) || title.includes(homeAbbr.toUpperCase())) return awayAbbr;
    }
    if (title.includes(homeKalshi) || title.includes(homeAbbr.toUpperCase())) {
      return homeAbbr;
    }
  }

  // Ticker format: KXMLBGAME-26APR04-TEAM → YES = TEAM wins
  const parts = market.ticker.split('-');
  const tickerTeam = parts[parts.length - 1].replace(/\d+$/, '').toUpperCase();
  if (tickerTeam === homeKalshi || tickerTeam === homeAbbr.toUpperCase()) return homeAbbr;
  if (tickerTeam === awayKalshi || tickerTeam === awayAbbr.toUpperCase()) return awayAbbr;

  return null;
}

function toKalshi(mlbAbbr: string): string {
  return MLB_TO_KALSHI[mlbAbbr.toUpperCase()] ?? mlbAbbr.toUpperCase();
}

export function toMLB(kalshiCode: string): string {
  return KALSHI_TO_MLB[kalshiCode.toUpperCase()] ?? kalshiCode.toUpperCase();
}
