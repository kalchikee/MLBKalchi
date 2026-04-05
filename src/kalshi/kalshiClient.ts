// MLB Oracle v4.0 — Kalshi API Client
// RSA-PSS authentication, market scanning, order placement, position management

import { createPrivateKey, createSign, constants as cryptoConstants, type KeyObject } from 'crypto';
import { logger } from '../logger.js';

const BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2';
const API_PREFIX = '/trade-api/v2';

// ─── Config ───────────────────────────────────────────────────────────────────

function getKeyId(): string {
  const id = process.env.KALSHI_API_KEY_ID;
  if (!id) throw new Error('KALSHI_API_KEY_ID not set in .env');
  return id;
}

function getPrivateKey(): KeyObject {
  let pem = process.env.KALSHI_PRIVATE_KEY ?? '';
  // Handle escaped newlines from .env
  pem = pem.replace(/\\n/g, '\n').replace(/^"|"$/g, '');
  if (!pem) throw new Error('KALSHI_PRIVATE_KEY not set in .env');
  return createPrivateKey(pem);
}

export const PAPER_TRADING = process.env.KALSHI_PAPER_TRADING !== 'false';
export const BET_SIZE_DOLLARS = parseFloat(process.env.KALSHI_BET_SIZE ?? '10');
export const MIN_MODEL_PROB = parseFloat(process.env.KALSHI_MIN_PROB ?? '0.65');
export const MIN_EDGE = parseFloat(process.env.KALSHI_MIN_EDGE ?? '0.05');
export const CASHOUT_LOSS_PCT = parseFloat(process.env.KALSHI_CASHOUT_LOSS_PCT ?? '0.20');

// ─── Auth ─────────────────────────────────────────────────────────────────────

function getHeaders(method: string, path: string): Record<string, string> {
  const keyId = getKeyId();
  const privateKey = getPrivateKey();
  const timestamp = String(Date.now());
  const message = timestamp + method.toUpperCase() + API_PREFIX + path;

  const sign = createSign('RSA-SHA256');
  sign.update(message, 'utf8');
  const signature = sign.sign({
    key: privateKey,
    padding: cryptoConstants.RSA_PKCS1_PSS_PADDING,
    saltLength: 32,
  });

  return {
    'KALSHI-ACCESS-KEY': keyId,
    'KALSHI-ACCESS-TIMESTAMP': timestamp,
    'KALSHI-ACCESS-SIGNATURE': signature.toString('base64'),
    'Content-Type': 'application/json',
  };
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

async function kalshiGet<T = unknown>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(BASE_URL + path);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));

  const resp = await fetch(url.toString(), {
    method: 'GET',
    headers: getHeaders('GET', path),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Kalshi GET ${path} → ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

async function kalshiPost<T = unknown>(path: string, body: Record<string, unknown>): Promise<T> {
  const resp = await fetch(BASE_URL + path, {
    method: 'POST',
    headers: getHeaders('POST', path),
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Kalshi POST ${path} → ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface KalshiMarket {
  ticker: string;
  title?: string;
  subtitle?: string;
  yes_sub_title?: string;
  no_sub_title?: string;
  status: string;
  // Dollar-denominated fields (Kalshi v2 API returns these as decimal strings)
  yes_bid_dollars?: string;
  yes_ask_dollars?: string;
  no_bid_dollars?: string;
  no_ask_dollars?: string;
  last_price_dollars?: string;
  // Computed cent values (populated by normalizePrices)
  yes_bid: number;   // cents
  yes_ask: number;   // cents
  no_bid: number;    // cents
  no_ask: number;    // cents
  last_price?: number;
  volume?: number;
  close_time?: string;
  result?: string;
  rules_primary?: string;
  event_ticker?: string;
  custom_strike?: Record<string, string>;
}

/** Convert dollar string fields → integer cents and populate yes_bid/yes_ask/no_bid/no_ask */
export function normalizePrices(m: KalshiMarket): KalshiMarket {
  const toCents = (s?: string): number =>
    s ? Math.round(parseFloat(s) * 100) : 0;
  return {
    ...m,
    yes_bid: toCents(m.yes_bid_dollars),
    yes_ask: toCents(m.yes_ask_dollars),
    no_bid: toCents(m.no_bid_dollars),
    no_ask: toCents(m.no_ask_dollars),
    last_price: toCents(m.last_price_dollars),
  };
}

export interface KalshiPosition {
  ticker: string;
  position_fp: string;             // "1.00" = 1 YES contract
  market_exposure_dollars: string; // total $ exposure as decimal string
  realized_pnl_dollars: string;
  resting_orders_count: number;
  total_traded_dollars: string;    // total paid as decimal string
}

export interface KalshiBalance {
  balance: number;          // cents
  portfolio_value: number;  // cents
  updated_ts: number;
}

export interface KalshiOrder {
  order_id: string;
  ticker: string;
  action: 'buy' | 'sell';
  side: 'yes' | 'no';
  type: 'limit' | 'market';
  status: string;
  yes_price: number;
  no_price: number;
  count: number;
  filled_count: number;
  remaining_count: number;
  created_time: string;
}

export interface PlaceOrderResult {
  order: KalshiOrder;
  paper?: boolean;
}

// ─── Account ──────────────────────────────────────────────────────────────────

export async function getBalance(): Promise<KalshiBalance> {
  const data = await kalshiGet<KalshiBalance>('/portfolio/balance');
  return data;
}

export async function getPositions(): Promise<KalshiPosition[]> {
  const data = await kalshiGet<{ market_positions: KalshiPosition[] }>('/portfolio/positions', {
    count_filter: 'position',
  });
  return data.market_positions ?? [];
}

// ─── Markets ──────────────────────────────────────────────────────────────────

export async function getMarket(ticker: string): Promise<KalshiMarket | null> {
  try {
    const data = await kalshiGet<{ market: KalshiMarket }>(`/markets/${ticker}`);
    return normalizePrices(data.market);
  } catch {
    return null;
  }
}

/**
 * Scan all open MLB game markets for today.
 * Returns ticker → market map.
 */
export async function scanTodayMLBMarkets(date?: string): Promise<Map<string, KalshiMarket>> {
  const today = toKalshiDate(date);
  const markets = new Map<string, KalshiMarket>();

  let cursor: string | undefined;
  let attempts = 0;

  // Collect unique KXMLBGAME tickers from multileg market legs
  const gameTickersFound = new Set<string>();

  while (attempts < 10) {
    attempts++;
    const params: Record<string, string> = { limit: '200', status: 'open' };
    if (cursor) params.cursor = cursor;

    const data = await kalshiGet<{ markets: KalshiMarket[]; cursor?: string }>('/markets', params);
    const batch = data.markets ?? [];

    for (const m of batch) {
      // Individual market matches
      const t = m.ticker ?? '';
      if (t.includes(today) && t.startsWith('KXMLBGAME') && !t.includes('TOTAL')) {
        gameTickersFound.add(t);
      }
      // Game tickers embedded as legs in multileg markets
      for (const leg of ((m as unknown as Record<string, unknown>)['mve_selected_legs'] as Array<{ market_ticker?: string }> ?? [])) {
        const lt = leg.market_ticker ?? '';
        if (lt.includes(today) && lt.startsWith('KXMLBGAME') && !lt.includes('TOTAL')) {
          gameTickersFound.add(lt);
        }
      }
    }

    cursor = data.cursor;
    if (!cursor || batch.length === 0) break;
  }

  // Fetch full market details for each game ticker (needed for prices)
  for (const t of gameTickersFound) {
    const full = await getMarket(t);
    if (full) markets.set(t, full);
  }

  logger.info({ count: markets.size, date: today }, 'Scanned Kalshi MLB markets');
  return markets;
}

// ─── Orders ───────────────────────────────────────────────────────────────────

/**
 * Place a buy order on Kalshi.
 * In paper trading mode, logs the action but does NOT hit the API.
 *
 * @param ticker   Market ticker (e.g. KXMLBGAME-26APR04-NYY)
 * @param side     'yes' or 'no'
 * @param priceCents  Limit price in cents (1–99)
 * @param contracts   Number of $1 contracts to buy
 */
export async function placeOrder(
  ticker: string,
  side: 'yes' | 'no',
  priceCents: number,
  contracts: number,
): Promise<PlaceOrderResult> {
  const yesPriceCents = side === 'yes' ? priceCents : 100 - priceCents;
  const noPriceCents = 100 - yesPriceCents;

  if (PAPER_TRADING) {
    const fakeOrder: KalshiOrder = {
      order_id: `paper-${Date.now()}`,
      ticker,
      action: 'buy',
      side,
      type: 'limit',
      status: 'filled',
      yes_price: yesPriceCents,
      no_price: noPriceCents,
      count: contracts,
      filled_count: contracts,
      remaining_count: 0,
      created_time: new Date().toISOString(),
    };
    logger.info(
      { paper: true, ticker, side, priceCents, contracts },
      '[PAPER] Would place order',
    );
    return { order: fakeOrder, paper: true };
  }

  const body = {
    ticker,
    client_order_id: `mlb-oracle-${Date.now()}`,
    type: 'limit',
    action: 'buy',
    side,
    count: contracts,
    yes_price: yesPriceCents,
    no_price: noPriceCents,
  };

  const data = await kalshiPost<{ order: KalshiOrder }>('/portfolio/orders', body);
  logger.info({ ticker, side, priceCents, contracts, orderId: data.order.order_id }, 'Order placed');
  return { order: data.order };
}

/**
 * Sell (exit) a position.
 * Submits a sell order at market (aggressive limit = 1¢ for YES, 99¢ for NO).
 */
export async function sellPosition(
  ticker: string,
  side: 'yes' | 'no',
  contracts: number,
  minPriceCents = 1,
): Promise<PlaceOrderResult> {
  if (PAPER_TRADING) {
    const fakeOrder: KalshiOrder = {
      order_id: `paper-sell-${Date.now()}`,
      ticker,
      action: 'sell',
      side,
      type: 'limit',
      status: 'filled',
      yes_price: side === 'yes' ? minPriceCents : 100 - minPriceCents,
      no_price: side === 'yes' ? 100 - minPriceCents : minPriceCents,
      count: contracts,
      filled_count: contracts,
      remaining_count: 0,
      created_time: new Date().toISOString(),
    };
    logger.info({ paper: true, ticker, side, contracts, minPriceCents }, '[PAPER] Would sell position');
    return { order: fakeOrder, paper: true };
  }

  const yesSellPrice = side === 'yes' ? minPriceCents : 100 - minPriceCents;
  const body = {
    ticker,
    client_order_id: `mlb-oracle-sell-${Date.now()}`,
    type: 'limit',
    action: 'sell',
    side,
    count: contracts,
    yes_price: yesSellPrice,
    no_price: 100 - yesSellPrice,
  };

  const data = await kalshiPost<{ order: KalshiOrder }>('/portfolio/orders', body);
  logger.info({ ticker, side, contracts, orderId: data.order.order_id }, 'Sell order placed');
  return { order: data.order };
}

// ─── Utility ──────────────────────────────────────────────────────────────────

/** Returns a date in Kalshi's format: e.g. "26APR04"
 *  @param isoDate Optional YYYY-MM-DD string; defaults to today local time
 */
export function toKalshiDate(isoDate?: string): string {
  const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
  const d = isoDate ? new Date(isoDate + 'T12:00:00') : new Date(); // noon to avoid tz drift
  const yy = String(d.getFullYear()).slice(2);
  const mon = months[d.getMonth()];
  const dd = String(d.getDate()).padStart(2, '0');
  return `${yy}${mon}${dd}`;
}

/** @deprecated Use toKalshiDate() */
export function getTodayStr(): string {
  return toKalshiDate();
}

/** Convert cents to dollars string */
export function centsToDollars(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}
