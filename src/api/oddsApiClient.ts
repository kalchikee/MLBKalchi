// MLB Oracle v4.0 — The Odds API Client
// Fetches live MLB moneylines from api.the-odds-api.com (free tier).
// Caches results for 30 minutes. Returns empty Map if API key not set.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes — odds move frequently

mkdirSync(CACHE_DIR, { recursive: true });

const ODDS_API_BASE = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds';

// ─── Result type ──────────────────────────────────────────────────────────────

export interface OddsApiLine {
  homeML: number;
  awayML: number;
}

// ─── Full team name → MLB abbreviation (all 30 teams) ────────────────────────
// The Odds API uses full names like "Detroit Tigers", "Los Angeles Dodgers", etc.

const TEAM_NAME_TO_ABBR: Record<string, string> = {
  // American League East
  'New York Yankees':      'NYY',
  'Boston Red Sox':        'BOS',
  'Tampa Bay Rays':        'TB',
  'Toronto Blue Jays':     'TOR',
  'Baltimore Orioles':     'BAL',
  // American League Central
  'Chicago White Sox':     'CWS',
  'Cleveland Guardians':   'CLE',
  'Detroit Tigers':        'DET',
  'Kansas City Royals':    'KC',
  'Minnesota Twins':       'MIN',
  // American League West
  'Houston Astros':        'HOU',
  'Los Angeles Angels':    'LAA',
  'Oakland Athletics':     'OAK',
  'Sacramento River Cats': 'OAK', // Athletics relocation alias
  'Seattle Mariners':      'SEA',
  'Texas Rangers':         'TEX',
  // National League East
  'Atlanta Braves':        'ATL',
  'Miami Marlins':         'MIA',
  'New York Mets':         'NYM',
  'Philadelphia Phillies': 'PHI',
  'Washington Nationals':  'WSH',
  // National League Central
  'Chicago Cubs':          'CHC',
  'Cincinnati Reds':       'CIN',
  'Milwaukee Brewers':     'MIL',
  'Pittsburgh Pirates':    'PIT',
  'St. Louis Cardinals':   'STL',
  // National League West
  'Arizona Diamondbacks':  'ARI',
  'Colorado Rockies':      'COL',
  'Los Angeles Dodgers':   'LAD',
  'San Diego Padres':      'SD',
  'San Francisco Giants':  'SF',
};

// ─── Raw API response types ───────────────────────────────────────────────────

interface OddsApiOutcome {
  name: string;
  price: number; // American moneyline
}

interface OddsApiMarket {
  key: string;
  outcomes: OddsApiOutcome[];
}

interface OddsApiBookmaker {
  key: string;
  title: string;
  markets: OddsApiMarket[];
}

interface OddsApiEvent {
  id: string;
  home_team: string;
  away_team: string;
  bookmakers: OddsApiBookmaker[];
}

// ─── Cache helpers ────────────────────────────────────────────────────────────

const CACHE_KEY = 'odds_api_mlb_moneylines.json';

interface CachedOdds {
  fetchedAt: number;
  data: Array<{ gameKey: string; homeML: number; awayML: number }>;
}

function readOddsCache(): CachedOdds | null {
  const path = resolve(CACHE_DIR, CACHE_KEY);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as CachedOdds;
  } catch {
    return null;
  }
}

function writeOddsCache(entries: CachedOdds['data']): void {
  const path = resolve(CACHE_DIR, CACHE_KEY);
  try {
    const payload: CachedOdds = { fetchedAt: Date.now(), data: entries };
    writeFileSync(path, JSON.stringify(payload), 'utf-8');
  } catch (err) {
    logger.warn({ err }, 'Failed to write Odds API cache');
  }
}

// ─── Team name lookup ─────────────────────────────────────────────────────────

function toAbbr(fullName: string): string | null {
  // Direct lookup first
  const direct = TEAM_NAME_TO_ABBR[fullName];
  if (direct) return direct;

  // Fuzzy: try matching the last word (city names vary)
  for (const [name, abbr] of Object.entries(TEAM_NAME_TO_ABBR)) {
    if (fullName.toLowerCase().includes(name.toLowerCase()) ||
        name.toLowerCase().includes(fullName.toLowerCase())) {
      return abbr;
    }
  }

  logger.warn({ teamName: fullName }, 'Odds API: unknown team name — no abbreviation found');
  return null;
}

// ─── Bookmaker preference order ───────────────────────────────────────────────
// Use DraftKings first, then FanDuel, then any available book.

const PREFERRED_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'pointsbet'];

function pickBestLine(bookmakers: OddsApiBookmaker[]): { homeML: number; awayML: number } | null {
  // Try preferred books in order
  for (const bookKey of PREFERRED_BOOKS) {
    const book = bookmakers.find(b => b.key === bookKey);
    if (!book) continue;
    const h2h = book.markets.find(m => m.key === 'h2h');
    if (!h2h || h2h.outcomes.length < 2) continue;
    const home = h2h.outcomes[0];
    const away = h2h.outcomes[1];
    if (home && away && home.price !== 0 && away.price !== 0) {
      return { homeML: home.price, awayML: away.price };
    }
  }

  // Fallback: use any bookmaker
  for (const book of bookmakers) {
    const h2h = book.markets.find(m => m.key === 'h2h');
    if (!h2h || h2h.outcomes.length < 2) continue;
    const home = h2h.outcomes[0];
    const away = h2h.outcomes[1];
    if (home && away && home.price !== 0 && away.price !== 0) {
      return { homeML: home.price, awayML: away.price };
    }
  }

  return null;
}

// ─── Primary fetch function ───────────────────────────────────────────────────

/**
 * Fetch MLB moneylines from The Odds API.
 *
 * Returns a Map keyed by "{awayAbbr}@{homeAbbr}" (e.g. "STL@DET").
 * Returns an empty Map if THE_ODDS_API_KEY is not set or the request fails.
 * Results are cached for 30 minutes.
 */
export async function fetchOddsApiLines(): Promise<Map<string, OddsApiLine>> {
  const apiKey = process.env.THE_ODDS_API_KEY;

  if (!apiKey) {
    logger.warn('THE_ODDS_API_KEY not set — skipping live odds fetch');
    return new Map();
  }

  // Check cache first
  const cached = readOddsCache();
  if (cached) {
    logger.debug({ entries: cached.data.length }, 'Odds API cache HIT');
    const map = new Map<string, OddsApiLine>();
    for (const entry of cached.data) {
      map.set(entry.gameKey, { homeML: entry.homeML, awayML: entry.awayML });
    }
    return map;
  }

  const url =
    `${ODDS_API_BASE}/?apiKey=${apiKey}&regions=us&markets=h2h&oddsFormat=american`;

  try {
    logger.info('Fetching MLB moneylines from The Odds API');
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'MLBOracle/4.0 (educational)' },
      signal: AbortSignal.timeout(15000),
    });

    if (!resp.ok) {
      const body = await resp.text();
      logger.error(
        { status: resp.status, body: body.slice(0, 200) },
        'Odds API returned error'
      );
      return new Map();
    }

    const events = (await resp.json()) as OddsApiEvent[];

    // Log remaining API quota if headers are present
    const remaining = resp.headers.get('x-requests-remaining');
    const used = resp.headers.get('x-requests-used');
    if (remaining !== null) {
      logger.info({ remaining, used }, 'Odds API quota');
    }

    const result = new Map<string, OddsApiLine>();
    const cacheEntries: CachedOdds['data'] = [];

    for (const event of events) {
      const homeAbbr = toAbbr(event.home_team);
      const awayAbbr = toAbbr(event.away_team);

      if (!homeAbbr || !awayAbbr) continue;

      const line = pickBestLine(event.bookmakers);
      if (!line) {
        logger.debug(
          { home: event.home_team, away: event.away_team },
          'Odds API: no valid h2h line for game'
        );
        continue;
      }

      // The Odds API outcomes[0] is home_team, outcomes[1] is away_team
      // but we need to verify by name match since outcome order may vary
      const gameKey = `${awayAbbr}@${homeAbbr}`;
      result.set(gameKey, line);
      cacheEntries.push({ gameKey, homeML: line.homeML, awayML: line.awayML });

      logger.debug(
        { gameKey, homeML: line.homeML, awayML: line.awayML },
        'Odds API line parsed'
      );
    }

    writeOddsCache(cacheEntries);
    logger.info({ games: result.size }, 'Odds API lines fetched and cached');
    return result;

  } catch (err) {
    logger.error({ err }, 'Failed to fetch from The Odds API');
    return new Map();
  }
}
