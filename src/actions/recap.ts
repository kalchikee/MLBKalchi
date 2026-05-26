// GitHub Actions entry point — Nightly Recap
//
// Walks every predictions/<date>.json file that hasn't been graded yet,
// fetches MLB Stats API final scores for each date, matches each pick
// to its final, and appends a graded row to data/grading_history.json.
//
// Idempotent: re-running on the same date is a no-op for already-graded
// games. Skips games not yet completed (picked up next run).

import 'dotenv/config';
import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync, renameSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const PRED_DIR = join(ROOT, 'predictions');
const HISTORY_FILE = join(ROOT, 'data', 'grading_history.json');
const MLB_BASE = 'https://statsapi.mlb.com/api/v1';

// MLB Stats API uses ATH (not OAK) post-rebrand and AZ (not ARI) for the
// Diamondbacks. The Oracle's predictions still emit OAK / ARI. Normalize
// both sides to whatever the API returns so the equality check works.
const ABBR_ALIASES: Record<string, string> = {
  OAK: 'ATH',
  ARI: 'AZ',
};
function normalizeAbbr(s: string): string {
  return ABBR_ALIASES[s] ?? s;
}

interface GradedPick {
  date: string;
  gameId: string;
  gamePk: number;
  away: string;
  home: string;
  pickedTeam: string;
  modelProb: number;
  actualWinner: string;
  correct: boolean;
  homeScore: number;
  awayScore: number;
  gradedAt: string;
}

interface GradingHistory {
  graded: GradedPick[];
}

interface PredictionPick {
  gameId: string;
  home: string;
  away: string;
  pickedTeam: string;
  pickedSide: 'home' | 'away';
  modelProb: number;
  extra?: { gamePk?: number };
}

interface PredictionFile {
  sport: string;
  date: string;
  picks: PredictionPick[];
}

interface MlbScheduleGame {
  gamePk: number;
  status: { abstractGameState: string };
  teams: {
    home: { team: { id: number; abbreviation?: string; name: string }; score?: number };
    away: { team: { id: number; abbreviation?: string; name: string }; score?: number };
  };
}

function loadHistory(): GradingHistory {
  if (!existsSync(HISTORY_FILE)) return { graded: [] };
  try {
    return JSON.parse(readFileSync(HISTORY_FILE, 'utf8')) as GradingHistory;
  } catch (e) {
    // Match the kalshi-safety pattern: preserve corrupt file, start fresh in-memory
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const backup = HISTORY_FILE.replace(/\.json$/, `.corrupt-${ts}.json`);
    try {
      renameSync(HISTORY_FILE, backup);
      logger.warn({ err: e, backup }, 'corrupt grading_history.json — preserved + starting fresh');
    } catch {
      logger.warn({ err: e }, 'corrupt grading_history.json — could not back up; starting fresh');
    }
    return { graded: [] };
  }
}

function saveHistory(h: GradingHistory): void {
  mkdirSync(dirname(HISTORY_FILE), { recursive: true });
  const tmp = HISTORY_FILE + '.tmp';
  writeFileSync(tmp, JSON.stringify(h, null, 2));
  renameSync(tmp, HISTORY_FILE);
}

async function fetchFinalsForDate(iso: string): Promise<Map<number, { home: string; away: string; hs: number; as: number; winner: string }>> {
  const url = `${MLB_BASE}/schedule?sportId=1&date=${iso}&hydrate=team`;
  const out = new Map<number, { home: string; away: string; hs: number; as: number; winner: string }>();
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!r.ok) {
      logger.warn({ status: r.status, iso }, 'MLB API non-2xx');
      return out;
    }
    const data = await r.json() as { dates?: { games?: MlbScheduleGame[] }[] };
    for (const d of data.dates ?? []) {
      for (const g of d.games ?? []) {
        if (g.status?.abstractGameState !== 'Final') continue;
        const home = g.teams.home.team.abbreviation;
        const away = g.teams.away.team.abbreviation;
        const hs = g.teams.home.score;
        const as = g.teams.away.score;
        if (!home || !away || typeof hs !== 'number' || typeof as !== 'number') continue;
        if (hs === as) continue;  // tie — extremely rare in MLB, skip
        out.set(g.gamePk, { home, away, hs, as, winner: hs > as ? home : away });
      }
    }
  } catch (e) {
    logger.warn({ err: e, iso }, 'MLB API fetch failed');
  }
  return out;
}

async function gradeDate(iso: string, history: GradingHistory): Promise<number> {
  const predFile = join(PRED_DIR, `${iso}.json`);
  if (!existsSync(predFile)) return 0;

  let preds: PredictionFile;
  try {
    preds = JSON.parse(readFileSync(predFile, 'utf8')) as PredictionFile;
  } catch (e) {
    logger.warn({ err: e, predFile }, 'could not parse predictions file');
    return 0;
  }

  if (!preds.picks?.length) return 0;

  const alreadyGraded = new Set(history.graded.filter(g => g.date === iso).map(g => g.gameId));
  const finals = await fetchFinalsForDate(iso);
  if (finals.size === 0) return 0;

  let newly = 0;
  for (const pick of preds.picks) {
    if (alreadyGraded.has(pick.gameId)) continue;
    const gamePk = pick.extra?.gamePk;
    if (!gamePk) continue;
    const game = finals.get(gamePk);
    if (!game) {
      // Game not yet final, postponed, or PK mismatch — pick up next run
      continue;
    }
    const correct = normalizeAbbr(pick.pickedTeam) === game.winner;
    history.graded.push({
      date: iso,
      gameId: pick.gameId,
      gamePk,
      away: pick.away,
      home: pick.home,
      pickedTeam: pick.pickedTeam,
      modelProb: pick.modelProb,
      actualWinner: game.winner,
      correct,
      homeScore: game.hs,
      awayScore: game.as,
      gradedAt: new Date().toISOString(),
    });
    newly++;
  }
  return newly;
}

async function main(): Promise<void> {
  const onlyDate = process.argv[2];  // optional YYYY-MM-DD
  const history = loadHistory();

  const today = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });

  let total = 0;
  if (onlyDate) {
    total += await gradeDate(onlyDate, history);
  } else {
    if (!existsSync(PRED_DIR)) {
      logger.info({}, 'no predictions directory yet');
      return;
    }
    for (const f of readdirSync(PRED_DIR).filter(x => x.endsWith('.json')).sort()) {
      const iso = f.replace(/\.json$/, '');
      if (iso >= today) continue;  // not yet played
      total += await gradeDate(iso, history);
    }
  }

  if (total > 0) saveHistory(history);

  const correct = history.graded.filter(g => g.correct).length;
  const t = history.graded.length;
  const acc = t > 0 ? (correct / t) * 100 : 0;
  logger.info({ newly: total, season: `${correct}/${t} (${acc.toFixed(1)}%)` }, 'recap complete');
}

main().catch((err) => {
  logger.error({ err }, 'recap failed');
  process.exit(1);
});
