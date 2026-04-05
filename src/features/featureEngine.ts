// MLB Oracle v4.0 — Feature Engineering
// Computes all 30 features as home-vs-away differences
// Phase 4: Umpire model, wind direction, DRS, catcher framing, bullpen model, Statcast

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type {
  Game, FeatureVector, PitcherStats, TeamStats, WeatherData,
  ParkFactor, SPScore, WindAdjustment, DrsTeam
} from '../types.js';
import {
  fetchPitcherStats, fetchTeamStats, computeRollingStats, TEAM_ID_TO_ABBR,
  fetchPitcherRecentForm, fetchTeamILPlayers, type ILPlayer,
} from '../api/mlbClient.js';
import { fetchParkWeather } from '../api/weatherClient.js';
import {
  getParkFactor, getParkFactorByTeam, computeWindAdjustment, getUmpireFactors
} from './parkFactors.js';
import { getEloDiff, seedElos, ALL_TEAM_ABBRS } from './elo.js';
import {
  getPitcherStatcast, getCatcherFraming
} from '../api/statcastClient.js';
import { getGameBullpenScores } from './bullpenModel.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DRS_PATH = resolve(__dirname, '../../data/drs.json');

const LEAGUE_AVG_WOBA = 0.320;
const LEAGUE_AVG_WRC_PLUS = 100;
const LEAGUE_AVG_FIP = 4.20;
const LEAGUE_AVG_RPG = 4.5;

// ─── DRS loading ──────────────────────────────────────────────────────────────

interface DrsFile {
  teams: Record<string, DrsTeam>;
}

let _drsData: DrsFile | null = null;

function loadDrs(): DrsFile {
  if (_drsData) return _drsData;
  if (!existsSync(DRS_PATH)) {
    _drsData = { teams: {} };
    return _drsData;
  }
  const raw = readFileSync(DRS_PATH, 'utf-8');
  _drsData = JSON.parse(raw) as DrsFile;
  return _drsData;
}

function getTeamDrs(teamAbbr: string): number {
  try {
    const drs = loadDrs();
    return drs.teams[teamAbbr]?.drs ?? 0;
  } catch {
    return 0;
  }
}

// ─── SP Score formula ─────────────────────────────────────────────────────────
// SP_score = 0.35 × K-BB% + 0.25 × xFIP_inverse + 0.20 × SIERA_inverse + 0.20 × rolling_game_score

function computeSPScore(pitcher: PitcherStats): SPScore {
  const kBBComponent = pitcher.kBBPct;
  const xfipInverse = 1 / Math.max(0.5, pitcher.xfip);
  const sieraInverse = 1 / Math.max(0.5, pitcher.siera);
  const gameScoreComponent = pitcher.rollingGameScore / 100;

  const kBBNorm = Math.max(0, Math.min(1, (kBBComponent - 0.02) / 0.23));
  const xfipNorm = Math.max(0, Math.min(1, (xfipInverse - 0.17) / 0.33));
  const sieraNorm = Math.max(0, Math.min(1, (sieraInverse - 0.17) / 0.33));

  const score = (
    0.35 * kBBNorm +
    0.25 * xfipNorm +
    0.20 * sieraNorm +
    0.20 * gameScoreComponent
  );

  let tier: 'elite' | 'average' | 'bad';
  let suppression_multiplier: number;

  if (score >= 0.65) {
    tier = 'elite';
    suppression_multiplier = 0.82;
  } else if (score >= 0.35) {
    tier = 'average';
    suppression_multiplier = 1.00;
  } else {
    tier = 'bad';
    suppression_multiplier = 1.18;
  }

  return { score, suppression_multiplier, tier };
}

// ─── Pythagorean win percentage ───────────────────────────────────────────────

function pythagoreanWinPct(runsScored: number, runsAllowed: number): number {
  if (runsScored <= 0 && runsAllowed <= 0) return 0.500;
  const rs = Math.max(0.1, runsScored);
  const ra = Math.max(0.1, runsAllowed);
  const exp = Math.pow(rs + ra, 0.285);
  return Math.pow(rs, exp) / (Math.pow(rs, exp) + Math.pow(ra, exp));
}

// ─── Log5 head-to-head probability ───────────────────────────────────────────

function log5Probability(teamAWinPct: number, teamBWinPct: number): number {
  const a = Math.max(0.01, Math.min(0.99, teamAWinPct));
  const b = Math.max(0.01, Math.min(0.99, teamBWinPct));
  const numerator = a - a * b;
  const denominator = a + b - 2 * a * b;
  if (Math.abs(denominator) < 0.001) return 0.500;
  return Math.max(0.01, Math.min(0.99, numerator / denominator));
}

// ─── SCI Formula ─────────────────────────────────────────────────────────────

function computeSCI(meanRuns: number, stdDevRuns?: number): number {
  const mu = Math.max(0.1, meanRuns);
  const sigma = stdDevRuns ?? Math.sqrt(mu);
  const cv = sigma / mu;
  const sci = (1 / (1 + cv)) * Math.sqrt(mu / LEAGUE_AVG_RPG);
  return Math.max(0, Math.min(2, sci));
}

// ─── Lineup wOBA estimate ─────────────────────────────────────────────────────

function estimateLineupWoba(teamStats: TeamStats): number {
  return teamStats.woba ?? LEAGUE_AVG_WOBA;
}

function estimateLineupWrcPlus(teamStats: TeamStats): number {
  return teamStats.wrcPlus ?? LEAGUE_AVG_WRC_PLUS;
}

// ─── Rest days estimation ─────────────────────────────────────────────────────

function estimateRestDays(_teamAbbr: string): number {
  return 1;
}

function estimateTravelTzShift(homeAbbr: string, awayAbbr: string): number {
  const tzMap: Record<string, number> = {
    LAA: -7, ARI: -7, SD: -7, SEA: -7, SF: -7, LAD: -7, OAK: -7,
    COL: -6, TEX: -5, HOU: -5, MIN: -5, KC: -5, MIL: -5, STL: -5,
    CHC: -5, CWS: -5, DET: -4, ATL: -4, MIA: -4, TB: -4, CIN: -4,
    CLE: -4, PIT: -4, PHI: -4, NYM: -4, NYY: -4, BOS: -4, BAL: -4,
    WSH: -4, TOR: -4,
  };
  const homeTz = tzMap[homeAbbr] ?? -5;
  const awayTz = tzMap[awayAbbr] ?? -5;
  return homeTz - awayTz;
}

// ─── Main feature computation ─────────────────────────────────────────────────

export async function computeFeatures(game: Game, gameDate: string): Promise<FeatureVector> {
  seedElos(ALL_TEAM_ABBRS);

  const homeAbbr = game.homeTeam.abbreviation ?? TEAM_ID_TO_ABBR[game.homeTeam.id] ?? 'UNK';
  const awayAbbr = game.awayTeam.abbreviation ?? TEAM_ID_TO_ABBR[game.awayTeam.id] ?? 'UNK';

  logger.debug({ homeAbbr, awayAbbr, gamePk: game.gamePk }, 'Computing features');

  const year = new Date(gameDate).getFullYear();

  // Fetch pitcher stats, team stats, IL rosters, and recent form in parallel
  const [homeSPStats, awaySPStats, homeTeamStats, awayTeamStats, homeIL, awayIL] = await Promise.all([
    game.homeTeam.probablePitcher?.id
      ? fetchPitcherStats(game.homeTeam.probablePitcher.id)
      : Promise.resolve(null),
    game.awayTeam.probablePitcher?.id
      ? fetchPitcherStats(game.awayTeam.probablePitcher.id)
      : Promise.resolve(null),
    fetchTeamStats(game.homeTeam.id),
    fetchTeamStats(game.awayTeam.id),
    fetchTeamILPlayers(game.homeTeam.id).catch(() => [] as ILPlayer[]),
    fetchTeamILPlayers(game.awayTeam.id).catch(() => [] as ILPlayer[]),
  ]);

  // ── IL/Injury adjustments ─────────────────────────────────────────────────
  // Check if either team's probable SP is on the IL
  const homeSPOnIL = homeIL.some(p =>
    p.position === 'P' && game.homeTeam.probablePitcher?.id === p.playerId
  );
  const awaySPOnIL = awayIL.some(p =>
    p.position === 'P' && game.awayTeam.probablePitcher?.id === p.playerId
  );

  if (homeSPOnIL) {
    logger.warn({ pitcher: game.homeTeam.probablePitcher?.fullName, team: homeAbbr }, '[IL] Home SP on IL — using league avg');
  }
  if (awaySPOnIL) {
    logger.warn({ pitcher: game.awayTeam.probablePitcher?.fullName, team: awayAbbr }, '[IL] Away SP on IL — using league avg');
  }

  // Count position players on IL (rough offensive impact estimate)
  const positionCodes = new Set(['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF', 'DH', 'UT']);
  const homeILBatters = homeIL.filter(p => positionCodes.has(p.position) && p.ilType !== '60-Day').length;
  const awayILBatters = awayIL.filter(p => positionCodes.has(p.position) && p.ilType !== '60-Day').length;

  // Each IL batter costs roughly 0.005 wOBA (avg batter ≈ 0.320; replacement ≈ 0.280)
  const IL_WOBA_COST_PER_PLAYER = 0.005;
  const homeILWobaAdj = homeILBatters * IL_WOBA_COST_PER_PLAYER;
  const awayILWobaAdj = awayILBatters * IL_WOBA_COST_PER_PLAYER;

  if (homeILBatters > 0 || awayILBatters > 0) {
    logger.info({ homeILBatters, awayILBatters }, '[IL] Lineup adjustments applied');
  }

  // Rolling stats + actual pitcher recent form (in parallel)
  const [homeRolling, awayRolling, homeSPRecentForm, awaySPRecentForm] = await Promise.all([
    computeRollingStats(game.homeTeam.id, gameDate),
    computeRollingStats(game.awayTeam.id, gameDate),
    game.homeTeam.probablePitcher?.id && !homeSPOnIL
      ? fetchPitcherRecentForm(game.homeTeam.probablePitcher.id, year).catch(() => null)
      : Promise.resolve(null),
    game.awayTeam.probablePitcher?.id && !awaySPOnIL
      ? fetchPitcherRecentForm(game.awayTeam.probablePitcher.id, year).catch(() => null)
      : Promise.resolve(null),
  ]);

  // Upgrade rollingGameScore with actual last-5-starts if available
  if (homeSPStats && homeSPRecentForm !== null) {
    homeSPStats.rollingGameScore = homeSPRecentForm;
    logger.debug({ pitcher: game.homeTeam.probablePitcher?.fullName, recentForm: homeSPRecentForm.toFixed(1) }, '[SP] Using actual recent game scores');
  }
  if (awaySPStats && awaySPRecentForm !== null) {
    awaySPStats.rollingGameScore = awaySPRecentForm;
    logger.debug({ pitcher: game.awayTeam.probablePitcher?.fullName, recentForm: awaySPRecentForm.toFixed(1) }, '[SP] Using actual recent game scores');
  }

  // If SP is on IL, reset to league average so model doesn't overrate a missing pitcher
  const defaultSPFallback = getDefaultSPStats();
  const effectiveHomeSP = homeSPOnIL ? null : homeSPStats;
  const effectiveAwaySP = awaySPOnIL ? null : awaySPStats;

  // Park factor
  const park = getParkFactor(game.venue.name) ?? getParkFactorByTeam(homeAbbr);

  // Weather
  let weather = game.weather;
  if (!weather || park.roof === 'dome') {
    try {
      weather = await fetchParkWeather(park);
    } catch {
      weather = { temperature: 72, windSpeed: 5, windDirection: 180, condition: 'Clear' };
    }
  }

  const wind = computeWindAdjustment(weather, park);

  // ── Umpire factors (Phase 4) ──────────────────────────────────────────────
  const umpireFactors = getUmpireFactors(game.umpireId, game.umpireName);
  const umpireRunFactor = umpireFactors.run_factor;

  // ── Statcast pitcher data (Phase 4) ───────────────────────────────────────
  let homePitcherStatcast = null;
  let awayPitcherStatcast = null;
  if (game.homeTeam.probablePitcher?.id) {
    homePitcherStatcast = await getPitcherStatcast(game.homeTeam.probablePitcher.id, year)
      .catch(() => null);
  }
  if (game.awayTeam.probablePitcher?.id) {
    awayPitcherStatcast = await getPitcherStatcast(game.awayTeam.probablePitcher.id, year)
      .catch(() => null);
  }

  // Enhance pitcher stats with Statcast data when available
  if (homeSPStats && homePitcherStatcast) {
    homeSPStats.xfip = homePitcherStatcast.xfip !== 4.20
      ? homePitcherStatcast.xfip
      : homeSPStats.xfip;
    homeSPStats.cswRate = homePitcherStatcast.csw_rate > 0
      ? homePitcherStatcast.csw_rate
      : homeSPStats.cswRate;
  }
  if (awaySPStats && awayPitcherStatcast) {
    awaySPStats.xfip = awayPitcherStatcast.xfip !== 4.20
      ? awayPitcherStatcast.xfip
      : awaySPStats.xfip;
    awaySPStats.cswRate = awayPitcherStatcast.csw_rate > 0
      ? awayPitcherStatcast.csw_rate
      : awaySPStats.cswRate;
  }

  // ── Catcher framing (Phase 4) ─────────────────────────────────────────────
  // Today's catcher IDs — use first non-pitcher from battingOrder if available
  // For now, use team-level framing estimates (individual catcher lookup requires lineup)
  let homeCatcherFramingAdj = 0;
  let awayCatcherFramingAdj = 0;

  // If we have lineup data, try to find the catcher (position 2 in batting order = catcher slot)
  const homeBattingOrder = game.homeTeam.battingOrder ?? [];
  const awayBattingOrder = game.awayTeam.battingOrder ?? [];

  if (homeBattingOrder.length > 0) {
    // Catcher is typically in the lower third of the order; try all lineup players
    for (const pid of homeBattingOrder.slice(0, 9)) {
      const framing = await getCatcherFraming(pid, year).catch(() => null);
      if (framing && Math.abs(framing.runs_extra_strikes) > 0) {
        // runs_extra_strikes per season / 162 games ≈ per game contribution
        homeCatcherFramingAdj = framing.runs_extra_strikes / 162;
        break;
      }
    }
  }

  if (awayBattingOrder.length > 0) {
    for (const pid of awayBattingOrder.slice(0, 9)) {
      const framing = await getCatcherFraming(pid, year).catch(() => null);
      if (framing && Math.abs(framing.runs_extra_strikes) > 0) {
        awayCatcherFramingAdj = framing.runs_extra_strikes / 162;
        break;
      }
    }
  }

  // ── Bullpen scores (Phase 4) ──────────────────────────────────────────────
  let homeBullpenScore = 1.0;
  let awayBullpenScore = 1.0;
  try {
    const bullpenScores = await getGameBullpenScores(
      game.homeTeam.id, homeAbbr,
      game.awayTeam.id, awayAbbr,
      gameDate
    );
    homeBullpenScore = bullpenScores.home.bullpenStrengthScore;
    awayBullpenScore = bullpenScores.away.bullpenStrengthScore;
    logger.debug({
      homeAbbr, awayAbbr,
      homeBullpen: homeBullpenScore.toFixed(3),
      awayBullpen: awayBullpenScore.toFixed(3),
    }, 'Bullpen scores computed');
  } catch {
    // graceful degradation
  }

  // ── DRS (Phase 4) ─────────────────────────────────────────────────────────
  const homeDrs = getTeamDrs(homeAbbr);
  const awayDrs = getTeamDrs(awayAbbr);
  // run_prevention_adj = team_drs / 162 (runs per game)
  const homeDrsAdj = homeDrs / 162;
  const awayDrsAdj = awayDrs / 162;

  // SP scores (computed for suppression multiplier; component stats used directly in features)
  const _homeSP = computeSPScore(effectiveHomeSP ?? defaultSPFallback);
  const _awaySP = computeSPScore(effectiveAwaySP ?? defaultSPFallback);
  void _homeSP; void _awaySP; // suppress unused variable warning — SP scores reserved for Phase 5

  // Team run rates for pythagorean
  const homeRPG = homeTeamStats?.runsPerGame ?? LEAGUE_AVG_RPG;
  const awayRPG = awayTeamStats?.runsPerGame ?? LEAGUE_AVG_RPG;
  const homeRA = homeTeamStats?.era ? homeTeamStats.era : LEAGUE_AVG_RPG;
  const awayRA = awayTeamStats?.era ? awayTeamStats.era : LEAGUE_AVG_RPG;

  const homePythPct = pythagoreanWinPct(homeRPG, homeRA);
  const awayPythPct = pythagoreanWinPct(awayRPG, awayRA);
  const log5 = log5Probability(homePythPct, awayPythPct);

  // wOBA and wRC+ — adjusted for IL position players
  const homeWoba = Math.max(0.260,
    estimateLineupWoba(homeTeamStats ?? getDefaultTeamStats(homeAbbr)) - homeILWobaAdj
  );
  const awayWoba = Math.max(0.260,
    estimateLineupWoba(awayTeamStats ?? getDefaultTeamStats(awayAbbr)) - awayILWobaAdj
  );
  const homeWrcPlus = estimateLineupWrcPlus(homeTeamStats ?? getDefaultTeamStats(homeAbbr));
  const awayWrcPlus = estimateLineupWrcPlus(awayTeamStats ?? getDefaultTeamStats(awayAbbr));

  // Bullpen strength diff: use model scores (Phase 4) vs FIP estimate fallback
  const bullpenStrengthDiff = homeBullpenScore - awayBullpenScore;

  // SCI
  const homeSCI = computeSCI(homeRolling.rpg);
  const awaySCI = computeSCI(awayRolling.rpg);

  // Rest & travel
  const homeRest = estimateRestDays(homeAbbr);
  const awayRest = estimateRestDays(awayAbbr);
  const tzShift = estimateTravelTzShift(homeAbbr, awayAbbr);

  // Statcast (team level)
  const homeXba = homeTeamStats?.xba ?? 0.248;
  const awayXba = awayTeamStats?.xba ?? 0.248;
  const homeBarrel = homeTeamStats?.barrelRate ?? 0.07;
  const awayBarrel = awayTeamStats?.barrelRate ?? 0.07;
  const homeHardHit = homeTeamStats?.hardHitRate ?? 0.37;
  const awayHardHit = awayTeamStats?.hardHitRate ?? 0.37;
  const homeEV = homeTeamStats?.exitVelocity ?? 88.5;
  const awayEV = awayTeamStats?.exitVelocity ?? 88.5;
  const homeGB = homeTeamStats?.gbRate ?? 0.43;
  const awayGB = awayTeamStats?.gbRate ?? 0.43;

  const features: FeatureVector = {
    // Pitcher differentials (uses effectiveSP which is null if pitcher is on IL → falls back to league avg)
    elo_diff: getEloDiff(homeAbbr, awayAbbr),
    sp_xfip_diff: (effectiveHomeSP?.xfip ?? 4.20) - (effectiveAwaySP?.xfip ?? 4.20),
    sp_kbb_diff: (effectiveHomeSP?.kBBPct ?? 0.14) - (effectiveAwaySP?.kBBPct ?? 0.14),
    sp_siera_diff: (effectiveHomeSP?.siera ?? 4.20) - (effectiveAwaySP?.siera ?? 4.20),
    sp_csw_diff: (effectiveHomeSP?.cswRate ?? 0.28) - (effectiveAwaySP?.cswRate ?? 0.28),
    sp_rolling_gs_diff: (effectiveHomeSP?.rollingGameScore ?? 50) - (effectiveAwaySP?.rollingGameScore ?? 50),

    // Bullpen & lineup
    bullpen_strength_diff: bullpenStrengthDiff,
    lineup_woba_diff: homeWoba - awayWoba,
    lineup_wrc_plus_diff: homeWrcPlus - awayWrcPlus,

    // Rolling
    team_10d_woba_diff: homeRolling.wobaEst - awayRolling.wobaEst,
    team_10d_fip_diff: homeRolling.fipEst - awayRolling.fipEst,

    // Win probability
    pythagorean_diff: homePythPct - awayPythPct,
    log5_prob: log5,

    // Defense (Phase 4: DRS fully implemented)
    drs_diff: homeDrsAdj - awayDrsAdj,
    // Catcher framing (Phase 4)
    catcher_framing_diff: homeCatcherFramingAdj - awayCatcherFramingAdj,

    // Park & weather
    park_factor: park.run_factor,
    wind_out_cf: wind.wind_out_cf,
    wind_in_cf: wind.wind_in_cf,
    temperature: weather.temperature,
    // Phase 4: full umpire run factor from umpires.json
    umpire_run_factor: umpireRunFactor,

    // Situational
    rest_days_diff: homeRest - awayRest,
    travel_tz_shift: tzShift,
    day_after_night: 0,
    is_home: 1,

    // Statcast
    statcast_xba_diff: homeXba - awayXba,
    statcast_barrel_diff: homeBarrel - awayBarrel,
    statcast_hardhit_diff: homeHardHit - awayHardHit,
    statcast_ev_diff: homeEV - awayEV,

    // Advanced
    gb_rate_diff: homeGB - awayGB,
    sci_adjusted_diff: homeSCI - awaySCI,
  };

  logger.debug({ gamePk: game.gamePk, features }, 'Features computed');
  return features;
}

// ─── Lambda adjustments for Monte Carlo ──────────────────────────────────────

/**
 * Apply Phase 4 adjustments to lambda values post-feature computation.
 * Called from pipeline to enrich the Monte Carlo inputs.
 */
export function applyAdvancedLambdaAdjustments(
  homeLambda: number,
  awayLambda: number,
  features: FeatureVector,
  umpireKModifier: number = 1.0,
  homeBullpenScore: number = 1.0,
  awayBullpenScore: number = 1.0
): { homeLambda: number; awayLambda: number } {
  // Umpire run factor: scales both lambdas
  const adjHome = homeLambda * features.umpire_run_factor;
  const adjAway = awayLambda * features.umpire_run_factor;

  // DRS: reduce runs allowed by opponents
  // homeDrsAdj reduces awayLambda (home team prevents more runs)
  const homeTeamDrsRunsPerGame = features.drs_diff > 0 ? features.drs_diff : 0;
  const awayTeamDrsRunsPerGame = features.drs_diff < 0 ? -features.drs_diff : 0;

  const finalHome = Math.max(2.0, adjHome - awayTeamDrsRunsPerGame);
  const finalAway = Math.max(2.0, adjAway - homeTeamDrsRunsPerGame);

  return { homeLambda: finalHome, awayLambda: finalAway };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getDefaultSPStats(): PitcherStats {
  return {
    playerId: 0,
    playerName: 'TBD',
    teamId: 0,
    era: 4.50,
    fip: 4.20,
    xfip: 4.20,
    siera: 4.20,
    whip: 1.30,
    kPct: 0.22,
    bbPct: 0.08,
    kBBPct: 0.14,
    cswRate: 0.28,
    rollingGameScore: 50,
    inningsPitched: 0,
    gamesStarted: 0,
    handedness: 'R',
  };
}

function getDefaultTeamStats(abbr: string): TeamStats {
  return {
    teamId: 0,
    teamAbbr: abbr,
    teamName: '',
    runsPerGame: LEAGUE_AVG_RPG,
    woba: LEAGUE_AVG_WOBA,
    wrcPlus: LEAGUE_AVG_WRC_PLUS,
    obp: 0.320,
    slg: 0.410,
    avg: 0.250,
    era: 4.50,
    fip: LEAGUE_AVG_FIP,
    whip: 1.30,
    kPct: 0.22,
    bbPct: 0.08,
  };
}

export { computeSPScore, pythagoreanWinPct, log5Probability, computeSCI };
