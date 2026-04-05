// MLB Oracle v4.0 — Park Factors, Wind Adjustment, Umpire Model
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { ParkFactor, ParkFactors, WindAdjustment, WeatherData } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PARK_FACTORS_PATH = resolve(__dirname, '../../data/park_factors.json');
const UMPIRES_PATH = resolve(__dirname, '../../data/umpires.json');

// ─── Park factor loading ──────────────────────────────────────────────────────

let _parkFactors: ParkFactors | null = null;

export function loadParkFactors(): ParkFactors {
  if (_parkFactors) return _parkFactors;
  const raw = readFileSync(PARK_FACTORS_PATH, 'utf-8');
  _parkFactors = JSON.parse(raw) as ParkFactors;
  return _parkFactors;
}

// ─── Umpire loading ───────────────────────────────────────────────────────────

interface UmpireRecord {
  name: string;
  run_factor: number;
  k_modifier: number;
}

interface UmpiresFile {
  umpires: UmpireRecord[];
}

let _umpires: UmpireRecord[] | null = null;

function loadUmpires(): UmpireRecord[] {
  if (_umpires) return _umpires;
  if (!existsSync(UMPIRES_PATH)) {
    _umpires = [];
    return _umpires;
  }
  const raw = readFileSync(UMPIRES_PATH, 'utf-8');
  const data = JSON.parse(raw) as UmpiresFile;
  _umpires = data.umpires ?? [];
  return _umpires;
}

// ─── Umpire run factor + K modifier ──────────────────────────────────────────

/**
 * Look up umpire by name (case-insensitive partial match).
 * Returns { run_factor, k_modifier } or defaults.
 */
export function getUmpireFactors(
  umpireId?: number,
  umpireName?: string
): { run_factor: number; k_modifier: number } {
  const DEFAULT = { run_factor: 1.0, k_modifier: 1.0 };

  if (!umpireName && !umpireId) return DEFAULT;

  const umpires = loadUmpires();
  if (umpires.length === 0) {
    // Legacy: use deterministic hash from ID
    if (umpireId) {
      const hash = (umpireId * 2654435761) >>> 0;
      const normalized = (hash % 1000) / 1000;
      return { run_factor: 0.95 + normalized * 0.10, k_modifier: 1.0 };
    }
    return DEFAULT;
  }

  if (umpireName) {
    const nameLower = umpireName.toLowerCase();
    const match = umpires.find(u => {
      const uLower = u.name.toLowerCase();
      return uLower === nameLower || uLower.includes(nameLower) || nameLower.includes(uLower);
    });
    if (match) return { run_factor: match.run_factor, k_modifier: match.k_modifier };
  }

  // Fallback to ID-based hash if no name match
  if (umpireId) {
    const hash = (umpireId * 2654435761) >>> 0;
    const normalized = (hash % 1000) / 1000;
    return { run_factor: 0.97 + normalized * 0.06, k_modifier: 0.98 + normalized * 0.04 };
  }

  return DEFAULT;
}

/** Legacy function kept for backward compatibility */
export function getUmpireRunFactor(umpireId?: number, umpireName?: string): number {
  return getUmpireFactors(umpireId, umpireName).run_factor;
}

// ─── Lookup by venue name ─────────────────────────────────────────────────────

export function getParkFactor(venueName: string): ParkFactor {
  const parks = loadParkFactors();

  // Exact match
  if (parks[venueName]) return parks[venueName];

  // Fuzzy match: find partial name overlap
  const normalizedVenue = venueName.toLowerCase().replace(/[^a-z0-9]/g, '');
  for (const [parkName, park] of Object.entries(parks)) {
    const normalizedPark = parkName.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (normalizedPark.includes(normalizedVenue) || normalizedVenue.includes(normalizedPark)) {
      return park;
    }
  }

  // Fallback: league average
  return getDefaultParkFactor();
}

// ─── Lookup by team abbreviation ──────────────────────────────────────────────

export function getParkFactorByTeam(teamAbbr: string): ParkFactor {
  const parks = loadParkFactors();

  for (const park of Object.values(parks)) {
    if (park.team === teamAbbr) return park;
  }

  return getDefaultParkFactor();
}

// ─── Wind adjustment computation ──────────────────────────────────────────────
// Wind direction: meteorological convention (direction wind is COMING FROM)
// CF heading: compass bearing TO center field
//
// Per spec: wind_angle_diff = (wind_dir - cf_heading + 360) % 360
//   wind_out_cf = 1 if wind_angle_diff between 315° and 45° (wind toward CF)
//   wind_in_cf  = 1 if wind_angle_diff between 135° and 225° (wind from CF)
//
// Wind factor applies park-specific multiplier from park_factors.json

export function computeWindAdjustment(
  weather: WeatherData,
  park: ParkFactor
): WindAdjustment {
  const windSpeed = weather.windSpeed;

  if (windSpeed < 2 || park.roof === 'dome') {
    return { wind_out_cf: 0, wind_in_cf: 0, net_wind_factor: 1.0 };
  }

  // Spec: wind_angle_diff = (wind_dir - cf_heading + 360) % 360
  const windAngleDiff = ((weather.windDirection - park.cf_heading) + 360) % 360;

  const isWindOutToCF = (windAngleDiff >= 315 || windAngleDiff <= 45);
  const isWindInFromCF = (windAngleDiff >= 135 && windAngleDiff <= 225);

  // Intensity: scale by wind speed (0 at 0 mph, 1.0 at 20 mph)
  const intensity = Math.min(1.0, windSpeed / 20);

  const wind_out_cf = isWindOutToCF ? intensity : 0;
  const wind_in_cf = isWindInFromCF ? intensity : 0;

  // Park-specific wind multipliers (from park_factors.json, with defaults)
  const parkData = park as ParkFactor & { wind_factor_out?: number; wind_factor_in?: number };
  const windFactorOut = parkData.wind_factor_out ?? 1.05;
  const windFactorIn = parkData.wind_factor_in ?? 0.95;

  let net_wind_factor = 1.0;
  if (isWindOutToCF) {
    // Interpolate between 1.0 and windFactorOut based on intensity
    net_wind_factor = 1.0 + (windFactorOut - 1.0) * intensity;
  } else if (isWindInFromCF) {
    // Interpolate between 1.0 and windFactorIn based on intensity
    net_wind_factor = 1.0 + (windFactorIn - 1.0) * intensity;
  } else {
    // Crosswind: mild effect based on component
    const crossComponent = Math.cos((windAngleDiff * Math.PI) / 180) * windSpeed;
    net_wind_factor = 1.0 + crossComponent * 0.002;
  }

  return {
    wind_out_cf: Math.min(1.0, wind_out_cf),
    wind_in_cf: Math.min(1.0, wind_in_cf),
    net_wind_factor: Math.max(0.85, Math.min(1.15, net_wind_factor)),
  };
}

// ─── Combined park + wind run factor ─────────────────────────────────────────

export function getTotalParkRunFactor(park: ParkFactor, weather: WeatherData): number {
  const wind = computeWindAdjustment(weather, park);
  return park.run_factor * wind.net_wind_factor;
}

// ─── Utility ──────────────────────────────────────────────────────────────────

function getDefaultParkFactor(): ParkFactor {
  return {
    name: 'Generic Stadium',
    team: 'MLB',
    run_factor: 1.00,
    cf_heading: 0,
    latitude: 39.5,
    longitude: -98.35,
    roof: 'open',
  };
}

export function getAllParkFactors(): ParkFactors {
  return loadParkFactors();
}
