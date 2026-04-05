// MLB Oracle v4.0 — Weather Client (Open-Meteo, free, no key)
import fetch from 'node-fetch';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type { WeatherData, ParkFactor } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 1)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

const OPEN_METEO_BASE = 'https://api.open-meteo.com/v1/forecast';

// WMO weather code → condition description
const WMO_CODES: Record<number, string> = {
  0: 'Clear', 1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
  45: 'Fog', 48: 'Fog', 51: 'Drizzle', 53: 'Drizzle', 55: 'Drizzle',
  61: 'Rain', 63: 'Rain', 65: 'Heavy Rain',
  71: 'Snow', 73: 'Snow', 75: 'Heavy Snow',
  80: 'Showers', 81: 'Showers', 82: 'Heavy Showers',
  95: 'Thunderstorm', 96: 'Thunderstorm', 99: 'Severe Thunderstorm',
};

interface OpenMeteoResponse {
  current: {
    temperature_2m: number;        // Celsius
    windspeed_10m: number;          // km/h
    winddirection_10m: number;      // degrees
    weathercode: number;
  };
}

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cacheKey(lat: number, lon: number): string {
  return `weather_${lat.toFixed(3)}_${lon.toFixed(3)}.json`;
}

function readCache(key: string): WeatherData | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as WeatherData;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: WeatherData): void {
  const path = resolve(CACHE_DIR, key);
  try {
    writeFileSync(path, JSON.stringify(data), 'utf-8');
  } catch (err) {
    logger.warn({ err }, 'Failed to write weather cache');
  }
}

// ─── Main weather fetch ───────────────────────────────────────────────────────

export async function fetchWeather(
  latitude: number,
  longitude: number
): Promise<WeatherData> {
  const key = cacheKey(latitude, longitude);
  const cached = readCache(key);
  if (cached) {
    logger.debug({ latitude, longitude }, 'Weather cache HIT');
    return cached;
  }

  const url = `${OPEN_METEO_BASE}?latitude=${latitude}&longitude=${longitude}` +
    `&current=temperature_2m,windspeed_10m,winddirection_10m,weathercode` +
    `&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto`;

  try {
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'MLBOracle/4.0 (educational)' },
      signal: AbortSignal.timeout(10000),
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const data = (await resp.json()) as OpenMeteoResponse;
    const current = data.current;

    const weather: WeatherData = {
      temperature: Math.round(current.temperature_2m),
      windSpeed: Math.round(current.windspeed_10m),
      windDirection: Math.round(current.winddirection_10m),
      condition: WMO_CODES[current.weathercode] ?? 'Clear',
    };

    writeCache(key, weather);
    logger.info({ latitude, longitude, weather }, 'Weather fetched');
    return weather;
  } catch (err) {
    logger.warn({ latitude, longitude, err }, 'Weather fetch failed; using defaults');
    return getDefaultWeather();
  }
}

// ─── Fetch weather for a park ─────────────────────────────────────────────────

export async function fetchParkWeather(park: ParkFactor): Promise<WeatherData> {
  // Dome stadiums: always controlled environment
  if (park.roof === 'dome') {
    return {
      temperature: 72,
      windSpeed: 0,
      windDirection: 0,
      condition: 'Dome',
    };
  }

  return fetchWeather(park.latitude, park.longitude);
}

// ─── Weather adjustment factor ────────────────────────────────────────────────
// Returns a multiplier for run scoring. Based on research:
// - Cold (< 50°F): reduces scoring ~8%
// - Hot (> 90°F): reduces scoring ~3% (fatigue/humidity)
// - Wind out: increases scoring
// - Wind in: decreases scoring
// Note: wind_out_cf and wind_in_cf are computed in parkFactors.ts

export function computeWeatherAdj(weather: WeatherData): number {
  let adj = 1.0;

  // Temperature adjustment (normalized around 72°F)
  const temp = weather.temperature;
  if (temp < 50) {
    adj *= 1.0 - (50 - temp) * 0.004; // -0.4% per degree below 50
  } else if (temp > 85) {
    adj *= 1.0 - (temp - 85) * 0.002; // -0.2% per degree above 85
  }

  // Rain adjustment
  const cond = weather.condition.toLowerCase();
  if (cond.includes('rain') || cond.includes('shower')) {
    adj *= 0.95;
  } else if (cond.includes('thunder')) {
    adj *= 0.90;
  }

  return Math.max(0.80, Math.min(1.20, adj));
}

function getDefaultWeather(): WeatherData {
  return {
    temperature: 72,
    windSpeed: 8,
    windDirection: 180,
    condition: 'Clear',
  };
}
