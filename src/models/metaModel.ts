// MLB Oracle v4.0 — ML Meta-Model (Phase 3)
// Loads Logistic Regression + isotonic calibration artifacts from JSON,
// applies them to FeatureVectors at prediction time, and falls back
// to Monte Carlo win probability if the model files are not present.

import { existsSync, readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../logger.js';
import type { FeatureVector } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODEL_DIR = resolve(__dirname, '../../data/model');

// ─── JSON artifact shapes ─────────────────────────────────────────────────────

interface CoefficientsJson {
  _intercept: number;
  [featureName: string]: number;
}

interface ScalerJson {
  feature_names: string[];
  mean: number[];
  scale: number[];
}

interface CalibrationJson {
  method: 'isotonic';
  x_thresholds: number[];
  y_thresholds: number[];
  n_thresholds: number;
}

interface ModelMetadataJson {
  version: string;
  model_type: string;
  feature_names: string[];
  train_dates: string;
  avg_brier: number;
  avg_accuracy: number;
  trained_at: string;
}

// ─── Internal model state ─────────────────────────────────────────────────────

interface LoadedModel {
  // Feature ordering from the scaler (defines the index for each feature)
  featureNames: string[];
  // Coefficients in the same order as featureNames
  coefficients: Float64Array;
  intercept: number;
  // StandardScaler params (parallel arrays, indexed by featureNames)
  scalerMean: Float64Array;
  scalerScale: Float64Array;
  // Isotonic calibration thresholds
  calX: Float64Array;
  calY: Float64Array;
  // Metadata
  version: string;
  trainDates: string;
  avgBrier: number;
  loadedAt: string;
}

// Singleton — loaded once on first call to predict() or loadModel()
let _model: LoadedModel | null = null;
let _loadAttempted = false;

// ─── Feature name list (must match FEATURE_COLUMNS in Python and types.ts) ───

const FEATURE_NAMES_ORDERED = [
  'elo_diff',
  'sp_xfip_diff',
  'sp_kbb_diff',
  'sp_siera_diff',
  'sp_csw_diff',
  'sp_rolling_gs_diff',
  'bullpen_strength_diff',
  'lineup_woba_diff',
  'lineup_wrc_plus_diff',
  'team_10d_woba_diff',
  'team_10d_fip_diff',
  'pythagorean_diff',
  'log5_prob',
  'drs_diff',
  'catcher_framing_diff',
  'park_factor',
  'wind_out_cf',
  'wind_in_cf',
  'temperature',
  'umpire_run_factor',
  'rest_days_diff',
  'travel_tz_shift',
  'day_after_night',
  'is_home',
  'statcast_xba_diff',
  'statcast_barrel_diff',
  'statcast_hardhit_diff',
  'statcast_ev_diff',
  'gb_rate_diff',
  'sci_adjusted_diff',
] as const;

// ─── Model loading ────────────────────────────────────────────────────────────

/**
 * Attempt to load all model artifacts from data/model/.
 * Silently returns false if any file is missing (fall back to MC).
 * On parse errors, logs a warning and returns false.
 */
export function loadModel(): boolean {
  if (_model) return true;    // already loaded
  if (_loadAttempted) return false;  // already tried and failed

  _loadAttempted = true;

  const coefPath  = resolve(MODEL_DIR, 'coefficients.json');
  const scalerPath = resolve(MODEL_DIR, 'scaler.json');
  const calPath   = resolve(MODEL_DIR, 'calibration.json');
  const metaPath  = resolve(MODEL_DIR, 'model_metadata.json');

  // Check all required files exist
  const missing = [coefPath, scalerPath, calPath].filter(p => !existsSync(p));
  if (missing.length > 0) {
    logger.debug(
      { missing: missing.map(p => p.split(/[\\/]/).pop()) },
      'ML model files not found — using Monte Carlo fallback'
    );
    return false;
  }

  try {
    // ── Load coefficients ──────────────────────────────────────────────────
    const coefJson: CoefficientsJson = JSON.parse(readFileSync(coefPath, 'utf-8'));
    const intercept = coefJson['_intercept'] ?? 0;

    // ── Load scaler ────────────────────────────────────────────────────────
    const scalerJson: ScalerJson = JSON.parse(readFileSync(scalerPath, 'utf-8'));
    const scalerFeatures = scalerJson.feature_names;

    if (scalerJson.mean.length !== scalerFeatures.length ||
        scalerJson.scale.length !== scalerFeatures.length) {
      throw new Error('Scaler JSON arrays have mismatched lengths');
    }

    // Build index: featureName → scaler array position
    const scalerIndex = new Map<string, number>(
      scalerFeatures.map((name, i) => [name, i])
    );

    // ── Build ordered arrays using the canonical feature name list ─────────
    // We use FEATURE_NAMES_ORDERED as the canonical order; the scaler may
    // have been fitted with a different ordering so we look up by name.
    const featureNames: string[] = [];
    const coefficients: number[] = [];
    const scalerMean: number[] = [];
    const scalerScale: number[] = [];

    for (const name of FEATURE_NAMES_ORDERED) {
      const scalerIdx = scalerIndex.get(name);
      const coef = coefJson[name];

      // If a feature is in our canonical list but not in the saved model,
      // use coefficient=0 and scaler mean=0, scale=1 (pass-through with no effect)
      featureNames.push(name);
      coefficients.push(coef ?? 0);
      scalerMean.push(scalerIdx !== undefined ? scalerJson.mean[scalerIdx] : 0);
      scalerScale.push(scalerIdx !== undefined ? Math.max(scalerJson.scale[scalerIdx], 1e-10) : 1);
    }

    // ── Load calibration ───────────────────────────────────────────────────
    const calJson: CalibrationJson = JSON.parse(readFileSync(calPath, 'utf-8'));
    if (!calJson.x_thresholds || !calJson.y_thresholds ||
        calJson.x_thresholds.length !== calJson.y_thresholds.length) {
      throw new Error('Calibration JSON is malformed');
    }

    // ── Load metadata (optional) ───────────────────────────────────────────
    let metaVersion = '4.0.0';
    let metaTrainDates = 'unknown';
    let metaAvgBrier = 0;
    if (existsSync(metaPath)) {
      try {
        const metaJson: ModelMetadataJson = JSON.parse(readFileSync(metaPath, 'utf-8'));
        metaVersion = metaJson.version ?? metaVersion;
        metaTrainDates = metaJson.train_dates ?? metaTrainDates;
        metaAvgBrier = metaJson.avg_brier ?? metaAvgBrier;
      } catch {
        // Metadata is optional — don't fail if it can't be parsed
      }
    }

    // ── Assemble and store model ───────────────────────────────────────────
    _model = {
      featureNames,
      coefficients: new Float64Array(coefficients),
      intercept,
      scalerMean: new Float64Array(scalerMean),
      scalerScale: new Float64Array(scalerScale),
      calX: new Float64Array(calJson.x_thresholds),
      calY: new Float64Array(calJson.y_thresholds),
      version: metaVersion,
      trainDates: metaTrainDates,
      avgBrier: metaAvgBrier,
      loadedAt: new Date().toISOString(),
    };

    logger.info(
      {
        version: _model.version,
        features: featureNames.length,
        calThresholds: calJson.x_thresholds.length,
        trainDates: metaTrainDates,
        avgBrier: metaAvgBrier.toFixed(4),
      },
      'ML model loaded successfully'
    );

    return true;

  } catch (err) {
    logger.warn({ err }, 'Failed to load ML model — using Monte Carlo fallback');
    _model = null;
    return false;
  }
}

/**
 * Check whether the ML model has been successfully loaded.
 */
export function isModelLoaded(): boolean {
  return _model !== null;
}

/**
 * Get model metadata for display/logging purposes.
 * Returns null if model is not loaded.
 */
export function getModelInfo(): {
  version: string;
  trainDates: string;
  avgBrier: number;
  features: number;
  loadedAt: string;
} | null {
  if (!_model) return null;
  return {
    version: _model.version,
    trainDates: _model.trainDates,
    avgBrier: _model.avgBrier,
    features: _model.featureNames.length,
    loadedAt: _model.loadedAt,
  };
}

// ─── Prediction ───────────────────────────────────────────────────────────────

/**
 * Generate a calibrated win probability for the home team.
 *
 * Pipeline:
 *   1. Extract feature values in canonical order from FeatureVector
 *   2. Standard-scale: z = (x − mean) / scale
 *   3. Compute logit score: β0 + Σ(βi × zi)
 *   4. Apply sigmoid: p = 1 / (1 + exp(−score))
 *   5. Apply isotonic calibration via linear interpolation on x_thresholds/y_thresholds
 *   6. Return calibrated probability
 *
 * If the model is not loaded, falls back to mcWinProb directly.
 *
 * @param features    FeatureVector from featureEngine.ts
 * @param mcWinProb   Monte Carlo win probability (0–1) as fallback
 * @returns Calibrated probability (0–1) for home team win
 */
export function predict(features: FeatureVector, mcWinProb: number): number {
  // Auto-load on first call
  if (!_model) {
    loadModel();
  }

  // If model still not available, fall back to MC
  if (!_model) {
    return mcWinProb;
  }

  const m = _model;
  const n = m.featureNames.length;

  // ── Step 1 & 2: Extract and scale features ────────────────────────────
  let logit = m.intercept;

  for (let i = 0; i < n; i++) {
    const name = m.featureNames[i] as keyof FeatureVector;
    const raw = (features[name] as number) ?? 0;
    // Replace NaN/Inf with 0 (safety guard for edge cases)
    const safeRaw = isFinite(raw) ? raw : 0;
    const scaled = (safeRaw - m.scalerMean[i]) / m.scalerScale[i];
    logit += m.coefficients[i] * scaled;
  }

  // ── Step 3: Sigmoid ───────────────────────────────────────────────────
  const rawProb = sigmoid(logit);

  // ── Step 4: Isotonic calibration ──────────────────────────────────────
  const calibrated = isotonicInterpolate(rawProb, m.calX, m.calY);

  // Clamp to [0.01, 0.99] — never output certainty
  return Math.max(0.01, Math.min(0.99, calibrated));
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/**
 * Sigmoid function: σ(x) = 1 / (1 + exp(−x))
 * Numerically stable for large |x|.
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  } else {
    // More stable for negative inputs
    const ex = Math.exp(x);
    return ex / (1 + ex);
  }
}

/**
 * Piecewise linear interpolation through the isotonic regression thresholds.
 *
 * The IsotonicRegression from sklearn stores knot points (xThresholds, yThresholds)
 * representing the fitted step function. We linearly interpolate between adjacent
 * knots to get a smooth (monotone) calibration mapping.
 *
 * @param rawProb     Raw sigmoid probability (0–1)
 * @param xThresholds Sorted ascending input probabilities from IsotonicRegression
 * @param yThresholds Corresponding calibrated output values
 * @returns Calibrated probability
 */
function isotonicInterpolate(
  rawProb: number,
  xThresholds: Float64Array,
  yThresholds: Float64Array,
): number {
  const n = xThresholds.length;
  if (n === 0) return rawProb;
  if (n === 1) return yThresholds[0];

  // Clamp to range of the calibration data
  if (rawProb <= xThresholds[0]) return yThresholds[0];
  if (rawProb >= xThresholds[n - 1]) return yThresholds[n - 1];

  // Binary search for the bracketing interval
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >>> 1;
    if (xThresholds[mid] <= rawProb) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  // Linear interpolation between [lo] and [hi]
  const x0 = xThresholds[lo];
  const x1 = xThresholds[hi];
  const y0 = yThresholds[lo];
  const y1 = yThresholds[hi];

  const dx = x1 - x0;
  if (dx < 1e-12) return y0;  // identical x values (flat step)

  const t = (rawProb - x0) / dx;
  return y0 + t * (y1 - y0);
}

// ─── Reset (for testing) ──────────────────────────────────────────────────────

/**
 * Reset the model singleton (for testing purposes).
 * Forces the next predict() call to reload from disk.
 */
export function _resetModel(): void {
  _model = null;
  _loadAttempted = false;
}
