// MLB Oracle v4.0 — Email Alerts via Resend API
// Optional: only active if RESEND_API_KEY is set in .env.
// Graceful no-op otherwise.

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { Prediction, AccuracyLog } from '../types.js';
import type { RecapGame, RecapMetrics } from './discord.js';

// ─── Config ────────────────────────────────────────────────────────────────────

function getResendConfig(): { apiKey: string; from: string; to: string } | null {
  const apiKey = process.env.RESEND_API_KEY;
  const from = process.env.RESEND_FROM ?? 'mlb-oracle@yourdomain.com';
  const to = process.env.RESEND_TO;

  if (!apiKey || !to) {
    return null;
  }

  return { apiKey, from, to };
}

async function sendEmail(subject: string, html: string): Promise<boolean> {
  const config = getResendConfig();

  if (!config) {
    logger.warn('Resend not configured (RESEND_API_KEY / RESEND_TO missing) — skipping email');
    return false;
  }

  try {
    const resp = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${config.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: config.from,
        to: [config.to],
        subject,
        html,
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Resend API error');
      return false;
    }

    logger.info({ subject }, 'Email sent via Resend');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send email');
    return false;
  }
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function formatRunLine(runLine: number): string {
  return runLine >= 0 ? `+${runLine.toFixed(1)}` : runLine.toFixed(1);
}

function getWinner(pred: Prediction): { team: string; winPct: number } {
  if (pred.calibrated_prob >= 0.5) {
    return { team: pred.home_team, winPct: pred.calibrated_prob };
  }
  return { team: pred.away_team, winPct: 1 - pred.calibrated_prob };
}

function isHighConviction(pred: Prediction): boolean {
  return pred.calibrated_prob >= 0.65 || (1 - pred.calibrated_prob) >= 0.65;
}

const baseStyle = `
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 0; }
  .container { max-width: 700px; margin: 0 auto; padding: 24px; }
  h1, h2, h3 { color: #f0f6fc; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .game-row { border-bottom: 1px solid #21262d; padding: 10px 0; }
  .game-row:last-child { border-bottom: none; }
  .matchup { font-weight: bold; font-size: 15px; }
  .winner { color: #3fb950; font-weight: bold; }
  .high-conv { color: #f85149; font-weight: bold; }
  .badge { display: inline-block; background: #1f6feb; color: #fff; border-radius: 4px; padding: 2px 7px; font-size: 12px; margin-left: 6px; }
  .badge-red { background: #da3633; }
  .meta { color: #8b949e; font-size: 13px; margin-top: 4px; }
  .footer { color: #6e7681; font-size: 12px; text-align: center; margin-top: 24px; border-top: 1px solid #21262d; padding-top: 12px; }
  table { width: 100%; border-collapse: collapse; }
  td, th { padding: 6px 8px; text-align: left; }
  th { color: #8b949e; font-size: 12px; text-transform: uppercase; border-bottom: 1px solid #30363d; }
  .correct { color: #3fb950; }
  .wrong { color: #f85149; }
`;

// ─── Morning Briefing Email ───────────────────────────────────────────────────

export async function sendMorningBriefingEmail(
  date: string,
  predictions: Prediction[],
  recentAccuracy?: AccuracyLog[]
): Promise<void> {
  if (!getResendConfig()) return;

  const highConv = predictions.filter(isHighConviction);

  let gamesHtml = '';
  for (const pred of predictions) {
    const { team, winPct } = getWinner(pred);
    const hc = isHighConviction(pred);
    const badgeHtml = hc ? '<span class="badge badge-red">⭐ TOP PICK</span>' : '';
    const edgeHtml = pred.edge !== undefined && pred.edge !== null
      ? `<span class="meta">Edge: ${pred.edge >= 0 ? '+' : ''}${(pred.edge * 100).toFixed(1)}%</span>`
      : '';

    gamesHtml += `
      <div class="game-row">
        <div class="matchup">${pred.away_team} @ ${pred.home_team}${badgeHtml}</div>
        <div class="winner">${team} wins — ${pct(winPct)}</div>
        <div class="meta">
          Score: ${pred.most_likely_score} &nbsp;|&nbsp;
          Total: ${pred.total_runs.toFixed(1)} &nbsp;|&nbsp;
          Run line: ${formatRunLine(pred.run_line)} &nbsp;|&nbsp;
          Upset%: ${pct(pred.upset_probability)}
        </div>
        ${edgeHtml}
      </div>
    `;
  }

  let brierHtml = '';
  if (recentAccuracy && recentAccuracy.length > 0) {
    const rows = recentAccuracy.slice(-7).map(a =>
      `<tr><td>${a.date}</td><td>${a.brier_score.toFixed(3)}</td><td>${pct(a.accuracy)}</td></tr>`
    ).join('');
    brierHtml = `
      <div class="card">
        <h3>📊 Brier Score Tracker (Last 7 Days)</h3>
        <table><thead><tr><th>Date</th><th>Brier</th><th>Accuracy</th></tr></thead>
        <tbody>${rows}</tbody></table>
      </div>
    `;
  }

  const html = `
    <!DOCTYPE html><html><head><style>${baseStyle}</style></head><body>
    <div class="container">
      <h1>⚾ MLB Oracle Morning Briefing</h1>
      <p style="color:#8b949e">${date} &nbsp;·&nbsp; ${predictions.length} games &nbsp;·&nbsp; ${highConv.length} top picks</p>
      <div class="card">
        <h3>Today's Games</h3>
        ${gamesHtml}
      </div>
      ${brierHtml}
      <div class="footer">MLB Oracle v4.0.0 &nbsp;·&nbsp; Monte Carlo 10k sims &nbsp;·&nbsp; Generated ${new Date().toISOString()}</div>
    </div>
    </body></html>
  `;

  await sendEmail(`⚾ MLB Oracle Morning Briefing — ${date}`, html);
}

// ─── High Conviction Email ────────────────────────────────────────────────────

export async function sendHighConvictionEmail(pred: Prediction): Promise<void> {
  if (!getResendConfig()) return;
  if (!isHighConviction(pred)) return;

  const { team, winPct } = getWinner(pred);
  const matchup = `${pred.away_team} @ ${pred.home_team}`;
  const edgeHtml = pred.edge !== undefined && pred.edge !== null
    ? `<p><strong>Market Edge:</strong> ${pred.edge >= 0 ? '+' : ''}${(pred.edge * 100).toFixed(1)}%</p>`
    : '';

  const html = `
    <!DOCTYPE html><html><head><style>${baseStyle}</style></head><body>
    <div class="container">
      <h1>🔴 High Conviction Pick</h1>
      <div class="card">
        <div class="matchup" style="font-size:18px">${matchup}</div>
        <p style="margin:8px 0">Venue: ${pred.venue}</p>
        <p class="high-conv" style="font-size:20px">${team} — <strong>${pct(winPct)}</strong></p>
        <table>
          <tr><th>Projected Score</th><th>Total (O/U)</th><th>Run Line</th><th>Upset%</th></tr>
          <tr>
            <td>${pred.most_likely_score}</td>
            <td>${pred.total_runs.toFixed(1)}</td>
            <td>${formatRunLine(pred.run_line)}</td>
            <td>${pct(pred.upset_probability)}</td>
          </tr>
        </table>
        ${edgeHtml}
      </div>
      <div class="footer">MLB Oracle v4.0.0 &nbsp;·&nbsp; Game starts in ~2 hours</div>
    </div>
    </body></html>
  `;

  await sendEmail(`🔴 High Conviction: ${matchup}`, html);
}

// ─── Evening Recap Email ──────────────────────────────────────────────────────

export async function sendEveningRecapEmail(
  date: string,
  games: RecapGame[],
  metrics: RecapMetrics
): Promise<void> {
  if (!getResendConfig()) return;
  if (games.length === 0) return;

  let gamesHtml = '';
  for (const g of games) {
    const markerClass = g.correct ? 'correct' : 'wrong';
    const marker = g.correct ? '✓' : '✗';
    const { team: predicted, winPct } = getWinner(g.prediction);
    const score = `${g.prediction.home_team} ${g.homeScore}–${g.awayScore} ${g.prediction.away_team}`;

    gamesHtml += `
      <div class="game-row">
        <span class="${markerClass}">${marker}</span>
        <strong>${g.prediction.away_team} @ ${g.prediction.home_team}</strong>
        &nbsp;${score}
        <span class="meta"> — predicted ${predicted} ${pct(winPct)}</span>
      </div>
    `;
  }

  const correct = games.filter(g => g.correct).length;
  const vegasRow = metrics.vegasBrierScore !== undefined
    ? `<tr><td>Vegas Brier</td><td>${metrics.vegasBrierScore.toFixed(4)}</td></tr>` : '';
  const seasonRow = metrics.seasonStats
    ? `<tr><td>Season Accuracy</td><td>${pct(metrics.seasonStats.accuracy)} (${metrics.seasonStats.correctPredictions}/${metrics.seasonStats.totalGames})</td></tr>` : '';

  const html = `
    <!DOCTYPE html><html><head><style>${baseStyle}</style></head><body>
    <div class="container">
      <h1>🌙 MLB Oracle Evening Recap</h1>
      <p style="color:#8b949e">${date} &nbsp;·&nbsp; <strong>${correct}/${games.length}</strong> correct</p>
      <div class="card">
        <h3>Results</h3>
        ${gamesHtml}
      </div>
      <div class="card">
        <h3>📊 Metrics</h3>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>Accuracy</td><td>${pct(metrics.dailyAccuracy)}</td></tr>
          <tr><td>Brier Score</td><td>${metrics.brierScore.toFixed(4)}</td></tr>
          <tr><td>Log Loss</td><td>${metrics.logLoss.toFixed(4)}</td></tr>
          ${vegasRow}
          ${seasonRow}
        </table>
      </div>
      <div class="footer">MLB Oracle v4.0.0 &nbsp;·&nbsp; ${new Date().toISOString()}</div>
    </div>
    </body></html>
  `;

  await sendEmail(`🌙 MLB Oracle Recap — ${date}`, html);
}
