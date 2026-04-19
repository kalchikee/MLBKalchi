// MLB Oracle v4.0 — Discord Webhook Alert Module
// Sends embeds for morning briefing, lineup updates, high-conviction picks, and evening recap.

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import {
  getPredictionsByDate,
  initDb,
} from '../db/database.js';
import type { Prediction, AccuracyLog } from '../types.js';

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_VERSION = '4.0.0';

const COLORS = {
  morning: 0x3498db,      // blue
  lineup: 0x2ecc71,       // green
  high_conviction: 0xe74c3c, // red
  recap: 0x95a5a6,        // gray
} as const;

// ─── Discord embed types ──────────────────────────────────────────────────────

interface DiscordField {
  name: string;
  value: string;
  inline?: boolean;
}

interface DiscordEmbed {
  title?: string;
  description?: string;
  color?: number;
  fields?: DiscordField[];
  footer?: { text: string };
  timestamp?: string;
}

interface DiscordPayload {
  embeds: DiscordEmbed[];
}

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;

  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }

  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });

    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }

    logger.info('Discord alert sent successfully');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
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

function predictionSummaryField(pred: Prediction): DiscordField {
  const { team, winPct } = getWinner(pred);
  const matchup = `${pred.away_team} @ ${pred.home_team}`;
  const conviction = isHighConviction(pred) ? ' ⭐' : '';
  const edgeStr = pred.edge !== undefined && pred.edge !== null
    ? `  Edge: ${pred.edge >= 0 ? '+' : ''}${(pred.edge * 100).toFixed(1)}%`
    : '';

  const value = [
    `**${team} wins** ${pct(winPct)}${conviction}`,
    `Score: ${pred.most_likely_score}  |  Total: ${pred.total_runs.toFixed(1)}`,
    `Run line: ${formatRunLine(pred.run_line)}  |  Upset%: ${pct(pred.upset_probability)}${edgeStr}`,
  ].join('\n');

  return { name: matchup, value, inline: false };
}

// ─── Morning Briefing (10 AM) — Predictions + Paper Trades in one message ─────

export async function sendMorningBriefing(
  date: string,
  bets?: import('../kalshi/betEngine.js').KalshiBetRecord[],
): Promise<void> {
  await initDb();

  const predictions = getPredictionsByDate(date);

  if (predictions.length === 0) {
    logger.info({ date }, 'No predictions for morning briefing — skipping Discord alert');
    return;
  }

  const highConv = predictions.filter(isHighConviction);
  const fields: DiscordField[] = [];

  // ── Section 1: All games ──────────────────────────────────────────────────
  const allGamesLines = predictions
    .slice()
    .sort((a, b) => {
      const aProb = Math.max(a.calibrated_prob, 1 - a.calibrated_prob);
      const bProb = Math.max(b.calibrated_prob, 1 - b.calibrated_prob);
      return bProb - aProb;
    })
    .map(pred => {
      const { team, winPct } = getWinner(pred);
      const star = isHighConviction(pred) ? ' ⭐' : '';
      return `**${pred.away_team} @ ${pred.home_team}** → ${team} ${pct(winPct)}${star}  _(${pred.most_likely_score})_`;
    });

  fields.push({
    name: `📋 All Games (${predictions.length} total)`,
    value: allGamesLines.join('\n') || 'No games today',
    inline: false,
  });

  // ── Section 2: Paper trades being placed ─────────────────────────────────
  if (bets && bets.length > 0) {
    const betLines = bets.map(b => {
      const matchup = `${b.away_team} @ ${b.home_team}`;
      const side = b.side.toUpperCase();
      const modelProbPct = (b.model_prob * 100).toFixed(1);
      const potentialProfit = ((1 - b.cost_basis) * 100).toFixed(0);
      return `**${matchup}** → ${side} @ ${b.entry_price}¢  |  Model: ${modelProbPct}%  |  Max profit: +${potentialProfit}¢`;
    });
    fields.push({
      name: `🎯 Paper Trades Today (${bets.length} bet${bets.length !== 1 ? 's' : ''} · 1 contract each)`,
      value: betLines.join('\n'),
      inline: false,
    });

    // Stop-loss note
    fields.push({
      name: '🛡️ Stop-Loss Rule',
      value: 'Auto-sell if any position loses **20%** of its value during the day',
      inline: false,
    });
  } else {
    fields.push({
      name: '🎯 Paper Trades Today',
      value: 'No bettable edge found today (need 65%+ model confidence AND market edge vs Vegas)',
      inline: false,
    });
  }

  const embed: DiscordEmbed = {
    title: `⚾ MLB Oracle — ${date}`,
    description: `**${predictions.length} games** · **${highConv.length}** high-conviction (65%+) · Paper trading`,
    color: COLORS.morning,
    fields,
    footer: { text: `MLB Oracle v${MODEL_VERSION} | Monte Carlo 10,000 simulations` },
    timestamp: new Date().toISOString(),
  };

  await sendWebhook({ embeds: [embed] });
}

// ─── Lineup Update Alert (~3 hrs pre-game) ───────────────────────────────────

export interface LineupChangeInfo {
  gamePk: number;
  homeTeam: string;
  awayTeam: string;
  oldProb: number;
  newProb: number;
  newPrediction: Prediction;
}

export async function sendLineupUpdateAlert(change: LineupChangeInfo): Promise<void> {
  // Only alert if a high-conviction pick (was OR is now 65%+) changed by 3%+
  const delta = Math.abs(change.newProb - change.oldProb);
  const wasHighConv = change.oldProb >= 0.65 || (1 - change.oldProb) >= 0.65;
  const isNowHighConv = change.newProb >= 0.65 || (1 - change.newProb) >= 0.65;

  if (delta < 0.03 || (!wasHighConv && !isNowHighConv)) {
    logger.debug(
      { gamePk: change.gamePk, delta },
      'Lineup change below threshold — skipping alert'
    );
    return;
  }

  const matchup = `${change.awayTeam} @ ${change.homeTeam}`;
  const { team: oldWinner } = change.oldProb >= 0.5
    ? { team: change.homeTeam }
    : { team: change.awayTeam };
  const oldPct = change.oldProb >= 0.5 ? change.oldProb : 1 - change.oldProb;

  const { team: newWinner, winPct: newPct } = getWinner(change.newPrediction);
  const direction = change.newProb > change.oldProb ? '📈' : '📉';

  const fields: DiscordField[] = [
    {
      name: matchup,
      value: [
        `**${direction} Probability shift: ${(delta * 100).toFixed(1)}%**`,
        `Before: ${oldWinner} ${pct(oldPct)}`,
        `After:  **${newWinner} ${pct(newPct)}**`,
        `New score: ${change.newPrediction.most_likely_score}  |  Total: ${change.newPrediction.total_runs.toFixed(1)}`,
        `Run line: ${formatRunLine(change.newPrediction.run_line)}`,
      ].join('\n'),
      inline: false,
    },
  ];

  const embed: DiscordEmbed = {
    title: `🔄 Lineup Update — ${matchup}`,
    description: 'Lineups confirmed. Prediction updated.',
    color: COLORS.lineup,
    fields,
    footer: { text: `MLB Oracle v${MODEL_VERSION}` },
    timestamp: new Date().toISOString(),
  };

  await sendWebhook({ embeds: [embed] });
}

// ─── High Conviction Alert (2 hrs pre-game) ───────────────────────────────────

export async function sendHighConvictionAlert(pred: Prediction): Promise<void> {
  if (!isHighConviction(pred)) {
    logger.debug({ gamePk: pred.game_pk }, 'Not high conviction — skipping alert');
    return;
  }

  const { team, winPct } = getWinner(pred);
  const matchup = `${pred.away_team} @ ${pred.home_team}`;

  const edgeStr = pred.edge !== undefined && pred.edge !== null
    ? `**Market Edge: ${pred.edge >= 0 ? '+' : ''}${(pred.edge * 100).toFixed(1)}%**`
    : '_No market data_';

  const fields: DiscordField[] = [
    {
      name: 'Prediction',
      value: `**${team}** to win  |  **${pct(winPct)}** calibrated probability`,
      inline: false,
    },
    {
      name: 'Projected Score',
      value: pred.most_likely_score,
      inline: true,
    },
    {
      name: 'Total (O/U)',
      value: pred.total_runs.toFixed(1),
      inline: true,
    },
    {
      name: 'Run Line',
      value: formatRunLine(pred.run_line),
      inline: true,
    },
    {
      name: 'Upset Probability',
      value: pct(pred.upset_probability),
      inline: true,
    },
    {
      name: 'Market Edge',
      value: edgeStr,
      inline: true,
    },
  ];

  const embed: DiscordEmbed = {
    title: `🔴 High Conviction Pick — ${matchup}`,
    description: `Game starting in ~2 hours at **${pred.venue}**`,
    color: COLORS.high_conviction,
    fields,
    footer: { text: `MLB Oracle v${MODEL_VERSION}` },
    timestamp: new Date().toISOString(),
  };

  await sendWebhook({ embeds: [embed] });
}

// ─── Evening Recap (11 PM) ────────────────────────────────────────────────────

export interface RecapGame {
  prediction: Prediction;
  homeScore: number;
  awayScore: number;
  correct: boolean;
  actualWinner: string;
}

export interface SeasonStats {
  totalGames: number;
  correctPredictions: number;
  accuracy: number;
}

export interface RecapMetrics {
  dailyAccuracy: number;
  brierScore: number;
  logLoss: number;
  vegasBrierScore?: number;
  seasonStats?: SeasonStats;
  eloChanges?: Array<{ team: string; change: number; newRating: number }>;
}

export async function sendEveningRecap(
  date: string,
  games: RecapGame[],
  metrics: RecapMetrics
): Promise<void> {
  if (games.length === 0) {
    logger.info({ date }, 'No recap games — skipping Discord alert');
    return;
  }

  const fields: DiscordField[] = [];

  // Results per game
  const resultLines = games.map(g => {
    const marker = g.correct ? '✓' : '✗';
    const { team: predicted, winPct } = getWinner(g.prediction);
    const actual = g.homeScore > g.awayScore ? g.prediction.home_team : g.prediction.away_team;
    const score = `${g.prediction.home_team} ${g.homeScore}–${g.awayScore} ${g.prediction.away_team}`;
    return `${marker} ${g.prediction.away_team} @ ${g.prediction.home_team}: ${score} (predicted ${predicted} ${pct(winPct)})`;
  });

  fields.push({
    name: 'Results',
    value: resultLines.join('\n'),
    inline: false,
  });

  // Daily metrics
  const metricsLines = [
    `Accuracy: **${pct(metrics.dailyAccuracy)}** (${games.filter(g => g.correct).length}/${games.length})`,
    `Brier Score: ${metrics.brierScore.toFixed(4)}`,
    `Log Loss: ${metrics.logLoss.toFixed(4)}`,
  ];
  if (metrics.vegasBrierScore !== undefined) {
    metricsLines.push(`Vegas Brier: ${metrics.vegasBrierScore.toFixed(4)}`);
  }
  fields.push({
    name: '📊 Daily Metrics',
    value: metricsLines.join('\n'),
    inline: false,
  });

  // Season accuracy
  if (metrics.seasonStats) {
    const s = metrics.seasonStats;
    fields.push({
      name: '📈 Season Running Total',
      value: `${s.correctPredictions}/${s.totalGames} games correct — **${pct(s.accuracy)}** season accuracy`,
      inline: false,
    });
  }

  // Elo changes
  if (metrics.eloChanges && metrics.eloChanges.length > 0) {
    const eloLines = metrics.eloChanges.map(e => {
      const dir = e.change >= 0 ? '+' : '';
      return `${e.team}: ${dir}${e.change.toFixed(0)} → ${e.newRating.toFixed(0)}`;
    });
    fields.push({
      name: '📡 Elo Rating Changes',
      value: eloLines.join('\n'),
      inline: false,
    });
  }

  // Vegas comparison
  if (metrics.vegasBrierScore !== undefined) {
    const vegasDiff = metrics.brierScore - metrics.vegasBrierScore;
    const comparison = vegasDiff < 0
      ? `Model outperforms Vegas by ${Math.abs(vegasDiff).toFixed(4)} Brier`
      : `Vegas outperforms model by ${vegasDiff.toFixed(4)} Brier`;
    fields.push({
      name: '🎰 Model vs Vegas',
      value: comparison,
      inline: false,
    });
  }

  const correct = games.filter(g => g.correct).length;
  const embed: DiscordEmbed = {
    title: `🌙 MLB Oracle Evening Recap — ${date}`,
    description: `**${correct}/${games.length}** predictions correct today`,
    color: COLORS.recap,
    fields,
    footer: { text: `MLB Oracle v${MODEL_VERSION}` },
    timestamp: new Date().toISOString(),
  };

  await sendWebhook({ embeds: [embed] });
}

// ─── Kalshi Betting Alerts ────────────────────────────────────────────────────

export async function sendNoBetsAlert(date: string, reason: string): Promise<void> {
  const embed: DiscordEmbed = {
    title: `⚾ MLB Oracle — No Bets Today (${date})`,
    description: reason,
    color: 0x95a5a6,
    footer: { text: `MLB Oracle v${MODEL_VERSION} · Paper Trading` },
    timestamp: new Date().toISOString(),
  };
  await sendWebhook({ embeds: [embed] });
}

export async function sendBetPlacedAlert(
  bet: import('../kalshi/betEngine.js').KalshiBetRecord,
  candidate: import('../kalshi/marketMatcher.js').MatchedBet,
  paper: boolean,
): Promise<void> {
  const mode = paper ? ' [PAPER]' : '';
  const matchup = `${candidate.prediction.away_team} @ ${candidate.prediction.home_team}`;
  const pickedTeam = candidate.side === 'yes' ? candidate.yesTeam : candidate.noTeam;
  const projScore = candidate.prediction.most_likely_score;
  const potentialProfit = (bet.contracts - bet.cost_basis).toFixed(2);
  const payout = (bet.contracts * 1.0).toFixed(2);

  const fields: DiscordField[] = [
    { name: '🏟️ Matchup', value: matchup, inline: true },
    { name: '✅ Pick', value: `**${pickedTeam}** (${bet.side.toUpperCase()})`, inline: true },
    { name: '📊 Model Prob', value: `**${pct(bet.model_prob)}**`, inline: true },
    { name: '💰 Entry Price', value: `${bet.entry_price}¢ / contract`, inline: true },
    { name: '🎟️ Contracts', value: `${bet.contracts}`, inline: true },
    { name: '💵 Cost Basis', value: `$${bet.cost_basis.toFixed(2)}`, inline: true },
    { name: '🏆 Max Payout', value: `$${payout} (+$${potentialProfit} profit)`, inline: true },
    { name: '🎯 Proj Score', value: projScore, inline: true },
  ];

  if (bet.edge && bet.edge !== 0) {
    fields.push({ name: '📈 Edge vs Vegas', value: `${(bet.edge * 100).toFixed(1)}pp`, inline: true });
  }

  const embed: DiscordEmbed = {
    title: `⚾ Bet Placed${mode} — ${matchup}`,
    description: `Backing **${pickedTeam}** at **${bet.entry_price}¢** · Ticker: \`${bet.ticker}\``,
    color: paper ? 0xf39c12 : 0x27ae60,
    fields,
    footer: { text: `MLB Oracle v${MODEL_VERSION}${mode} · Order: ${bet.order_id}` },
    timestamp: new Date().toISOString(),
  };
  await sendWebhook({ embeds: [embed] });
}

export async function sendCashoutAlert(
  bet: import('../kalshi/betEngine.js').KalshiBetRecord,
  exitPriceCents: number,
  pnl: number,
  pctChange: number,
  paper: boolean,
): Promise<void> {
  const mode = paper ? ' [PAPER]' : '';
  const matchup = `${bet.away_team} @ ${bet.home_team}`;
  const pnlStr = pnl >= 0 ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`;
  const pctStr = `${(pctChange * 100).toFixed(1)}%`;

  const embed: DiscordEmbed = {
    title: `🚨 Stop-Loss Triggered${mode} — ${matchup}`,
    description: `Position lost **${pctStr}** of value — auto-sold`,
    color: 0xe74c3c,
    fields: [
      { name: '🎟️ Ticker', value: bet.ticker, inline: true },
      { name: '📌 Side', value: bet.side.toUpperCase(), inline: true },
      { name: '📉 Entry / Exit', value: `${bet.entry_price}¢ → ${exitPriceCents}¢`, inline: true },
      { name: '📊 P&L', value: pnlStr, inline: true },
      { name: '📉 Loss %', value: pctStr, inline: true },
    ],
    footer: { text: `MLB Oracle v${MODEL_VERSION}${mode}` },
    timestamp: new Date().toISOString(),
  };
  await sendWebhook({ embeds: [embed] });
}

export async function sendEODSummaryAlert(
  date: string,
  bets: import('../kalshi/betEngine.js').KalshiBetRecord[],
  summary: { totalCost: number; totalPnl: number; wins: number; losses: number; open: number },
  paper: boolean,
  recapGames?: RecapGame[],
  recapMetrics?: RecapMetrics,
): Promise<void> {
  const mode = paper ? ' [PAPER]' : '';
  const roi = summary.totalCost > 0 ? (summary.totalPnl / summary.totalCost) * 100 : 0;
  const pnlStr = summary.totalPnl >= 0
    ? `+$${summary.totalPnl.toFixed(2)}`
    : `-$${Math.abs(summary.totalPnl).toFixed(2)}`;

  if (bets.length === 0) {
    const noBetFields: DiscordField[] = [];
    if (recapGames && recapGames.length > 0) {
      const correct = recapGames.filter(g => g.correct).length;
      const gameLines = recapGames.map(g => {
        const marker = g.correct ? '✅' : '❌';
        const { team: predicted, winPct } = getWinner(g.prediction);
        return `${marker} **${g.prediction.away_team} @ ${g.prediction.home_team}** ${g.homeScore}–${g.awayScore} · predicted ${predicted} ${pct(winPct)}`;
      });
      noBetFields.push({
        name: `🎯 Game Predictions — ${correct}/${recapGames.length} correct (${pct(correct / recapGames.length)})`,
        value: gameLines.join('\n'),
        inline: false,
      });
      if (recapMetrics?.seasonStats && recapMetrics.seasonStats.totalGames > 0) {
        const s = recapMetrics.seasonStats;
        noBetFields.push({
          name: '📈 Season Running Total',
          value: `${s.correctPredictions}/${s.totalGames} · **${pct(s.accuracy)}**`,
          inline: true,
        });
      }
    }
    const gameAccStr = recapGames && recapGames.length > 0
      ? ` · **${recapGames.filter(g => g.correct).length}/${recapGames.length}** games correct`
      : '';
    const embed: DiscordEmbed = {
      title: `📋 MLB Oracle EOD${mode} — ${date}`,
      description: `No bets placed today (no games cleared the 75% threshold)${gameAccStr}`,
      color: 0x95a5a6,
      fields: noBetFields.length > 0 ? noBetFields : undefined,
      footer: { text: `MLB Oracle v${MODEL_VERSION}${mode}` },
      timestamp: new Date().toISOString(),
    };
    await sendWebhook({ embeds: [embed] });
    return;
  }

  // Per-bet breakdown
  const betLines = bets.map(b => {
    const matchup = `${b.away_team} @ ${b.home_team}`;
    const entryStr = `${b.side.toUpperCase()} @ ${b.entry_price}¢`;
    let resultStr: string;
    if (b.status === 'open') {
      resultStr = '⏳ Still open';
    } else if (b.exit_reason === 'stop_loss_20pct') {
      resultStr = `🛑 Stop-loss  ${b.pnl !== undefined ? (b.pnl >= 0 ? `+${(b.pnl * 100).toFixed(0)}¢` : `${(b.pnl * 100).toFixed(0)}¢`) : ''}`;
    } else if (b.pnl !== undefined && b.pnl > 0) {
      resultStr = `✅ Won  +${(b.pnl * 100).toFixed(0)}¢`;
    } else if (b.pnl !== undefined && b.pnl < 0) {
      resultStr = `❌ Lost  ${(b.pnl * 100).toFixed(0)}¢`;
    } else {
      resultStr = '—';
    }
    return `**${matchup}** · ${entryStr} · ${resultStr}`;
  }).join('\n');

  const fields: DiscordField[] = [
    {
      name: '🎟️ Bet Results',
      value: betLines,
      inline: false,
    },
    {
      name: '📊 Record',
      value: `${summary.wins}W  ${summary.losses}L  ${summary.open > 0 ? `${summary.open} open` : ''}`.trim(),
      inline: true,
    },
    {
      name: '💵 Total Wagered',
      value: `$${summary.totalCost.toFixed(2)}`,
      inline: true,
    },
    {
      name: '💰 Net P&L',
      value: `**${pnlStr}**`,
      inline: true,
    },
    {
      name: '📈 ROI',
      value: `${roi.toFixed(1)}%`,
      inline: true,
    },
  ];

  // ── Game prediction accuracy section ─────────────────────────────────────
  if (recapGames && recapGames.length > 0) {
    const correct = recapGames.filter(g => g.correct).length;
    const total = recapGames.length;
    const acc = correct / total;

    // Per-game result lines — split into chunks to respect Discord's 1024 char/field limit
    const gameLines = recapGames.map(g => {
      const marker = g.correct ? '✅' : '❌';
      const { team: predicted, winPct } = getWinner(g.prediction);
      const score = `${g.homeScore}–${g.awayScore}`;
      return `${marker} **${g.prediction.away_team} @ ${g.prediction.home_team}** ${score} · predicted ${predicted} ${pct(winPct)}`;
    });
    const MAX_FIELD_LEN = 1000; // leave headroom under 1024
    const chunks: string[] = [];
    let current = '';
    for (const line of gameLines) {
      if (current.length + line.length + 1 > MAX_FIELD_LEN) {
        chunks.push(current);
        current = line;
      } else {
        current = current ? `${current}\n${line}` : line;
      }
    }
    if (current) chunks.push(current);
    chunks.forEach((chunk, i) => {
      fields.push({
        name: i === 0
          ? `🎯 Game Predictions — ${correct}/${total} correct (${pct(acc)})`
          : `🎯 Game Predictions (cont.)`,
        value: chunk,
        inline: false,
      });
    });

    // High-conviction accuracy
    const hc = recapGames.filter(g => g.prediction.calibrated_prob >= 0.65 || (1 - g.prediction.calibrated_prob) >= 0.65);
    if (hc.length > 0) {
      const hcCorrect = hc.filter(g => g.correct).length;
      fields.push({
        name: '⭐ High-Conviction (65%+)',
        value: `${hcCorrect}/${hc.length} correct — **${pct(hcCorrect / hc.length)}**`,
        inline: true,
      });
    }

    // Season running total
    if (recapMetrics?.seasonStats && recapMetrics.seasonStats.totalGames > 0) {
      const s = recapMetrics.seasonStats;
      fields.push({
        name: '📈 Season Running Total',
        value: `${s.correctPredictions}/${s.totalGames} · **${pct(s.accuracy)}**`,
        inline: true,
      });
    }

    // Model vs Vegas
    if (recapMetrics?.vegasBrierScore !== undefined && recapMetrics.brierScore > 0) {
      const diff = recapMetrics.brierScore - recapMetrics.vegasBrierScore;
      const vsVegas = diff < 0
        ? `✅ Beat Vegas by ${Math.abs(diff).toFixed(4)} Brier`
        : `📉 Vegas beat us by ${diff.toFixed(4)} Brier`;
      fields.push({ name: '🎰 vs Vegas', value: vsVegas, inline: true });
    }
  } else if (recapGames) {
    fields.push({
      name: '🎯 Game Predictions',
      value: 'No final scores available yet',
      inline: false,
    });
  }

  const emoji = summary.totalPnl > 0 ? '🟢' : summary.totalPnl < 0 ? '🔴' : '⚪';
  const gameAccStr = recapGames && recapGames.length > 0
    ? ` · **${recapGames.filter(g => g.correct).length}/${recapGames.length}** games correct`
    : '';
  const embed: DiscordEmbed = {
    title: `📋 MLB Oracle EOD${mode} — ${date}`,
    description: `${emoji} **${bets.length} bet${bets.length !== 1 ? 's' : ''} placed** · P&L: **${pnlStr}**${gameAccStr}`,
    color: summary.totalPnl >= 0 ? 0x27ae60 : 0xe74c3c,
    fields,
    footer: { text: `MLB Oracle v${MODEL_VERSION}${mode} · 20% stop-loss` },
    timestamp: new Date().toISOString(),
  };
  await sendWebhook({ embeds: [embed] });
}
