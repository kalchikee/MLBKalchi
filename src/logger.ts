// MLB Oracle v4.0 — Shared Logger
import { pino as createPino } from 'pino';

export const logger = createPino({
  level: process.env.LOG_LEVEL ?? 'info',
  transport: {
    target: 'pino-pretty',
    options: { colorize: true },
  },
});
