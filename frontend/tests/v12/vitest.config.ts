import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/v12/setup.ts'],
    include: ['./tests/v12/unit/**/*.test.{ts,tsx}', './tests/v12/integration/**/*.test.{ts,tsx}'],
    exclude: ['./tests/v12/e2e/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      reportsDirectory: './tests/v12/coverage',
      exclude: ['node_modules/', 'tests/v12/setup.ts'],
      thresholds: {
        lines: 90,
        functions: 90,
        branches: 90,
        statements: 90,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
