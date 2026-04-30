/**
 * build.mjs
 *
 * Bundles pneumonia-surrogate.js with esbuild, injecting the jsDelivr CDN
 * base URL so the component resolves its own asset URLs automatically when
 * loaded from the published package.
 */

import { build } from 'esbuild';
import { readFileSync } from 'fs';

const pkg = JSON.parse(readFileSync('./package.json', 'utf8'));
const cdnBase = `https://cdn.jsdelivr.net/npm/${pkg.name}@${pkg.version}`;

const shared = {
  entryPoints: ['pneumonia-surrogate.js'],
  bundle: true,
  format: 'iife',
  platform: 'browser',
  define: {
    // Replaced at build time so the component knows its own CDN location.
    ASSET_CDN_BASE: JSON.stringify(cdnBase),
  },
};

await build({
  ...shared,
  outfile: 'dist/pneumonia-surrogate.js',
  minify: true,
  logLevel: 'warning',
});

await build({
  ...shared,
  outfile: 'dist/pneumonia-surrogate.dev.js',
  sourcemap: true,
  logLevel: 'warning',
});

console.log(`Built dist/ for ${pkg.name}@${pkg.version}`);
console.log(`Default asset CDN base: ${cdnBase}`);
