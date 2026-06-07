// postbuild: strip dev-only parity fixtures (golden.*) from the production build.
// The runtime only fetches capsule.json / config.json / weights.bin; golden.bin
// (~12 MB) + golden.json are used solely by the local parity test suite (which
// reads them from public/weights/, not dist/), so they should not ship.

import { readdirSync, rmSync, existsSync, statSync } from 'node:fs';
import { join } from 'node:path';

const root = 'dist/weights';
if (!existsSync(root)) process.exit(0);

let removed = 0;
let bytes = 0;
for (const ent of readdirSync(root, { withFileTypes: true })) {
  if (!ent.isDirectory()) continue;
  const dir = join(root, ent.name);
  for (const f of readdirSync(dir)) {
    if (f.startsWith('golden.')) {
      const p = join(dir, f);
      bytes += statSync(p).size;
      rmSync(p);
      removed++;
    }
  }
}
if (removed) console.log(`trim-golden: removed ${removed} dev fixture(s) (${(bytes / 1e6).toFixed(1)} MB) from ${root}`);
