# CLAUDE.md

## Project

**dotscraper**: a browser app (to be served from GitHub Pages) that extracts dot locations from images of
graphs, especially scatterplots whose dots overlap and are drawn with alpha transparency (`testgraphs/dog-ages.png`
is the main target), where ordinary circle or blob detection fails. The approach is analysis by synthesis: estimate
the dots' style (color, size, alpha, edge blur), mask the non-dot layers, then iterate on a set of dot locations
whose rendering reproduces the image, ignoring the masked pixels. Pixel locations become data values through the
detected axis scales.

The spec, decisions and milestones are in **PLAN.md**. Read it before starting a phase, tick its checkboxes as
work is completed, and record measured results there (what was tried, the numbers, what was decided).

## Hard rules

- **No bundler, no build step.** Plain ES modules served as static files. Do not add Vite, webpack, esbuild,
  TypeScript, Babel or Jest. Ask before adding any runtime dependency; dev dependencies for testing are fine.
- **Every relative import uses an explicit `.js` extension** (required by both browsers and Node ESM).
- **`src/core/` is pure**: no DOM, no `window`, no canvas, no bare-specifier imports, no network. It must run
  unchanged in Node for tests and in the browser (main thread or worker). Images enter it as
  `{ width, height, data }` with `data` a `Uint8ClampedArray` of RGBA, the shape of `ImageData`.
- Bare imports, if any are ever approved, are allowed only in `src/ui/`, resolved by the import map in
  `index.html` (jsDelivr ESM URLs, pinned versions).
- Long-running fits run in a module Web Worker (`src/fit-worker.js`) so the page stays responsive; the worker
  only wraps `src/core/` functions.
- Do not add Claude attribution to commit messages.

## Layout (see PLAN.md §7 for what's planned)

```
index.html              layout, loads src/ui/main.js as a module
styles.css
src/core/               pure algorithms: image.js (hash, pixel access), render.js (forward model), style.js (dot
                        style), layers.js (classes, morphology, mask), synth.js (synthetic graphs with known
                        dots), metrics.js (matching against ground truth), fit.js (the fit), rng.js;
                        planned: axes.js
src/ui/                 main.js (state, panel), view.js (canvas zoom/pan/click, dot marks), overlays.js, input.js
src/fit-worker.js       module worker running fitDots, posting progress
src/fit-worker.js       module Web Worker wrapping the fit
testgraphs/             test images, index.json, ground truth in truth/<name>.json: LOCAL ONLY, git ignores all
                        but the README (third-party figures, not cleared for redistribution). Never commit them,
                        and never let a test fail just because they're absent (skip instead).
tools/check-page.mjs    headless-Chrome page check (copied from lafspline)
tools/png.mjs           PNG read/write for tools and tests (pngjs)
tools/measure-dots.mjs  phase 0 measurement of a translucent dot family (levels, alpha, radius, blur)
tools/                  bench and index scripts
test/*.test.js          node:test + node:assert/strict
```

## Commands

- Tests: `npm test` (runs `node --test`). Node 22+ (`tools/check-page.mjs` uses Node's built-in WebSocket).
- Serve locally: `npm run serve` (python3 on port **8003**, leaving 8000–8002 to related projects), then open
  http://localhost:8003/
- Page check (server must be running): `npm run check:page -- http://localhost:8003/ --until '<js expr>'
  --eval '<js expr>' --screenshot out.png [--dark]`. Exits 1 on any console error, exception, or failed load.
- Test image list: after adding an image to `testgraphs/`, `npm run images:index` (the test suite fails if
  `testgraphs/index.json` is stale). Servers that don't list directories (WebStorm's built-in one) need it.
- Consistency check without ground truth: `node tools/compare-fits.mjs a.png b.png` fits two renderings of one
  graph, estimates the scale and offset between them, and reports how well the dots agree.
- Model vs Chrome canvas (server running): `npm run check:page -- http://localhost:8003/tools/render-check.html
  --until 'window.renderCheckResult' --eval 'JSON.stringify(window.renderCheckResult)' --screenshot out.png`.
- Measure a dot family: `npm run measure -- testgraphs/<image> [--seed r,g,b] [--out dir]` (writes the overlap
  count image D to `dir`).
- Benchmark: `npm run bench [-- --seeds n --preset name --verbose]` runs style, mask and fit on the synthetic
  presets and on every local image with `testgraphs/truth/<name>.json`, printing precision, recall, count error,
  position RMSE, Hough-only precision/recall, flags and time. Record results in PLAN.md.

## Conventions

- Code style: 4-space indentation, single quotes, semicolons. The body of every `if`, `else`, `for` and `while`
  goes on its own line, even single statements and guard clauses; `else`, `catch` and `finally` go on their own
  line after the closing brace. Put `return`, `break` and `continue` on their own lines too.
- Plain functions and simple data (arrays, objects, typed arrays). JSDoc type annotations on every exported
  function (parameters, return, units: px, data units, 0–1 or 0–255).
- **Pixel coordinates:** pixel (i, j) covers [i, i + 1) × [j, j + 1), so its center is (i + 0.5, j + 0.5); x right,
  y down. Data coordinates have y up. Dot positions are subpixel `Float64Array`s or `{ x, y }` in px.
- Image planes for computation are `Float32Array` (width × height, row-major). Say whether values are sRGB or
  linear-light wherever it matters. Compositing is in sRGB values (measured, PLAN.md phase 0).
- Images are decoded without color-profile conversion (`colorSpaceConversion: 'none'`) so browser and Node see
  the same values; `pixelHash` in `src/core/image.js` checks that.
- Name magic numbers as constants at the top of the file with units in a comment, e.g.
  `const EDGE_DILATE = 1.5; // px`. They are tuning knobs.
- Comments explain why and state constraints the code can't show; a short block comment above each nontrivial
  function gives its purpose and design rationale. Don't narrate the next line.
- Randomness only through `src/core/rng.js` (seeded). Tests must be deterministic.
- Each core module gets a test file. Prefer synthetic images with known dots (`synth.js`) for tests; state
  tolerances explicitly (px, counts). Keep a regression test for every real bug found, using a small crop of the
  real image when the bug came from one.
- State the complexity of anything worse than O(pixels · dot area) and keep slower paths behind size limits.
- **Display honesty:** never show a number or color that misrepresents the data. The app says which dots are
  ambiguous (count or position not well determined by the visible pixels), shows the residual image on request,
  and reports pixel resolution in data units. Simplified displays keep the exact values inspectable (tooltip,
  table, fine print).

## Working style

- Diagnose with evidence before changing code: inspect the actual pixels (crops, histograms, the residual image),
  reproduce the problem, then fix it.
- Work one phase (or milestone within a phase) at a time. At the end of each, run the tests and the benchmark,
  make sure the page loads without console errors (screenshot it with `check:page`), summarize what changed and
  what was measured, and stop for review.
- Report results plainly: if something fails or is unverified, say so.
- Don't refactor unrelated code. Don't reformat files you didn't otherwise change. Delete dead code outright.
- No defensive code for impossible cases; validate at real boundaries (user images, user-entered tick values).
- Markdown: hard-wrap prose at sentence and clause boundaries (~100–115 chars per line). Leave code blocks and
  table rows on single lines. American English; go easy on em-dashes.
- The user works in WebStorm on macOS and is strong in C++, less so in the JavaScript and web ecosystem: explain
  web-platform specifics (workers, canvas color handling, clipboard events) briefly when they matter.
