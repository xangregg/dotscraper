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
- Long-running work runs in module Web Workers so the page stays responsive: the fit in `src/fit-worker.js`, the
  scan for dot families (and re-measuring them) in `src/scan-worker.js`. The workers only wrap `src/core/` functions.
- Do not add Claude attribution to commit messages.

## Layout (see PLAN.md §7 for what's planned)

```
index.html              layout, loads src/ui/main.js as a module
styles.css
src/core/               pure algorithms: image.js (hash, pixel access, color tests and distance), render.js (forward model), style.js (dot
                        style: the families of an image and one family's style, with the band and box rules), scan.js
                        (page, flat pixels, tolerances, the Scan passed through the search), levels.js (translucent
                        families: levels, alpha, color), opaque.js (opaque families: measureOpaque, opaqueFamilies, radius
                        from shapes), shape.js (isolated dots, components, edge blur), layers.js (classes, morphology, mask), synth.js (synthetic graphs with known
                        dots), metrics.js (matching against ground truth), fit.js (the fit), axes.js (lines, a
                        filled panel's edges, ticks, labels, scales), labels.js (reading tick labels' numbers: glyphs, digits by template, raised
                        exponents, scientific notation ("1e+05", "2 × 10³"), the check by the axis's scale; digit-templates.js, generated: the shapes of the digits and of the signs e, E, +, × in 18 fonts), jpeg.js (chroma restoration for JPEGs, JPEG-noise detection), unmix.js (separating
                        overlapping translucent families), background.js (shaded regions over or under the
                        dots), colormap.js (dots colored by a continuous scale, or by a categorical palette, split by each dot's own color after the fit), outline.js (outlined dots fitted by their rings in drawing order:
                        whole images whose fills vary, and single outlined families), annotations.js (text lines and legend markers, masked), families.js (all of the
                        above, per image: the families to fit), extract.js (from an image to its dots: the scan with the choice of color
                        matching, mask options, fit input, dots' fill colors, and extractAll, all of it at once: the steps the app takes
                        and the bench and tests run), stats.js (medians, quantile), rng.js
src/ui/                 main.js (state, panel), view.js (canvas zoom/pan/click, dot and axis marks), overlays.js,
                        axes-panel.js (tick values, read from the labels or typed, scales, data coordinates), input.js
src/fit-worker.js       module worker running fitDots, posting progress
src/scan-worker.js      module Web Worker running the scan for dot families (extract.js scanFamilies), posting progress
testgraphs/             test images, index.json, ground truth in truth/<name>.json (px) or <name>.csv (data
                        cells, ticks in the header): LOCAL ONLY, git ignores all
                        but the README (third-party figures, not cleared for redistribution). Never commit them,
                        and never let a test fail just because they're absent (skip instead).
bench1/ bench2/ bench3/  the user's own JMP renderings (PNG and JPEG) of one data set each: data.csv in data units
                        with scale, tick and plot-area header lines (bench1: 100 dots, bench2: 204 with more ties,
                        bench3: 204 as beeswarms, x a lane). Exact ground truth; the user's to commit or not.
                        Tests using them skip when they're absent.
bench-r/                R's renderings with tick labels in scientific notation, and of ggplot2's gray theme, drawn by
                        bench-r/make.R, each with a CSV of its points and tick values: for both benchmarks and
                        tests (which skip without it)
tools/check-page.mjs    headless-Chrome page check (copied from lafspline)
tools/png.mjs           PNG read/write and JPEG read for tools and tests (pngjs, jpeg-js)
tools/measure-dots.mjs  phase 0 measurement of a translucent dot family (levels, alpha, radius, blur)
tools/truth-csv.mjs     truth sets in data units (benchN/data.csv) and their pixel positions through the plot area
tools/truth-json.mjs    truth placed by hand (Save truth JSON, px, in testgraphs/truth/ and bench-nature/truth/); run
                        directly, it converts an old Save CSV export to one
tools/label-bench.mjs   benchmark of the tick-label reader against the tick values the truth files list
tools/make-digit-templates.mjs  writes src/core/digit-templates.js from tools/digit-fonts.html (digits and signs drawn by Chrome)
tools/                  bench and index scripts
test/*.test.js          node:test + node:assert/strict
```

## Commands

- Tests: `npm test` (runs `node --test`). Node 22+ (`tools/check-page.mjs` uses Node's built-in WebSocket).
- Serve locally: `npm run serve` (python3 on port **8003**, leaving 8000–8002 to related projects), then open
  http://localhost:8003/. Dev mode (test image list, `#image=` URL state, Save truth, the Count, Mask, Classes and
  Model views) is on for a local host and off on the deployed site (https://xangregg.github.io/dotscraper/, GitHub
  Pages from the branch root); `?dev` turns it on anywhere, `?dev=0` off.
- Page check (server must be running): `npm run check:page -- http://localhost:8003/ --until '<js expr>'
  --eval '<js expr>' --screenshot out.png [--dark] [--width px]` (`--width` for narrow layouts; default 1280). Exits 1 on any console error, exception, or failed load.
- Test image list: after adding an image to `testgraphs/`, `npm run images:index` (the test suite fails if
  `testgraphs/index.json` is stale). Servers that don't list directories (WebStorm's built-in one) need it.
- Consistency check without ground truth: `node tools/compare-fits.mjs a.png b.png` fits two renderings of one
  graph, estimates the scale and offset between them, and reports how well the dots agree.
- Model vs Chrome canvas (server running): `npm run check:page -- http://localhost:8003/tools/render-check.html
  --until 'window.renderCheckResult' --eval 'JSON.stringify(window.renderCheckResult)' --screenshot out.png`.
- Measure a dot family: `npm run measure -- testgraphs/<image> [--seed r,g,b] [--out dir]` (writes the overlap
  count image D to `dir`).
- Benchmark: `npm run bench [-- --seeds n --preset name --set bench1 --only name --verbose --jobs n]` runs style, mask and
  fit on the synthetic presets, every local image with `testgraphs/truth/<name>.json`, and every image of the truth
  sets (bench1–3; `--set` picks one, `--set testgraphs` the test images with a CSV of their own, `--set bench-nature` the Nature figures, `--set bench-plos` the PLOS ONE figures, `--set bench-r` R's renderings, `--only` filters by `set/file`, `--seeds 0` skips the presets), printing precision, recall, count error, position RMSE, Hough-only precision/recall, flags and
  time. `--set none` runs the synthetic presets alone. It fits 10 images at a time in worker threads (`--jobs n`
  for another count, `--jobs 1` for one at a time): the rows are the same, each image's time is longer than alone
  (14% in all at 10), so compare times only between runs at one job count.
  Record results in PLAN.md.
- Label benchmark: `npm run bench:labels [-- --set dir --verbose]` reads the tick labels of every image whose truth
  lists tick values and prints, per set, the labels read right and the axes solved, wrong (a scale accepted with a
  wrong value: there should be none), unchecked and left alone. Record results in PLAN.md.
- Digit templates: `node tools/make-digit-templates.mjs` (server running) rewrites `src/core/digit-templates.js`
  from the fonts of the machine it runs on; rerun the label benchmark after it.
- Results table: `npm run results [-- --jobs n]` runs the full benchmark (about 3.5 minutes at the default 10 jobs,
  18.5 at `--jobs 1`) and rewrites RESULTS.md, a table per set with every image's counts, precision, recall and
  notes (`--json file` on the bench gives its rows); its header says how many images were fitted at a time. The numbers are
  regenerated; the Notes column is hand-written and kept. Regenerate it after a change that moves bench numbers, and
  keep each image's note current (what it shows, what fails and why).

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
