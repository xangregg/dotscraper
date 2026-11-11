# PLAN.md — dotscraper

## 0. Goal

Given an image of a graph, recover the location of every dot, in pixels and in data units, as faithfully as the
visible pixels allow, and say where they don't. The main target is scatterplots whose dots overlap and are drawn
with alpha transparency (`testgraphs/dog-ages.png`), where Hough circles, Laplacian-of-Gaussian blobs and
watershed on a thresholded mask all fail: overlapping dots merge into blobs, and coincident dots vanish.

The core idea is **analysis by synthesis**. With a forward model of how the plotting library drew the dots, any
candidate set of locations renders to an image; the extraction is the set whose rendering best reproduces the
original, at the smallest number of dots, ignoring pixels that belong to other layers. The classic detectors
remain useful, but as initializers for that fit, not as the answer.

Pipeline (the order of the user's outline; §8 builds it in a different order to retire the biggest risk first):

1. Identify the graph scales (plot area, axes, ticks, tick values, linear/log/categorical).
2. Identify the graph layers (background, dot families, lines, text, shaded regions).
3. Mask the non-dot layers.
4. Estimate each dot family's style: color, radius, alpha, outline, edge blur.
5. Fit: (a) candidate locations; (b) render them; (c) compare with the image on the unmasked pixels;
   (d) refine (move, add, remove, split, merge) and repeat from (b).

## 1. The test images

What each one asks of the extractor (from viewing them; pixel measurements are a phase 0 task):

| image | size (px) | dots | alpha | outline | overlap | other layers | axes |
|---|---|---|---|---|---|---|---|
| dog-ages | 832 × 1078 | one green family, r = 24.0 px | α = 0.31 (measured) | no | heavy; internal arcs visible; many near-coincident dots at x ≈ 0 | black fit curve and gray dashed band drawn over the dots; equation text | linear x and y, ticks and labels |
| economist-canada | 870 × 1368 | red and blue families, r = 8.4 px | α = 0.30, 0.29 (measured) | no | moderate, red over blue | thick red and blue smooth lines over the dots, gridlines, labels, title | linear, labels on the right, year axis |
| fibers-distance | 432 × 448 | one blue family, small | slight | no | light, some exact ties | regression line and light blue CI band (dots over it?) | linear |
| pathogen-trade | 742 × 1040 | one family, fill color varies with y (colormap) | slight? | thin dark ring | dense band at y = 0 | gridlines, purple CI band and black line over dots, panel letter | log y with a 0 row, linear x |
| Laser-Asyllable | 532 × 778 | black, teal, purple families | no | no | few touching | bar outlines, significance brackets, text | categorical x, linear y |
| cantons-haemocytes | 1562 × 540 | four colors, reused across groups | no | no | touching, beeswarm-like | median lines, brackets, text | categorical x, linear y |
| nicest-language | 866 × 1146 | gray, pink, red, light and dark blue | no | no | packed beeswarm, touching | beige panel background, a thin baseline through the middle, text below | linear x only |
| violin-beeswarm | 1950 × 1328 | one blue family | no | no | packed, touching | light violin shading under dots, black whisker lines over them, brackets | categorical x, linear y |
| beeswarm-boxplot | 970 × 2374 | gray and red families; outliers with a dark ring | no | ring on some | packed, touching | boxes, dashed whiskers and reference lines, text | categorical x, linear y |

**bench1/, bench2/, bench3/** (added in phase 5): the user's JMP renderings of one data set each, with exact ground
truth (`data.csv` in data units; its header gives the scales, ticks and the plot area in px, whose edge pixels'
centers are the scales' min and max). bench1 has 100 dots with one coincident pair; bench2 the same styles with
204 dots, five coincident pairs and a tight pile of three; bench3 204 dots as beeswarms and jitter in two lanes
(x is the group). Some renderings are also JPEGs (4:2:0 chroma, typical export quality), and medium-dots-grid1/2
add gridlines. All are 624 × 515 with the plot area at (70, 30), 546 × 436 (bench3: 442 tall). The bench1 and
bench2 styles:

| rendering | what it tests |
|---|---|
| tiny-, small-, medium-, large-dots | one 50% family at r 2.6, 3.8, 6.7 (orange), 12.6 (green); the pair is the only overlap at small sizes |
| smoother, smoother-behind | 50% black dots and a blue smoother over or under them; black text on the dots' own color line |
| smoother-ci, smoother-behind-ci | a translucent blue CI band under or over orange dots |
| group, group-large, group-smoother(-ci) | red and blue translucent families; the pair is one of each (purple) |
| ellipse | a translucent density ellipse under orange dots |
| stroked-dots | light gray dots with a darker ring |
| medium-dots-grid1, -grid2 | gridlines under the dots |

bench3: dots-swarm, -packed, -random (jitter), and swarms with a box plot (its outlier marker drawn over a dot),
violin, grid, and an ANOVA mean with a translucent CI band over the dots.

dog-ages, zoomed, also shows that the image was upscaled from a lower resolution (blocky, slightly blurred edges),
that each additional overlapping dot darkens the green by a visibly constant step, that circle edges inside a
cluster are visible as arcs, and that there are a few small white specks where several edges meet
(antialiasing artifacts, so the model must tolerate small local misfits).

Two regimes follow:

- **Translucent dots** (dog-ages, economist, maybe fibers): overlaps are visible, so the count inside a blob is
  recoverable, even for coincident dots. This is where the generative fit earns its keep.
- **Opaque dots** (the beeswarms, dot plots): the top dot hides what's under it. Dots are mostly separate or
  touching, and the problem is packing: a cluster of touching circles of known radius. The same fit applies with
  alpha = 1 and a drawing order, but coincident dots are undetectable and the app must say so.

## 2. Concepts

### 2.1 Forward model (stage 5b)

A dot family k has fill color c_k, alpha a_k, radius r_k, optional outline (color, width), and the image has an
edge blur (PSF, Gaussian σ) from antialiasing and resampling. Dot i covers pixel p with fraction m_i(p) ∈ [0, 1]:
the exact area of the pixel square inside the disc, then blurred. Drawn with source-over compositing onto a
background B(p):

    I(p) = c + (B(p) − c) · Π_i (1 − a · m_i(p))          (one family)

For one family the product commutes, so **drawing order doesn't matter**. Where m is 0 or 1 (everywhere but the
edges), the log transform linearizes it:

    D(p) = log((I(p) − c) / (B(p) − c)) / log(1 − a) = Σ_i m_i(p)

D is the **overlap count image**: a sum of discs. Extraction from D is a sparse deconvolution with a known disc
kernel, which is a far better-posed problem than finding circles in the raw image. The fit uses the exact
forward model; D is for initialization, display and diagnosis.

For several families or opaque dots, order matters (over is not commutative across colors). The model then carries
a drawing order: by family (common: one series after another) or per dot (opaque beeswarms, where the visible
arc boundaries reveal which dot is on top). Outlined dots are a two-part template (ring and fill); a fill that
varies by dot (pathogen) is a per-dot color parameter.

### 2.2 Estimating the style (stage 4)

- **Background B:** the mode of the plot area's pixels, then a smooth field where it varies (a panel, a CI band,
  violin shading drawn under the dots), estimated from pixels not assigned to dots or masks.
- **Color and alpha from overlap levels:** pixels of a translucent family lie on the segment from B toward c,
  quantized at levels L_n = c + (B − c)(1 − a)^n for n overlapping dots. Projecting pixels onto the direction
  (L_1 − B) gives a histogram with peaks at L_0 = B, L_1, L_2, …; then 1 − a = (L_1 − L_2) / (L_0 − L_1) and c
  follows from L_1. Three levels (background, one dot, two dots) suffice; more levels overdetermine it and test
  the model. Fitting per channel checks that one alpha explains all three.
- **Compositing space:** most renderers (browsers, matplotlib, R's Cairo/Quartz devices) composite in sRGB
  values, not linear light. Measured in phase 0: sRGB makes the levels geometric within 8-bit rounding, and linear
  light does not (§8, phase 0).
- **Radius:** from isolated dots (components at level 1 that are nearly circular: area, and the radius of the
  best-fitting circle to the edge), from curvature of internal arcs, or from the first minimum of D's
  autocorrelation. Report the spread; a family with two radii (size-encoded scatterplots) is out of scope at
  first.
- **Edge blur σ:** from the edge profile of isolated dots (a blurred step); upscaled images like dog-ages have a
  larger σ, sometimes anisotropic blockiness that a Gaussian only approximates.
- **User confirmation:** the app shows each family's swatch, radius, alpha and σ with a sample isolated dot
  rendered next to a real one, and lets the user click a dot to seed or correct the estimate.

### 2.3 Layers and masks (stages 2–3)

Each pixel is classified as background, explained by a dot family (on its color segment, at any level), or other.
"Other" pixels get weight 0 in the fit, dilated by about the blur width so their antialiased edges don't count.

- **Lines** (fit curves, dashes, whiskers, gridlines, axis spines): thin relative to the dot diameter.
  Morphological opening with a disc somewhat smaller than a dot removes them from a dot mask; what the opening
  removes, and isn't the dot color, is a line.
- **Text**: compact components of a non-dot color, mostly outside the plot area or in a corner.
- **Shaded regions**: large smooth areas. If dots show through them as if composited over them, they are part
  of the background field; if they are over the dots (pathogen's purple band), they are a translucent layer of
  their own and the model composites them on top, or masks them where that's too uncertain.
- Gridlines and spines also feed stage 1.

A dot under a mask is only partly visible. Masked pixels don't penalize the fit, so the model can't hallucinate
dots there unless the visible part supports them; a dot whose visible fraction is below a threshold is flagged
as poorly determined rather than silently placed or dropped.

### 2.4 Fitting (stages 5a–5d)

- **Objective:** Σ_p w(p) ρ(I(p) − Î(p; dots, style)) + λ · (number of dots), with w = 0 on masks, ρ a robust
  loss (Huber, so specks and JPEG noise don't drive the fit), and λ the cost of a dot (an MDL/BIC-style penalty
  scaled by the pixel noise σ and the dot area). λ is calibrated on synthetic images so that a true dot is always
  worth adding and a phantom never is.
- **Candidates (5a):** peaks of D correlated with the disc kernel (a matched filter), with non-maximum suppression;
  a Hough vote with the known radius on D's edges, where each internal arc votes for its own center (this is what
  finds dots inside a blob); and, as an alternative, a sparse non-negative solve on a subpixel grid (D ≈ A·w,
  w ≥ 0, L0 by adaptive ridge as in lafspline). Phase 3 compares them.
- **Render and compare (5b–5c):** incremental. Each dot touches about π(r + 2σ + 1)² pixels, so moving, adding or
  removing a dot updates the model and residual on that patch only. Keep a per-pixel product (or, for opaque
  dots, a per-pixel top-dot index) so a dot's contribution can be divided out.
- **Refine (5d):**
  - Continuous: Gauss–Newton or Levenberg–Marquardt on positions, one dot at a time given the others (block
    coordinate descent) looped to convergence; jointly for small clusters that move together.
  - Discrete moves, accepted when the objective drops: **birth** where the residual says the image is darker
    than the model by about one level over a dot-sized patch; **death** of a dot whose removal costs less than λ;
    **split** of a dot into two nearby ones (coincident and near-coincident dots in dog-ages); **merge** of two
    dots that would fit as one.
  - Alternate with refitting the style (c, a, r, σ, B) with the dots fixed; a few rounds.
  - Later, if needed: simulated annealing or reversible-jump MCMC over the same moves, which also gives posterior
    uncertainty.
- **Uncertainty:** per-dot position standard error from the local Hessian; per-dot count ambiguity from how much
  the objective changes if the dot is removed or duplicated. Dots below a margin are flagged, and the summary
  reports the count as a range when flagged dots could go either way.

### 2.5 Scales (stage 1)

- **Plot area and axes:** long horizontal and vertical runs of a dark color (row and column projections), or the
  frame of a panel; gridlines are long light lines spanning the plot area at regular spacing.
- **Ticks:** short perpendicular runs at regular spacing along an axis, or the gridline positions.
- **Tick values:** first, the user types the values of two ticks per axis, with the app proposing a guess from
  regular spacing; later, optional OCR of the labels (open question 2).
- **Scale type:** linear, log (pathogen: 1, 2, 4, …, 64 equally spaced, plus a special 0 row) or categorical
  (beeswarms, dot plots: the output is the group, found by clustering dot x positions, and x within a group is
  jitter, not data). The user can override.
- **Output resolution:** one pixel in data units for each axis, shown with the results, since no extraction is
  more precise than that.

### 2.6 Validation

- **Synthetic graphs** (`synth.js`): known dots with chosen count, clustering, ties, radius, alpha, colors, outline,
  blur, upsampling, JPEG-like noise, and lines and text drawn over them. These give exact ground truth at any
  difficulty. Because the forward model and the generator share the rasterizer, the generator alone could hide a
  model error, so phase 1 checks it against images rendered by Chrome's canvas (and, if convenient, matplotlib
  and R) with the same dots.
- **Real images:** ground truth hand-annotated in the app's edit mode and saved to
  `testgraphs/truth/<name>.json`, local like the images (pixel positions, with a note on how confident the annotator was in dense
  clusters). For dog-ages, also look for the published data (open question 4).
- **Truth sets in data units** (`tools/truth-csv.mjs`, bench1–3): `data.csv` with `# x scale: linear, min=…,
  max=…`, `# x ticks: …` and `# plot area: left=…, top=…, width=…, height=…` header lines, then x, y and an
  optional group per dot. Every image of a set maps through the plot area (exact; detected ticks are off by up to
  0.25 px because JMP snaps them to whole pixels). Groups have no colors, so each found family is linked to the
  groups it explains and dots are matched within each linked set. When the x tick list is exactly the set of x
  values (bench3), x is a lane whose layout is unknown: found dots go to the nearest lane center and are matched
  by y alone. JPEGs are decoded with jpeg-js, which agrees with macOS on 97% of values but differs by up to 31
  levels at color edges (chroma upsampling); browsers may differ similarly.
- **Metrics** (`metrics.js`): match extracted to true dots by optimal assignment within a radius (r/2); report
  precision, recall, count error, position RMSE (px) and time. `npm run bench` prints the table; results go into
  this file as phases complete.

## 3. App

- **Image input:** choose a test image (from `testgraphs/index.json`), upload a file, paste from the clipboard
  (Cmd-V anywhere on the page), or drag and drop. Very large images are processed at full resolution but displayed
  scaled.
- **Steps panel** following the pipeline, each with its findings and a way to correct them: Axes → Layers →
  Dot style → Fit → Review. Each step can run automatically and be revisited.
- **View:** canvas with zoom and pan; overlay toggles for the original, the mask, the overlap count image D, the
  rendered model, the residual (signed, with a legend), and the extracted dots (circles, flagged ones marked).
  Hovering a dot shows its pixel and data position, standard error and flags.
- **Tuning:** pick a family's color with an eyedropper, set radius, alpha and blur, paint mask additions or
  removals, set the dot cost λ.
- **Manual edits:** click to add, select and delete, drag to move; "refit nearby" reoptimizes the neighborhood.
  The same edit mode creates ground-truth files.
- **Fit progress:** the worker posts intermediate dot sets, so the view animates the iteration and can be
  stopped.
- **Output:** a table and CSV/JSON export (pixel x, y, data x, y, family, standard errors, flags, plus the
  calibration and style), copy to clipboard. Fine print: pixel resolution in data units, the number of flagged
  dots, and the fraction of plot-area pixels masked.
- URL state (image, settings), light and dark themes, an About panel.

## 4. Alternatives considered

- **Hough circles / LoG blobs / template matching on the raw image:** fail in translucent clusters (the reason
  for this project) and can't count coincident dots; kept as initializers on D, where they work far better.
- **Distance-transform watershed:** splits touching opaque dots reasonably; useful as an initializer for the
  beeswarms, not for translucent clusters.
- **A trained CNN detector:** needs training data, a model file and an inference runtime (a dependency), and its
  failures are hard to explain. The synthetic generator could train one later if the generative fit falls short.
- **Pure sparse deconvolution on a grid:** elegant (linear in D, convex with an L1 penalty), but grid-bound and it
  ignores the edge nonlinearity; good for candidates, followed by the continuous fit.

## 5. Open questions

1. ~~Test image licensing.~~ **Decided:** `testgraphs/` stays local and git-ignored (all but its README), with
   everything derived from the images (ground truth, crops, `index.json`). The deployed site will offer synthetic
   examples instead, and a benchmark on the real images runs only on machines that have them.
2. **OCR for tick labels.** Still open after phase 4: label crops beside the value boxes make typing quick.
   tesseract.js would be a runtime dependency (large, with a WASM file); a small
   recognizer for digits, minus signs, decimal points and commas (single font per image, so glyphs can be
   clustered and the user labels a cluster once) would not. Start with typed values; decide after phase 4.
3. ~~Compositing space.~~ **Decided: sRGB values** (phase 0 measurements below). Recheck against canvas renders in
   phase 1.
4. **Ground truth for dog-ages:** is the underlying data published? If not, hand annotation, with the dense
   clusters marked as uncertain.
5. **Dot cost λ:** derived from the noise σ and dot area (MDL), or calibrated on the synthetic set? Probably the
   first, checked by the second.
6. **Drawing order** for several translucent families (economist red over blue): by family, estimated from
   overlaps of different colors, or per dot?
7. **Size-encoded dots** (bubble charts): out of scope for now; revisit if a test image needs it.
8. ~~License.~~ **Decided: 0BSD**, like lafspline.

## 6. What to borrow from the sibling projects

- From **lafspline**: the no-build static layout (`index.html`, `styles.css`, `src/core`, `src/ui`, a module
  worker), `tools/check-page.mjs` (headless-Chrome check with screenshots, dependency-free), the `index.json`
  approach for servers that don't list directories, `rng.js`, theme and URL-state handling, the habit of
  recording measured results and decisions in PLAN.md, and display honesty (every result says how well it is
  determined).
- From **comma**: `npm start`-style local serving, the GitHub Pages workflow (`.github/workflows/deploy.yml` with
  the build number), clipboard and drag-and-drop input handling, and node:test with thorough tests of pure
  functions plus sanity checks for stochastic ones.

## 7. Planned layout

```
index.html, styles.css
src/core/image.js       RGBA → planes, color conversions, sRGB/linear, histograms, crops
src/core/render.js      exact pixel coverage of a disc and ring, blur, compositing, incremental patch updates
src/core/style.js       background, family candidates, overlap levels → α, c, level table, count image D, radius, blur
src/core/layers.js      pixel classes, distance transform and morphology, thin marks, mask (later: background field)
src/core/fit.js         objective, Hough candidates, compass refinement, births/deaths/merges, radius refit, assess
src/core/axes.js        lines, ticks, labels, gridline ticks, scale fitting, pixel ↔ data
src/core/synth.js       synthetic graphs with known dots
src/core/metrics.js     assignment, precision/recall, RMSE
src/core/rng.js         seeded random numbers
src/ui/                 main.js, view.js (canvas, zoom), overlays.js, input.js; later steps.js, export.js, urlstate.js
src/fit-worker.js
testgraphs/             images, index.json, truth/*.json (local only: git ignores all but the README)
tools/                  check-page.mjs, png.mjs, images-index.mjs, measure-dots.mjs, bench.mjs, render-canvas page
test/                   *.test.js; PNG decoding in tests via tools/png.mjs (pngjs dev dependency)
```

## 8. Phases

The extractor works in pixels until phase 4: scales are well understood and mostly mechanical, while the
translucent-dot fit is the risk, so it comes first. Each phase ends with tests, the benchmark, a clean page check
with a screenshot, and a stop for review.

### Phase 0 — Scaffold and risk checks
- [x] `package.json` (`"type": "module"`, scripts, `pngjs` 7.0.0 dev dependency), `.gitignore` (test images
      local, open question 1), `index.html` and `styles.css` (light and dark), `tools/check-page.mjs` and
      `src/core/rng.js` copied from lafspline (port changed to 8003), `tools/png.mjs` (PNG read/write for tools and
      tests), `tools/images-index.mjs`, LICENSE (0BSD); the placeholder `index.js` is gone.
- [x] Image input in the app: test image list (Python's directory listing, else `testgraphs/index.json`), file
      picker, paste, drag and drop; canvas view with zoom at the pointer, pan, Fit and 1:1, pixel readout.
      Decoding uses `createImageBitmap(…, { colorSpaceConversion: 'none' })`, so the values are the file's: the
      browser's pixel hash equals pngjs's for all nine test images. Checked headless: key and wheel zoom, the
      hover readout (191, 226, 198 at a dog-ages dot center), paste and drop.
- [x] **Risk check** (`npm run measure -- <image> [--seed r,g,b] [--out dir]`, `tools/measure-dots.mjs`). Results
      below; the §2.1 model holds.

**Measured (phase 0).** Levels are histogram peaks of flat pixels (3×3 neighborhood within 3 units) projected on
the line from the background toward a seed color; the geometric model L_n = c + (B − c)(1 − α)^n is fitted with
one α for all channels.

| family | levels found (t along the line) | α (sRGB) | fit rms / max, sRGB | fit rms / max, linear light (×255) | radius (isolated dots) | edge σ |
|---|---|---|---|---|---|---|
| dog-ages green | 90, 151, 194, 223, 244 (5 levels) | 0.310 | 0.36 / 0.71 | 3.53 / 7.43 | 24.04 px (4 dots, 23.92–24.10) | 1.16 px |
| economist red | 82, 139 | 0.302 | 0.05 / 0.09 | 0.25 / 0.47 | 8.37 px (12 dots, 8.30–8.39) | 0.77 px |
| economist blue | 84, 142, 185 | 0.289 | 0.34 / 0.72 | 1.98 / 4.49 | 8.35 px (8 dots, 8.32–8.40) | 0.75 px |

- **sRGB compositing, constant alpha:** the sRGB fits are within 8-bit rounding (max 0.7 units); linear light
  misfits by up to 7 units. Open question 3 is settled.
- **The first step is a little large:** fitting levels 1 and up alone implies a background of about 252 rather
  than 255 in dog-ages and economist blue. That's 2–3 units, small but systematic. The forward model can absorb
  it with an empirical level table, or by fitting the first level's color separately from the geometric tail.
- **Deep stacks:** in dog-ages, pixels sit at predicted levels 6 (225 px) and 7 (123 px). Beyond about 8, each
  step is under 7 units and shrinks by 0.69 per level, so counts in the x ≈ 0 stack will be uncertain, and the app
  must say so.
- **Lines in the dot color:** the economist trend lines are the dots' full color c (red: within 0.8 units of the
  fitted c; blue: 10.6, from fewer level-3 pixels). An opaque mark at the family's c is a separate layer to mask,
  and it confirms c independently.
- **Small dots need flat-pixel voting:** at r ≈ 8 px, antialiased edge pixels outnumber interiors and made false
  level peaks until only flat pixels voted. Near the background, the off-line tolerance must also be relative to
  t: one-step-from-white grays passed an absolute tolerance and created a false first level.
- **Radius:** whole isolated dots (aspect ≤ 1.05, moment radius within 3% of the area radius) agree to ±0.1 px.
  Partly covered dots fail those tests and are excluded rather than biasing the radius.
- **Blur and upscaling:** edges are erf-shaped with σ ≈ 0.75–1.2 px (all the test images are macOS screenshots,
  tagged sRGB, no ICC profile). dog-ages looks blocky when zoomed, but its pixel changes show no integer-
  replication phase, so it was resampled smoothly by a non-integer factor; a Gaussian edge model fits it.

### Phase 1 — Forward model and synthetic data
- [x] `render.js`: exact disc coverage per pixel (closed form: quarter-plane areas of the disc, combined over the
      pixel's corners), sRGB compositing for one family, Gaussian blur of the composited image, and
      `createDotModel` with incremental add, move and remove over a spatial grid. Tests: coverage sums to πr²
      (1e-9) and matches 400×400 supersampling (2e-4), mirror symmetry, overlap levels, a blurred edge has
      σ² + 1/12 (the pixel box), random edits equal a full rerender (1e-3, with and without blur), and
      `influenceRect` bounds every changed pixel.
- [x] `synth.js`: `scatterPoints` (uniform, curve, clusters; ties with jitter), `synthGraph` (families
      composited in order, solid or dashed lines over the dots, native resolution then bilinear upsampling,
      blur, noise, 8-bit rounding), presets `sparse`, `dense` and `curve` (a dog-ages imitation, 832 × 1078,
      rendered at half size; 0.65 s). `metrics.js`: `matchDots`, optimal one-to-one matching within a tolerance
      (Hungarian on connected groups), with precision, recall, count error and RMSE. Tests: determinism, dot
      centers exact under upsampling by 2 and 1.7 (0.02 px), ink conserved by upsampling (0.5%), dashes, the
      Hungarian against brute force, a cluster where greedy matching fails, coincident dots.
- [x] Agreement check against Chrome's canvas (`tools/render-check.html`; run it through `check:page` with
      `--until 'window.renderCheckResult'`). It draws the same dots with `arc` and `fill` and compares in the
      page, since `src/core` runs unchanged there.

**Measured (phase 1).**

- **Synthetic round trip:** `measure-dots` on the `curve` preset recovers α = 0.311 (true 0.31), c = 50.3,
  161.1, 73.2 (true 50, 161, 73), r = 23.94 px (true 24) and σ = 1.22 px (true 1.17: 2× bilinear upsampling plus
  0.6 blur). On `dense`, α 0.300, r 10.94 (true 11) and σ 0.83 (true 0.85). The tool's radius reads 0.06 px
  low. Real dog-ages measured σ = 1.16, so the upsampling imitation is realistic. The synthetic levels imply
  a background of 255.1, so dog-ages' oversized first step (phase 0) comes from whatever rendered it.
- **Model vs Chrome canvas** (headless Chrome, software raster), five cases with r from 3.3 to 24 px and α from
  0.3 to 1:
  - Interior pixels more than 1 px from any edge: max |diff| 1.47, mean ≤ 0.21 (0–255). That's 8-bit
    rounding, which accumulates with depth: Chrome's stacked levels run darker than the exact geometric series
    by 0.5 (1 dot), 1.3 (3), 2.1 (5) and 2.5 (8 dots).
  - Edge pixels: Chrome's coverage is close to exact area but deviates by up to about ±0.15 (25–50 units in
    single pixels, mean 3.6–7.8). The deviations are largest along near-horizontal edges, consistent with
    coarse vertical subsampling, and the antialiasing reaches about a pixel past the exact disc. The effect on
    ink is small: opaque dots have an effective radius 0.08 px smaller (r 8.37 and 24.04) or 0.02 px larger
    (r 2.5).
- **Consequences for phases 2 and 3:**
  - The level colors should come from the data, not from a strict geometric series. Measure each level
    present, and extrapolate geometrically beyond them. The model maps the effective count
    n = log T / log(1 − α) through that table, which is exact on plateaus and smooth at edges.
  - The fit must expect edge residuals of a few tenths of a level that no position change can remove. Use a
    robust loss, or weight edge pixels by their expected error, and let the fitted radius absorb the ~0.1 px
    bias.

### Phase 2 — Style and masks for one translucent family
- [x] `style.js`: background (mode color); family candidates by clustering flat pixels on direction from B (2°
      bins), each measured along its color line; levels, the geometric fit (golden-section search on α, c in
      closed form), an empirical level table (B, the measured levels, then 12 geometric levels); the overlap count
      image D; isolated whole dots for the radius; the erf edge fit for the blur (the pixel box's 1/12 variance
      removed, since the renderer supplies it). `render.js` takes the table as `levels`, interpolating linearly in
      transmittance: exact on plateaus, and identical to the formula when the table is geometric (test).
- [x] `layers.js`: classes (background, dot color, other), exact Euclidean distance transform (Felzenszwalb),
      disc dilation, erosion and opening, thin marks in the family's full color, and the mask (other ∪ thin,
      grown by 1 + 2·blur px). dog-ages: every black and dash-gray pixel in the plot area is masked, and the
      dots and background are not (test).
- [x] UI: Dot style (family menu, Pick a dot's color, level swatches with measured and extrapolated ones
      distinguished, full color, fit, isolated count, editable radius, α and blur with Reset, notes, and an isolated
      dot beside its model rendering and their difference) and Layers (overlays Image, Count, Mask and Classes
      with legends, mask growth, class and mask statistics, timing). The hover adds the pixel's count and class.

**Measured (phase 2).**

| image | family | α | radius | blur | isolated dots | levels | time (style + mask, page) |
|---|---|---|---|---|---|---|---|
| synth sparse | blue | 0.347 (0.35) | 10.001 (10) | 0.14 (0) | 15 | 2 | |
| synth dense | green | 0.300 (0.30) | 11.007 (11) | 0.80 (0.80) | 3 | 5 | |
| synth curve | green | 0.310 (0.31) | 23.992 (24) | 1.20 (upsampled) | 1 | 4 | |
| dog-ages | green | 0.310 | 24.03 | 1.15 | 4 | 5 | 122 + 55 ms |
| economist | red (1st of 2) | 0.302 | 8.37 | 0.72 | 9 | 2 | 198 + 79 ms |
| economist | blue (2nd) | 0.288 | 8.38 | | | 3 | |

The opaque test images are found as one-level families and say so, along with the fact that no isolated whole
dot was found (their dots touch); they are phase 5.

What the real images taught:

- **Text can look like a family:** on the economist chart, black and gray text gave two plateaus with a
  shrinking step. Its fitted c was −204, outside the RGB cube. A translucent family now needs a fit within 2 units
  rms and c within 10 units of the cube.
- **Rims are off the line in resampled images:** dog-ages' dot rims sit up to perp/t ≈ 0.14 off the color line,
  which made 3 of its 4 isolated dots look partly hidden. The count image accepts perp/t ≤ 0.2 (gray text is at
  0.31) and any perp ≤ 2 units. Doing so also removed the 0.06 px low radius from phase 1.
- **Lines in the dot color are wide:** the economist trend lines are about 9 px wide against r = 8.4, so the
  opening needs radius 0.75 r. "Deep" must mean saturated (within 8 units of the line's measured color, which
  is a better anchor than the extrapolated c), not just beyond the measured levels. Otherwise the core of a deep
  pile is taken for a line (synthetic curve: 867 px). Thin components must also be at least 3 r long.
- **dog-ages' dots are stair-stepped:** the isolated-dot sample shows blocky rims against the model's smooth
  ones (mean difference 1.39, max 27.8, 0–255). The figure looks enlarged by nearest-neighbor resampling at a
  non-integer factor, which fits phase 0 seeing no integer-replication phase. The fit will see rim residuals that
  no subpixel position removes, and should weight rims down (as the canvas check also suggested).

### Phase 3 — The fit
- [x] Candidates: Hough vote with the known radius on D's gradient (`houghCenters`), then births from a box
      filter of the residual (the matched filter of §2.4, applied to what the model doesn't yet explain). The
      sparse grid solve wasn't needed: Hough plus births reach recall 1.000 on the synthetic set, so it is left
      out.
- [x] Refinement (`createFit`, `fitDots`): compass search per center over active dots, births, deaths and merges,
      a global radius refit, and `assess` (position standard errors from the curvature, removal margin in λ,
      visible fraction, depth; flags weak, hidden and deep). There's no separate split move, because births at a
      dot's own site do it: a coincident pair is born as a second dot. Blur and α are not refit; the style
      estimate's values are used.
- [x] Worker (`src/fit-worker.js`) with progress drawn live and Stop (a stopped fit keeps its dots, marked
      unassessed); Dots panel (count, flags, fitted radius, residual sd, Copy CSV, Save truth JSON); Model and
      Residual overlays with a legend; Edit mode (click to add or remove, then `refitNear`); dot readout on hover.
- [ ] Ground truth for dog-ages: **needs a person**. The app's Edit mode and Save truth write
      `testgraphs/truth/dog-ages.json`, which `npm run bench` then scores. The fit's own output must not be used
      as truth without review.

**Measured (phase 3)**, `npm run bench` (5 seeds per preset, matching tolerance r/2), Node 22 on this Mac:

| preset | dots | precision | recall | count error | position RMSE | Hough alone (P/R) | time per image |
|---|---|---|---|---|---|---|---|
| sparse (400 × 400, r 10, few overlaps) | 30 | 1.000 | 1.000 | 0 | 0.013–0.020 px | 1.00/1.00 | 0.2–0.3 s |
| dense (500 × 400, r 11, clusters, 8% ties) | 150 | 1.000 | 1.000 | 0 | 0.061–0.116 px | 0.97–1.00/0.78–0.88 | 2.0–2.2 s |
| curve (dog-ages imitation, 832 × 1078, r 24) | 104 | 1.000 | 1.000 | 0 | 0.096–0.278 px | 1.00/0.56–0.67 | 7.0–9.7 s |

Real images, no ground truth yet:

- **dog-ages:** 95 dots in 8.4 s (fitted r 23.92, residual sd 1.28). Flags: hidden 42 (under or beside the
  curve and dashes), deep 15 (the x ≈ 0 pile and the top cluster). Hough gave only 20 starting points (the
  stair-stepped rims weaken the edge votes); births found the rest. In the residual overlay, every dot interior
  is white. What remains is rim fringes from the nearest-neighbor enlargement: mean |residual| 3.8 units over
  unmasked ink pixels, 12% of them beyond 10.
- **dog-ages-actual-size** (288 × 372, added during phase 3): the native rendering. α 0.332 from only 2 levels,
  r 7.99, blur 0.35. **Cross-check** (`node tools/compare-fits.mjs testgraphs/dog-ages-actual-size.png
  testgraphs/dog-ages.png`): dog-ages is this image enlarged exactly 3.000× (x 3.0002, y 3.0007) and cropped by
  (8.1, 4.7) px. Fitted independently, both give 95 dots, and all 95 match one to one, none extra on either side.
  The 41 dots unflagged in both agree within 0.072 px RMSE (native px; max 0.149), and all 95 within 0.163. Both
  fits could share a systematic error (the deep pile's count above all), so this doesn't replace ground truth, but
  it is strong evidence for the count and the positions.
- **economist, red:** 110 dots in 3.5 s; separated dots are matched cleanly. Where the red and blue lines and
  dots overlap at the left, it builds clumps of 5–10 red dots flagged deep or weak. That is a red–blue mixture
  explained as deep red; it needs the second family (phase 5).

What the fit needed, found on the way:

- **Depth-aware birth units and λ:** births first missed dots in deep stacks. At depth n one more dot changes
  the color only (1 − α)^n as much as the first, but the residual was measured in first-level steps and λ was 20
  pixels one first step wrong. Recall on dense rose from 0.97 to 1.000 once both were made local: the residual
  in units of the next level at each pixel, and λ = 20 px × 10 units (absolute).
- **A neighborhood decides a birth:** while a dot is missing, its neighbors drift to cover it. Adding the
  missing dot alone raised the loss (dog-ages: +13 000); with the neighbors refined it fell by 44 000. Births are
  now judged after the neighbors re-settle, within half a radius, under a snapshot that undoes a rejected
  birth. curve's position RMSE fell from 0.6–1.0 px to 0.1–0.3, and its last misses went.
- **Single-level families:** with no overlaps, α isn't measured. The level table once repeated the single level
  12 times, so every dot read as 13 deep and the radius was lost (sparse #3: 137 dots at r = 1). It is now kept
  at two entries (regression test).
- **Speed:** 21–55 s on curve at first. Per-dot coverage stamps, snapshot/undo instead of re-rendering a
  rejected trial, rendering only the value along the color line (blur commutes with the projection), sweeps over
  active dots only, and incremental birth search brought it to 7–10 s. Most of what's left is the blur and
  re-render per trial move. A Newton step per dot could roughly halve the trials.
- **Not done:** deaths and merges are still judged without re-settling neighbors, which errs toward keeping
  dots. A dot almost entirely under a mask (12 of 448 px visible in a test) is not found, which is correct
  given λ, but it is not reported either.

### Phase 4 — Scales
- [x] `axes.js`: background colors (any color on ≥ 5% of the image, so panels count), gray ink, long thin straight
      lines (`findLines`), ticks along dark lines (`findTicks`: dark-cored perpendicular strokes, the majority side,
      corner strokes kept only with a label), label words (`findLabels`), gridlines standing in for ticks with a
      baseline in step and labels at the lines' common end, and `fitScale`, `toData`, `toPixel`, `resolution`,
      `guessValues` (linear and log). Tests: synthetic L axes (lines and ticks within 0.2 px, label boxes),
      gridline-only y axis with a baseline, scale math, guesses, and the test images (regressions below).
- [x] Axes panel (`src/ui/axes-panel.js`): per axis a type menu (linear, log, categories), a row per tick with its
      label crop, a value box with guesses after two values, the pixel position, and a summary (data units per px,
      typed ticks' max distance from the fitted scale, flagged beyond 1 px); values remembered per image (pixel
      hash) in localStorage; axis and tick marks drawn on the view. Data coordinates in the Pixel readout, the dot
      hover and the CSV (`x`, `y`, `se_noise_x`, `se_noise_y`).
- [ ] OCR of tick labels (open question 2) is still deferred; the label crops make typing quick.

**Measured (phase 4).** Detection takes 5–125 ms.

| image | x axis | y axis |
|---|---|---|
| dog-ages | 6 ticks, 88.94 px apart, labeled | 6 ticks, 147.61 px apart, labeled |
| dog-ages-actual-size | 6 ticks, 29.63 px | 6 ticks, 49.20 px |
| fibers | 4 ticks | 7 ticks |
| pathogen | 5 ticks | 8 ticks, log (64 … 1, and a 0 row), uneven |
| economist | 11 ticks, labels between them (left unassigned) | no line: 3 gridlines plus the zero baseline, labels at their right ends |
| Laser | 3 category ticks (rotated labels only partly boxed) | 6 ticks, including the zero line crossing the axis |
| cantons | 4 category ticks | 5 ticks |
| violin | 4 category ticks | 5 ticks (frame) |
| beeswarm-boxplot | none (no bottom axis line) | 9 ticks (top label cut off by the image edge) |
| nicest language | 6 ticks (axis inside a beige panel) | none (no y scale) |

What the test images taught:

- **Thin antialiased axes are light on average.** cantons, violin and beeswarm-boxplot axes averaged lighter than
  the dark threshold; a line's luminance is now its darkest row's. Ticks may be any gray but need a dark core:
  pathogen's light gridline stubs inside the frame were taken for ticks.
- **Words, not glyphs:** labels are merged into words before being assigned to ticks. Glyphs merge only if most
  of their heights overlap and their heights are within 2.5×. Ink touching the image border never joins a label
  (pathogen: a 1 px border line chained all x labels and the title into one box). A word must sit in the nearest
  column or row of labels (rotated axis titles in cantons and pathogen).
- **Gridline labels share an end:** economist's zero line took a dot rim at its left end until each gridline
  took its label from the end most gridlines use.
- **Tick positions are only as good as the renderer's pixel snapping.** dog-ages-actual-size's ticks are 29.3–30.0
  px apart, so a scale from two typed ticks can be off by a few tenths of a px (enlarged dog-ages: 0.59 px with 0,
  3 and 15 typed). Typing every tick averages it out; the summary shows the residual.
- **The dots' standard errors understate the real error.** Fitted separately and calibrated from their own
  ticks, dog-ages and dog-ages-actual-size agree on all 95 dots within 0.011 years RMS in x and 0.052 in y
  (0–100). That's about 0.1 native px, against noise-only standard errors of 0.0005 and 0.0018 (big) and 0.0027
  and 0.011 (native). The app labels them "noise only" and says the real error is a few tenths of a px.
- **Native or enlarged dog-ages:** the native image fits in 2.4 s against 8.2 s, and its ticks are cleaner, but it
  shows only 2 overlap levels against 5. So α is less certain (0.332 against 0.310) and more dots are flagged
  deep (44 against 15). In data units the results agree to the precision above, so neither is more accurate.

### Phase 5 — Generalize to the other test images
Done in milestones, each stopped for review.

**Milestone 1: opaque dots and several families.**

- [x] Opaque families (`style.js`): color clusters (8-unit bins, 3×3×3 smoothing) rather than directions, since a
      swarm's dots and the violin shading under them share a hue. Each cluster's pixels are read as coverage against
      the background where they are: the nearest pixel of a background color or another family's color
      (`nearestTransform`, a feature transform in `layers.js`). Its usual background is the most common color around
      its thick parts. Radius comes from isolated dots, else distance-transform peaks; blur from isolated dots, else
      the boundary profile. A cluster is a dot family if r ≥ 3, at least 3 filled centers see 75% of their edge, and
      either the discs explain ≥ 35% of its pixels or it has ≥ 3 isolated whole dots. Translucent families come
      first, then opaque ones; antialiased blends of a family with its background are not families.
- [x] The fit for opaque families compares coverage (in color units) with the renderer's two-entry table at α = 1
      (`createFit` takes `imgT`). An opaque dot must be at least 15% visible and have a filled interior (85% of its
      visible interior at ≥ 0.75 of the dot color). Thin marks for opaque families: every family pixel is a
      candidate, opened together with other marks (which may hide the rest of a dot), with 1 px slivers removed.
- [x] Several families: each fitted in turn, the others masked (they are off its color line, or other colors).
      The worker takes a list; the panel shows per-family counts and radii; dots carry their family and radius;
      edits, the model and residual overlays act on the current family; the CSV gets a family column.
- [x] Synthetic `beeswarm` preset (three opaque swarm columns, touching dots, median lines over them) and
      family-aware benchmark rows.
- [x] Drawing order and compositing of several *translucent* families (economist: red over blue): unmixed pixel
      by pixel (`unmix.js`, below), each family then fitted alone on its own image.

**Measured (milestone 1).** Benchmark, 5 seeds each: sparse, dense and curve are unchanged (precision and recall
1.000). beeswarm (3 opaque families, 84 dots) has precision and recall 1.000, count error 0, position RMSE
0.028–0.033 px, 0.6–1.0 s. Real images, no ground truth:

| image | families found | dots | time |
|---|---|---|---|
| cantons | 4 (r 6.36–6.39) | 60 + 60 + 60 + 59 = 239, none flagged | 2.4 s in the page |
| Laser | 3 (r 5.69) | 32 + 23 + 25, 1 flagged | 7 s |
| nicest language | 4 of 5: gray, pink, dark blue, red (r 8.4–8.5) | 69 + 61 + 10 + 9, 2 flagged | 7.7 s |
| violin | 1 (r 7.57) | 734, 17 flagged | 19 s |
| economist | 2 translucent | 110 + 85, 90 flagged (see above) | 4.8 s |
| dog-ages | 1 translucent | 95, unchanged | 8.7 s |

Not yet: nicest language's light blue dots (soft edges: only 2 centers see 75% of their edge; Pick finds them
manually). beeswarm-boxplot and pathogen (outlined dots, milestone 2). fibers is found as an opaque family on its
band (milestone 3 will treat the band).

What the images taught, in the order it came up:

- **Text shares dot colors.** Black and gray text (and its antialiasing) form color clusters, and round glyphs pass
  a Hough vote. Text is told apart by hollowness (a filled-disc test on each center), thinness (thick parts for the
  background, distance peaks ≥ 3 for the radius), coverage, and in the fit by the filled-interior test (Laser's
  black family drew 118 dots on labels and bars before it, 23 after).
- **Blends look like coverage.** The antialiased rim of one family's dots, or a lighter family that is a blend of
  another with the background (nicest's pink is its red at 0.47 on beige), reads as partial coverage. Remedies:
  blend clusters are not families; off-mix tolerance relative to the distance along the mix (12%, with a floor of
  2.5× the family's color noise for compressed images); other families' colors away from the family are unknown;
  dots must be filled at 0.75 of the color and at least 15% visible.
- **A line over a dot is not its background.** Only background colors and other families' colors may be a local
  background; otherwise a median line across a dot read as "no dot" and the dot failed the fill test.
- **A cut dot is not a thin mark.** Opening the family together with other marks keeps half-discs on either side
  of a line as dots.
- **Thresholds are close on small families.** violin's coverage is 0.43–0.44 and nicest's red has 3 strong
  centers. The family test is a heuristic tuned on these images; Pick remains the way to add a family it misses.
- The test suite now takes about 20 s (the opaque fits, Laser among them).

**bench1: ground truth, and what it changed in earlier phases.** `npm run bench` now also scores every bench1
image (`--set bench1`, `--only name`). First run: tiny-dots crashed, every full image lost its axes, smoother drew
15 dots on its text in 12 s, and the 98s missed the coincident pair. Fixes, each with a regression test:

- **Axes (phase 4).** JMP draws axes and ticks in gray 165, above the dark threshold of 160 (other figures: axes
  0–125, gridlines 194–244): now 180. A tick axis needs two labeled ticks (beeswarm-boxplot's gray bracket at 160
  became an x axis otherwise), and unlabeled ticks beyond the other axis are dropped (the y label "30" crossing
  the x axis line left of the corner made three).
- **Small dots (phase 5).** r ≥ 2 px counts as a dot family when at least 5 isolated whole dots show it round
  (text clusters measure r 2.5–2.74 and never have one); isolated dots may be 12 px² (r 2.6 is 21 px²).
- **Weak second level (phase 2).** Sparse translucent dots overlap in one pair, whose second level is 2 flat
  pixels at r 2.6 and 12 at r 3.8. With one level found, a peak of ≥ 2 pixels at t₂/t₁ in [1.05, 1.95] and at
  least the level separation beyond the first is the second level. (Without the separation, shoulders of the first
  peak at ratios 1.05–1.07 made light shading on four compressed images a "translucent family".)
- **Text in the family's color (phase 2 masks).** A pixel at count ≥ 1.5 is inside a dot, so a disc of 0.9 r
  around it must lie in the family's pixels or the mask; text strokes fail that and are masked. Pixels masked
  anyway count as support (dog-ages' line halos read as background and lost one dot until they did). On the
  economist this also masks a stretch of the blue trend line that isn't saturated: its blue family drops from 85
  to 78 dots, and the 10 dots it lost were phantoms on red dots and on the line.

**Measured (bench1, first version).** Positions were scored two ways: interior images by their bounds, full images
through the detected ticks. JMP snaps ticks to whole pixels, so the tick scale is off by up to 0.17–0.25 px at the
ends; scored with the exact mapping, the full images' fits matched the interior ones (tiny 0.018, medium 0.028 px).
Every full image held its interior pixel for pixel at (70, 30) except the axis lines over its left column and
bottom row, so the interiors were dropped for a plot-area header line.

| rendering | found / 100 | recall | RMSE interior / full (px) | time (full) |
|---|---|---|---|---|
| tiny-dots | 100 | 1.000 | 0.018 / 0.227 | 0.7 s |
| small-dots | 100 | 1.000 | 0.019 / 0.134 | 0.4 s |
| medium-dots | 100 | 1.000 | 0.028 / 0.135 | 0.6 s |
| large-dots | 100 | 1.000 | 0.028 / 0.135 | 0.9 s |
| smoother, smoother-behind | 100 | 1.000 | 0.032 / 0.137 | 0.6 s |
| group, group-large, group-smoother | 98 | 0.980 | 0.021–0.029 / 0.133–0.135 | 0.8–1.2 s |
| smoother-ci, smoother-behind-ci | 80, 83 | 0.800, 0.820 | 0.023–0.041 / 0.13 | 0.9 s |
| group-smoother-ci | 131 interior, 77 full | 0.91, 0.77 | 0.043 / 0.134 | 1–2.3 s |
| ellipse | 194 | 0.970 | 0.017 / 0.126 | 1.5 s |
| stroked-dots | no radius | – | – | – |

Still failing, and where they belong: the group pair (red over blue: translucent multi-family compositing, above);
dots on the CI bands and the ellipse (milestone 3: the ellipse's orange-on-lavender and orange-on-white become two
families, each fitting every dot); stroked dots (milestone 2).

**Measured (bench1–3 through the plot area).** `npm run bench -- --seeds 0` (no code changes to src/core since
the table above; the full images now score as the interiors did). Found / truth, recall, RMSE px:

| rendering | bench1 (100) | bench2 (204) | bench3 (204, y only) |
|---|---|---|---|
| tiny-dots | 100, 1.000, 0.018 | 207, 1.000, 0.032: 3 extra in the "100" label below the plot | |
| small-dots | 100, 1.000, 0.019 | 204, 1.000, 0.030 | |
| medium-dots, -grid1, -grid2 | 100, 1.000, 0.028–0.029 | 204, 1.000, 0.038–0.044 | |
| large-dots | 100, 1.000, 0.028 | 204, 1.000, 0.042 | |
| smoother, smoother-behind | 100, 1.000, 0.032–0.034 | 204, 1.000, 0.034–0.035 | |
| group, group-smoother | 98, 0.980, 0.021–0.023 | 193, 0.946, 0.047–0.049 | |
| group-large | 98, 0.980, 0.029 | no radius (no isolated dot) | |
| smoother-ci, smoother-behind-ci | 80 / 83, 0.80 / 0.82 | 290 / 180, 0.853 / 0.868, precision 0.60 / 0.98 | |
| group-smoother-ci | 77, 0.770, 0.023 | 163, 0.799, 0.047 | |
| ellipse | 194, 0.970, 0.017 | 178, 0.667, 0.353 | |
| stroked-dots | no radius | no radius | |
| dots-swarm, -grid, -violin, -box | | | 204, 1.000, 0.014–0.016 |
| dots-packed, dots-random | | | 204, 1.000, 0.017 / 0.019 |
| dots-swarm-anova | | | 185, 0.907, 0.039: the 19 missed are under the CI band |
| JPEG: smoother | 97, 0.970, 0.025 | 133, 0.652, 0.300 | |
| JPEG: medium-dots | no family | 129, 0.627, 0.091 | |
| JPEG: large-dots, group-smoother, smoother-ci | no family or no radius (all three) | no family or no radius (all three) | |
| JPEG: dots-packed, dots-random | | | 204, 1.000, 0.017; no family |

New findings:
- **JPEG** is its own problem: on 8 of the 12 JPEGs, style estimation finds no family or no radius, where the
  PNGs work; where it does fit, recall drops (bench2 smoother 0.652, medium 0.627). bench1 smoother.jpg's 50%
  black reads as opaque, so its overlap levels aren't found. Opaque gray dots-packed.jpg is exact. Fixed below.
- **Dots outside the plot area:** text below the axes can make dots (bench2 tiny-dots: the zeros of "100"). When
  axes are detected, dots beyond them should be dropped or flagged. Fixed below.
- **bench2 group** misses 11, each within 6 px of a dot of the other color (multi-family compositing);
  group-large finds no isolated dot for the radius at 204 dots.
- **bench2 smoother-ci** finds 290, 116 of them unmatched (phantoms on the band), where bench1 only missed dots:
  milestone 3.
- **Beeswarms work:** bench3 swarms, packed and jittered layouts are exact apart from the dots under the ANOVA band.

**JPEG.** Diagnosis, on the bench JPEGs against their PNGs:

- *Flat pixels vanish.* Inside isolated dots the RGB values are 3–9 units (rms per channel) from the PNG's, so
  few 3×3 neighborhoods pass the flat test (FLAT_TOL 3): 6 of 3246 dot pixels on bench1 medium-dots, against
  2880 in the PNG. No flat pixels, no level peaks, no family.
- *The error is mostly chroma.* In YCbCr it is 2 units rms in luma but 4–5.6 (up to 16) in Cb and Cr inside the
  dots: JPEG keeps chroma at half resolution and the decoder upsamples it smoothly, so the background's color
  bleeds into small dots. It is a smooth bias, not noise, so an edge-preserving blur (bilateral, tried) doesn't
  remove it.
- *Ringing.* The background next to every edge has near-white pixels slightly off every color line ((252, 252,
  248) around orange dots). In the count image they were "off the line", so every isolated dot looked partly
  hidden; their small positive counts made a halo that failed the disc test (moment ratio 1.06–1.07 against
  1.03); and on smoother.jpg they made a false first level at t = 5.

Fix:

- `src/core/jpeg.js` `restoreChroma`: a guided filter (He, Sun and Tang 2010) refits Cb and Cr as a linear
  function of luma in 9 × 9 windows (radius 4, ε 4). In a window with two colors that is exact, so chroma
  follows the sharp luma edges again. On isolated dots it halves the error at the rims (medium-dots 12.0 → 6.2,
  large 10.9 → 6.0, dots-random 14.6 → 8.0, rms) and cuts it 25–45% inside colored dots; on 50% black it costs
  a little (2.9 → 3.2). Smoothing luma too changed nothing. The mask and fit also use the restored image
  (positions 0.046 against 0.084 px RMSE on bench1 large-dots when only the style saw it).
- `JPEG_TOLERANCES` in style.js, chosen by `estimateStyle(image, { jpeg: true })`: flat 6, levels start at
  t ≥ 15, off-line distance 8 accepted near B in the count image, and counts below 0.1 weigh nothing in an
  isolated dot's moments. Swept one at a time over the 12 JPEGs: flat 5–8, minLevelT 15–20, nearB 5–12 and
  isolatedFloor 0.1–0.2 work alike; flat 3, minLevelT 10, nearB 2 or isolatedFloor 0.05 each lose images; the
  level-vote distance off the line is best left at 4 (6 lost bench2 smoother-ci). Both parts are needed:
  restoration with the exact tolerances gets 1 of 12 exactly right, and the JPEG tolerances without restoration
  get 3, with 9 finding no family or no radius.
- Who says "JPEG": the app sets it from the file type (or name), with a "JPEG-compressed" checkbox under Image
  for pasted images (the clipboard hands over PNG) and JPEGs saved as PNG; the bench from the extension. Detecting
  it from the pixels was tried: a smooth-region noise estimate puts antialiased translucent PNGs (bench3
  dots-random.png) above some JPEGs, and an 8×8 block-boundary statistic didn't separate them either.

Measured (found / truth, RMSE px; jpeg-js in Node):

| JPEG | before | now | the PNG |
|---|---|---|---|
| bench1 medium, large, smoother | no family, no radius, 97 | 100, 100, 100 (0.053, 0.050, 0.033) | 100 each |
| bench2 medium, large, smoother | 129, no radius, 133 | 204, 204, 204 (0.060, 0.080, 0.040) | 204 each |
| bench3 dots-packed, dots-random | 204, no family | 204, 204 (0.017, 0.030) | 204 each |
| bench1 smoother-ci | no family | 74 | 80 |
| bench2 smoother-ci | no radius | 171, precision 1.000 | 290, precision 0.600 |
| bench1 group-smoother | no family | 90 (recall 0.880) | 98 |
| bench2 group-smoother | no family | 131 (recall 0.637): red reads as opaque | 193 |

In the browser (bench1 medium-dots.jpg through the file input): the box is checked from the file type,
translucent α 0.487 (Node 0.50: the decoders differ), 100 dots, 0 flagged, 0.048 px RMSE against the truth.
Fitted JPEG dots are flagged more often (bench2 medium-dots: 44 against 1), since the residual is noisier.
Still short: bench2 group-smoother's red level 2 keeps 77 on-line pixels near t 208 against 755 in the PNG (the
overlaps of a saturated color are small lenses that the chroma bleed covers), so a deeper peak becomes "level 2",
the fit is impossible and red falls back to opaque. That is the multi-family case again.

Also found while checking the page: "Find dots" was enabled before any image was loaded, and clicking it threw
(no estimate yet); it now starts disabled.

**Beyond the axes.** `axes.plotBounds` turns each axis line with tick marks into a side of the plot area when
all its labels lie on one side of it (x labels below: the plot is above), and `buildMask(…, { bounds })` masks
pixels more than one dot radius beyond those sides, so a dot centered on an axis line (data at the scale's end)
keeps every pixel. Gridline axes and axes with labels on both sides bound nothing. The app and the bench pass the
bounds from the detected axes; the Layers panel reports the share masked "beyond the axes".

Bounds found: on all ten test images the left and bottom axes where they exist (nicest language: bottom at y 400
of 1146, its labels and the page text below; economist: bottom only; beeswarm-boxplot: left only), none on top or
right; bench3: left only (its category axis has no tick marks). Measured: bench2 tiny-dots 207 → 204 (precision
1.000); every other bench row and every real image's dot counts unchanged (Laser 32 + 23 + 25 in 4.1 s instead of
7.7 s). In the browser, bench2 tiny-dots masks 19.0% of the image beyond the axes and finds the 204 dots
(0.032 px RMSE against the truth).

**Several translucent families.** Each family was fitted alone, so pixels where a red and a blue dot overlap
(on neither family's color line) were masked for both, and a coincident red–blue pair had no dot in either.

- *Drawing order interleaves the families.* At every red–blue overlap in bench1 and bench2 group (11 and 33 of
  them) the dot of the later data row is on top; the two orders' predicted colors differ by about 69 units and the
  observed color matches one within 0.5. So no family-wide order (all blue, then all red) holds.
- *Unmixing* (`src/core/unmix.js`). With one or more dots of A and of B over a pixel, its color is B plus a
  weighted sum of the two families' color offsets, whatever the order; the weights solve a 3 × 2 least-squares
  problem from the pixel alone, so no prior fit is needed (the coincident pair has none). The order only changes
  how the weights become transmittances, and only the true order gives whole dot counts inside a lens (one dot
  each at α 0.5: 0.5 and 0.5 against 0.75 and 0.33), so each pixel takes the order nearer whole counts. Pixels on
  one family's line belong to it alone; pixels nothing explains (text, lines, other marks, stacks of three or
  more dots in both colors) are left as they are for the masks. Each translucent family then gets its own image
  (its transmittance over B) and count image, and the existing mask and fit run on it unchanged. The app and the
  bench do this when two or more families are translucent; the style panel says so.
- *A flat pixel on one family's line must be a whole number of its dots* (within 0.3): blue dots over the light
  steel-blue CI band read as 1.5 blue dots and made 20 phantom blue dots on bench2 group-smoother-ci. The same
  test on pixels a pair explains changed nothing on the bench, so it isn't applied there.

Measured (found / truth, recall, RMSE px):

| chart | alone | unmixed |
|---|---|---|
| bench1 group, group-smoother, group-large | 98, 0.980 each | 100, 1.000 each (0.028–0.033) |
| bench1 group-smoother.jpg | 90, 0.880 | 100, 1.000 (0.134) |
| bench2 group | 193, 0.946 | 204, 1.000 (0.101) |
| bench2 group-smoother | 193, 0.946 | 203, 0.995 (0.086) |
| bench1 group-smoother-ci | 77, 0.770, precision 1.000 | 85, 0.840, precision 0.988 (1 phantom, flagged) |
| bench2 group-smoother-ci | 163, 0.799, precision 1.000 | 190, 0.882, precision 0.947 (10 phantoms on the band, all flagged) |
| bench2 group-large | no radius | no radius (no isolated dot at 204 large dots, in either family's image) |

Every other bench row is unchanged (no other image has two translucent families). Positions near the other
color are less precise than elsewhere: on bench2 group, 0.042 px RMSE for dots with no other-color neighbor
against 0.186 for those with one (one dot in a crowd with four is off by 1.9 px). bench2 group-smoother.jpg is
unchanged (131): its red family reads as opaque in the JPEG, so only one family is translucent. In the browser,
bench2 group: 204 dots (113 + 91, the truth's group sizes), none flagged, 0.101 px RMSE.

economist (no truth): 110 + 78 → 116 + 82 dots, flags 94 → 63. Of the dots that changed, the 18 lost red "dots"
stacked along the red trend line were phantoms (all flagged weak, hidden and deep); the gains sit in the dense
red–blue cluster, mostly on visible discs, but include a stack of 5 red dots at one spot where the two trend
lines meet (all flagged weak and deep), likely phantoms.

**Milestone 2: outlines and per-dot fill color** (pathogen, beeswarm-boxplot outliers, bench1 stroked-dots).

- [x] Outlined dots with one fill: a two-part template (ring and fill).
- [x] A fill that varies per dot, translucent (bench1–2 gradient-dots, `colormap.js`).
- [x] Outlined opaque dots of one fill (bench1–2 solid-opaque-stroke).
- [x] A fill that varies per dot, opaque (bench1–2 gradient-opaque, gradient-opaque-stroke).
- [x] Translucent fills that vary per dot under a dark gray outline, fitted by the rings (pathogen; below).

**Outlined dots.** bench1 stroked-dots: a light gray fill (229 over white; α 0.31 toward (170, 170, 170), from the
fill-on-fill level) and a darker ring about 2 px wide at the rim (196 at r ≈ 9). The ring read deeper than one
dot, so no dot passed as isolated and the radius was unknown.

- *Render* (`render.js` `style.ring`): a dot is α opaque inside and α + ring α between inner × radius and the radius,
  in the family's one color; the annulus's coverage is the disc's less the inner disc's, so antialiasing stays
  exact. It is exact for one dot; where outlines cross other dots it is an approximation (the true ring has its
  own color and is composited after its own fill).
- *Isolated* (`style.js`): a component deeper than one dot still counts when all of its deeper pixels lie beyond
  0.6 radius from its center; the area (counts clamped at 1) still gives the outer radius.
- *Measured* (`measureRing`): when at least half the isolated dots are ringed, the inner radius (0.5–0.97 × r) and
  the ring's extra opacity are fitted to their pixels along the color line (opacity by linear least squares for
  each inner radius). bench1: a ring from 8.3 to 10.3 px, 69% opaque against 31% inside; bench2: 8.3–10.3 px, 63%
  against 25%. The style notes describe it.

Measured: bench1 stroked-dots 100/100, bench2 204/204 (was: no radius), 29 and 48 flagged deep (ring pixels read
as deeper stacks than the measured levels). RMSE 0.62 and 0.71 px, which is JMP's drawing, not the fit: its
stroked dots sit up-left of the data, the fill by 0.5 px and the ring by 0.13 px (centroids of isolated dots
against the truth; plain dots are exact), and the fitted centers carry a mean offset of (−0.44, −0.44) px with
0.03 px scatter around it. Other rows: three JPEG rows' RMSE moved by 0.001–0.007 px (a few JPEG dots now pass
as ringed isolated dots); everything else, and every real image, unchanged. In the browser: 100 dots, no errors.

**Translucent dots colored by a scale** (bench1–2 gradient-dots: JMP's "Muted Yellow to Red" on
mean(Score, Response) + row mod 37, α 0.5; plain and JMP's "Outlined" marker mode, PNG and JPEG). Color scales are
rare, so this is a side path (`colormap.js`, an early return in `families.js`) that leaves the main design alone.

- *Model:* a translucent dot keeps its offset's direction from B, so a pixel's direction names its dot's color and
  its distance over one dot's distance in that direction (m₁) counts the dots. Against the truth: medians 1.49–1.50
  at two dots and 1.72–1.77 at three (1.5 and 1.75 expected); stacks of unlike colors stay within 0.02–0.04 of the
  one-dot curve.
- *Curve:* m₁ per direction cell from flat pixels within 4 px of the background (the outer, one-deep part of
  dots); cells reading over 1.2 × their neighbors are coincident pairs (bench2: 123 against 85) and are dropped;
  points link into a chain within 0.15 and 1.35 × in m₁ (bench2's lone darkest dot is 0.116 from the next color);
  the scale is the largest chain, spanning at least 0.2 (gradients: 0.6). m₁ is a kernel regression over the points.
- *JPEG:* directions are read from offsets summed over a 5 × 5 window; per-pixel chroma noise had spread the one-dot
  level over 64–96. α comes from the ratio histogram (the second peak at 2 − α), since the level finder took a
  shoulder of the broad first peak for level 2 (α 0.88).
- *Fit:* the image is rewritten as one virtual family (each pixel B + r·(P − B)) and the usual pipeline runs on it.

Measured (found / truth, recall, RMSE px): bench1 plain 100/100, 1.00, 0.035; JPEG 100/100, 1.00, 0.085; outlined
101/100, 1.00, 0.116; outlined JPEG 104/100, 1.00, 0.154. bench2 plain 200/204, 0.98, 0.247 (median error 0.040 px;
the 4 misses and the two worst positions, 2.6 and 1.9 px, are in clusters of 4–6 mixed-color dots); JPEG 196,
0.956, 0.522; outlined 207, 0.971, 0.532; outlined JPEG 221, 0.951, 1.031. Every other bench row is unchanged, and
the detector stays off on all ten testgraphs images.

Tried and dropped: a ring less opaque than the fill (JMP's outlined markers are an outline-colored disc under a
smaller fill, both α 0.5: fill radius 9.6, outline 11.3 px, measured correctly). It made bench2 outlined worse
(211 found) and fitted a false 1 px ring on the plain JPEGs, so outlined markers are fitted as one disc at about
the outline's radius (11.1). Not done: each dot's color isn't reported (a legend mapping would be needed to make it
data); opaque scale-colored dots are handled below; a scale sharing the plot with another family isn't
handled. In the browser: not checked (the page check couldn't launch Chrome in this session's sandbox).

**Outlined opaque dots** (bench1–2 solid-opaque-stroke: JMP's outlined markers at α 1, a fill (171, 8, 252) of
radius 8.78 over an outline disc (105, 2, 156) of radius 10.47, centered together; plain solid-opaque has the
same fill radius). Before: "No dot family found", since every outline pixel read as neither the fill nor its
background, so each fill sat in a ring of unknown pixels and the Hough vote saw no edges.

- *Model* (`measureOpaque`, `outline`): a two-color opaque family. Pixels between the fill and the outline are
  covered; the outer rim is the outline's coverage of the background; a dot is one disc at the outline's radius,
  and the opaque fit runs unchanged. The boundary between fill and outline isn't used.
- *Detection* (`opaqueFamilies`): only a cluster that fails the dot test is retried, when its ring color is not a
  page background, not a translucent family's level (economist: a red–blue lens inside red dots, with red's
  one-dot level around it, made a phantom family), and thin: an opening of radius 1.5 px keeps under 25% of it
  (JMP's outlines keep 0.5–4.8%; bench1 ellipse's lavender region 98% and bench2 group-smoother.jpg's red level
  94%, both phantoms before this test). The blend of fill and outline is a rim, not a family (synthetic).
- *Display*: the dot sample and the Model view draw the outline disc and then the fill (sample difference: mean
  10.21 → 0.58, max 95.2 → 6.3); the Model view draws all outlines before all fills and says so.

Measured: bench1 98/100, recall 0.98, RMSE 0.041 (plain solid-opaque: 98, 0.035); bench2 165/204, 0.760, 0.862
(plain: 180, 0.863, 0.625). About 8 of bench2's dots are under 0.1% visible in either rendering. Against the plain
rendering, the outlined one misses 15 dots that are at least 60% visible (plain: 7), in packed clusters, where the
union of the discs is one blob and only the inner outlines show where each dot is. Using them needs a renderer that
respects drawing order (a later outline paints over an earlier fill); not done. Every other bench row and all ten
testgraphs images' families are unchanged. In the browser: bench2 165 dots, no console errors.

**Splits: opaque clusters covered by too few dots** (bench2 solid-opaque, 180 of 204 found). Of the 24 misses,
about 10 can't be recovered: coincident pairs (rows 120, 140, 144, 200; 198/201 0.7 px apart) and dots under 10%
visible. For one opaque color the edges between dots don't show, so what the image holds of a dot is the part of
its circle outside every other dot: of the 23 dots with under 10% of their edge shown, 16 were missed; above 20%,
only the coincident pairs. But that information isn't exhausted: scored with the fit's own objective, the truth
(204 dots) has data loss 8 800 and total 112 800, the fit's 180 dots 89 700 and 181 500; started from the truth, the
fit keeps 194 (recall 0.951). The fit's discs cover the image to 27 px, so births, which look for missing ink, found
no site; the loss is in boundaries everywhere slightly wrong, a dot stretched over two.

- *Split* (`fit.js` `splits`, opaque families only): a dot whose patch loss is at least λ and 4 × the median dot's
  is tried as two, 0.4 r either side of it in 4 directions; the neighbors re-settle as for a birth, and the best
  variant is kept if the loss falls by more than λ and each half would cost more than λ to remove (as deaths
  require). A dot that fails isn't tried again until it moves 0.25 r.
- *Thresholds, measured:* kept splits had patch loss 4.4–12 × the median (bench2: 3.1–15.5 λ against 0.7; violin, a
  JPEG with noise in every patch: 59–107 λ against 13). Trying every patch over λ made violin 5 × slower (21 → 101 s)
  and fibers +2, violin +4 dots. Without the each-half test, a Laser dot half under a purple bar was split into
  itself and a dot on the bar, and a later death removed the real one.

Measured: bench1 solid-opaque 98 → 100 (recall 1.00, RMSE 0.056); bench1 solid-opaque-stroke 98 → 101 (recall
1.00; median error 0.021 px, but one true dot is fitted as two, one of them 4.7 px off, RMSE 0.551); bench2
solid-opaque 180 → 196 (recall 0.863 → 0.926, RMSE 0.625 → 0.357, 8 extra flagged dots); bench2 solid-opaque-stroke
165 → 179 (0.760 → 0.848). Times: bench2 1.9 → 3.9 s, 2.5 → 4.3 s. The synthetic presets, every other truth-set row
(bench3's opaque beeswarms included) and all ten testgraphs fits are unchanged, at the same times.

**Opaque dots colored by a scale** (bench1–2 gradient-opaque and gradient-opaque-stroke: the same "Muted Yellow
to Red" at α 1, plain and with JMP's outlined markers). Before: bench1 114 found (recall 0.82) and 46 (0.33), as
an opaque family of the commonest fill; bench2 "The dot radius is unknown" and "No dot family found".
`findColormap` already found the curve (spread 0.54–0.60) but required a second level, which opaque dots never show.

- *Detection:* a curve with no second level is an opaque scale when it spans at least 0.4. Across all bench and
  testgraphs images the only other such curve is bench1 smoother-ci.jpg's band edges, at 0.215.
- *Model* (`opaqueImage`): the image becomes one virtual opaque family, each dot pixel B + a·(P − B), so the opaque
  pipeline (radius, Hough, fit, splits) runs unchanged on the union of the discs. The ray trick of the translucent
  path doesn't give a here: JMP's outline of a light fill points the way the fill does but lies farther out (yellow
  fill 155 from white, its outline 255, 0.01 apart in direction), so a full fill and a partial outline look alike.
  Position settles it: a pixel is partial only next to one under 0.45 of its full color's offset (the lightest
  fill is 0.61 of its outline's), and then a is its offset over the largest one nearby (5 × 5) in its direction.
- *Three fixes, each found in the difference from the one-color rendering of the same dots:* the direction
  tolerance grows by 1/m for rounding (a faint rim pixel, (253, 247, 245), was 0.045 off its fill and counted
  itself as full: whole rings over-covered); "next to a much lighter pixel" rather than "touching the background"
  (2 px gaps between dots hold no background pixel, so they read as covered); outline pixels up to 0.25 off the
  curve count when within 3 px of a curve pixel (dark red outlines, (109, 47, 45), are 0.15 off and as close to
  black text, and pale yellow outlines left as gaps inside clusters).
- *Coverage against solid-opaque(-stroke), RMS over dot pixels:* plain 0.019 (97 pixels off by more than 0.1,
  where three colors meet at a rim), outlined 0.019 (was 0.080 before the outline pass).

Measured (found / truth, recall, RMSE px):

| Image | Now | Same dots in one color |
|---|---|---|
| bench1 gradient-opaque | 101/100, 1.00, 0.380 | solid-opaque 100, 1.00, 0.056 |
| bench1 gradient-opaque-stroke | 100/100, 0.99, 0.478 | solid-opaque-stroke 101, 1.00, 0.551 |
| bench2 gradient-opaque | 192/204, 0.926, 0.499 (precision 0.984) | solid-opaque 196, 0.926, 0.357 (0.964) |
| bench2 gradient-opaque-stroke | 177/204, 0.828, 0.885 | solid-opaque-stroke 179, 0.848, 0.944 |

bench1's RMSEs come from one dot, (179.5, 355.3), almost hidden by three neighbors 7.6–9.6 px away: the fit
merges it with a visible neighbor (3.8 px off; 7.6 outlined). Median errors are 0.020 px. The synthetic
presets, every other truth-set row and all ten testgraphs images are unchanged, at the same times (gradient-opaque:
1.0–3.4 s). In the browser (the app's pane; headless Chrome can't start in this sandbox): bench2 192 and 177
dots, as in the bench, no console errors.

Not done: the edges between dots of different colors, and the outlines inside clusters, show where each dot is,
but the virtual image keeps only the union (the same limit as solid-opaque-stroke). Using them needs per-dot
colors in the model. Each dot's color isn't reported. pathogen is still not detected: `findColormap` finds no
chain for it (not investigated further).

**Outlined dots whose fills vary: the rings** (pathogen: 583 dots, ggplot's shape 21 colored by a scale from
near-white to crimson, under a thin dark gray ring; gridlines under them, a translucent purple band and a black line
over them; log(y + 1) with zero at its own place; jittered by about ±1 year in x and ±25 px in y; the image noisy,
as if resampled from a JPEG). Truth: `testgraphs/pathogen-trade-time-scatter.csv` (data cells, ticks in the header,
mapped through the detected ticks: `truth-csv.mjs` `truthPixelsByTicks`). With the jitter only cells are known, so
the bench matches within a box of ±14 × ±27 px and reports no RMSE. Rows 0–2 are over 50 px apart, so per-row counts
are exact checks (286, 86 and 38 dots).

Why nothing was found: the fills vary per dot, are noisy (3% of fill pixels pass the exact flat test), and the scale
runs from near-white, so no color line or curve held them; the curve finder's largest chain was the gridlines' gray.

- *Projection* (`outline.js`): the fill colors lie in a plane through white (11 106 flat colored pixels within 2.8
  units). Its normal n gives P = n·(X − B) / n·(K − B), K black: 0 for every fill and every composite of fills, and
  a ring composites in P like one translucent family. The darkest ring pixels read P 0.26–0.33 on fills from pale to
  crimson. Gridlines read 0.082, the band −0.077.
- *Background in P*: the largest of 12 directional linear openings longer than a ring is across keeps gridlines,
  the band's level and the black line, and drops the rings. Ring ink A = (P − P_bg) / (1 − P_bg); P_bg above 0.5
  (the line, text, frame) is masked.
- *Style* from 12 isolated rings (square, hollow components of P): outer radius 8.625, width 1.5, blur 0.5, αk 0.39
  (0.33–0.38 per ring by hand, across fill colors). αf from 110 pairs of strong candidates: along one ring, its ink
  inside the other's fill over its ink outside is bimodal, about 1 (the other under it) and 0.2 (on top). The low
  mode gives αf 0.74. The fills are nearly opaque, so a ring under one fill keeps a quarter of its ink.
- *Model* (`createOutlineModel`): in drawing order over the base P_bg, each dot lays its fill
  (P ← (1 − αf)·P + αf·f) and then its ring. Each dot has a z, and its own fill value f, fitted per dot from the
  interior it is on top of. The scale is a curve, not exactly a plane: P inside isolated fills runs from −0.04 to
  +0.08. Without f, hidden phantom rings under fills paid for themselves by explaining that offset (row 1: 112 found
  for 86).
- *Fit changes* (`fit.js`): ring-only styles (α 0 with a ring) take candidates from `ringCenters`, a matched filter
  of the mean ring count on the ring's mid-radius circle. Hough on D's gradient fails there: a 1.5 px ring's two
  edges cancel in the Sobel kernel, giving arcs of 0.25 for whole rings. Births use the same filter on the
  residual and try the new dot at the best gap in its neighbors' order. A dot needs 30% of itself unmasked: dots
  beside the y-axis ticks explained their gray fringes.
- *Tried and dropped*: an order pass trying each dot on top of or under its neighbors (468 found against 469
  without it, the same recall). Adding the first candidates weakest first (461, recall 0.768).

| Step | Found | Precision | Recall | Rows 0 / 1 / 2 | Time |
|---|---|---|---|---|---|
| Rings alone (no fills, no order) | 281 | 0.972 | 0.468 | 72 / 61 / 25 | 1.3 s |
| + ring births by matched filter | 291 | 0.969 | 0.484 | 78 / 62 / 25 | 41 s |
| + fills αf 0.8 in drawing order | 513 | 0.914 | 0.804 | 232 / 112 / 28 | 143 s |
| + each fill's own value | 438 | 0.973 | 0.731 | 186 / 84 / 27 | 106 s |
| + first candidates at arc 0.5, births at 0.15 | 456 | 0.974 | 0.762 | 204 / 85 / 27 | 35 s |
| Pipeline (measured style, visibility ≥ 0.3) | 469 | 0.972 | 0.782 | | 43 s |

Of the 583 truth dots, 114 are unmatched: 83 in the band at 0 (up to 41 dots to a cell, under two or more fills),
28 above row 2, 13 in row 2. Of the 53 cells near the purple band, 27 are missed: the band is drawn over the
dots and lets part of their ink through, which isn't modeled. (The overlay detector of `background.js` needs a
region of one exact color, and this image's noise gives none.) Every other bench row is unchanged, at the same
times, and the detector stays off on every other image: the bench gradients have a fill plane but no ring off it,
and JMP's outlines share their fill's hue. In the browser: 469 dots, 20 flagged weak, no console errors, but 349 s
against 43 s in Node, with the pane hidden (a throttled background page, probably; not checked).

Not done: the band (divide its transmission out of P: s from the rings' contrast inside it, needing a region
detector for noisy images); a single fill color under a gray ring (the plane is then undetermined: the normal
should be the gray direction made orthogonal to the fill's, untested for want of an image); two color families
under gray rings are fitted as one (their colors aren't reported); the fit's time.

**Not done, and why.**

- *pathogen*: done below, by its rings, except the dots under its purple band and the deepest stacks.
- *beeswarm-boxplot* fails for another reason than its few outlined outliers: its gray swarm is packed (no
  isolated dot, the discs explain 0.31 of the color's pixels against the 0.35 required), and its red swarm's local
  background reads as dark red (57, 0, 0), probably where the box plot's black lines cross red dots, so no coverage
  is read. That is milestone 1's opaque-family test.

**Milestone 3: varying backgrounds and translucent overlays** (fibers' CI band, violin shading, pathogen's band,
bench1's CI bands and ellipse).

- [x] Background field under the dots; translucent layers over them.

A shaded region is a large flat area of one color B₂ that isn't a dot family's (`src/core/background.js`): at
least 1500 flat pixels of the color, surviving an opening by 0.75 dot radius (dog-ages' black curve and text,
52 000 flat pixels after upscaling, are strokes, not areas; bench3's ANOVA band, its two boxes mostly covered by
dots, survives 0.75 but not 1 radius), and not itself a dot's color over or under a region found before it
(orange dots under the band are a large flat color of their own). Each pixel belongs to the region of its nearest
background-colored pixel, so a dot inside a band is in the band and a dot across its edge is split along it.

- *Over or under.* Drawn under the dots, a region changes a dot's color to c + (B₂ − c)·T: its offset from B₂ points
  toward c. Drawn over them as a translucent layer, it leaves B₂ + s·(U − B): the offset points along c − B, scaled
  by s = 1 − the layer's α. The flat pixels inside the region vote; at least 50 votes and 60% agreement decide,
  else the region is "unknown" and changes nothing. s is the commonest scaled offset against one dot's, and an
  "overlay" keeping more than 95% of what's under it isn't one (nicest language's white page read as s = 1).
- *Overlays are divided out* first, U = B + (I − B₂) / s, and everything is estimated again on the result. Pixels on
  the region's edge are only partly under the layer and are left as they are (inverted as if wholly under it, the
  slivers where bench1 smoother-ci's band edge crosses a dot made an "opaque family" that took every dot again).
- *Regions under translucent dots* give unmix.js a per-pixel background: a dot there is read along the line from B₂.
  Opaque families already read coverage against their local background.
- *One family over two backgrounds* (`families.js`): with too few overlaps to show its levels, a translucent family
  looks like two opaque ones (bench1 ellipse: orange on white and on lavender, the only overlap inside the ellipse).
  Their colors differ by q·(B − B₂), which gives α = 1 − q and c without any overlap; such a pair becomes one
  translucent family, and an opaque cluster that is a translucent family's dot over B₂ is dropped.
- `estimateFamilies` (`src/core/families.js`) does all of it (estimate, regions, overlays, merges, unmixing); the
  bench and the app both call it, and the style panel's notes say what was done.

Measured (found / truth, recall, RMSE px); every other bench row and every real image's counts unchanged:

| chart | before | now |
|---|---|---|
| bench1 smoother-ci (band over) | 80, 0.800 | 100, 1.000 (0.028) |
| bench1 smoother-behind-ci (band under) | 83, 0.820 | 100, 1.000 (0.026) |
| bench1 group-smoother-ci | 85, 0.840 | 100, 1.000 (0.034) |
| bench1 ellipse | 194, 0.970, precision 0.500 | 100, 1.000 (0.046) |
| bench2 smoother-ci | 290, 0.853, precision 0.600 | 203, 0.995 (0.046) |
| bench2 smoother-behind-ci | 180, 0.868 | 205, 1.000, precision 0.995 (0.064) |
| bench2 group-smoother-ci | 190, 0.882, precision 0.947 | 205, 0.990, precision 0.985 (0.117) |
| bench2 ellipse | 178, 0.667 | 204, 1.000 (0.042) |
| bench3 dots-swarm-anova (band over opaque dots) | 185, 0.907 | 204, 1.000 (0.038) |
| bench1 smoother-ci.jpg | 74, 0.740 | 97, 0.960 (0.258) |
| bench2 smoother-ci.jpg | 171, 0.838 | 187, 0.907 (0.191) |

In the browser: bench1 smoother-ci 100 dots (0.028 px), with the note that a layer letting 74% through was
divided out; bench2 group still 204 (113 + 91).

Not done: fibers' band (3 votes: its opaque dots barely enter it; left "unknown", 39 dots as before); pathogen's
band over outlined dots (milestone 2 first); the JPEG band charts are short of their PNGs, and bench1
smoother-ci.jpg's positions are worse (0.258 px against 0.038 before; not examined).

**Milestone 4.**

- [ ] Ground truth and benchmark rows for each test image. (bench1–3 have exact truth for their 59 renderings.)

### Phase 6 — Polish and deploy
- [ ] About panel, CSV/JSON export, copy to clipboard, URL state, dark theme.
- [ ] Optional OCR of tick labels (open question 2).
- [ ] GitHub Pages workflow and live-site check.
