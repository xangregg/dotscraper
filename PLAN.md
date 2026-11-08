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
2. **OCR for tick labels.** tesseract.js would be a runtime dependency (large, with a WASM file); a small
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
src/core/style.js       background, families, overlap levels → color and alpha, radius, blur
src/core/layers.js      pixel classification, morphology, masks, background field
src/core/fit.js         objective, candidates, continuous refinement, discrete moves, uncertainty
src/core/axes.js        plot area, axis lines, ticks, gridlines, scale fitting, pixel ↔ data
src/core/synth.js       synthetic graphs with known dots
src/core/metrics.js     assignment, precision/recall, RMSE
src/core/rng.js         seeded random numbers
src/ui/                 main.js, view.js (canvas, zoom, overlays), steps.js, input.js, export.js, urlstate.js
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
- [ ] `render.js`: exact disc coverage per pixel, blur, compositing for one family; incremental updates. Tests:
      coverage sums to the disc area, symmetry, patch updates equal a full rerender.
- [ ] `synth.js` with a seeded generator for the §2.6 variations; `metrics.js`.
- [ ] Agreement check against Chrome canvas renders of the same dots (fixture images made by a tools page via
      `check-page.mjs`), within a stated tolerance per pixel.

### Phase 2 — Style and masks for one translucent family
- [ ] `style.js`: background, color and alpha from overlap levels, radius, blur. Tests on synthetic images
      (recovered within tolerance) and on dog-ages (against the phase 0 numbers).
- [ ] `layers.js`: classification, line removal by opening, dilated mask. dog-ages: the black curve, gray dashes
      and text are masked; dot pixels are not.
- [ ] UI: Layers and Dot style steps with overlays (mask, D) and correction controls.

### Phase 3 — The fit
- [ ] Candidates: matched filter on D, Hough with known radius, sparse grid solve; compare on the synthetic set.
- [ ] Continuous refinement, birth/death/split/merge, style refit, uncertainty and flags.
- [ ] Worker, progress animation, residual overlay, manual edits with "refit nearby".
- [ ] Ground truth for dog-ages; benchmark table for the synthetic set and dog-ages, recorded here.

### Phase 4 — Scales
- [ ] `axes.js`: plot area, axis lines, ticks, gridlines; typed tick values with guesses; linear, log and
      categorical scales; pixel ↔ data. Tests on synthetic axes and the test images' axes.
- [ ] Axes step in the UI; data coordinates in the table and export; pixel resolution in the fine print.

### Phase 5 — Generalize to the other test images
- [ ] Several families and drawing order (economist, Laser-Asyllable, cantons, nicest-language).
- [ ] Opaque dots with occlusion (the beeswarms); packing initializer.
- [ ] Outlines and per-dot fill color (pathogen, beeswarm-boxplot outliers).
- [ ] Varying background and translucent overlays (fibers' CI band, violin shading, pathogen's band).
- [ ] Ground truth and benchmark rows for each test image.

### Phase 6 — Polish and deploy
- [ ] About panel, CSV/JSON export, copy to clipboard, URL state, dark theme.
- [ ] Optional OCR of tick labels (open question 2).
- [ ] GitHub Pages workflow and live-site check.
