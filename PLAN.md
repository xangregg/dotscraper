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
9. **Non-dot mark shapes** (triangles, squares, plus, ×, diamonds): future. The forward model renders discs; a
   shape would be a family attribute with its own coverage function (and orientation, for ×), estimated from
   isolated marks like the radius is now.
10. **Rings as a family attribute.** The ring readers decide globally (outline.js: one fill plane, or each pixel's
   darkness against its local fill) whether the image is outlined dots, before any family is known. Real figures
   mix them (hair2–3: black outlines around pink and gray fills, a gray fill putting black in the fill plane, and
   outlines as dark as text, which the ink mask removes). Rethink: each family gets its fill and its ring (color,
   width, opacity), measured from its own isolated dots, so outlined and plain families can share an image.
   phase1 and engagement1's PDF capture (2026-11-14, below) add two cases: black outlines around gray fills, where
   white, fill and stroke are collinear and the outline shares its color with text and a black family; and five
   discrete fills read as one scale. Outline pixels should count only next to a fill of the family. Done for opaque
   families (2026-11-14, below); outline.js's global readers still decide first for JMP-style and scale outlines,
   and kite2 (translucent, own-shade ring) is still open.
   engagement1 (2026-11-13) shows how the global view fails. Five blue fills (navy 4, 3, 87 to lavender
   188, 186, 234) share one dark stroke, under a pixel wide and blurred, so it never shows its own color (on
   lavender its darkest pixel is about 80, 80, 105). (a) Lavender and cyan never become families: their edge
   pixels mix three colors (white, stroke, fill), the opaque test accepts two, so the edge is unknown and no dot
   shape is left (lavender: radius 5.5, the fill alone, 0 dots). (b) The stroke is close to the navy fill, so
   every dot's ring reads as navy coverage: the one family found (50, 55, 99) takes all 20 navy dots, 7 of 10
   mid-blue and 1 of 6 cyan, and no lavender. Prototype: coverage as the weight off white inside the triangle
   white–fill–stroke, the stroke taken as the most extreme ring pixel. With lavender's stroke read from its own
   rings, 8 Hough centers (10 in truth, 6 with ¾ of the arc) where there were 0; but rings alone don't say whose
   dot it is, and the dark families' triangles claim 9–13 of the other families' dots each. A dot's family has to
   come from its fill: the interior inside the ring.

   **Stages 1–2 done** (2026-11-13, style.js). (1) A color that fails as a plain or outlined family is tried with a
   stroke: `strokeColor` reads it from the rings of isolated fills (pixels up to 4 px past the fill's color, the
   fill isolated when 90% of the band 5–7 px out is background; engagement1's stroke is dark 3–4 px out and
   background from 5), as the mean of the ring pixels in the top 20% by distance off the background–fill line, and
   needs at least 30 units off it. Coverage is then the weight off the background inside the triangle
   background–fill–stroke, and a center is the family's only if its inner disc is the fill's weight (a darker
   family's fill can lie in the triangle). In the final pass, when any family is stroked, every family's coverage
   gives a ring pixel within 4 px of another fill, and nearer it by 0.5 px, to that dot (it reads 0). Both masks are
   the fills' thick parts: the stroke is within 40 units of navy, so navy's color mask took the lavender rings, and
   lavender's coverage handed its own rings to navy (17 dots for 10, at r 6.84) until only thick parts counted.
   Tried and reverted: accepting pixels darker than the stroke estimate (it averages the darkest mixes seen, so
   some go beyond it); black text went with them and lavender was lost. (2) The fit needs nothing new: it works in
   coverage, where an outlined dot is one disc of the outer radius. For display, a stroked family carries
   `style.outline` (the stroke estimate and the fill's radius), so the Model view draws a ring under each fill, and
   the list calls it "opaque, outlined".

   Results. engagement1 (against the user's 46-dot export): 33 dots in one family, P 0.848 R 0.609 → 44 in two,
   P 0.841 R 0.804; lavender 10 of 10, no extra, r 7.58, and the navy family takes none of them (it took 7 of the
   mid-blue before and 6 now). hair3 is now found without the JPEG fallback, its pink fill inside a black stroke:
   recall 0.735 → 0.824 (28 of 34; the 6 gray dots are no family). Every other bench row is unchanged (synthetic,
   bench1–3, testgraphs, bench-nature; diffed without times). Tests: synthetic light and dark fills in a 0.8 px
   stroke (light 14+ of 15, dark 15, none across, radius within 0.3 px; before, the light dots were never found)
   and engagement1 (skipped without bench-nature). Still open: engagement1's mid-blue and cyan (mid-blue has 3
   isolated fills and no strong arc; cyan's six dots never make a color cluster), hair2 and tumour1 (white fills:
   the fill is the background), and folding the global ring readers (outline.js) into this (stage 3).

   **Editable outlines** (2026-11-13). Every opaque family of its own color, estimated or picked, has an Outline row
   under Radius, Alpha and Blur: on or off, a color (typed, or Pick: click the darkest part of a ring) and a width.
   A change measures every opaque family again (`opaqueFamilyEstimate`): the edited one with its outline (the fill
   radius is the outer radius less the width), all with the others' fills known and, when any is outlined, each
   ring given to the fill it surrounds; the edited family's dots are dropped (Undo restores them), the others keep
   theirs. Adding a picked family does the same when any family is outlined. Reset restores the estimate. On
   engagement1 in the app: cyan picked (55, 169, 218, radius 5.5, the fill alone), outlined with lavender's
   94, 94, 116 at 2 px and radius 7.5: 60 dots, against the user's export recall 0.804 → 0.913 and precision
   0.841 → 0.700 (cyan took 3 of its 6 and 4 mid-blue; mid-blue picked as well would claim those). Without the
   radius set, a crowded family's outer radius isn't measured (cyan: 4.97, from the distance transform).
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
- [x] Ground truth for dog-ages: **needs a person**. The app's Edit mode and Save truth write
      `testgraphs/truth/dog-ages.json`, which `npm run bench` then scores. The fit's own output must not be used
      as truth without review. Done by the user (2026-11-12, 95 dots, reviewed in the app). Its positions are the
      fit's (they agree to 0.004 px), so the row checks the count and which dots, not sub-pixel accuracy; bench1–3's
      exact truth and the dog-ages-actual-size cross-check cover positions.

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
- [x] One fill color under a gray outline (bench1–2 solid-graystroke; below).
- [x] Two color families under gray outlines (bench1–2 group-graystroke, group-solid-graystroke; below).

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

Deferred for pathogen: the band, until bands over dots turn up more often (it would mean dividing its
transmission out of P, with s from the rings' contrast inside it and a region detector for noisy images).
Deferred indefinitely: the rest of pathogen's misses, the deep stacks in the band at 0, and the fit's time (43 s in
Node).

**Gray outlines** (bench1–2, three JMP renderings each, drawn as filled circles plus open circles of the same
data): solid-graystroke, one opaque light green (106, 255, 107) under opaque gray rings, marker size 8;
group-graystroke, the two groups' colors at α 0.4 under black rings at α 0.7, size 7; group-solid-graystroke, the
same opaque. Before: "No dot family found" or "The dot radius is unknown" on all six.

- *One fill color*: its pixels lie on a line through B, so any plane holding the line will do. The normal is K's
  direction made orthogonal to the line (`fillPlane`), which sees black best (kOff 0.58).
- *Opaque fills*: αf measured 1.00 (the pairs' low mode is 0: a later fill hides the ring under it, so JMP draws
  each dot's fill and then its ring). The fill fit divided by 1 − αf, every loss was NaN, and the fit froze on
  its 172 first candidates. It now uses f = f0 + (t − t̂) / αf.
- *Text*: the title's letters looked like rings in P (81 phantoms piled on "Response vs. Score"). Ink at P ≥ 0.8 is
  masked with 2 px around it.
- *Rings over every fill*: when the pairs show no dimmed mode, αf is 0 (none of these images needed it).
- *JMP offset*: these dots sit 0.497 px up and left of the plot area's mapping in both x and y. The fill centroids
  and the fitted rings agree to 0.05 px, so it is the rendering (marker size 8 against 7 elsewhere?), not the fit.
  RMSE is 0.70 px for that reason.

| Image | Found / truth | Precision | Recall | RMSE px | Time |
|---|---|---|---|---|---|
| bench1 solid-graystroke | 98 / 100 | 1.000 | 0.980 | 0.703 | 5.0 s |
| bench2 solid-graystroke | 189 / 204 | 0.995 | 0.922 | 0.717 | 7.1 s |
| bench1 solid-opaque-stroke, by rings | 98 / 100 | 1.000 | 0.980 | 0.034 | 4.8 s |
| bench1 solid-opaque-stroke, opaque path (before) | 101 / 100 | 0.990 | 1.000 | 0.551 | 1.6 s |
| bench2 solid-opaque-stroke, by rings | 172 / 204 | 0.988 | 0.833 | 0.284 | 8.1 s |
| bench2 solid-opaque-stroke, opaque path (before) | 179 / 204 | 0.966 | 0.848 | 0.944 | 4.4 s |

With the one-color plane, JMP's outlined markers (a darker purple outline, off the fill's line) read as rings too,
and solid-opaque-stroke moved to the ring path: positions much better (the inner outlines are used now, in drawing
order), precision better, recall 0.02 lower, 2–3 times slower. Kept; the two regression tests were updated.
pathogen is unchanged (468, 0.972, 0.780).

*Two color families* (group-graystroke, group-solid-graystroke). Blue and orange are nearly complementary, so
the plane through white that holds both almost holds gray: black is 0.05 of |K − B| off it (pathogen: 0.32), and
the plane test (K at least 0.2 off) refused it. But the projection is linear, so every composite of white, the
fills and the ring still maps exactly; what a small kOff costs is noise, and these pixels are exact. The test is
now P's expected noise: the fills' scatter off the plane, with the pixels' rounding (0.5), over K's distance from
the plane, at most 0.03 (pathogen 0.020, these 0.022). Only these four images newly pass it.

- *Style*, measured: ring 7.75 px, width 1.25–1.37, αk 0.37–0.59, αf 0.44–0.45 for JMP's α 0.4 and 1.00 for the
  opaque ones.
- *Colors*: the fit sees rings only, so each dot's color is read afterwards (`dotFamilies`). The fill colors
  (`fillFamilies`) are the largest bins of flat colored pixels when a few hold 0.75 of them (two colors hold 0.87;
  pathogen's scale has no bin above 0.02). A dot takes the nearest color to the median of its fill where no other
  dot reaches. A dot with no such pixels is read where the fit draws it on top, and marked unsure.
- *Unsure colors*: all 128 and 137 sure dots of bench2 take the right color; of the unsure, 52 of 70 and 41 of 50.
  The wrong ones are mostly bench2's near-coincident pairs, where the lower fill doesn't show at all. Reading which
  of two dots is on top from the rings, instead of the fit's z, did no better (50 and 37), so it was dropped.
- The bench splits the fit into one result per color, so the group scoring checks the colors. The app keeps one
  family and gives each dot a fill color, shown on hover and in the CSV ("2? #b8d7ff" when unsure).

| Image | Found / truth | Precision (by group) | Recall | Sure colors right |
|---|---|---|---|---|
| bench1 group-graystroke | 100 / 100 | 1.000 | 1.000 | all |
| bench1 group-solid-graystroke | 98 / 100 | 1.000 | 0.980 | all |
| bench2 group-graystroke | 198 / 204 | 0.914 | 0.887 | 128 of 128 |
| bench2 group-solid-graystroke | 187 / 204 | 0.957 | 0.877 | 137 of 137 |

Pooled over groups, precision is 1.000 on all four (recall 1.000, 0.980, 0.971, 0.917). RMSE is 0.70 px from
JMP's half-pixel offset, as in solid-graystroke (at marker size 7 too). In the browser: bench2 group-graystroke
198 dots, 128 sure colors, no console errors.

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

**How many of bench2's misses can be found at all** (measured 2026-11-12). For each truth dot, the share of its
disc that no other truth dot covers (for one color with no ring, where drawing order can't be seen), or that no
later row covers (with rings or several colors), sampled every 0.25 px at the fitted radius. The rings confirm
that JMP draws in row order: in close pairs the later row's ring shows 85 times to 3 (solid-opaque-stroke), 83 to 9
(gradient-opaque-stroke) and 25 to 1 (solid-graystroke). Each image has 5 exact duplicates.

| image | missed | under 10% shown | 10% or more shown |
|---|---|---|---|
| solid-opaque (one color, any dot) | 15 | 14 | 1 |
| gradient-opaque (any dot) | 15 | 14 | 1 |
| solid-opaque-stroke (ring, later rows) | 34 | 22 | 12 |
| gradient-opaque-stroke (ring, later rows) | 35 | 15 | 20 (14 show ≥ 35%) |
| solid-graystroke (ring, later rows) | 16 | 8 | 8 |
| group-smoother.png (later rows) | 1 | 0 | 1 |
| group-smoother.jpg (later rows) | 74 | 11 | 63 (41 show ≥ 35%) |

So the opaque rows without rings are at their limit: their misses are dots hidden under others. (gradient-opaque
counted by later rows has 7 misses that show ≥ 35% on top, but neighbors' colors may be too close to tell; not
checked.) The stroked rows have 8–20 misses that show, gradient-opaque-stroke the most. group-smoother.jpg is the
real gap: its misses are visible dots that the PNG finds, so the loss is in the JPEG path, not in occlusion.

**bench2 group-smoother.jpg fixed** (2026-11-12). Its red family was taken for opaque pink (240, 155, 168), the
single-dot tint, so overlaps were hidden dots. The two-dot lenses are narrow, and JPEG noise left few flat pixels
in them: 29 at level 2, spread over 7 one-unit bins, none reaching PEAK_MIN_PIXELS (20). The weak-level search
then took a deeper level for level 2, and the fit put c outside the cube. Two changes in `style.js`:

- JPEG tolerances sum the level histogram over ±3 bins (`levelSmooth`) before finding peaks; exact images don't.
  The weak-level search keeps the raw counts: on the smoothed ones it took the first peak's shoulder 10 units on
  for a level (bench3 dots-packed.jpg and bench2 smoother-ci.jpg lost their radius).
- When the levels' geometric fit fails as consecutive counts, one count may be missing (`MAX_MISSING_LEVELS`): the
  red's levels at t 137, 205 and 258 are counts 1, 2 and 4. The missing level gets the fitted color, marked
  `inferred`, and the panel says so ("3 measured, 1 inferred", a dim swatch, a note).

Red α 1 → 0.495 (truth 0.5); group-smoother.jpg 131 → 200 found, precision 0.992 → 0.995, recall 0.637 → 0.975,
flagged 57 → 21. Every other row is unchanged but for small shifts in JPEG RMSE (±0.013 px) and flags (bench2
large-dots.jpg 14 → 2, smoother.jpg 27 → 15). In the page: 204 dots (89 + 115; the browser decodes JPEGs a little
differently from jpeg-js), 24 flagged, no console errors.


**The stroked rows' visible misses, looked at** (2026-11-12). The table above overcounts them: of a coincident pair
(5 exact duplicates per image, more within 1 px), the dot on top is found and the matcher may give it to the hidden
one, so the top one reads as a visible miss. Leaving out misses with a found dot within 1 px:

- solid-graystroke: 1.
- solid-opaque-stroke: 7. Three are near pairs 2.7–4.4 px apart (the lower dot a crescent of 10–16% of its ring);
  four are a stack near (175, 330), where adding three of them at their true places raises the loss (their
  neighbors are fitted wrong), and the fourth was tried in round 1, failed, and not proposed again.
- gradient-opaque-stroke: 18, on the color-scale path, which counts every dot pixel as covered and ignores the
  outlines, so a dense stack's centers are poorly determined; 8 extras sit 6–8 px from truth in the same stack.

Added to the finished solid-opaque-stroke fit at their true places, four of its misses lower the loss by 3 300–17 500,
far above λ = 510: the births never proposed them, since ring births look only at missing ring ink (0.16 of a ring
there, at RING_MIN_ARC 0.15). Tried: adding to the birth score the mean negative residual inside the dot (rings the
model draws where a missing opaque fill hides them). It found three of those four, but lost three others, mostly the
partner of a near pair: solid-opaque-stroke 172 → 176 (RMSE 0.284 → 0.426 px), solid-graystroke 189 → 187, the group
graystrokes +1 and +2. Not kept.

gradient-opaque-stroke's outlines are a darker, more saturated shade of each dot's fill (fill 238, 218, 145 → ring
224, 188, 57; 189, 90, 78 → 120, 53, 45). Its fills lie 4.2 units rms off their best plane through white, and the
rings' distance off that plane ranges from 2 (yellow fills) to 55 (dark red), so the fill-plane projection can't
separate them.

**Outlines in each fill's own shade, fitted by their rings** (2026-11-12): `outline.findHueOutlines`. P is read
against the local fill instead of a plane: the RGB distance of a pixel from F, F per channel the largest of linear
closings in 12 directions with a 5 px segment (`relativeDarkness`), over white's length. A closing fills thin dark
structures and keeps edges between areas, so every fill and every edge between fills reads 0. The rest is the ring
path as it was (isolated rings, ring style, αf from pairs, the ordered fit). Tried first, on the isolated rings:

| P | ring, dark red to pale yellow |
|---|---|
| 1 − S / S_fill, S = R + G + B | 0.22–0.39 (0.61 on the darkest fill) |
| \|F − X\| / \|F\| | 0.27–0.38 (0.61) |
| \|F − X\| / \|white\| (kept) | 0.19–0.23 (85–101 units) |

The first two read pale fills' rings weakly and dark fills' strongly; a dark fill's ring, stronger than the one
fitted αk, drew a second dot stacked on the first. It is used only when the color-scale path finds opaque fills:
it also finds the rings of stroked-dots (translucent, already 1.000), and nothing on gradient-opaque, solid-opaque,
gradient-dots-stroke or medium-dots.

| image | before (color scale) | now (rings) |
|---|---|---|
| bench1 gradient-opaque-stroke | 100, precision 0.990, recall 0.990, RMSE 0.478 px, 1.3 s | 98, 1.000, 0.980, 0.062 px, 5.2 s |
| bench2 gradient-opaque-stroke | 177, 0.955, 0.828, 0.885 px, 3.5 s | 171, 0.982, 0.824, 0.126 px, 9.9 s |

bench2's misses that show (a found dot more than 1 px away, at least 10% of the ring on top) went from 18 to 9, most
showing 10–24% of their ring, about where solid-opaque-stroke is (7); its 8 dots 6–8 px off in the stacks are gone.
No other bench row changed. In the page: 171 dots, 2 flagged, no console errors.

**Milestone 4.**

- [ ] Ground truth and benchmark rows for each test image. (bench1–3 have exact truth for their 59 renderings.)
      Status 2026-11-14: every testgraphs image has truth but dog-ages-actual-size (derivable from dog-ages.json
      through the measured 3× scale and offset); every bench-nature image has truth, but four lane CSVs (cells1,
      hair1, protrusion3, tumour1) can't be scored until the bench pairs category ticks with lanes.

### Phase 6 — Polish and deploy
- [x] Layout and editing review (2026-11-12, below).
- [x] About panel (2026-11-14, below).
- [x] CSV and truth JSON export, copy to clipboard (Copy table), URL state (`#image=`), dark theme.
- [ ] Optional OCR of tick labels (open question 2).
- [x] Move dot, Clear all, Copy table and Save CSV (2026-11-13, below).
- [x] README.md (2026-11-14).
- [ ] GitHub Pages deploy (from the branch root, no workflow needed) and live-site check.

**About panel and README** (2026-11-14). An About button at the toolbar's end opens a modal `<dialog>` (Esc and its
Close button close it; the view's keys are ignored while it's open): what the app does and how (analysis by
synthesis), the steps of using it, keys and mouse, what to trust (hidden opaque dots, noise-only standard errors, tick
values), privacy (nothing is uploaded) and the license. README.md covers the same briefly, development commands, why
the test images are local only, and deployment: GitHub Pages from the branch root, nothing to build. Without local
test images (the deployed site) the test-image list is hidden and the empty view says to open, paste or drop an
image; checked by serving only the deployable files (no console errors). The site is
https://xangregg.github.io/dotscraper/ and the About panel links the repository.

**Dev mode** (2026-11-14). The test-image list, the `#image=` URL state and Save truth are for development: they show
only in dev mode, which is on for a local host (localhost, 127.0.0.1, ::1, *.localhost) and off elsewhere, and `?dev`
or `?dev=0` overrides it. Out of dev mode the app doesn't probe for test images, so the public site makes no failing
requests. A red "dev" badge by the title marks it. Opening a file keeps the query string (`?dev`) when it clears the
hash. Checked headless: localhost (dev: 10 test images, Save truth), `?dev=0` (neither, the public message) and
`?dev#image=dog-ages.png` (opens it), no console errors.

**cells1's magenta family** (2026-11-15). cells1 found only its green dots (38 of 72). Magenta (196, 78, 174), pink
fills in the same black outline, failed all three tries: no isolated whole dot (the gray lines joining each session's
two dots chain them, and most sit over a gray bar), so its radius came from the distance transform (4.5 for 20), and
the fill-disc fallback found no round fill because isolatedDots judges a component in a window 4 px around it, which in
a cluster takes in the neighbors' fills 2 px away across the outlines. Now:

- `roundFills` judges a fill component's roundness on its own pixels and antialiased rim, not a window.
- Stroked families use the fill-based radius and center search too (their fill weight, as outlined ones use the fill's
  coverage).
- The local outline model (try 3) needs a ring darker than the fill: with round fills found, bench3 ANOVA's pink under
  a pale blue band became a second family "outlined" in the band's color (31 dots; it had 204 in one).

Tried and reverted: letting any background color (the bar's gray, 2 isolated magenta fills on white against 8 with it)
surround an isolated fill in strokeColor. It wasn't needed, and it let bench1 group-smoother-ci's red–blue overlap
purple become a stroked family (109 for 100). cells1: 38 → 73 for 72, P 0.904 R 0.917 (rmse 2.2 px: the truth is data
values in jittered lanes); hair2 9 → 10 (P 0.900); phase1 65 → 75 (P 0.387 R 0.299); everything else unchanged.

**degradation1** (2026-11-15). Mint and orange fills in gray outlines, a thick gray regression line, two legend dots.
Only a translucent gray "family" was found, with no radius: its levels were the line (166, 2330 px) and the outlines
(136, 6936 px). The fills' ring, the outline gray, lay on that family's line, which ruled out the outline model for
them (a guard against economist's red–blue lens). The local outline model (try 3) no longer takes that guard: its own,
a ring darker than the fill and a thick fill, already stop the lens (the economist test passes either way). And a
translucent candidate with a level in an opaque family's outline or stroke color is dropped as those outlines. Now two
outlined families at r 21.6–21.7, 58 dots; every other bench row unchanged.

The source CSV's first column is the predicted value, plotted on y: the columns are swapped in the file now, with a
note (the user's original is kept aside). The data has no train/test split and no legend dots, and many rows are
hidden in the upper cluster, so the bench's group scoring linked neither family to the single 127-row group (each
must take 20% of it) and scored 0. A family now links when it takes 20% of the group or of its own matches, whichever
is fewer. degradation1: 58 for 127, P 0.672 R 0.307; rmse 6.3 px from cluster pairings (isolated dots are within 1–2.5
px, about what the published values and the drawing agree to).

**Close pairs toggle** (2026-11-15). A "Close pairs" box beside Dots and Axes hides the magenta marks and shared rings
of possible duplicates; off, such a dot is drawn by its other flags (red if any, else blue). The close flag itself stays
in the Flagged count, the table and the hover line; the Dots legend drops the magenta item while it's off.

**Outlined families fitted by their rings** (2026-11-15). With the user's hand truth for degradation1 (70 dots with
some interior visible, legend dots included), 58 found, P 0.966 R 0.800. Of the 7 missed mint dots, 6 were pairs of one
family merged: fitted as a union of discs, fill and outline alike are "dot", so two overlapping mint dots are one blob,
and the upper one's outline crossing the lower, which tells them apart, is invisible (a dot added at a missed truth
position had removal margin 0). Now an outlined opaque family (a visible ring of its own color) gets a ring image
(style.familyRings: each pixel's weight of the outline in the triangle page–fill–outline, 0.45 for a full outline;
unknown off the triangle, on other families' fills and the rings nearer them, and beyond 4 px of this family's
thick fill) and is fitted by outline.js's layered model with opaque fills (families.ringEstimate), when
outlinesInP measures a consistent ring in it (the existing limit, rms 0.06). Tried and not kept:

- A looser ring limit (0.12) let the mint fit by its 3 poorly measured rings (0.095): recall 0.843 → 0.771; where most
  of each ring is hidden or another family's, the union, which also reads the visible fill, places dots better.
- Sharing the best-measured ring (width, opacity, blur) among families of one outline color: no change.
- Stroked families too: engagement1 (strokes under a pixel wide) 54 dots for 49 → 40, R 0.980 → 0.816.

degradation1: the orange family by its rings, 58 → 61, P 0.967 R 0.843 (rmse 2.7 px; against the source data, 61
for 127, P 0.705 R 0.339); every other bench row unchanged (phase1's and hair2's rings aren't measured consistently,
so they stay unions). In the app, such a family's Model view draws each dot outline then fill in its fitted order
(sharp, blurred once: blurring each pass blurred the dots under it again), the Residual compares in ring units with
the ordered model, and the style panel says "fitted by its outlines in drawing order" (no α of its own).

**Diagnostic views are dev-only** (2026-11-15). Count, Mask, Classes and Model show only in dev mode (class
`dev-only`); Image and Residual stay for everyone, the residual being how a user sees what the dots don't explain
(the display-honesty rule). Checked headless: all six views on localhost, Image and Residual with `?dev=0`.

**Two toolbar rows** (2026-11-15). The header is two explicit rows, each a left and a right side pushed apart: row 1
the name and the image source | export and About; row 2 Find dots, Stop and the status | the edit buttons. A narrow
screen wraps each row's right side onto a line of its own. check-page takes `--width` for that; checked at 1280, 900,
600 and 400 px and in the public view, no console errors.

**beeswarm-boxplot** (2026-11-15). A JPEG-noisy screenshot (31%) of packed red and gray beeswarms, a few outliers in
black outlines; the user's truth has 711 dots in four families (red, gray, and the outlined of each). No family was
found. Five changes, each from a measured cause:

- With exact colors, JPEG fringes gave red a dark red "background" and no dots; the app turned on noisy color
  matching only when exact colors found nothing, and they found gray. Now, on an image with JPEG noise, noisy color
  matching is also tried when exact colors find families, and kept if it finds more (app and bench alike; the hint
  says why). It costs a second analysis on noisy images only.
- In noisy color mode the colormap took gray and its noise for a color scale: one curve point held 97% of the votes,
  15 noise points of 5–27 votes reached a spread of 0.23. A curve whose largest point holds over half its votes is no
  scale now (colormap.js MAX_POINT_SHARE; the gradient renders: at most 11%).
- Gray's radius came from the distance transform, 6.2 for about 9: no dot is isolated in a packed swarm. When that
  radius finds too few strong centers, Hough tries larger radii (to 1.6×) and keeps the largest with 75% of the most
  strong, filled centers (the evidence leans small in a pack): 9.0. Only for a fill on the page: one surrounded by
  another ink is an outlined or stroked dot (hair3 and cells1 passed as plain fills, 26.4 px for 28.3).
- Gray then had 30 strong centers but coverage 0.27 (Hough finds part of a packed swarm): 10 strong, filled centers
  now pass a family without the coverage or isolated dots (DOT_MANY_STRONG).
- At 8.2–8.7 px the fit put two dots in nearly every gray dot, and its radius refit settled small because the pairs
  covered the dots (474 dots, 352 close; at a fixed 9.2, 282 and none). fitDots refits an opaque family 0.5 px larger
  while over 40% of its dots are close, up to three times.

beeswarm-boxplot: radius unknown → 682 for 711, P 0.956 R 0.917 (gray 284 at r 9.09, no pairs; red 398; the outlined
outliers aren't separated yet). Also: delta1 no family → 23 of 48, P 0.957 R 0.458; phase1 75 → 50 found, P 0.387 →
0.580 (the doubled refit), R 0.299 the same; everything else unchanged. Fits on noisy images take longer (this one 129 s
in the bench, both modes).

**RESULTS.md** (2026-11-15). `npm run results` runs the full bench (`--json` writes its rows and failures) and rewrites
RESULTS.md: a table per set, synthetic presets summarized over seeds, every image with truth kind, color mode, counts,
precision, recall, RMSE, flags, radius, time, and a hand-written note that regeneration keeps. First run: 119 rows, 10
failures (radius unknown, no family, category ticks not paired).

**Layout and editing review** (2026-11-12, from the user's notes and a review of my own).

- Edit did nothing: the first click threw (`editAt` passed `families: undefined, dots: undefined` as extras to
  `setFits`, which spread them over the new values), and every click after it only panned. Edits are now two
  modes, **+ Add dot** and **− Remove dot** (A, R; Esc ends), so a click near a dot is never ambiguous. Remove
  highlights the dot it would take. **Undo** (⌘Z) keeps 100 steps, a Find dots included.
- Edits re-added every outlined dot without its drawing order, and an added one went on top: `refitNear` now keeps
  each dot's z and places an added one with `bestZ` (regression test in outline.test.js).
- The header holds the actions: source, Find dots and Stop with a progress bar and a status line of fixed width
  (the stage, dot count and elapsed time, ticking between the fit's messages; the bar is indeterminate until the
  first stage), the edit modes, and Copy CSV and Save truth. Above the image: the layer choice, Dots and Axes
  toggles, zoom. Under it: a readout of the pixel and dot under the pointer and the layer's legend.
- The panel is 360 px wide, in collapsible sections that remember their state: Dot families and Dots open; Axes
  (with a summary line), Mask and Image closed.
- Dot families are a list with a checkbox each (Find dots fits the checked ones), a swatch (the fill colors side
  by side, or a hatch when the dots vary in color), and their dot counts once fitted. Selecting one shows its
  style and overlays and no longer throws the fit away (it did). **+ Add family by color** adds a picked color as a
  family of its own instead of replacing the list. The level swatches are labeled "Overlap levels" 1, 2, 3, … with
  a tooltip; outlined and color-scale families hide the levels rows and the α box, whose values are internal.
- A resize refits the view until the user zooms or pans: the image was sometimes drawn at 3% in a corner, fitted
  before the stage had its size.

Not done: an automated test of the page's edit flow (checked by hand in the Browser pane with clicks: remove,
add, add on top of another dot, ⌘Z, an outlined family's add); fit speed in the app, which a hidden Browser
pane throttles (bench2 group-graystroke 57 s there against 6 s in Node).

**Manual truth editing and bench-nature** (2026-11-13).

- **Move dot** (M): click a dot of the selected family, then the arrow keys move it 1 px, or 0.1 px with Shift. It
  isn't refit, which would pull it back; its fit numbers (standard errors, visibility, flags) are dropped and it is
  marked moved. A run of nudges to one dot is one undo step. **Clear all** removes every dot, undoably.
- Copy CSV became **Copy table** (tab-separated, as before) beside **Save CSV** (RFC 4180, a file); missing data
  values are empty cells.
- `bench-nature/` (git-ignored): Nature figures, DOI in each file name, CSV truth in data units. Headers added
  (ticks, scales, source); tick lists include unlabeled minor ticks, which detectAxes counts (protrusion1–3).
  Category-name ticks (escape2: early, late) now become lanes 1, 2, … in truth-csv.mjs. Not paired yet: hair1,
  hair2 and protrusion3 have no x tick marks (hair2's six "ticks" are dots resting on the axis), so their lanes
  need a plot area or lane centers. The bench reads the set (`--set bench-nature`, below).
- **No dot family** on 8 of 19 images, all PNGs. Most are JPEGs in their history (screenshots of web figures): the
  edges' colors leave the line between dot and background (primates1's isolated dot: 207, 233, 243 then
  128, 182, 213 between white and 33, 118, 183), so no edge passes the opaque test (0 isolated dots, 1 strong
  Hough center). With the JPEG-compressed box checked, autocor1, kite1 (both colors), primates1 and hair1 are
  found. Still none or wrong: fanc1 (a dozen colors of 1–4 dots each, under the family thresholds), hair2 and hair3
  (black outlines on pink and gray fills; JPEG mode finds hair3's light gray instead, open question 10), kite2
  (translucent, no isolated dot for a radius).
- **JPEG history detected** (`jpegNoise` in jpeg.js). Tried first: how far restoreChroma moves the chroma at luma
  edges. It doesn't separate: clean PNGs with outlines score high (three colors meet at a ring: bench1
  solid-graystroke p90 37.7 units) and JPEGs of gray dots score 0 (no chroma to bleed). Kept: of the background
  pixels 3–6 px from any mark, the share that isn't exactly the background color. A render leaves them exact; JPEG
  ringing reaches across its 8 px blocks. Over 98 images: the bench renders 0%, the testgraphs PNGs at most 0.013%
  (dog-ages), the bench JPEGs 19–51%, the Nature PNGs 4.8–51%, and four testgraphs PNGs 27–37% (pathogen,
  beeswarm-boxplot, fibers-distance, violin-beeswarm). Threshold 1%. Noisy synth.js presets read 29–40%: noise of
  any kind looks the same, so the app calls it "JPEG-like".
- **When it acts:** only when the exact analysis finds no family. Always on would cost: pathogen in JPEG mode found
  867 dots for 583 (precision 0.972 → 0.637, recall 0.780 → 0.947, 3.7× slower), restored chroma pulling its fills
  off their plane beside the rings. The app measures the noise on loading, shows it under Image ("Noise near
  marks"), and if no family shows, turns JPEG mode on and says so in the status and the family notes; the box stays
  the user's. The bench's extractAll does the same and marks the row "(J)". Every existing bench row is unchanged
  (synthetic, bench1–3, testgraphs: diffed against HEAD without times).
- **bench-nature in the bench:** the CSV sets are testgraphs and bench-nature; lane data paired through ticks is
  scored by lane (laneRow; truthPixelsByTicks now returns lane centers). Baseline:

| case | truth | found | precision | recall | rmse px | note |
|---|---|---|---|---|---|---|
| autocor1 (J) | 804 | 675 | 0.948 | 0.796 | 0.762 | no family before |
| escape1 | 38 | 33 | 1.000 | 0.868 | 0.121 | lanes |
| escape2 | 145 | 43 | 1.000 | 0.297 | 0.486 | lanes; dense translucent gray |
| fanc2 | 52 | 37 | 1.000 | 0.712 | 0.148 | many ties at y = 0 |
| hair3 (J) | 34 | 25 | 1.000 | 0.735 | 0.427 | lanes; the light-gray fills only, scored by y in the lane |
| tumour1 | 44 | 13 | 0.231 | 0.068 | 2.501 | white fills, black outlines: the wrong family |

  Not scored: hair1, hair2, protrusion3 (no x tick marks to pair the lanes with), protrusion1–2 (a translucent
  family with no isolated dot: "the dot radius is unknown"). primates1 (no CSV) finds its 27 dots in the app,
  every dot in the image (regression test in jpeg.test.js).
- **JPEG mode whenever the noise shows?** (2026-11-13, asked because pathogen is hard for other reasons.) Every
  noise-flagged PNG, exact mode (with the fallback) against JPEG mode, truth paired through the original image's
  ticks (the restored image's ticks differ: autocor1 4 y ticks for 7, tumour1 none on x):

| image | exact | JPEG | |
|---|---|---|---|
| escape1 | P 1.000 R 0.868 | P 1.000 R 1.000 | better |
| escape2 | 43 of 145, R 0.297 | 136 of 145, R 0.938 | better |
| autocor1, fanc2, hair3 | | | the same |
| pathogen | 468, P 0.972 R 0.780 | 867, P 0.637 R 0.947, 3.7× slower | worse |
| engagement1 (the user's 46-dot export) | 33, P 0.879 R 0.630 | 40 in 2 families, P 0.650 R 0.565 | worse |
| violin-beeswarm (no truth) | 734 at r 7.4, 23 s | 3190 at r 4.4, 497 s | worse |
| tumour1 | a wrong family, P 0.231 | no family | neither |

  Without truth, fibers, kite1, molar1, primates1–2, wave1 and hair1 come out within a dot or two either way;
  beeswarm-boxplot, fanc1, kite2, hair2 and protrusion3 fail both ways. So JPEG mode helps some and hurts as many,
  pathogen not the only one: kept as a fallback for now. Better: run both and keep the one the image supports (the
  residual), since analysis by synthesis can score itself. Also seen: in JPEG mode the bench's extractAll finds the
  plot bounds on the restored image, whose ticks differ; the app uses the original's.
- To make the fallback visible, the option moved from the Image section to Dot families as **Noisy color matching**
  (JPEG named in its tooltip), with a line under it saying why it is set: on by itself (no family with exact
  colors, and the noise), on for a JPEG file, or, when the image has JPEG-like noise and the box is off, "try this
  if dots are missed".

**Hand-made truth** (2026-11-13). Truth placed by hand is the app's Save truth JSON (image px, family and fill per
dot, the image's pixel hash), in `testgraphs/truth/` and `bench-nature/truth/`; `<name>.csv` stays for published
data in data units. A Save CSV export under that name isn't truth (fit-dependent columns, labels from the fit's
colors): the bench skips it with a note, and `node tools/truth-json.mjs <export.csv> <image>` converts one (done for
engagement1). The bench used to fit only the first family for the JSON truth and read only testgraphs PNGs; it now
fits every family, scores them by group (each truth family, and fill) like the CSV group rows, reads both sets and
JPEGs, and skips a PNG whose pixel hash isn't the truth's. The old rows understated the multi-family images:

| image | before (family 0 only) | now (every family) |
|---|---|---|
| cantons-haemocytes-dotplot | 60 of 239, P 1.000 R 0.251 | 239 of 239, P 1.000 R 1.000 |
| Laser-Asyllable-Maturity | P 0.813 R 0.371 | 80 for 70, P 0.875 R 1.000 |
| nicest language | P 0.971 R 0.414 | 149 for 162, P 0.987 R 0.907 |
| dog-ages (one family) | P 1.000 R 1.000 | the same |
| bench-nature engagement1 | – | 44 for 46, P 0.841 R 0.804 |

**PDF captures of bench-nature** (2026-11-14). The user re-captured the Nature figures from the PDFs (2–3× the
pixels of the web captures, clean: no JPEG noise but fanc2's 16%), replaced primates2 and added cells1 (data in
CSV), comment1, crystal1, delta1 and phase1 (truth JSON). cells1's header had "(implicit from categorical labels)"
after its ticks, which made them names; it is now `1, 2` with the names in a note. From the user's notes:

- **Small families beside regular ones** (comment1). The dotted gridlines' black specks passed as a small-dot family
  (r 2.53, 9 isolated) beside the orange dots (r 9.2), and fitting it over all the text took over 5 minutes (2923
  dots). `findCandidates` now keeps families under DOT_MIN_RADIUS only when no regular family exists (translucent
  ones count as regular: their levels already show dots). comment1: 102 dots for 94, P 0.922 R 1.000, 3 s.
- **Near-duplicate dots** (comment1). Its dots have a thin white outline that cuts into the dot below; the opaque
  model has none, so the fit puts a second dot 0.1–2 px from the first to absorb the edge, and it is worth more than
  λ. All 8 extras are such pairs (10 pairs; the truth has 2, one 0.13 px apart, probably kept from a fit). `assess`
  now flags a dot within MERGE_RADII (0.35 r) of another of its family `close` (`flagClose`, also rerun after a
  nudge), and the view draws it in magenta with a second ring 4 CSS px outside. Real ties get it too (bench1: one
  pair; bench2 large-dots: 60 of 204), so its wording is neutral: translucent pairs show in their levels, opaque ones
  can't. Modeling the white outline (open question 10) would remove the cause.
- **No dots on crystal1.** Nature frames draw ticks into the plot, and labels were sought on the ticks' side only:
  pink dots "labeled" 2 of 5 y ticks, the plot was bounded on the wrong side (right = 213, the frame's left edge) and
  86% of the image was masked. `findLabels` now searches both sides and keeps the one labeling more ticks; the far
  side needs at least 3 (OTHER_SIDE_MIN_LABELS: beeswarm-boxplot's "***" labeled 2 of a significance bracket's 3 ends
  and made it an x axis). crystal1: 0 → 397 dots for 391, P 0.980 R 0.995 (rmse 0.79 px: hand-clicked truth).
- **phase1: no family.** Gray fills (82, 136, 187, 228) and a black one, all in black outlines about 2.5 px wide, with
  fitted curves of the same grays drawn through them. The curves (13 px wide) outvote the dots in the distance
  transform (radius 6.33), no fill is isolated (beyond the black ring, white is off the black–gray line), the stroke
  model needs the stroke off the white–fill line (here all are gray) and the outline model is gated by isThin(black),
  which fails because the black family, text and a colorbar share the color. Forcing the outline model makes every
  black glyph a dot. Left for open question 10.
- **engagement1's capture** is now claimed by outlineEstimate (outlines a darker shade of each fill, on a color
  scale): one virtual family by rings, 22 of 49. Its regression test is marked todo; primates1's JPEG-fallback test
  skips while primates1 is clean (a JPEG made with jpeg-js, full chroma, doesn't reproduce the failure).
- UI: Add mode makes the added dot the chosen one, so arrow keys nudge it (refitNear returns its index; the dots come
  back sorted by x). Number fields were muted gray, as if disabled: `input[type="number"] { color: inherit }` beat
  `.params input` at equal specificity; they are ink now, and disabled ones muted and dashed.

bench-nature (old code → now, the new captures): comment1 2923 → 102 for 94; crystal1 0 → 397 for 391; every other
row the same but its flag count. bench1–3, the presets and testgraphs: counts, precision, recall and rmse identical to
HEAD; only flags rise (the `close` pairs). Still wrong on the new captures, not yet looked at: fanc2 1143 for 52,
wave1 931 for 43, hair3 722 for 34, primates2 97 for 15, hair2 170 for 18 (overcounts at r 4.5–12, probably
another small family over text or lines); autocor1 R 0.833; cells1 (no x ticks found for its two lanes), delta1,
kite2 and phase1 (no family), kite1, molar1 and protrusion1–2 (no radius).

**fanc2 and wave1 overcounts** (2026-11-14). comment1 was re-truthed (92 dots, no near-duplicates): its 10 extra
dots are now exactly the 10 `close` pairs. wave1's legend has three dots, counted in its truth.

- **wave1** (931 for 43): a black family at r 3.23, the legend's and ticks' decimal points (27 isolated, coverage
  0.19), beside the three dot families at r 14.9, fitted over all the text in 132 s. The small-family rule now also
  drops a family under SPECK_RADIUS_RATIO (0.4) of the largest's radius whose discs explain less than
  DOT_MIN_COVERAGE of its color. wave1: 44 for 43, P 0.955 R 0.977 (the miss is a yellow dot half under the x axis),
  15 s.
- **fanc2** (1143 for 52): its PDF draws every overlap of the gray dots (86) in one flat black (19), as dark for three
  dots as for two, the text's black. Two failures. (a) The antialiased rim gray inside the overlaps (59) became a
  family: on the white–black line, black is 1.2 of the way to 59 (MAX_MIX allows 1.3), so black glyphs and lenses
  read as full coverage and Hough found "dots" there. A center now counts only if its inner disc is the family's own
  color, not just covered. (b) The lenses were unknown to the gray family, hiding the stacked dots at y = 0 (38 of
  52): `fillSaturatedOverlaps` marks covered any region of unknown pixels beyond the family's color on its line, up
  to OVERLAP_MAX_MIX (2), within OFF_MIX_ABS of the line, not another family's color, at least OVERLAP_BORDER_SHARE
  (0.9) enclosed by the family, with a 5 × 5 square inside it. Each guard came from a failing test: at border 0.6,
  the synthetic swarm's black median line over its light gray dots was filled (a line also has no square inside); with
  the noise allowance and no cap, economist's pink and blue dots were filled for a phantom light gray family. With the
  fill, fanc2 finds 50 rather than 39. A synthetic gray-and-black image didn't reproduce this (it reads as a
  translucent family of two levels; fanc2 doesn't, because its light confidence band at 251 is the first level on
  the gray line from white), so the regression tests use fanc2 and wave1 themselves.

bench-nature: fanc2 1143 → 50 for 52 (P 1.000 R 0.962, 136 s → 4 s); wave1 931 → 44 for 43; primates2 97 → 15 for 15
(the same small-family rule); every other row unchanged. Synthetic presets, bench1–3 and testgraphs: every row
identical to the last commit's, flags included. Still overcounting: hair2 (170 for 18) and hair3 (722 for 34) at r
4.5–4.8.

**hair2 and hair3** (2026-11-14). Both had one family: black decimal points (and a colon's dots), r 4.9–5.0,
coverage 0.02, 4–5 isolated. The real dots are pink fills in a black stroke (r 28–30), plus gray (hair3) and
white (hair2) fills in the same stroke. Four fixes, each a place where the stroke's black, which is also all the
text, counted for the pink family or against it:

- An isolated dot must have its fill at its middle (ISOLATED_FILL_RADII 0.5 × r, DOT_MIN_FILL of own color, or fill
  weight for a stroked family): in the stroke model the black specks read as covered and outvoted the one whole pink
  dot (median r 4.9 against 30.3).
- A stroked family's coverage counts its fill, not D, which covers all black ink (hair3: 0.26 → 0.67).
- The small-family rule runs inside opaqueFamilies, before the families are measured again with the others' colors
  excluded: the black speck family's color is the pink dots' stroke, and excluding it cut their radius to 18.
- A family needs coverage ≥ DOT_MIN_COVERAGE_ANY (0.1) even with isolated dots: a color 98% text is no family, even
  alone (real families sharing text's color measured 0.18).

hair3: 722 → 29 for 34, P 0.966 R 0.824, one outlined pink family at r 28.25 (its 5 gray dots in black strokes are
not found: phase1's case). hair2: 170 → "No dot family found": its 9 pink dots sit in two heavy clusters (one
strong arc) and its 9 white fills are the page's color. comment1: 102 → 103 (one more extra, flagged weak, by its
thick x axis: the speck family's black, excluded from the orange family's final measure until now, kept the axis
out). Everything else, synthetic, bench1–3, testgraphs and the other bench-nature rows, unchanged.

**Hand edits and the refit** (2026-11-14). An add or remove refines the family's dots within NEIGHBOR_RADII (2.2 r)
of it, since neighbors drift into a missing dot's place and spread into a removed one's. That refit used to treat a
dot placed or nudged by hand like any other (editAt passed only x, y and z, dropping the `moved` mark), so a nearby
edit silently undid a nudge. Now dots added or moved by hand are pinned (`fit.pin`: refineAll skips them, also when
the refinement spreads from a neighbor), refitNear reports which input each output dot came from so the marks
survive, and Option-click (Alt) in Add or Remove mode edits without any refit: an added dot goes exactly where
clicked (sub-pixel), marked placed. The hover line and the dot table (`by_hand`: added, moved, placed) say which
dots are the user's; a moved dot's standard errors and removal margin are dropped, as they describe the fit's
position, not the user's.

**Labels across the axis, economist** (2026-11-14). With the user's new truth, economist scored 0 of 210: the label
search on the far side of the ticks (for inward frames) found dots "labeling" 3 ticks above its baseline, one a chain
of dots 566 px wide, against its 2 year labels below, so the baseline became the plot's top and every dot was masked.
Far-side words wider than OTHER_SIDE_MAX_WIDTH (1.5) tick spacings are no labels now. A ratio of the two sides'
counts doesn't work: economist's 3 to 2 is the synthetic inward frame's 6 to 4. economist: 0 → 198 for 210,
P 0.874 R 0.824; crystal1 and beeswarm-boxplot's axes unchanged; every other bench row unchanged. New hand-truth rows:
fibers-distance 39 for 44 (P 0.897 R 0.795), violin-beeswarm 734 for 702 (P 0.956 R 1.000), pathogen (hand truth,
493 dots) 468, P 0.989 R 0.939; beeswarm-boxplot: no radius (no family).

**Per-family outlines** (2026-11-14, open question 10). An opaque family's outline is now read with the family, not
before any family is known, in three tries when the plain fill fails the dot test: (1) the outline model when the
ring's color is thin over the whole image (JMP's own-color outlines, as before); (2) the stroke model (a stroke too
thin to show its color, off the line from the page to the fill; it places colored fills better: cells1 rmse 0.28 px
against 0.95 by the outline model); (3) the outline model again for a thick fill whose outline color is also other
ink. For (3) to work:

- Outline pixels count only within OUTLINE_REACH (4 px) of the fill's thick parts: phase1's black outlines share
  their color with text, axes, a black family and a colorbar block, which all read as dots. Thick parts, because an
  antialiased edge from black to anything lighter passes through each gray (phase1: 82-gray pixels on every black
  edge put every outline within reach, and the 82 family took the light and black dots).
- Without an isolated whole dot, the radius comes from the fills: each dot's fill is bounded by its own outline, so
  the top dots' fills are whole discs even on a same-gray curve drawn behind them (phase1: the curves' distance ridge
  gave 6.3 for 19). Fill radius plus the outline width (the covered area in a ring outside each fill disc). The
  center search runs on the fill's antialiased coverage (the position between outline and fill), at the fill radius:
  a binary mask's stair-steps gave whole fills arcs of only 0.6.
- A cluster between a found family's fill and its outline is a rim only if mostly edge (interior < 0.3) or within
  the fill's color tolerance (a synthetic rim 39 units from its fill): phase1's darker grays sit between the lighter
  one and black, 62% interior.
- mergeAcrossRegions no longer merges outlined families: any two grays fit "one translucent family over white and
  over black" (phase1's black colorbar block was the region).
- The colormap's hue-outline reading (outlines a darker shade of each fill, on a scale) yields to families found one
  at a time when two or more are stroked or outlined and their fills hold 60% of the colored ink (engagement1: 4,
  0.87; the gradient renders: none). Translucent families far smaller than the others (SPECK_RADIUS_RATIO) are text
  read as levels (engagement1: black at r 3.5, 1362 "dots"); one without a measured radius stays for the user.

Results: engagement1 22 → 54 for 49, P 0.889 R 0.980 (cyan is no family; mid-blue fits its dots); hair2 no family →
9 of 18, P 1.000 (its pink; the white fills are the page's color); phase1 no family → 65 for 97, P 0.431 R 0.289 (grays
187 and 82 found; 136 fails coverage, 0.29; 228's ring is the page, its curve's white surroundings; black shares
text's color; about 7 dots per family fill the colorbar's blocks, and the 187 family puts near-duplicates where its
same-gray curve meets dots); hair3 and cells1 unchanged; every other row identical. Bounding the plot by where its
axis lines end would have masked the colorbar, but lines stop short of data in cantons (86 dots), comment1 (35, its
axis broken by dots) and delta1 (1).


**Bench: tick pairing for data truth** (2026-11-15). Four bench-nature images with data CSVs couldn't be scored
("ticks unmatched" in RESULTS.md): the header's tick values pair with detected ticks only when the counts agree.
- More ticks detected than the header lists: its values pair with the labeled ticks when those are as many (cells1's y
  has unlabeled ticks between 0, 100, 200, 300).
- A perpendicular line starting at the axis, at its end, is that end's tick when it continues the ticks' spacing
  within 3% (`CORNER_SPACING_TOL`); like other ticks on a line, it is kept only if labeled. cells1's "0" tick is
  hidden by the bars' baseline, which runs right from the y axis's bottom end; without it only 3 of 4 labels paired.
  This is in detectAxes, so the app gets the tick too.
- Lane data whose x axis has no ticks to pair (category axes: cells1, hair1, protrusion3; tumour1 detects 4 x ticks
  for 8 bars) takes its lane centers from the fitted dots: their x split at the widest gaps into the header's number of
  lanes, each lane's median. Only x's lane matters to laneRow, so this costs no accuracy when lanes are farther apart
  than their spread.
- laneRow ordered lanes by their first appearance in the truth rows, while lane centers go left to right: hair1's
  first row is lane 2, which would have swapped its lanes. Lanes are now sorted.

Results: cells1 data truth scored, 73 for 72, P 0.959 R 0.972 (the hand truth gives 0.904, 0.917); hair1 16 for 29,
P 0.875 R 0.483 (only the pink family: the white-filled dots aren't found); protrusion3 and tumour1 now pair their
ticks and fail in detection instead (radius unknown; no family). Every other row identical (full bench, diffed
without times).

**Fills in the page's color** (2026-11-16). hair1's white PBS dots and every tumour1 dot are white in black
outlines on a white page. The color clusters leave out the page's color, so no fill showed (tumour1: no family).
- Each such dot's fill is a page region its outline encloses (not touching the image border). Round enclosed regions
  of one size (within 10%, at least 3) are whole fills; hair1 has 11 at about 930 px² (35 × 35, fill share 0.76–0.78,
  a disc's being 0.785), tumour1 about 30 at 207 px². The page color is tested within INK_TOL (15): bench2
  stroked-dots' translucent fills, about 230 on white, passed the cluster tolerance (40) and replaced its family.
- Enclosed regions up to 1.2 × a whole fill's area are fills, whole or partly hidden, when they lie in the disc of a
  Hough center at the fill radius: a hidden dot's crescent shows an arc of its own edge. Without that test hair1 got
  26 dots, 15 of them in letters' counters (P, B, D, the title's) and in the gaps among the pink dots, which are
  bounded by arcs curving the other way.
- The outline must be thin beside the fill (at most 0.4 × the fill radius): comment1's geometric font made a family of
  11 counters (o, 0, d, a) whose stroke is 0.56 × the counter's radius; hair1's outlines are 0.19, tumour1's 0.27.
- measureOpaque takes a page-colored fill with an outline as this kind (so the app's re-measure works too): coverage
  1 on the fills and the outline next to them, the outline's mix with the page beyond, faint pixels beyond the reach
  page (they read unknown at first, which broke isolated dots: tumour1's radius fell to 10.3 and the fit doubled 9
  dots). The fit can't see a white fill on white, so it fits the outline's color (style.fitColor); the display
  keeps the white fill. Another family's page-colored fill is no one's ring owner or "other color".

Results: tumour1 no family → 41 for 44, P 0.976 R 0.909; hair1 gains its white family, 11 dots, the pink unchanged at
16. hair1's CSV had its lane codes swapped against its note, which the old lane-order bug had hidden; with the CSV
fixed (2026-11-15) hair1 scores 27 for 29, P 1.000 R 0.931, RMSE 0.49 px: all 11 white PBS dots, 16 of 18 pink DT
(one of three ties at 5.88 and one of four zeros missed). Every other row identical (full bench, without times).

**bench-plos: first look** (2026-11-15). Ten PLOS ONE figures (CC BY) in bench-plos/, seven with hand truth
(truth/<name>.json) and four with a source-data CSV. The bench and tools/results.mjs now include the set (`--set
bench-plos`). The CSVs came without a `#` header; their tick values and scales were read off the images and added.
adults1 lists only the x ticks 200–600: the panel letter C overlaps the 100 label, so that tick isn't detected.
Placed through the detected ticks, grader1's CSV matches its 40 hand-truth dots at 0.55 px RMSE, and adults1's 138
of 139 at 0.60 px; adults1's CSV has 142 rows (the figure says n = 142), 4 of them not within 4 px of a hand-truth
dot (presumably behind others; not checked). Hand truth, with the code as of 06a5518:

| Image | Truth | Found | P | R | What it shows |
|---|---|---|---|---|---|
| adults1 | 139 | 146 | 0.952 | 1.000 | opaque black; the 7 extras are in the dense cluster, and the CSV has 142 rows, so some may be coincident dots the hand truth lacks |
| boxdots1 | 268 | 152 | 0.645 | 0.366 | small black dots (r 4.9) over red and green boxes, JPEG noise; most dots over the red box are missed |
| grader1 | 40 | 46 | 0.870 | 1.000 | translucent green with a ring; the 6 extras are in overlaps |
| microplastics1 | 24 | – | – | – | opaque black (0, 0, 0), crisp, r ≈ 11, many isolated, yet "no isolated whole dot" (read as α 0.81, color −2) |
| myeloid1 | 18 | 9 | 1.000 | 0.500 | orange found; teal (20, 154, 155), stacked in one column under a black mean bar, is no family |
| score1 | 168 | 316 | 0.528 | 0.994 | teal (α 0.72, white edge) takes 166 of 168; a second, gray family (131) from the label leaders and text adds 150 false dots |
| score2 | 92 | 99 | 0.929 | 1.000 | viridis fills, darker outlines (the colormap path); a few dots fitted twice; 54 s |

Data truth: adults1 142, found 146, P 0.945 R 0.972 (4 data dots missed); grader1 40, found 46, P 0.870
R 1.000; droplet1 and landing1 find no family; microplastics1 has no CSV.

Without hand truth: droplet1 (black outlines about 3 px around flat fills of a diverging scale, purple to white to red)
and landing1 (translucent blue and orange dots over translucent violins of their own hue, gray paired lines) find no
family; step1 finds the blue family's color (123, 168, 229, α 0.56) but no radius, and nothing for the hollow red
circles.

Hollow dots (step1): the forward model already draws them, a ring of its own opacity over a fill of opacity 0
(render.js: `1 − α·m − ring.alpha·m_ring`), and the ring fit (outline.js, ringCenters) fits rings. What's missing is
finding and measuring such a family: the scan starts from flat fill pixels, and a hollow ring has none. A ring
family would be found from thin round strokes of one color (Hough on the distance to the stroke color, as pageFills
validates its fills) and measured from isolated rings (radius, width, opacity; outline.js measureRing). Its inside
must stay transparent, not the page's color as in pageFilledFamily, since step1's rings show the band and the blue
dots through them. In reach, but step1 is a hard first case: rings about 6 px in radius and 1.5 px wide, blurred,
JPEG-noisy, over a translucent band, a curve and another family.

**bench-plos: microplastics1, myeloid1, score1** (2026-11-16). Three scan failures, three changes in style.js:
- microplastics1 (radius unknown): a 3 px gray border bar down the right edge (48, 820 flat pixels) and the black
  dots fitted a translucent black family as levels 1 and 3 (α 0.81, level 2 inferred), which had no isolated dot,
  and black was then never tried as an opaque family (on the translucent line). A translucent candidate whose
  first level's color is thin over the whole image (isThin, as for outlines) is no family: level 1 fills the insides
  of single dots. 24 of 24.
- myeloid1 (teal no family): its 9 dots touch in pairs and a stack and gray pair lines (unknown pixels) reach each,
  so the best arcs were 0.72–0.76 at 13–13.84 px and no center was strong (0.75); the dot test needs 3. A plain
  fill rejected by the dot test now counts when its radius is within 10% of a family found (SIBLING_RADIUS_TOL;
  figures draw every group with one marker size) and it has DOT_MIN_STRONG filled centers of half an arc and
  coverage DOT_MIN_COVERAGE (teal: 5, 0.64). 18 of 18. This also finds nicest language's cyan (59, 186, 209, 15
  dots at 8.4 px): 149 → 164 found for 162, recall 0.907 → 1.000.
- score1 (a gray family of 150 false dots): the gray of the dashed line and the label leaders, 130.9, 131.1, 131.1,
  lies 39.9 units from the teal dots' 105, 151, 154, inside CLUSTER_TOL, so its color mask took every teal dot.
  measureOpaque's own pixels now exclude those nearer one of `others`, and opaqueFamilies passes the translucent
  families' measured level colors among the others. 166 of 168, no extra. Nearest-color membership also moved
  engagement1: no family takes another color's dots any more, 54 → 47 found for 49, P 0.889 → 1.000, R 0.980 →
  0.959 (mid-blue 19 → 15 dots, navy 13 → 10).
- Two regressions on the way, both from a family seen through a translucent band: bench3 ANOVA's pink under the
  lavender band (197, 106, 136) passed the sibling rule, and bench1 group-smoother-ci's red levels under the CI band
  (197, 106, 135) passed the ordinary dot test once the levels' pixels stopped counting as its own; either way the
  band was no longer divided out (ANOVA 204 → 203 in two families; group-smoother-ci 100 → 108 found, P 0.907).
  An opaque cluster c' is now no family when c' − P = s·(c − page), s in (0.05, 0.95), within CLUSTER_ON_LINE_TOL,
  for c a found family's color or a translucent level and P another cluster's color 20 units or more off the line
  from the page to c (LAYER_MIN_OFF: on one line, as phase1's grays, every color is any other seen through a
  third; the bands are 48 off, their mauves within 0.5–0.9 units of the relation).

Results (full bench, diffed without times against the run before): the three images as above, nicest language and
engagement1 as above, degradation1 within noise (data P 0.705 → 0.721, R 0.339 → 0.346); every other row identical.
Regression tests for microplastics1, myeloid1 and score1 (skipped without bench-plos). In the app (check:page,
dropped image, Find dots): microplastics1 24, myeloid1 18 (two families, r 13.8), score1 166; no console errors.

**bench2 group-large: overlaps read as families** (2026-11-16)

- At α 0.33 the red–blue overlaps cover many flat pixels, and two of them became translucent "families" of their own
  with no isolated dot and no radius: blue over red's level 1 (181, 160, 213) and red over blue's (202, 154, 193).
  Unmixing then gave them the overlap pixels, so the true families' images lost their stacks: red fitted 70 dots
  (65 matched) for 113, blue 39 for 91.
- A translucent candidate is now dropped when every measured level is within BLEND_TOL (12) of a stack of up to
  BLEND_DEPTH (4) dots of two or more larger candidates already kept, composited in sRGB over the page (dropBlends
  in findCandidates). Measured: the purple levels within 0.4–1.3 of a stack, the real families' 86–102 away; bench1
  group-large, bench2 group and group-smoother-ci have no such candidates.
- group-large: radius unknown → 203 of 204 found, P 1.000, R 0.995, RMSE 0.24 px (red 112 of 113, blue 91 of 91).

Results (full bench, diffed against the run before): group-large as above; every other row identical apart from
times. Regression test in test/unmix.test.js (skipped without bench2). In the app (dropped image, Find dots): two
families, 112 + 91 dots in 2.4 s; no console errors.

**bench-plos boxdots1: a box edge as the x axis, gray box lines, dots over the red box** (2026-11-16)

- The figure's tick labels are cropped, so its frame (bottom at y 683, two ticks) has none, and the green box's lower
  edge (y 618.5, x 341–555) became the x axis: dots touching it were its "ticks" and dots its "labels". plotBounds put
  the plot's top there, and every dot above it was masked: 152 found for 268, recall 0.366.
  - A tick is a straight stroke: its shortest stroke across its width is 0.8–1.0 of its longest on every real axis
    (median over the labeled ticks; step1's 5 px ticks 0.8), while dots touching the edge were 4 of 14 and 11 of 22
    px (TICK_MIN_STRAIGHTNESS 0.6). One real tick in nicest language is 0.3 (a dot on it), hence the median.
  - That left the box's upper line (y 640.3; its "ticks" 0.71) with dots above the box as labels. A label lies in
    plain view of the axis: the ink between them, across the label, is 0.00 (median) on every real axis and 0.62
    and 0.74 for these, through the box's fill (LABEL_PATH_MAX_INK 0.5; single real labels reach 0.84).
  - Every other image's axes are unchanged; boxdots1 has none. Without them: 332 found, P 0.675, R 0.836.
- The gray box outlines and median lines (52, 4–5 px wide) lie on the line from the page to the black dots and read
  as 81% covered; they border the masked box fills, so the thin-mark test (opening with other ink) kept them, and
  the fit put 64 dots along them. A cluster rejected as a blend of a family and the page but with interior pixels
  (BLEND_INK_MIN_INTERIOR 0.05; rims are 0.00 everywhere, the gray 0.18) is ink: the final measure takes it as one
  of the others, unless it is within CLUSTER_TOL of a background or a family's color (the near-page 236 cluster;
  bench2 gradient-opaque's neighboring shade). 232 found, P 0.858, R 0.743. It also applies to violin-beeswarm's
  violin fill (182, 187, 224, interior 0.62): 734 → 712 found for 702, P 0.956 → 0.985, R 1.000 → 0.999.
- The misses were over the red box (228, 24, 73), found as a region under the dots but read as other ink by the
  opaque measure, so the mask took the dots' edges. Region colors under opaque dots are now backgrounds for
  findCandidates (estimateStyle `regions`), when solid: an opening by the dots' radius keeps half the region's pixels
  (REGION_MIN_SOLID; boxes, bands and the page 0.71–0.99). engagement1's cyan and delta1's black dots, families not
  found, read as regions too (0.00), and as backgrounds they lost 7 and 8 dots; phase1's black outlines (0.06) left
  no family at all.
- boxdots1: 152 → 275 found for 268, P 0.645 → 0.938, R 0.366 → 0.963. Extras: 7 on the bold "4d", about 10 along
  the red box's dark edges.

Results (full bench, diffed against the run before): boxdots1 and violin-beeswarm as above; every other row
identical apart from times. Regression tests in test/axes.test.js and test/opaque.test.js.

**bench-plos droplet1: opaque black rings around a diverging scale** (2026-11-16)

- No family: the fills (purple to white to red) lie in a plane through the page, so the outline path (findOutlines)
  ran, and found 9 isolated rings, but the ink mask, meant for black text (MASK_INK 0.8 in P), took the rings too:
  their cores are 0.93 in P, and αk came out 0.028. Rings measured so far were translucent or gray: the 90th
  percentile of P over the isolated rings' ink is 0.33–0.53 on the fill plane (pathogen 0.40), droplet1 0.97.
- When no ring shows with the mask, the fills vary (the plane isn't one color) and the isolated rings' cores reach
  MASK_INK, the rings are now read without the mask; text is left to the plot bounds and the ring fit.
  - The style first came out radius 9, αk 1.26, rms 0.11 (over MAX_RING_RMS 0.06): one "isolated ring" was a glyph
    below the axis at r 15 beside rings of 9.75–10.5, and a free αk let a thin blurred annulus stand in for a sharp
    one. The rings used for the style are now within 20% of the median radius (ISO_RADIUS_TOL), and black rings
    are fitted with αk fixed at 1 (radius 9.0, inner 0.78, blur 0.5, rms 0.096, against BLACK_RING_MAX_RMS 0.1:
    the clean rings 0.052, two crowded ones more). With αk clamped instead, the fit stacked 2–3 dots on single
    dots to darken the model's faint ring: 68 found, P 0.603.
  - Not for one fill color: hair1's pink and page-filled dots in black outlines took the retry as one outlined
    family (82 found for 29); the opaque families read them better. Laser, primates2 and tumour1 reach 0.92–0.93
    from glyphs, which fit no ring, so they are unchanged.
- droplet1: no family → 46 found for 57 (data truth), P 0.891, R 0.719, RMSE 0.64 px. The misses are the two piles
  near the origin (about 12 and 3 dots nearly coincident), whose rings merge into dark areas that the openings
  (underRings) keep as lines and that are masked; 5 extras beside pairs.

Results (full bench, diffed against the run before): droplet1 as above; every other row identical apart from times
(hair1 checked again after the one-color rule). Regression test in test/outline.test.js.

**bench-plos step1: the blue dots under a band of their own hue** (2026-11-16). No truth yet, so nothing is scored.

- The blue family (translucent, α 0.56) had no radius: its levels were the CI band (181, 207, 240, 4800 flat pixels)
  and the dots (149, 185, 234, 820). Solving a dot over the band (124, 168, 229) gives the dots' color 95, 149, 223
  at α 0.66 and the band the same hue at α 0.46: on the family's line but no level of it, so every dot read as
  count 2 and none as a whole one.
- withoutBand (style.js findCandidates): when a translucent candidate shows no lone dot, it is measured again
  without its first level's flat pixels; if that gives lone dots (DOT_MIN_ISOLATED) and, at their radius, the first
  level's color is a solid region (solidRegion), the level is a band, recorded on the candidate. With exact colors
  the image's JPEG noise leaves no lone dot either way; with noisy matching: levels 149, 185, 234 / 113, 161, 225 /
  96, 150, 224, α 0.634, color 89, 145, 221, 15 lone dots, r 5.43.
- solidRegion moved to style.js, REGION_MIN_SOLID 0.5 → 0.3: the band keeps 0.52 (every dot's antialiased rim
  passes through its color); boxes and the page 0.71–0.99, regions of dots 0.00–0.06.
- The band then read as partial coverage and the fit tiled it with false dots (194). regionColors no longer takes a
  family's band for the family's color, so it is a region under the dots (on the family's line, "under" and "over"
  predict the same colors; the tie goes to under): 133 dots.
- The app and bench analyze a JPEG-noisy PNG as a JPEG when that finds more families with a radius (was: with a
  style); with exact colors step1's family had a style and no radius, so noisy matching was never tried.
- In the app: one family, α 0.63, r 5.4, 133 dots, 89 flagged (77 hidden: under the dark curve, the red band's
  dashes and the masked hollow rings); unverified without truth. The hollow red rings are no family yet (see the
  hollow-dots note above).

Results (full bench, diffed against the run before): every row identical. Regression test in test/opaque.test.js.

**Hollow rings, first family (step1)** (2026-11-16). No truth yet, so nothing is scored.

- step1's red rings: r about 6.4, a 1.5–2 px stroke at about 0.31 of white's distance at its core, (232, 167, 153) on
  the page. Under the red band each ring's darkening is 0.61, 0.59, 0.60 of the page's per channel: the band is a
  layer over them letting 60% through. Where the bands overlap the color fits the blue band over the red one (blue
  band 46% opaque): ggplot drew the points, then the red ribbon, then the blue.
- outline.js findHollowRings: localFill (split out of relativeDarkness, unchanged) gives the fill around each pixel;
  fill minus pixel is thin ink's darkening. Its directions with at least 300 pixels of 40 units or more (up to 4,
  6° bins) are tried, skipping the found families' colors' directions from the page (hair1's black outlines,
  step1's blue rims). Along a direction u, P = darkening·u / white's distance, and a pixel off u by more than
  max(10, 0.3 × along) is unknown (text, the curve, the dashed line, the blue dots' rims). outlinesInP on that P:
  12 isolated rings, r 6.38, inner 0.73, αk 0.37, rms 0.016. A family only when the isolated rings' insides match
  their surroundings (median distance ≤ 12 units; a filled outlined dot's inside is its fill): fill α 0, and fit.js
  reads no fill for such dots. The direction histogram on step1: gray 19 000 px (text), red 3000, blue 600.
- families.js: hollow families are added after the others (and are the only ones when no other family shows),
  fitted by their rings in drawing order as outlined dots are; the app calls them "hollow rings" in the ring color.
- step1: 41 rings, every one on a real ring by eye. Missed: rings under the red band (60% of their ink, and the
  band is read as a region under the dots: the band-over-band area votes "under", 1388 to 248), and rings crossing
  blue dots, whose pixels are unknown in P. Undoing the two bands in drawing order would recover the first.
- Cost: localFill on every image analyzed, about 0.55 µs per pixel; the full bench 1200 → 1311 s. Every bench row
  identical: no other image gets a hollow family.

**step1: two bands over the points, divided out in order** (2026-11-16). Truth by hand: 100 blue dots, 75 rings.

- Before: 174 found (133 blue, 41 rings), P 0.655, R 0.651; blue P 0.549 (the bands' overlap tiled), rings R 0.547.
- The image's flat colors fit one drawing order: points, then the red band (s 0.65: a blue dot under it, 168, 144,
  165, is 237, 190, 179 + 0.65·(dot − white) per channel), then the blue band over both (overlap 169, 165, 191 = blue
  band + 0.65·(red band − white); a dot under both 124, 135, 181 = overlap + 0.65²·(dot − white)). The blue dots are
  44, 118, 212 at α 0.50, the blue band's own color: what read as their "level 2" (112, 161, 226) is a dot under the
  blue band.
- background.js: stacked regions. For each ordered pair of region colors, flat pixels on B₂ᵗᵒᵖ + s·(B₂ᵇᵒᵗᵗᵒᵐ − B)
  within STACK_TOL 4, a 1000-pixel peak in s, and an overlap color 15 units or more from B and every region color
  (kite1 and cells1: near-white pairs made "stacks" whose overlap was the page again): the upper band is a layer
  with that s whatever the dots' votes (a band in their own hue can't be told by them), and the overlap is a region
  of its own (base B, s the product; or the lower band's color, if that is under the dots), so its pixels don't vote
  for the lower band (red: 1388 "under" to 248 before, 2 to 226 after).
- Region seeds are the regions' bodies, pixels within 10 units of a region color that survive an opening by 1 px:
  a dot's rim passes through a band's color and, as a seed, took the dot's middle for the band. (2 px removed bench3
  ANOVA's band where it shows between packed dots: 204 → 202. A closed outline instead of nearest seeds was tried
  and dropped: under JPEG noise it left parts of the bands.)
- style.js findLevels: a first level whose color is thin over the image is dropped (with the bands divided out, the
  bands' edges and the dots' rims at 206, 222, 246 came ahead of the dots' own color, which the step ratio then put
  beyond the sequence). findCandidates: a translucent candidate without a lone dot whose first level is a solid
  region at the other families' radius is a band, dropped (the red band with 2 flat pixels of a ring's core as a
  weak second level).
- jpeg.js restoreChroma: the guided filter's windows are weighted by how well their chroma-on-luma line fits, and a
  pixel keeps its own chroma where they fit badly (three colors meet: a blue dot touching a red ring went from 148,
  187, 236 to 178, 178, 204, off the family's line). GUIDE_RES_SCALE 800: at 100, beeswarm-boxplot's red was no
  family; at 400–600, bench1 group-smoother.jpg's blue had no lone dot. At 800 it is worth 3 blue dots and 3 rings on
  step1 and is neutral on the bench's JPEGs.
- step1: 128 found for 175, P 0.992, R 0.726 (blue 73 of 100, rings 55 of 75 in the per-family match).
  Misses: dots and rings under the dark curve and the dashed line (masked), blue dots touching rings, and dots on
  the bands' edges, which the nearest-seed labels split.

Results (full bench, diffed against the last commit): step1 new; bench2 smoother-ci.jpg 187 → 195 found (P 0.995,
R 0.951), bench1 smoother-ci.jpg 97 → 98; bench2 group-smoother-ci 205 → 203 (P 0.985 → 0.995); bench2 smoother-ci
203 → 201 (R 0.995 → 0.985: two dots on the band's edge, the body seeds); bench2 gradient-dots.jpg 196 → 195 (P 0.995
→ 1.000); beeswarm-boxplot 682 → 687 (R 0.917 → 0.920); the rest identical or within 0.1 px RMSE. Tests: a synthetic
two-band overlay in test/background.test.js and step1 against its truth in test/opaque.test.js.

**bench-plos landing1: dots over violins of their own hue** (2026-11-16)

- No family. Two translucent families, α 0.75 (blue 76, 113, 176; orange 222, 132, 81), over violins drawn under
  them in their own hue at about 0.35 of the dots' offset (193, 206, 227 and 243, 212, 195): a dot over the page,
  121, 149, 196, and over its violin, 105, 137, 189, differ by (1 − α)·(violin − white) (likewise orange 230, 163, 125
  → 227, 152, 110). Each violin was its family's one level, the dots' colors beyond it at steps no sequence takes,
  so no candidate was translucent, and with no candidate no region was ever looked for.
- style.js overBand: a one-level candidate whose level isn't thin, with two colors beyond it a level or more apart
  (LEVEL_SEPARATION; beeswarm-boxplot's gray dots under a gray panel gave a pair 5 apart, 2129 dots for 711) along
  the band's offset by q in (0.05, 0.95), is a family over a band of its hue: α = 1 − q, color B + (X₁ − B)/α, X₁
  the first color beyond, X₂ the one of most pixels that puts the color inside the cube (blue: 105, 137, 189, α 0.747;
  the deeper colors gave −210 and −22). The band is recorded and read as a region under the dots.
- No dot is isolated, even with the violins read per pixel (every dot touches a gray paired line, a box, a whisker or
  a violin's outline, and the dots' darker rims read near 2). unmix.js: when a family over a band of its hue has no
  radius, style.js shapeRadius gives one: the distance transform's, then the Hough radius that keeps 0.75 of the
  most centers with half an arc (strong arcs: 2 and 3; half arcs 9 and 11 at 10–11 px). Only for those families:
  elsewhere it gave protrusion3, whose radius is the user's to set, one that found 120 dots for 50.
- landing1: 77 found for 70 (data, lanes), P 0.909, R 1.000; blue 38 for 34, orange 39 for 35, the extras doubled
  dots where they are densest.

Results (full bench, diffed against the last commit): landing1 as above; every other row identical. Regression
test in test/opaque.test.js.

**bench-nature kite1: dots on a line of their own color, a band over the dots** (2026-11-16)

- The first copy (a print rendering) was replaced by the online figure, with a data CSV (25 rows). The print copy's
  colors fit no compositing model: its red band lightened the red dots under it (234, 114, 107 → 249, 128, 118) and
  turned any teal under it into one flat 121, 177, 176, one dot deep or two; its bands crossed in a color neither
  order makes. Probably transparency flattened in another color space (CMYK for print). Work aimed at it (a band as a
  translucent family's only level, over/under decided by decisive votes alone, band crossings as regions, a radius
  through the layers for a lone translucent family) went unused on the online copy, and one part broke phase1; it was
  removed.
- Online kite1: opaque cyan (1, 191, 198) and red (249, 118, 110), r 12.3, JPEG noise (analyzed as a JPEG), a cyan
  and a red band. Before: 10 found, red only.
- measureOpaque: the cyan line, in the dots' color, set the distance-transform radius to its half width (3.5 for
  12.3), and the scan to 1.6 times that found no dot. When it finds none, it scans again around the radii the
  family's connected pieces give (componentRadii: the size-weighted median of their depths; see kite2 for the other).
- families.js: the red dots, most inside the red band, had the band as their background (254, 227, 225), and regions
  were told from the first family's background: the page became a region "under" the dots and the band none, so the
  cyan dots under it were missed. Regions are now told from the page first; without a layer over the dots, from the
  first family's background as before (from the page, violin-beeswarm's violins gave 18 more extras, P 0.985 → 0.960,
  and bench1 ellipse 15 flagged for 2). The red band then reads as over the dots, s 0.82 (0.81 measured).
- kite1: 26 found for 25, P 0.923, R 0.960 (data CSV and hand truth alike). The miss is a cyan dot 8 px from another;
  the extras two dots found twice, a cyan one and an opaque red one.
- Speed, results unchanged: houghGradient computes the Hough votes' gradient once per radius scan (doubles: Float32
  directions moved votes enough to change two bench2 fits), and the fit's births keep a list of candidate centers
  instead of scanning the image per attempt (6 of 13 s on the 2.8 Mpx print copy). The test suite: 754 → 405 s.

Results (full bench, diffed against 5377fb5): kite1 as above; every other row identical. Regression test in
test/opaque.test.js (it replaces the print copy's, a speck filter that once dropped a translucent family without a
radius: no image here has one now).

**bench-nature kite2: fills in rings of their darker shade, beside a curve of the fill's color** (2026-11-16)

- No family. Orange fills (209, 131, 82) in rings of 192, 85, 28, about 3 px; the fill is the ring's color at α 0.74,
  but overlaps (lenses 195, 98, 46) don't composite as an sRGB over (α 0.67–0.78 by channel; 6.8 off the line, past
  PERP_TOL), so no translucent reading. A curve of 211, 136, 88 joins the fill's color cluster (mean 210, 134, 85).
- As opaque: the cluster's surroundings were the page (the curve borders it longer than the dots' rings do), so no
  outline reading; the ring is on the line from the page through the fill (6 off), so no stroke (strokeColor wants
  one off it); and the radius was the curve's half width, 6.9 for 19.5.
- style.js opaqueFamilies, reading (4): when the fill's surroundings are the page, a color cluster in a darker shade
  of its hue (hueRings: on the page–fill line within CLUSTER_ON_LINE_TOL, beyond the fill; not only thin ones, as the
  ring and the lenses make a 3 px cluster) is tried as its outline, the fill read just inside the rings (ringFill:
  209, 131, 82; the cluster's mean made the fills 0.94 inside and none passed as round). The round fills in rings are
  the evidence, so the coverage need only be DOT_MIN_COVERAGE_ANY (0.32: half the fill-colored pixels are the
  curve's). The family is fitted by its rings in drawing order (r 19.0).
- Only for a fill with a hue (chroma, max − min channel, 30 or more; kite2's 127): a gray's darker shade is black,
  which is also text and axes. Without it, reading (4) took black as the outline of gray antialiasing (primates2: 136
  found for 15), of fanc2's gray (P 1.000 → 0.894) and economist's (radius 8.4 → 8.7), and of phase1's gray 228
  (50 → 69 found, P 0.580 → 0.696: real dots, but not ones black can be told from text for).
- And only for a shade next to the page: an overlap of translucent dots is the same darker shade, but lies inside
  them, where a ring lies at their rims. At least 0.3 of the cluster within 3 px of the page (kite2: 0.50; bench1
  medium-dots.jpg read as exact, its α 0.5 family unseen through JPEG noise, its overlaps taken for rings: 0.11).
- componentRadii also returns the deepest compact piece (area within 2.5 times a disc's at its depth): the curve,
  15 700 px at 6.4–7 px deep, made the size-weighted median 9.6 against dots 16–16.6 deep.
- kite2: 21 found for 21, P 1.000, R 1.000, RMSE 0.11 px. The app: one family, opaque, outlined, r 19.0, 21 dots,
  none flagged.

Results (full bench, diffed against 22712e8): kite2 as above; every other row identical. Regression test in
test/opaque.test.js.

**bench-nature molar1: black dots beside a gray label that fits as their level 1** (2026-11-17)

- Radius unknown. Black dots (19, 20, 19) on a black curve, and a label, "K_d = 119 nM", in gray 62, 62, 62: the
  two fit levels 1 and 2 of a translucent family (α 0.78, rms 0.17), so the dots read as two-dot overlaps, none whole.
  isThin doesn't catch the label: its strokes are 4 px, and an opening by 1.5 px keeps 70% of them.
- A translucent family's level 2 lies inside its level 1, since two dots overlap inside each dot. style.js
  levelsNest counts level 2's flat pieces (3 px or more) that have level 1's flat pixels within 6 px (flat pixels stop
  1–2 px short of an edge on each side, so 3 px found too few even in real families). Over every bench image's
  translucent candidates: real families 9–100% of their pieces (step1's band 1 of 42, but withoutBand reads that one
  first, and the check comes after it), and the ones at 0% had 7 pieces or fewer; text's levels 0% over 73–281 pieces
  (molar1 0 of 191; engagement1, protrusion1–3's black text). A candidate is dropped with 20 or more pieces and under
  5% of them near level 1.
- A dropped candidate's levels stay other ink for the opaque families (`ink`, except for a family of their color):
  engagement1's black text candidate had been other ink for the blues (and their strokes) until withoutSpeckLevels
  dropped it; without it, 55 found for 49, P 1.000 → 0.873.
- molar1: black as an opaque family, r 9.64 (9.59 in the truth), 16 found for 16, P 1.000, R 1.000, RMSE 0.12 px.
- protrusion3, radius unknown before (its black text candidate came first, with no radius): now the green family,
  r 22.0, 114 found for 50, R 1.000. The extras are mostly in the green box (89 of the 114 found lie in it): not yet investigated.
  protrusion1–2 stay radius unknown, for other reasons.

Results (full bench, diffed against 87a1a04): molar1 and protrusion3 as above; every other row identical. Tests in
test/style.test.js (synthetic) and test/opaque.test.js (molar1).

**bench-nature protrusion1–3: translucent dots over box plots' boxes in their own hue** (2026-11-17)

- Each box is a flat color on its family's line, where a deeper level or a color beyond the levels is expected:
  protrusion2's green box (198, 237, 219) as level 2, 6 units from the dots' own 193, 235, 216, and its orange box as
  level 3; protrusion3's (150, 183, 149) as level 2. protrusion1's dots are very light (α 0.16–0.36): each box sits 6
  units past level 2, took its peak, and the step ratio then ended the sequence at level 1.
- A level's boundary is arcs of the dots' radius, so no straight run of it is longer than a dot's diameter; a box's
  edges are. style.js straightShare: the share of a color's boundary in horizontal or vertical runs of 2 radii or
  more. Over every translucent candidate's levels on the bench: dots' levels 0 (a few grays at radius under 4, axis
  text: up to 0.063), the protrusion boxes 0.145–0.274, score2's colorbar 0.124. Threshold 0.08. A box must also be an
  area (solidRegion at 4 px: boxes 0.92–0.99; landing1's black axes and text, straight too, 0.000).
- boxBand: a level after the first, or a color beyond them, that is a box is the family's band; the family is measured
  again without its pixels (protrusion2: green 220 | 193 | 177 | 163 | 155, orange 232 | 214). When no second level
  shows without it the levels stay (protrusion3: two dots on the page, 142, 178, 142, lie 8.4 units off the line to
  level 1, past PERP_TOL 4). With no lone dot anywhere (protrusion1) the radius is unknown, and a box is told by
  comparison instead: straighter than level 1 by 2 times in 20 px runs (protrusion1: 3.5–4.0 times).
- beyondBox, before overBand: protrusion1's one-level candidates read as dots over a band (overBand, for landing1's
  violins) with the roles swapped, the dots' level 1 as the band and the box as one dot; both readings fit the colors.
  The band is the straighter of the two: landing1's violins 0.155 against the dots' 0.000; protrusion1's boxes 3.5–4.0
  times the dots'.
- A box is a region under the dots (background.js regionMode). In the dots' own hue c − B₂ and c − B nearly agree, so
  their votes can't tell (protrusion3: 7.7° and 6.8°, "over" by a hair, s 0.46), but the box's outline can: lightened
  where a dot crosses it, 19, 20, 19 to 87, 94, 87.
- The box holds its rectangle (merged pieces of its exact color) whatever seed is nearer: protrusion1's boxes are 37
  px tall and its dots 39 px across, and pixels in such a dot nearer the page above or below were read against the
  page. Outside the rectangle nearest seeds decide as before: protrusion2's green box shows only above its median,
  under a stack below; leaving the box's pieces out of the seeds there found 33 for 30 (P 0.848). That version found
  all 50 of protrusion3's.
- These Nature figures don't composite in sRGB: in protrusion3 one dot over the box is 116, 163, 122 against
  128, 167, 127 predicted from c and α on the page (closer in linear light for the green, but not for protrusion2's
  orange), which read as 1.56 dots and was masked as fractional. unmix.js regionLevels measures a family's levels over
  a box: the commonest flat color of the region within half a predicted step of the prediction, then findLevels on the
  line from the box through it (ignoring colors nearer the box than half that: two dots on the page in the region's
  margin, 12 units from the box, came first). Pixels there count by those levels. Boxes only: over landing1's violins
  it gave 87 found for 70 against 75.
- No orange dot in protrusion2 is whole (the whisker crosses each, or the box's edge). families.js withSiblingRadii:
  a translucent family with no radius takes another's when DOT_MIN_STRONG of its dots show half their edge at it, as
  opaque siblings do (3 at the green's 30.1 px, none at 25 or 35).
- protrusion1 went down the color-scale path: green and lavender chained through the mixes of their rims with the
  gray whiskers, points of 5–89 votes. colormap.js: a scale's top 3 points hold at most 0.7 of the votes (protrusion1:
  0.92; the gradients 0.23–0.29, score2 0.36, bench2 group-graystroke 0.48).
- protrusion1's radius: every dot on a whisker read as two halves, and the distance transform gave 9.5–10 for 19.5,
  below the Hough scan's reach. style.js radiusFromDistance: masked pixels are no family edge, and no peaks or
  neighbors that hide one. Radii now 18.0–19.5 (pink by sibling).
- protrusion1: 241 found for 246, P 0.892, R 0.874 (radius unknown before). Misses by row: DMSO 10 (its box shows
  226 px), Cyto C 8, Pseudo A 6, Cyto E 4; extras mostly in the orange stacks (23 of 26). protrusion2: 26 for 30,
  P 1.000, R 0.867 (radius unknown before; misses a dot under three at the box's edge and three in the densest
  stack). protrusion3: 48 for 50, P 1.000, R 0.960, RMSE 0.43 px (114 found before). The app: protrusion1, four
  families, 241 dots in 19 s, no console errors.

Results (full bench, diffed against 5463870): protrusion1–3 as above; landing1 77 → 75 found for 70 (P 0.909 →
0.933: its radii 12.0 and 11.5 became 12.05 and 12.0 with the distance transform's change, which alone gives 75);
beeswarm-boxplot 687 → 667 (P 0.952 → 0.982, R 0.920 → 0.921) and violin-beeswarm RMSE 0.20 → 0.26 px
(same counts, radius 7.43 → 7.48), both from the distance transform's change; delta1 23 → 24; RMSE and radius in the
last digit for curve and dots-swarm-box. Tests: test/style.test.js (box band, cut dots' radius, sibling radius),
test/unmix.test.js (regionLevels), test/background.test.js (box rectangle), test/families.test.js (protrusion1–3).

**The scan for dot families in a worker, with progress** (2026-11-17)

- On loading, estimateFamilies ran on the main thread: 3–56 s on large images (Node: score2 56 s, protrusion1 25 s;
  beeswarm-boxplot 19 s and then 33 s more for the noisy-color retry), long enough for the browser's Page Unresponsive
  warning. Its results survive structured cloning, so src/scan-worker.js runs it (and a picked color's estimateStyle,
  and re-measuring opaque families after an outline edit). One request at a time: a newer image or rescan terminates
  the one running.
- estimateFamilies takes onProgress(label, fraction) over stages weighted by measured time on 11 bench images: the
  dots' colors 0.31–0.97 of it, outlines and hollow rings 0.18–0.49, the colors over regions 0.20–0.32 where they run,
  a color scale's check most of score2 (0.68). findCandidates reports each opaque color cluster; its translucent part
  before them was 0.07–0.28 of it on five images (0.73 on protrusion1).
- Headless Chrome: score2 scanned in 47.7 s, the main thread's longest gap 347 ms; beeswarm-boxplot (both passes, the
  bar's halves) 44.5 s, longest gap 197 ms; the same families as before. The bar sits at 0.01–0.05 for 16 s on score2
  while it checks for a color scale (fast on every other image). Bench rows checked unchanged (landing1, protrusion3,
  bench2 group).
- The Claude browser pane ran the scan about 4 times slower: it reports the page hidden, and Chrome lowers hidden
  pages' and their workers' priority. A cached main.js from before the change also ran there until the scripts were
  fetched with cache: 'reload'.

**bench-nature delta1: black dots joined by their line to the axes, beside as much black text** (2026-11-17)

- No black family: measured at r 3.5 (text strokes), 2 strong centers and coverage 0.075. Hough on black's coverage
  shows the dots at 13.5–14.2 px (15–17 strong centers), but the rescan never reached them. componentRadii's pieces:
  dots, line, error bars and axes are one piece 14.1 px deep and 47% of the black; text and tick labels, 3–5 px deep,
  49%; so the size-weighted median fell on the bold panel label "g" (8.1 px), and the deepest compact piece was a
  scrap of text (the dots' piece is 32.8 times a disc's area). The rescan covered 6.5–10.8 px: no strong center.
- componentRadii also offers the deepest piece of all: dots are usually their color's thickest ink.
- delta1: 24 → 44 found for 48, P 0.958 → 0.909, R 0.479 → 0.833, RMSE 0.32 → 0.19 px; black r 14.4 (truth 14.0),
  26 found. Purple went 24 → 18: with black a family, its pixels mask the edges of purple dots partly behind black
  ones (8 missed, 2 placed beside them). Before, black was read as a region under the dots, which likely let purple's
  edges against it count (not checked). 2 black extras where the curves cross.

Results (full bench, diffed against 81f7e35): delta1 as above; every other row identical but for run times. Test in
test/opaque.test.js.

**bench-nature delta1: purple dots under black ones, the drawing order of opaque families** (2026-11-17)

- Black is drawn over purple. 8 purple dots partly under black ones were missed (18 of 24 found). measureOpaque reads
  pixels near a family against their nearest other color, so the black pixels at a seam read 0, the page, for purple.
  (1) In layers.js buildMask's thin test that one-pixel seam cut the purple remnant from the black dot over it, and
  the remnant, joined to the purple line, was masked as thin (the whole visible part of the dot at 666, 613). (2) A
  purple dot under black paid for every seam pixel it covered, so the fit dropped the Hough centers it had there
  (arcs 0.30–0.51 at four of the misses).
- measureOpaque now records those pixels (edgeInk, by which other color: inkColors) and returns its centers. (1) The
  thin test counts them as other ink: one more purple found (45, P 0.956, R 0.896). (2) Masking them for the fit found
  every purple dot but one, but let black put dots where purple shows (4 black extras): a seam is the edge of the dot on
  top. So opaqueFamilies decides the drawing order (drawnUnder): the seam lies on the upper family's circles and inside
  the lower one's discs, and the upper's dots are whole, their centers found. Seam pixels within 2 px of each family's
  found circles are counted; f is under g with 20 or more on g's and 3 times as many as on f's. The seam pixels are
  masked only for a family under the other's ink (layers.js overInk).
- Disc interiors first (a top dot holds no other color): by the truth's centers 0.142 against 0.000, but the found
  centers are the filled ones, mostly not covered (0.007, 40 px), and a synthetic copy gave nothing.
- delta1: 44 → 50 found for 48, P 0.909 → 0.940, R 0.833 → 0.979, RMSE 0.19 → 0.31 px (the purple under black placed
  less exactly); purple 24 for 24 but one doubled (flagged), one miss almost wholly behind a black dot, 2 black extras
  where the curves cross.

Results (full bench, diffed against a92c0d5): delta1 as above; nicest language 164 → 163 for 162 (P 0.988 → 0.994);
kite1 26 → 27 for 25 (P 0.923 → 0.889): a cyan dot doubled 5 px away, flagged, from the thin test's change (no order
found there); every other row identical but for times. Tests in test/opaque.test.js: a synthetic black-over-purple
copy of delta1's layout (5 of 10 purple before), and delta1's thresholds raised.

**bench-nature hair2 and tumour1: white dots in black outlines, mostly hidden** (2026-11-18)

- Both have dots filled with the page's color, found as the page regions their outlines enclose (pageFills). A dot
  mostly behind others shows a crescent, under half its edge. The in-disc test wanted a Hough center with half its
  edge seen, so crescents read as page: a dot behind them cost the fit more than it gained. tumour1's lane 8 (six dots
  at zero, four crescents) found 2. hair2's white dots (two rows of five and four, 0.55 × the fill radius apart) had
  only two whole fills, under the three needed, so they were no family, and the pink family's mask took them.
- At a Hough arc of 0.2 the crescents pass the in-disc test, but so do digits' counters (0s and 8s, arcs 0.42–0.49).
  New test: a region's edge must be its own circle (within 1.5 px of the fill radius, 0.2 of the edge at least) or
  the outlines of fills over it (within 1.4 × the fill radius of another center), 0.9 of it in all. Measured, every
  real fill and crescent of tumour1, hair1 and hair2 scores 1.00 (own share 0.43 or more), counters 0.61 at most. The
  old false dot, the 6 of tumour1's 0.9964 (arc 0.69), scores 0.
- Dots closer than their radius: a crescent also lies in the disc of the dot beside it. Each region takes the center
  whose circle is most of its edge, not the first one whose disc holds it.
- A family needs two whole fills now, and three dots in all, counted as distinct centers. pageFilledFamily takes those
  as its evidence when too few dots show half their edge (hair2: 2 strong).
- The fit still lost hair2's middle crescents: by the union of discs, a dot 0.47 r from both neighbors changes the
  model by less than it costs, and its Hough never proposes it. The crescents' centers are now the fit's seeds (the
  estimate's `seeds`, added to Hough's first centers). Seeded, all 9 survive. Half as large (a synthetic copy at fill
  r 14.5), two of them die: a limit of fitting outlined dots by the union.
- The rim of a page-filled dot was read as a mix of the outline with the page, but over tumour1's gray bars (171, on
  the line from white to black) the bar read 0.36 out to the outline's reach. Dots over bars read a pixel larger, and
  the fit paired them. Its 10.54 radius came from fitDots' doubled-dot bump (22 of 52 close, just over 0.4); with the
  crescents found it was 20 of 54, under, and 10 extras showed. The rim is now read against what the dot lies on: the
  nearest flat pixel beyond the family, not of the outline's color (style.js surroundings).
- Rings (ringEstimate with the fills as the fill mask) were tried for both: hair2 unchanged (no isolated ring to
  measure), tumour1 37 for 44. Not kept.
- Seeding gave tumour1 two flagged extras in lane 8 (a double paid for by the tick under it, and a dot inside the
  union beside the axis line, both outline-colored ink within the outline's reach). Choosing each crescent's center by
  its edge removed them.
- tumour1: 41 → 44 for 44, P 0.976 → 1.000, R 0.909 → 1.000, RMSE 0.43 → 0.34 px. hair2: 10 → 19 for 18, P 0.900 →
  0.947, R 0.500 → 1.000 (the white 9 of 9; the pink's one doubled dot, flagged, as before).

Results (full bench, diffed against 9bbaa9f): hair2 and tumour1 as above; every other row identical but for run times
(hair1 unchanged, 27 for 29). App: hair2 "Found 19 dots, 2 flagged" (pink 10, white 9), tumour1 "Found 44 dots", no
console errors. Tests in test/opaque.test.js: synthetic rows of crescents beside 0s at hair2's scale (no family
before), dots over a gray bar (D read 0.36 on the bar before), and the hair2 and tumour1 regressions tightened.

**bench-nature phase1: all five families, four grays in black outlines and black dots** (2026-11-18)

- Found were 187 and 82 (50 for 97, P 0.580 R 0.299). Each missing family failed at its own step:
  - 136 (27 dots): the outline model measured r 19.2 and 8 strong dots, but coverage 0.29, under 0.35. Its curves,
    13 px wide beside fills of 17, held most of its pixels. An outlined family's coverage now counts its fill's
    color only where a disc of 0.5 × the fill radius fits (THICK_FILL_RADII): 0.29 → 0.63. 187 and 82 rose likewise
    (0.42 → 0.73, 0.36 → 0.66).
  - 228 (20): its curve outweighed its dots, so its surroundings read as the page and the outline model was never
    tried. With black forced: r 19.5, 18 strong, coverage 0.43. After the main loop (clusters go largest first, so
    228 came before any outlined family), a rejected fill is tried in each found family's outline, if darker.
  - black (16): fill and outline one color, also text's, axes', the colorbar's and every gray dot's ring. Measured
    as plain ink: r 4.86 (periods), coverage 0.01. A rejected color that is a found family's outline is now measured
    at those families' radius (measureOpaque `opts.radius`), their rings theirs (the outline width plus RING_SLACK,
    1.5 px), and kept on 3 or more strong centers filled with it and coverage 0.1 (6 strong, 12 centers, 0.21).
- Then 82 put pairs on its dots (31 for 16, 6 pairs under 0.15 r apart). Its radius, 19.1, is a little small; before,
  16 of 35 close (over the 0.4 share) bumped it to 19.6, now 12 of 32 didn't. fitDots now also refits 0.5 px larger
  when 3 or more pairs are that close, keeping the result only if the objective (loss + λ · dots) is lower: for 82,
  25 dots, none close, objective 5% lower. Black keeps 5 such pairs (no larger radius is better: its radius is the
  grays').
- A synthetic copy (all five, curves under 228 and 136, black periods and a block) found 187 alone before, and showed
  two more failures phase1 doesn't: (1) without curves, 187's background is black, its outline's color, and 82, on
  the line from black to 187, was a blend; an outline-colored background is now a rim only when rimOf says, as the
  outline is. (2) One pure gray's two histogram peaks, 5 units apart, made two families; a cluster within INK_TOL of a
  found family is skipped.
- phase1: 50 → 132 found for 97, P 0.580 → 0.606, R 0.299 → 0.825, every family found. About 30 of the 35 extras are
  the colorbar's blocks (6 per family, beyond the axes' end; bounding the plot there would cut cantons', comment1's
  and delta1's dots, as noted before), the rest doubles where a same-gray curve meets dots and black's pairs. App:
  "Found 132 dots, 30 flagged", five families, no console errors.

Results (full bench, diffed against 4e7bba6): phase1 as above; every other row identical but for run times. Tests in
test/opaque.test.js: the synthetic copy, and phase1's regression (all five families, P ≥ 0.55, R ≥ 0.8).

**testgraphs wages2025: a categorical palette, and navy axes** (2026-11-18)

A 4000 × 4000 scatterplot of 766 occupations (772 rows with both values) in 22 group colors, translucent (α 0.58)
with white rims, log axes, a legend, a dashed median line and annotations. The scan made one family "colored by a
continuous scale" and found no axes.

- Truth CSV (tools/truth-csv.mjs): ticks as printed (1K, 10M) are numbers, "log" is a scale, and tick-placed truth
  keeps each row's group (truthPixelsByTicks dropped it, so groupRow pooled every family; kite1 and landing1, the
  other CSVs with a group, are now scored by group too).
- Axes (axes.js): the ticks and labels are dark navy, 24, 37, 50, chroma 26 over GRAY_CHROMA (24), so no tick was
  found and unlabeled gridlines stood in. A dark pixel (luminance under 64) is gray ink up to chroma 40. Then the y
  labels were all dropped: the x axis's "100", under the corner, reached into the y labels' search beside the corner
  tick, clipped at gap 0, and the labels 19 px out fell outside the column (minimum gap + 17 px). A word wholly
  beyond the axis line's ends (by 2 px) is now no label of its ticks. Two rules tried first broke JMP renderings:
  skipping words cut by the search's edge lost bench1 tiny-dots' bottom y label, drawn right at the ticks; requiring
  a word to span its tick lost medium-dots-grid2's last x label, whose "00" runs off the image, and then the 1 px
  gridlines above the axis outnumbered the labels, the plot was read as below the axis and masked whole (0 found in
  bench1 and bench2). Both axes: 6 labeled ticks; the plot bounded left and below.
- A palette, not a scale (colormap.js): its colors chain through the mixes of overlapping dots of unlike colors, but
  the votes pile up at its colors. wages2025: 26 peaks (points with the most votes within 0.08), the top 10% of
  points 0.87 of the votes; scales 3–6 peaks, 0.32–0.41 (bench1–2 gradients) and 0.61 (score2's viridis). A palette
  is 10 peaks or more and over 0.75.
- Rejected from the scale path, it split one family at a time: 9 of the 22 (7 read as opaque at their one-dot
  colors), 860 found, P 0.823 R 0.917 pooled, in 6 minutes against 1. So it is fitted as a scale is, one virtual
  family that reads overlaps of unlike colors, and each fitted dot's color is read afterwards:
  - paletteColors: each dot's own pixels (3 px inside it and beyond the other fitted dots' rims), their median, from
    dots with 20 or more such pixels, 90% of them within 8 of the median, clustered greedily within 8 (each color's
    dots lie within 0–11 of it; the nearest two colors, both blues, are 17.6 apart). Without the uniformity test, 38
    colors: two-dot stacks (a fitted dot over a missed one), gridline grays, the median line's purple. Testing
    instead that a color lay on the one-dot curve lost both grays, whose direction the subtitle's slate text holds
    (its peak, 68 741 votes, is the largest).
  - paletteFamilies: the dot's own pixels vote for the color they are within 8 of. By their median, 41 of 652 matched
    dots were in another family, all marked sure: missed dots cover part of the "own" pixels, and a stack's nearest
    color is another's. By votes, 10, all unsure.
  - A palette's color needn't chain: pink (8 dots) is 0.38 from the nearest color in direction and was off the curve,
    never fitted. Points outside the chain are added when at least 0.5 × the nearest peak's distance from the
    background (pink 0.77; the gridlines' gray 0.28).
- The bench (splitByFill) and the app (labelFills) split the fitted dots by color as for outlined dots of a few
  fills; the app names the family "23 colors (a palette)" and shows each dot's color, unsure ones marked.
- wages2025: 23 families (the 22 and the median line's purple), 767 found for 772, P 0.851 R 0.846 by group (0.860
  and 0.845 pooled before); 72 flagged; 64 s. Extras still on the legend's markers (fitted twice each), the title,
  subtitle and the median line's label: no top or right bound, and no text or legend masking yet.

Results (full bench, diffed against dd5b19a): wages2025 added as above; every other row identical but for run times
(kite1 and landing1 alike by group). App: "Found 767 dots, 72 flagged", "23 colors (a palette), α 0.58, r 17.6", no
console errors. Tests: a synthetic palette at wages2025's scale (colormap.test.js: 13 colors, none of 177 sure dots in
another color's family), wages2025's palette and axes, navy axes, and the truth CSV's printed ticks, log scale and
groups.

**Text and legends masked** (2026-11-18, wages2025)

wages2025's 105 extras of 767 lay on its legend (each marker fitted twice), title, subtitle and the median line's
two-line label; no axis bounds them (right and top). annotations.js finds text lines and legend markers on the image
as analyzed, and buildMask masks their boxes (`opts.exclude`) for every family, in the bench and the app.

- Ink is what differs from the page (by INK_TOL); its 8-connected pieces get a box, a pixel count, an inscribed
  radius (distance transform) and the color at that deepest pixel.
- A glyph is a piece thin for its height (inscribed radius ≤ 0.3 × height; a disc's is 0.5), thinner than the dots
  (≤ 0.6 × their radius), at least 6 px tall, filling 0.08 of its box and no wider than 4 × its height. Measured in
  wages2025: subtitle and legend strokes 4–6 px, median label 8–10, title 15–17 beside dots of 17.6. Each bound came
  from a regression in the full bench:
  - against height only, a chain of touching dots is thin too: bench-plos adults1's black cluster (strokes 13.5–20
    beside dots of 13) made a "line" and lost 18 dots; bench2 medium-dots lost 40 in a box of translucent clusters.
    So the dots' radius: wages2025's bold title, as thick as its dots, stays unmasked (670 found against 664, P 0.978
    against 0.986).
  - testgraphs violin-beeswarm: its frame, thin for its height, made the whole plot "text" (0 found). Fill 0.02.
  - bench3 dots-swarm-violin and -box (51 and 7 lost): the dots' own gray covers 5% of the image, a "background"
    color, so only their rims were ink, thin arcs in rows. Text is now read against the page alone.
- A line: glyphs overlapping by half the shorter's height, apart by at most 0.6 × the taller's, heights within 2.5×,
  of one ink (stroke colors within 60: a dot's piece joined "Police officers" and its box covered dots), 3 or more,
  60% of them on the median bottom (within 0.08 × height, 2 px at least: white rims cut wages2025's overlapping dots
  into crescents, thin like glyphs, which chained in its dense middle), and not 80% one piece repeated (box within
  2 px and pixel count within 5%: a row of hollow rings; o, n and e share a box, not a count).
- A legend marker: a round piece (aspect 1.25) filled like a disc (inscribed radius ≥ 0.35 × height), within 2 line
  heights before a line and centered on it, 3 or more in a column (centers within 3 px). wages2025's markers are the
  dots' size (radius 17.1), so only their column tells them from data.
- wages2025: 670 found for 772, P 0.851 → 0.978, R 0.846 → 0.848; 22 legend markers, 79 text boxes, 0.8 s.

Results (full bench, diffed against 815cb69): wages2025 as above (22 families: the median line's purple went with its
label); cells1 73 → 74 (one more extra, not located: its boxes are labels outside the plot); phase1 132 → 135 (3
more extras; its boxes are "System", above the colorbar where its extras are, and the x axis title); primates2 RMSE
0.01 → 0.03 px; every other row identical but for run times. App: "Found 670 dots, 34 flagged in 23.9 s", "22 colors
(a palette)", no console errors; 3 dots left on the title. Tests: test/annotations.test.js (a synthetic page: a
title, a note beside a dot, a legend column, crescents, a row of rings, touching black dots and a frame; wages2025's
boxes), and wages2025's palette regression at P ≥ 0.96.

**bench-nature autocor1: dots clipped at the plot's edges; the misses that remain** (2026-11-19)

With the user's hand truth (727 dots, the scan's result adjusted: most changes additions from slivers of color, a
few dots cut at the plot's edges, a few interior dots moved or removed): 703 found, P 0.979 R 0.946; 39 missed, 15
extras. Against the source CSV (804 rows, many wholly hidden) 0.953 and 0.833.

- The plot clips its dots: at the y axis line and where the x axis line ends (x 2033; no line there), a straight
  cut, the family's coverage full just inside and white just beyond. Read as the dots' edge: at the right end the fit
  moved the cut dots in and stacked three where two were. Beyond the axes the mask keeps a margin of a radius
  (other plots draw whole dots across them), so the cut had to be read: axes.clipEdges gives the bounding axis
  lines and their far ends, and buildMask masks everything beyond one where the coverage is full (≥ 0.9) just
  inside and empty (≤ 0.1) 1.5 px past the line (or past the boundary pixel, within 3 px of a line's end) in 70% of
  the rows full inside, 2 radii of them at least. autocor1: cuts at the y axis (x 266) and at x 2032; the three
  stacked extras went, and two cut dots were found (702 found, P 0.980, R 0.946). Still missed there: two dots
  more than half past the cut and mostly under others, and at the y axis two dots 6 px apart, fitted as one.
- The other 31 misses lie in the dense band, most along the opaque curve drawn over the dots (35–50% of their disc
  masked as its ink). Measured: 0–11 px of each missed dot's disc show the family's color uncovered by a fitted
  dot. The neighbors' discs already cover the slivers, so a dot there explains almost nothing, and the fit rightly
  won't pay for it: the evidence the hand truth used is the shape of the edges, which the union of discs can't
  read. Not attempted (births on slivers this small would add dots wherever a fit's discs fall short elsewhere).
- The extras now cover little page (0–2% of their discs white): mostly one dot standing for two in the dense band.

Results (full bench, diffed against 9ceaf57): autocor1 by hand 703 → 702 for 727, P 0.979 → 0.980, R 0.946; by the
CSV P 0.953 → 0.960, R 0.833 → 0.838; phase1 135 → 129 for 97, P 0.593 → 0.620 (gray 136 clipped where the x axis line
ends, x 1287, masking the colorbar beyond; 228 and 187 at the x axis line, recall unchanged); every other row
identical but for run times. Test in test/layers.test.js: a synthetic plot clipping its dots at the y axis and the x
axis line's end (cut dots found whole; without the clip, one at the end is misplaced).

**The remaining "not yet investigated" testgraphs; bars cut by the dots on them** (2026-11-20)

- Laser-Asyllable-Maturity (80 found for 70, all matched, 10 extras): opaque dots and brackets drawn in the dots'
  colors, bars about 5 px thick against dots 11 px across. The dots sitting on a bracket's bottom bar cut it into
  pieces of 5–10 px, shorter than the length test (3 radii), so they weren't thin and the fit put dots on them.
  buildMask now joins such pieces (layers.js barPieces): short components longer than they are thick (≥ 1.3 times),
  no thicker than the opening removes, filling ≥ 0.7 of their boxes (a crescent doesn't), lined up in a row or a
  column (one's rows within the other's, 1 px either side), with gaps up to a dot's diameter + 1 px, together at
  least 3 radii long; or one such piece in line with a thin mark across such a gap (purple's bar: one short piece
  beside the long rest). The bar's deep pixels in the gaps, under the dots, are marked too (else the strip of bar
  below a dot's edge drew a second dot). Laser: 73 for 70, P 0.875 → 0.945, R 1.000 → 0.986. Without the
  elongation test a dotted line's round specks were joined too, and one masked speck cut into adults1's cluster
  (a dot added there).
- fibers-distance (39 for 44): translucent matplotlib blue at α 0.5, r 4, with JPEG noise, read as opaque: no
  level has flat pixels. Coincident pairs show darker and count once; dots on the regression line (the family's
  full color) are missed. The noisy path (restoreChroma, JPEG tolerances) finds α 0.50 and all 44, but stacks 9
  extras where the dots cross the line: there the line, thickened by the dot, survives the opening, and its JPEG
  noise (up to 21 units off the full color) keeps it from reading as deep. The app keeps the exact path because both
  find one family; preferring the noisy one when it finds the same families translucent would also flip fanc2,
  whose opaque dots draw every overlap one dark color (2 levels, alike in both cases). Left as is.
- economist-canada-unfriendly-chart (198 for 210): nearly all misses and extras in the dense pile at lower left,
  where both translucent families and both full-color trend lines cross; the hand truth is a guess there too.
  Left as is (one stack there is fitted as 5 coincident dots).
- violin-beeswarm (712 for 702): the 11 extras lie along the violins' dark center lines, between dots that the line
  shows through. Small, and no rule found that wouldn't risk the packed dots.

Results (full bench, diffed against b523049): Laser above; cells1 74 → 73 for 72 (P 0.892 → 0.904 by hand, 0.946 →
0.959 by data); kite1 27 → 26 for 25 (P 0.889 → 0.923); phase1 129 → 128 for 97 (P 0.620 → 0.625); every other row
identical but for run times. Tests in test/layers.test.js: a synthetic bracket with dots centered on its bar (the
bar between them masked), and Laser's regression (≤ 5 extras, ≤ 2 missed; skipped when absent). Both fail without
the rule.

**Whole-app review: where the time goes, and two fit speed-ups** (2026-11-20)

Profile (`node --cpu-prof` on 20 images, the slowest of each kind, 876 s of samples): the fit takes 53%, the scan for
families 40% (findCandidates 23%, opaqueFamilies 21%, measureOpaque 16%, the hollow-ring search 7%), and mask, axes,
annotations and chroma restoration under 3% together. Two costs in the fit had nothing to do with the image:

- createDotModel and createOutlineModel copied the whole dot Map in every snapshot() and walked it in every
  restore(), once per trial move, birth, split, death and merge: 10% of all time. They now save the records of the
  dots that can reach the snapshot's rectangle (the ones recompute would read), sorted by id so restore() refills the
  grid cells in the order a full copy did, and restore only those. A dot put back after a removal lands at the Map's
  end, so dots() sorts by id.
- ringCenters' bilinear sampler built four small arrays per sample, 48 samples per pixel of the image, each round of
  ring births: 11.5% of all time in the sampler alone. It now adds four corner terms without allocating.

Results: every fitted dot of all 118 local images (18,242 dots: radius, position and flags per family) is identical
before and after, by digest. Full bench 1649.5 → 1255.3 s (−24%), every row unchanged but for its time, none slower:
score2 215.2 → 62.0 s, beeswarm-boxplot 105.8 → 83.4, pathogen 46.3 → 25.6, step1 26.4 → 10.5, kite2 28.1 → 16.1,
autocor1 31.4 → 21.4, violin-beeswarm 51.5 → 41.9, bench2 gradient-dots-stroke 44.5 → 35.0. phase1 (115 s) does
not move: its time is radius bumps refitting from scratch. In the app pathogen fits to the same 468 dots in 17.5 s.
Tests: snapshot and restore among 50–60 dots in test/render.test.js and test/outline.test.js (moves, removals,
births and fill values undone exactly, dots in id order, the grid still right afterwards).

Found by the review and not done (each needs its own check against the bench):

- Failed births are retried every round (the tabu list is cleared each round): bench2 gradient-dots-stroke made 805
  failed attempts for 17 births. (Tried next, below: remembering failures gains nothing.)
- A doubled or coincident radius bump refits from scratch (phase1's gray family: 65 s, about 5 s of it the last
  fit). Deciding sooner changes behavior.
- findHollowRings runs on nearly every image (7% of all time; one bench image has hollow rings), and computes ring
  centers and a fill alpha it discards before the cheap inside test rejects the family. (Reordered, below.)
- 44–184 whole-image distance transforms per image (7%), 10–65% of them on a mask already transformed; isThin runs
  up to 10 times for 3 colors. backgroundColor is computed 5–9 times per image, flatMask 2–5, backgroundColors 4–7:
  one per-image context would hold these and shorten the signatures that thread (image, B, flat, tol). (The context: done,
  below.)
- findCandidates runs again when regions lie under the dots, and the whole scan again for the noisy trial; in the
  app the two scans, and the families' fits, could run in parallel workers.
- Math.hypot in per-pixel color tests is 7–9 times slower than a squared distance (about 27 sites); the same sites
  are one "pixels within a tolerance of a color" helper, and the "between two colors on a line" projection test is
  written out in eight places. (Done, below.)
- measureOpaque's radius scan calls houghCenters up to 128 times (delta1: 27 of 47 s); the linear openings
  (underRings, localFill) are O(length) per pixel (6.6%).
- Structure: measureOpaque braids four reading modes through one loop; opaqueFamilies, findCandidates and
  estimateFamilies are chains of rescue passes; the two incremental models share a duplicated scaffold; six medians
  (two variants), five component labelers; tools/bench.mjs extractAll mirrors the app's scan orchestration; style.js
  holds three subjects in 3,077 lines.

**Hollow-ring reorder; failed births: remembering them gains nothing, the tabu scan did** (2026-11-20)

- Hollow rings: outlinesInP takes the hollow test (hollowInside) and runs it as soon as the ring's style is known,
  and for a hollow ring skips the ring candidates and the fill alpha, which findHollowRings threw away. Worth less
  than the profile said, now that ringCenters is cheap: over the 20 profiled images the search takes 35 s, 3.4 s
  of it the skipped part (engagement1 1.6 of 4.6 s). The rest is localFill and underRings, the linear openings.
- Remembering failed births across rounds, tried and dropped. A failure was kept with the rectangle it read (the
  scored rectangle and the dots within NEIGHBOR_RADII · r) and skipped while no dot changed there. Measured hits:
  1 of 1104 failures on bench2 gradient-dots-stroke, 113 of 1585 on cells1, 84 of 364 on pathogen, 0 on dog-ages,
  kite2 and violin-beeswarm. The failures are not repeats: most are at new sites (the best box moves when anything
  near it does), the dots near the rest really moved between rounds (a sixteenth of a pixel is a change), and the
  radius refit starts over (b2 gradient-dots-stroke: 11.1200 to 11.1197 px, 299 failures after it). Letting only
  births and deaths count as changes, which would no longer be exact, barely raised the hits (cells1 179).
- It also needed a rejected trial to leave no trace, and it does leave one: restore() puts moved dots back at the
  end of their grid cells, so later renders multiply the same stamps in another order and differ in the last bit.
  Rendering in id order removes that, and by itself changed the dots of 6 of the 118 images (pathogen 468 to 464,
  phase1 128 to 130, four others in position only): the fit is sensitive to rounding at ties. Not kept; worth
  knowing before any change that reorders arithmetic in the model.
- What was slow in births was the scan for the best center, once per attempt, testing every candidate center
  against every failed site (the profile's 19.8 s closure in birthsWithBox, and much of its own 16.8 s).
  birthsWithBox now keeps a count per pixel of the failed sites within BIRTH_TABU_RADII · r, updated when a site
  is added or forgotten: the same test, made once per site.

Results: every fitted dot of the 118 local images identical again, by digest. Full bench 1255.3 → 1202.4 s (−4%),
every row unchanged but for its time, none slower: beeswarm-boxplot 83.4 → 68.9 s, cells1 38.4 → 28.3,
bench2 gradient-dots-stroke 35.0 → 32.2. In the app dog-ages fits to the same 95 dots. Tests: births keep away from
failed sites and forget the ones an accepted birth makes stale (test/fit.test.js); a hollow family reports no
measured fill (test/outline.test.js).

Where failed births still cost: an attempt is about 13 ms (the new dot and each neighbor refined by compass
search), and gradient-dots-stroke spends 15 of its 32 s on 1104 of them. Cheaper only by a rule that changes
results (stop a pass after a run of failures, skip the pass after a radius refit that moved nothing), to be judged
on the bench.

**One scan context, and helpers for the two color tests** (2026-11-20)

- Color tests (image.js): nearColor(data, color, tol) is the test "this pixel is within tol of this color" that was
  written out with Math.hypot at some thirty places; nearAnyColor takes several colors, colorMask gives the mask.
  They compare squared distances. By the profile's line counts the per-pixel Math.hypot tests took 39 s of the 830 s
  profiled (style.js's cheap dot test alone 9.4 s, measureOpaque's membership 5.6 s, isThin's mask 4.8 s). The
  squared test agrees with Math.hypot on every pixel, also on pixels exactly tol away, when tol² is exact, which
  holds for every tolerance used (3, 8, 15, 40, minLevelT 5 or 15); for a tolerance like √48 a pixel exactly that
  far is in by Math.hypot and out by the squares, so a computed tolerance would make the boundary case differ.
  Used in style.js and background.js. Left as they were: tests that compare two distances (measureOpaque's nearer,
  levelsNest), pixel against pixel (surroundings), sites that keep the distance itself (colormap.js), and
  axes.js isInk, unmix.js and colormap.js's palette lookup, none of them hot.
- alongLine(v, w) (style.js) is the projection every "is this color a mix of those two" rule made by hand: how far
  along w and how far off it. isMixOf(c, P, F) is the commonest form, with its range named (MIX_SHARE). Used by
  isBlend, throughLayer, hueRings, overBand, strokeColor's final measure and families.js mergeAcrossRegions. The
  per-pixel projections in measureOpaque and strokeColor stay inline: an object per pixel would cost more than the
  repetition.
- Scan (style.js scanOf): { image, tol, background, B, flat, pageBackgrounds, backgrounds, memo }, made once per
  image by estimateFamilies and passed to every estimateStyle call on it (opts.scan), and below estimateStyle in
  place of the image, page, flat mask, backgrounds and tolerances as separate arguments: findCandidates(scan),
  measureFamily(scan, seed), measureOpaque(scan, color, others, outline, opts), opaqueFamilies, findLevels,
  familyCandidates and the band and box rules (twenty signatures). A rule that measures without some pixels passes
  { ...scan, flat: rest }; regions under the dots extend `backgrounds` for that call. scan.memo remembers, per
  color, isThin, solidRegion and straightShare, which several rules ask of the same colors. The page color,
  the backgrounds and the flat mask are no longer recomputed by each estimateStyle call.
  findColormap and backgroundRegions keep their arguments (the tests call them directly).
- Not done, measured: a cache of distance transforms by mask content. The repeats (10–65% of 44–184 transforms
  per image) come from different callers handing in equal masks, not from one question asked twice, so it would
  have to key on content: a hash of every mask, and 10 MB per entry kept on a 2.6 Mpx image, for 2–3% at best.

Results: every fitted dot of the 118 local images identical, by digest. Full bench 1202.4 → 1126.5 s (−6%), every
row unchanged but for its time, none slower (score2 60.6 → 52.7 s, beeswarm-boxplot 68.9 → 61.1, violin-beeswarm
41.8 → 37.2, kite1 20.1 → 16.9). In the app Laser scans and fits to the same 73 dots. style.js is no shorter
(3077 → 3142 lines: the context's description and the remembered wrappers outweigh the arguments removed); what
changed is that the rules read alike. Tests: nearColor against Math.hypot over a cube of colors at whole and half
tolerances, colorMask and nearAnyColor (test/image.test.js); alongLine, and estimateStyle with a scan made
earlier, with regions, and with another image's scan (test/style.test.js).

Since the review: full bench 1649.5 → 1126.5 s (−32%) with no fitted dot changed.

**measureOpaque split into its four steps** (2026-11-20)

measureOpaque was one function of 336 lines: the family's pixels and surroundings, a per-pixel loop that braided
four readings by `if (paged) … if (stroke) … if (fo) …` with `continue`s, the radius, the centers with their radius
scan, and the coverage. It is now 43 lines that name the steps, with the code of each moved as it was:

- opaqueGround (74 lines): where the family's pixels are and what they lie on (own, mask, reach, the usual and
  the local background, the off-line tolerance from the color's noise, other families' pixels).
- opaqueCoverage: one of four readings, each a loop of its own: readPlain (11 lines), readOutlined (29),
  readStroked (43), readPageFills (35). The plain and outlined readings share mixReader (a pixel as the edge color
  mixed with its background, and which other ink it touches), and all but the page fills share noDotAt (away from
  the family on a background color, or on another family's ring).
- opaqueRadius (18): given, isolated dots, fills, or the distance transform, in that order.
- dotCenters (82): the dot test at that radius and the scan of larger radii; shareInDiscs for the coverage.

The order of the readings is as it was (page fill, then stroke, then outline, then plain), so a family given both
a stroke and an outline is still read by its stroke, with fillK left at zero.

Results: every fitted dot and every family estimate (style, notes, D, edge ink, radius, centers, coverage: all
fields, typed arrays by content) of the 118 local images identical, by digest. Full bench 1126.5 → 1110.0 s, every
row unchanged but for its time; a function call per pixel for the mix costs nothing measurable. In the app Laser
scans and fits to the same 73 dots. Test: measureOpaque called directly on the synthetic swarm (its dots covered,
the page 0, another family unknown, radius, centers and coverage) in test/opaque.test.js; the outlined, stroked and
page-fill readings are covered by the existing synthetic tests there.

**style.js split by subject** (2026-11-20)

style.js held three subjects in 3,283 lines. It is now five modules, each function and constant moved as it was
(by a script that checked no line was lost), importing in one direction: scan.js → shape.js and levels.js →
opaque.js → style.js.

| Module | Lines | What |
|---|---|---|
| scan.js | 226 | the page and background colors, flat pixels, tolerances, the Scan and its remembered answers (isThin, solidRegion, straightShare) |
| shape.js | 332 | components, isolated dots and their rings, edge blur, median and quantile |
| levels.js | 498 | translucent families: color lines, levels, the geometric fit, the level table and count image |
| opaque.js | 1,862 | opaque families: measureOpaque and its readings, page fills, strokes and rings, opaqueFamilies, the radius from shapes (shapeRadius, siblingRadius) |
| style.js | 420 | estimateStyle, findCandidates and the band and box rules, opaqueFamilyEstimate |

- The 130 constants went with their comments to the module that uses them; the eleven used by more than one are
  exported from the lowest (CLUSTER_TOL, LEVEL_SEPARATION and BOX_COLOR_TOL from scan.js, the ISOLATED_ limits from
  shape.js, FIT_C_TOL from levels.js, CLUSTER_ON_LINE_TOL, DOT_MIN_ISOLATED and STROKE_RING from opaque.js).
  BACKGROUND_TOL was used nowhere and is gone.
- shapeRadius and siblingRadius, which unmix.js and families.js use for translucent families, live in opaque.js:
  they are the dot test (Hough centers at a radius), and share its constants.
- Importers name the module that defines what they use (families.js, unmix.js, background.js, jpeg.js,
  colormap.js and six test files); style.js re-exports nothing.
- Tests: scan.test.js, shape.test.js and levels.test.js are new (each module its own file), with the tests of
  fitGeometric, alongLine and scan reuse moved out of style.test.js and direct ones added for backgroundColor,
  flatMask, solidRegion and its memo, colorLine and projectOnLine, components, isolatedDots and discShare.

Results: every fitted dot and every family estimate of the 118 local images identical, by digest. Full bench
1110.0 → 1110.8 s, every row unchanged but for its time. 208 tests pass (1 standing skip). In the app Laser fits
to the same 73 dots and dog-ages to the same 95.

**One store under both incremental models** (2026-11-20)

createDotModel (render.js) and createOutlineModel (outline.js) each had their own copy of the records Map, the
spatial grid, the lookup of dots near a rectangle, rectangle copy and paste, and snapshot and restore: about 75
lines apiece, kept alike by hand (the local-snapshot change of this review was made twice). They now share
createDotStore (render.js): newId, get, put, drop, near, entries, blurredRect, snapshot and restore, over the
buffers each model names (T, the composite, the blurred output, with their strides). A model is what differs:
what a record holds (a stamp; or z, a fill value and two coverages) and how a rectangle is rendered from the
records. The outline model's snapshot adds the top of its drawing order.

- put, drop and restore make the same grid changes in the same order as before: a grid cell lists its dots in the
  order they were put there, which is the order the dot model multiplies their stamps in, and the fit is sensitive
  to that at ties.
- render.js 570 → 625 lines, outline.js 1155 → 1080: 20 fewer in all, and one copy of the part that is easy to get
  wrong.

Results: every fitted dot and every family estimate of the 118 local images identical, by digest; 209 tests pass
(1 standing skip). The full bench run that followed read 6% slower on every image, scan-heavy ones included, with
other load on the machine; run back to back, three rounds each, the committed code and this take the same time
(bench2 gradient-dots-stroke 24.0 and 24.2 s, pathogen 24.2 and 24.3, dog-ages 11.1 and 11.1), so RESULTS.md
keeps the earlier run's times. In the app pathogen fits to the same 468 dots and dog-ages to the same 95. Test:
createDotStore directly (records, near, a trial with a move, a drop and an add undone, buffers of each kind
restored, ids never reused) in test/render.test.js.

**Small duplicates: medians, component labelers, color distance, the bounds mask** (2026-11-20)

Four things the review found written more than once, each now in one place. No rule changed.

- **Medians.** Eighteen places spelled a median, in two variants that differ at an even count: the mean of the
  middle two (shape.js median, a local one in axes.js, inline in unmix.js, a copy in tools/measure-dots.mjs) and
  the upper of the two (four local lambdas in annotations.js, colormap.js and outline.js, and inline after a sort
  in ten more places of outline.js, colormap.js and fit.js). Both are kept, since the rules were measured with
  them, and named: `median` and `upperMedian` in a new stats.js, with `quantile`. Neither sorts its argument (most
  of the old spellings did, in place; nothing read the sorted array afterwards).
- **Component labelers.** Four 8-neighbor flood fills (layers.js components8, outline.js components, axes.js
  inkComponents, annotations.js inkPieces) are one: components8, exported from layers.js. The callers derive what
  they need from its pixel lists and boxes: a count, a half-open box, the piece's deepest pixel. axes.js now tests
  each pixel of a label rectangle for ink once instead of up to nine times. The 4-neighbor `components` of shape.js
  stays: it labels the pieces of a coverage image, which mustn't join at corners.
- **Color distance.** The distance between two colors was written three ways at 43 places
  (`Math.hypot(a[0] - b[0], …)`, `Math.hypot(...a.map((v, ch) => v - b[ch]))`, `Math.hypot(...sub(a, b))`), with
  local `dist`, `near` and `close` lambdas beside them. It is `colorDistance` in image.js, still Math.hypot, so the
  values are the same to the last bit. The per-pixel tests on scalars stay as they are (nearColor has the hot
  ones). `sub`, `dot` and `unit`, each defined in two or three of background.js, families.js and unmix.js, moved
  there too.
- **The bounds mask.** The loop masking pixels beyond the plot area was in buildMask and in outlineLayers; it is
  `maskOutside` in layers.js.

src/core is 42 lines shorter with stats.js and the new doc comments counted (10,766 → 10,724).

Results: every fitted dot and every family estimate of the 118 local images identical, by digest; so are the axes
(lines, ticks, label boxes) and the annotation boxes at three radii, compared with the code before the change on
the same 118.
216 tests pass (1 standing skip). In the app dog-ages fits to the same 95 dots with 6 ticks on each axis, and
pathogen to the same 468 (78 flagged), with no console errors. No speed change was looked for or claimed. Tests:
test/stats.test.js (both medians, quantile, the argument left as it was), colorDistance against the three old
spellings and sub, dot and unit in test/image.test.js, components8 in test/layers.test.js.

**The long family functions, as named stages** (2026-11-20)

The three functions that decide what an image's families are had grown by a rule at a time, each rule a block of
code under its comment in one long body. They are now a list of stages, each a function named for what it decides,
with its comment above it. No rule changed, and the stages run in the same order on the same values.

- **opaqueFamilies** (opaque.js, 217 lines → 33). The search's state is one object passed through the stages: what
  is given (the scan, the translucent families and their level colors, other ink, the color clusters) and what is
  found (`out`, `rejected`, `blendInk`). The stages: `tryCluster` for each cluster (the screens, cheapest first:
  `onLevelLine`, a found family's outline or color, `interiorShare`, `isBlend`, `throughLayer`; then
  `outlineReadings`, the plain measure and readings (1) to (4)); the three retries of the rejected clusters,
  `fillsInFoundOutlines`, `solidsInOutlineColor` and `siblingFills`; the page-colored fill; specks; and
  `measuredForFit`. The screens were closures over `out` before; they still read the families found so far, through
  the state.
- **findCandidates** (style.js, 41 → 10): `translucentFamilies` (`levelCandidates`, `withoutSolidBands`, the box
  bands, the blends), then the opaque families, then the translucent ones that are an opaque family's outlines.
- **estimateFamilies** (families.js, 98 → 41): the three readings that can settle an image (outlined dots,
  `scaleReading`, no family by color), then `shadedRegions`, `familyEstimates` and `withRingFits`.

The files are longer for it (opaque.js 1,863 → 1,930, style.js 421 → 439, families.js 450 → 485: function
headers, and a line or two above each stage saying what it takes and gives).

Results: every fitted dot and every family estimate of the 118 local images identical, by digest; 216 tests pass
(1 standing skip). In the app, where the scan runs in its worker with progress reported by stage, Laser fits to 73
dots, bench1 group-smoother-ci to 100 in two families with its band divided out and noted, and dog-ages to 95,
with no console errors. No test was added: the stages are private to their modules, and the tests of
opaqueFamilies, findCandidates and estimateFamilies through real and synthetic images cover them as before.

**One path from image to dots for the app and the bench** (2026-11-20)

tools/bench.mjs extractAll was a copy of what the app does between its workers: scan with exact colors and, for a
PNG with JPEG noise, again with noisy matching; mask each family beyond the axes, at clip edges and in text and
legend boxes; fit; tell the dots of an outlined or palette family apart by fill color. The app had the same steps
in src/scan-worker.js and src/ui/main.js, each written a second time, and nothing checked that they agreed. They
are now in src/core/extract.js, and both sides call them:

- `scanFamilies(source, { jpeg, tryNoisy, onProgress })`: the scan and the choice of color matching, with the
  reasons for the rule in one place. The scan worker's own copy is gone; it passes the result on without the image
  as analyzed, which the page makes for itself.
- `excludedBoxes(work, radius)` and `maskOptions(axes, exclude)`: what buildMask is told. The app keeps its cache
  of the boxes by radius and adds the user's dilation.
- `fitInput(image, est, mask, style)`: what fitDots is given, with the user's style when there is one.
- `dotFills(work, est, dots, radius)`: each dot's fill color for an outlined or palette family. The app labels its
  dots with it, extractAll makes a result per color (now marked with `fill`, the color's index).
- `extractAll(source, jpeg)`: all of it at once, for the bench and the tests, which import it from core now.

The bench keeps what is only the bench's: the Hough-only candidates, added to extractAll's results, and `extract`,
the one-family path through estimateStyle that the synthetic one-family presets and compare-fits use.

One difference between the two paths showed, and is settled. The bench read the axes on the image as analyzed (a
JPEG's chroma restored), the app on the image as decoded, before the scan. Of the 25 local images where the two
differ (16 JPEG files, 9 PNGs with JPEG noise), the plot's sides differ on six of the PNGs, by 1e-6 to 0.08 px
(droplet1's bottom). extractAll now reads them as the app does, on the image as decoded: the app's axes carry the
tick values the user types, so they can't follow the choice of color matching.

Still different, and left: the synthetic one-family presets go through `extract` (estimateStyle, bounds only), not
through extractAll. Moving them would change what those rows measure, which is a decision of its own.

Results: every fitted dot and every family estimate of the 118 local images identical, by digest, both as moved
(axes on the image as analyzed) and with the axes read as the app reads them. In the app, six images through each
changed path give extractAll's counts with no console errors: dog-ages 95, step1 128 in two families (the noisy
scan wins and turns noisy matching on), bench2 group-smoother.jpg 200 (a JPEG file), bench1 group-graystroke 100
(fills told apart after the fit), wages2025 670 (a palette), pathogen 468. Tests: test/extract.test.js, among them
the app's steps taken in turn (scanFamilies, buildMask with maskOptions and excludedBoxes, fitDots with fitInput)
giving extractAll's masks and dots on a synthetic graph of three families, and the progress of the two scans.

**The review's three speed items, measured** (2026-11-20)

Measured on the 118 local images through extractAll, six at a time: 1,040 s summed (464 s the scan for families,
500 s the fits, by CPU profile; refits timed in an instrumented copy of the code). Nothing in the repository changed.

1. **Radius bumps refit from scratch: 98 s (9%), no remedy in the fit.** fitDots fits an opaque family again at
   r + 0.5 px when most of its dots come in close pairs ("doubled") or enough coincide ("coincident", kept only if
   the objective improves), up to three times. 18 refits in 8 images: 9 doubled (70.8 s) and 9 coincident (27.3 s;
   2 kept, 7 rejected, 21.2 s). phase1 alone is 76 s, 10 refits over its families, one of 37 s that starts from no
   Hough candidate and builds 28 dots by births; the other seven images total 22 s (beeswarm-boxplot 7.6, adults1
   5.1, delta1 4.3). Starting a refit from the previous fit's dots instead of the Hough candidates was tried: 93.8 s
   cold, 93.2 s warm on the eight images (phase1 72.1 → 73.6 s), and the dots changed in three of them (phase1 128
   → 132, kite1 26 → 27, delta1's positions). The time is in the rounds, not in the start. What phase1 pays for is
   a radius measured 1 to 1.5 px small for several families: a question of the estimate, not of speed.
2. **Linear openings: 112 s (11%), halved exactly by a plain change.** outline.js underRings (12 directions, a
   segment longer than a ring) takes 56 s and localFill (12 directions, 5 px, three channels) 56 s. A prototype of
   lineOpening that visits each offset of the segment once and skips the bounds tests away from the image's edges
   gives the same bytes (504 random planes with zeros of both signs and NaN, sizes down to 1 × 1; three real images
   at lengths 5, 21 and 41 in all 12 directions) and runs 1.8 to 2.3 times as fast (score2, 2404 × 2088: 0.91 →
   0.46 s at 5 px, 6.25 → 2.73 s at 41). About 58 s of the 1,040, 5–6%. Not tried: a running minimum along the
   segment, exact only in the four directions where the rounded segment is a straight run of pixels.
3. **Math.hypot at the remaining sites: 43 s (4%) on every line that calls it, as an upper bound.** 21 s are color
   tests per pixel (levels.js projectOnLine's distance off the line 4.7, opaque.js mixReader 1.7, axes.js isInk 1.5,
   annotations.js 1.4, colorDistance 1.3, and smaller); 19 s are geometry. Of the color tests, those that compare
   with a whole tolerance could become squared comparisons with the same result, about 6 s (0.6%); the rest store
   the distance or compare two of them with fractional colors, where a square root in place of hypot can differ in
   the last bit. Left as they are. The largest line is none of these: houghCenters' peak selection (fit.js, `kept.
   every(k => Math.hypot(…) >= nms)`, 7.9 s, and ringCenters' 0.6 s) tests each peak against every peak kept, O(n²);
   a grid of the kept peaks, cells of the suppression distance, asks the same question of the same pairs.

For scale, the largest self times in the same profiles: render.js blurPlaneRect 97 s and recompute 95 s (the fit's
re-rendering), layers.js distanceTransform 92 s, fit.js houghCenters 66 s.

**The two exact speed-ups from those measurements** (2026-11-20)

- **lineOpening** (outline.js, under underRings and localFill): each pixel of the rounded segment is visited once,
  and the bounds are tested only in the band near the image's edges, where a tap can fall outside; elsewhere a
  loop for the minimum and one for the maximum read the taps unchecked. Math.min and Math.max are kept, since
  comparisons would order −0 and 0 differently and drop NaN.
- **Hough peak selection** (fit.js, houghCenters and ringCenters): `suppressNear` keeps the kept peaks in a grid
  and tests a peak against its own cell and the eight around it, instead of against every kept peak. The cells are
  1.01 × the suppression distance wide, so rounding in the division can't put two peaks closer than the distance
  two cells apart; the grid is a Map of the occupied cells (a 4000 px image at 2 px would be 4 million cells).

Results: every fitted dot and every family estimate of the 118 local images identical, by digest. The same run,
six images at a time: 1,037.9 s summed before, 978.0 s after, 5.8% less (four earlier runs of the unchanged code
under the same load gave 1,037 to 1,040 s). Most on the images with outlined or ringed dots at full size:
beeswarm-boxplot 66.5 → 60.1 s, violin-beeswarm 40.0 → 35.2, score2 56.6 → 51.9, kite2 16.7 → 13.3, degradation1
18.5 → 15.3. In the app dog-ages fits to the same 95 dots, pathogen to 468 and bench1 group-graystroke to 100,
with no console errors. Tests: lineOpening against its definition, every tap tested against the bounds, to the
bit (360 cases: sizes from 1 × 1, lengths 1 to 51, 12 directions, zeros of both signs and NaN) in
test/outline.test.js; suppressNear against testing every kept peak (random peaks at four densities, and a lattice
exactly the distance apart, where "closer than" decides) in test/fit.test.js.

**The benchmark, several images at a time** (2026-11-20)

`npm run bench -- --jobs n` (and `npm run results -- --jobs n`) fits n images at a time. tools/bench.mjs's main
loop, four nested loops that read, extracted, scored and printed as they went, is now a list of jobs (listJobs: one
per synthetic graph and per image with truth, in the order of the rows), a function that runs one (runJob: the rows
it adds, the failures, the lines to print) and a runner (runJobs) that takes them one at a time in the process, as
before, or n at a time in worker threads, each worker taking the next job as it finishes one. Results are reported
in the jobs' order whatever order they finish in, so the table, the --json rows and the messages are the same. The
default is 10 at a time (or the machine's cores, if fewer), chosen from the measurements below; `--jobs 1` is the
serial run. No dependency: node:worker_threads, with the file itself as the worker.

Measured, on the full bench (144 rows; 12 performance and 4 efficiency cores):

| Jobs | Wall clock | RESULTS.md's times, summed |
|---|---|---|
| 1 | 1,115 s | 1,058 s |
| 6 | 269 s | 1,166 s |
| 10 | 207 s | 1,205 s |

- 4.1 times as fast at 6 jobs and 5.4 at 10. The floor on the wall clock is the slowest image, phase1 (117 s at 6
  jobs).
- The rows are identical: 1, 6 and 10 jobs, every number of RESULTS.md's 128 table rows but the time (and the 144
  --json rows of 6 against 10).
- The times are not: an image's time grows with what runs beside it, 10% in all at 6 jobs and 14% at 10. So times
  compare only between runs at one job count, and RESULTS.md's header now says how its images were fitted.

RESULTS.md is regenerated from the serial run (`--jobs 1`): every row as before but the times, which sum to 1,058 s
against 1,111 s before the two speed-ups of the last entry, 5% less. The next `npm run results` without `--jobs 1`
will give times at 10 jobs, about 14% longer, and say so in its header.

Test: test/bench.test.js runs the bench as a program on four synthetic graphs, alone and with three workers, and
compares the rows, their order and the printed table apart from the times.

**Every bench image through extractAll** (2026-11-20)

The synthetic one-family presets (sparse, dense, curve) still went through the bench's own `extract`: estimateStyle,
a mask with the plot's bounds only, the fit. Measured first: through extractAll all 15 graphs (3 presets × 5 seeds)
give the same dots and the same mask, pixel for pixel, 15 to 30% slower, since the full path also looks for
outlined dots and, for the two presets with noise, scans a second time with noisy color matching. So they are
moved, `extract` is deleted, and tools/compare-fits.mjs takes each image's first family from extractAll (dog-ages
against its actual-size rendering: 95 dots each, all matched, as before). A one-family graph that extractAll reads
as several families is now scored by family, and its row's name says how many.

**phase1: the plot's clip edges, read on all the families together** (2026-11-20)

The reading of phase1's refits in "The review's three speed items, measured" above (a radius measured 1 to 1.5 px
small) was a guess from the refits, and wrong. Measured on the pixels, an isolated dot's outline is 38.0 px across, r 19.0; the estimates are 19.1 to 19.5.
(The radii in the hand truth, 18.3 to 19.5, are a fit's at the time it was saved, not a measurement.)

What the refits came from: 27 of the 45 extra dots were on the colorbar, a strip of the five grays at the image's
right edge, beyond where the x axis line ends. Those packed false dots are close pairs, and an opaque fit with over
40% of its dots in close pairs is fitted again at a larger radius. The plot clips its dots where the x axis line
ends, one dot of each of four families cut in half there, and layers.js already masks what lies beyond such a cut;
but it asked each family alone for 2 r of cut rows, and one half dot gives 1.9 to 2.0 r: one family passed (39
rows for 2 × 19.24), three didn't (37, 37, and black's 42 of 81, its rows shared with a tick).

- **The rule** (layers.js plotClips; extract.js familyClips for the app and extractAll): an edge clips a family
  where its own coverage is cut, as before, or where the coverage of all the families together is. A plot clips
  every series alike. Together phase1's families show 7.6 r at the x axis line's end, and 5.0 r at the y axis
  line, where dots of the first x value are cut in half and no family alone showed it.
- **Not the sum over families.** engagement1's four families share a dark stroke, the color of its text, and the
  top of the line "Post-sipper" reads as 1.9 r of cut in each family's coverage: 104 of 104 rows summed, a false
  clip. In the union those pixels count once, 1.9 r, no clip.
- **Surveyed on the 118 local images**: the union clips nowhere that every family doesn't already clip alone, but
  in phase1. (The other clipped images have one family: autocor1, comment1, adults1.)

Tried and left: accepting a "doubled" refit only when the objective improves. The refits that remain in phase1
(its top row, five families' dots drawn over one another) mostly make it worse, but so does delta1's, which is
plainly right (52 dots, 51 in close pairs, to 26): the rule is for what the objective can't tell.

Results. phase1: 128 found for 97, P 0.625, R 0.825, RMSE 1.30 px, becomes 101, P 0.812, R 0.845, RMSE 1.15, and no
dot on the colorbar; 112 s alone becomes 74 s. Of the 16 extras left, most are doubles in the top row, where the
five families' dots lie over one another, and two families' radii still end at 20.2 and 20.4 px after the refit
for close pairs. Every other image of the 118: the same dots and estimates, by digest (phase1's estimates too: only
its masks changed). 227 tests pass (1 standing skip). In the app phase1 fits to the same 101 dots (20 + 19 + 24 +
19 + 19), none on the colorbar, and dog-ages to 95, with no console errors.

RESULTS.md is regenerated at the bench's default, 10 images at a time, in 172 s: every row as before but phase1's,
and the times, which are now those of a 10-job run (10% more in all than the serial run's) and say so in the header.

Tests: plotClips on planes made by hand (two families with one cut dot each, 1.5 r alone and 3 r together; a
family with no dot at the edge gets the cut too; the same cut dot in two families' coverage is no cut; a family
without a radius; a large dot lying across the edge) in test/layers.test.js; phase1's regression now asks for P
0.75, no more than 108 dots and none beyond x 1300; test/extract.test.js takes the app's steps with familyClips.

**phase1's top row: curve stubs and outlines that read as dots** (2026-11-20)

After the clip edges, phase1 had 19 extras counted by family: black's 8 (16 dots fitted as 24), and 5 dots of gray
82, 4 of gray 187 and 2 of gray 136 where no dot of that gray is. Two causes, seen in each family's coverage
image, and neither is the overlap of the five families as such.

- **A gray's own curve, showing between dots.** Each gray has a curve in its fill's color, 12 px wide, under its
  dots. Along the top row the dark gray curve shows for 5 to 8 px between two mid gray dots. Away from dots the
  curve is a thin mark and masked, but a stub this short isn't: it is opened together with the other families'
  dots around it, which may hide the rest of a dot, and so it read as the visible part of a dot hidden under them.
  The fit put a dark gray dot under the two neighbors for each stub (flagged hidden, 30 to 40% visible), and a
  second dot where a curve leaves a dot. What tells a stub from a dot's visible part: an outlined dot's fill never
  meets the page, its outline lies between. **bareFill** (layers.js): for an opaque family with an outline at least
  2 px wide, a connected patch of fill pixels that meets the page bare (the page within 2 px, at 6 of its pixels
  and a tenth of its border at least) is masked as a thin mark. A dot's own fill is a patch of its own, closed off
  by its outline. The page beyond a clip cut doesn't count: a cut dot's fill meets it along the cut.
- **A gray dot's outline beside a black dot.** Black is a family and the grays' outline color. In black's coverage
  a thin black pixel within an outline's reach of a gray fill was the gray's only when it was nearer that fill than
  black's own dots, the rule for a stroke between two fills; an outline running beside a black dot was black dot,
  1 to 3 px beyond its disc along an arc, and a second black disc a pixel or two off explained it. (The fit at the
  bumped radius, 19.9 px, had 16 clean dots; its own radius refit brought the discs back to 19.4, and the births
  after it added 8.) For a family in the outlines' color every such pixel is now the gray's (opaque.js
  outlineRings, `whole`), whether the family was found by its own dots or by solidsInOutlineColor.

Tried and left: fitting the grays by their rings (families.js ringEstimate), the path made for outlined dots that
overlap. It declines in phase1 because the ring image reads the 2.6 px outline as the black family's fill wherever
it is "thick" at 1.5 px, eight places around each ring, so no whole ring is left to measure. Read whole (black
counted as dots only where thicker than the outline), the light gray was fitted by its rings at r 18.9, the true
radius, but lost its half dot at the y axis, and the mid gray got worse (5 extras for 2): no gain overall (P 0.833
against 0.863 for the black fix alone).

Results. phase1: 101 found for 97, P 0.812, R 0.845, RMSE 1.15 px, becomes 85, P 0.965, R 0.845, RMSE 0.89; 3
extras (two black pairs, one of gray 82) for 19. No fit is refitted larger for its close pairs any more: the radii
end at 19.2 to 19.8 px (19.0 on the pixels; 20.2 and 20.4 for two families before), 15 dots are flagged for 34, and
the image takes 21 s alone for 74 (8 s of fitting in the app for 65). The 15 missed are as before: 9 dots of gray
136 that the truth places under others in the top row, where nothing of them shows, the three half dots where the
x axis line ends, and 3 more. Every other image of the 118: the same dots and estimates, by digest. 228 tests pass
(1 standing skip). In the app phase1 fits to the same 85 dots (20 + 15 + 18 + 17 + 15), bench1 group-graystroke to
100 and dog-ages to 95, with no console errors. RESULTS.md is regenerated (10 jobs, 156 s): only phase1's row
changes.

Test: a scene painted as the left of phase1's top row (test/opaque.test.js: the dark gray curve under mid grays
43 px apart, dark gray dots over two of them, black dots over the mid grays' outlines), every family's dots found
with none extra. Without bareFill the dark gray gets two hidden dots under its neighbors, as in phase1; without
outlineRings 18 of 30 outline pixels beside the black dots read as black dot, and 8 with it (the wedge where an
outline runs into the disc). phase1's own regression now asks for P 0.9, at most 18 black dots, and every fitted
radius within 0.6 px of 19.4.

**The noisy-matching switch has a test that runs again** (2026-11-20)

The one test asserting that a PNG with JPEG noise is switched to noisy color matching was written on primates1's web
capture, skipped itself since a clean PDF capture replaced that image (2026-11-14), and so nothing that ran checked
the switch. It is now on step1 (bench-plos), a noisy PNG (13.75% of the background pixels near marks slightly off):
with exact colors the blue dots alone, their radius unknown, nothing to fit; with noisy matching the blue dots at
5.4 px and the red rings at 6.4, 128 dots for 175 by hand, P 0.992, R 0.726. The test asks for the switch (0
fittable families against 2), both radii, and P 0.95, R 0.68; it skips if step1 is absent or no longer noisy. The
suite has no skipped test now.

**A lint-like pass: JSDoc, vague types, and whether the medians should keep a sort** (2026-11-20)

Checked with a throwaway script over src/ and tools/ (no linter is installed, and none was added): every function
with a JSDoc block documents each parameter; exported functions have a block and a `@returns`; no `{object}` or
`{object[]}` stands alone; the control-flow rules of CLAUDE.md; relative imports end in .js; nothing is declared
or imported and unused.

- Found: 26 functions with undocumented parameters, 32 without `@returns`, 10 exported without a block, about 20
  bare object types, and two JSDoc blocks stranded above the wrong function (colorLine's above alongLine, which had
  been inserted between them; ratioAlpha's above paletteColors). No control-flow or import slip, nothing unused.
- The recurring shapes are typedefs now, where they are made, and the docs name them: `LevelMeasure` (levels.js) and
  `OpaqueMeasure` (opaque.js), a family as measured; `Estimate` (style.js), a family ready to mask and fit, with
  the fields families.js adds; `Axis`, `Axes` and `ClipEdge` (axes.js); `FitInput`, `FittedDot` and `FitResult`
  (fit.js); `RingReading` (outline.js); `AxisEntry` and `AxisScale` (ui/axes-panel.js). The factories that return
  an object of functions (createFit, createDotModel, createOutlineModel, createDotStore, createView) list their
  members, each documented where it is defined.
- One line of code: a translucent family's `kind` was set after the fact, in two places in style.js
  (`for (const c of translucent) c.kind = 'translucent'`: `const` fixes the name, not the object, which the loop
  tagged). measureFamily now returns it, as measureOpaque returns 'opaque', and so does the one measure built by
  hand (style.js overBand), which the loop had been covering: without it landing1's two families lost their kind,
  which the comparison of estimates over the 118 images showed (their dots were the same).

**The sort in median, upperMedian and quantile**: each call sorts a copy. Over the profiles of all 118 images, the
three with their sorts take 0.17 s of 1,024 s (0.016%): 0.06 s under fitFill, 0.04 under findColormap, 0.03 under
radiusFromDistance, 0.02 under paletteColors, and no array is sorted twice for two statistics. A variant taking a
sorted array would save nothing that shows, so there is none.

**Category axes without tick marks; where axis detection stands** (2026-11-21)

Surveyed on the 118 local images: both axes found on 102, no x axis on 13, no y axis on 2, neither on 1. Each of the
219 axes found has at least two labeled ticks, so finding the label boxes is not the weak point; reading them is
missing (the user types the values), and so were category axes, which have labels and no tick marks. Label boxes are
small: a median of 13 px tall, 141 of the 219 axes under 16 px (the JMP renderings); published figures 28 to 74 px.

Of the 13 with no x axis, ten are one case: an axis line, no tick marks, a row of labels beside it (bench3's nine
renderings: "1" and "2" under the swarms, with the title "group" below; cells1: "L5PTs" and "GrCs" under its bars'
baseline). An axis needed tick marks, or gridlines standing in for them, so the line was dropped.

- **labelAxis** (axes.js), tried when an axis is still missing and the other one has a line: a dark line that meets
  an end of the other axis's line within 3 px (the plot's corner) and has at least two words in the row nearest it,
  on the side away from the plot, between the other axis and the line's far end. Each word's center is a position:
  a tick of length 0 with `source: 'label'`. The row is the words at the smallest gap, as for tick labels, so a title
  farther out is not a category; one word alone is taken for a title; the other axis's own label at the corner lies
  beyond its line (JMP's bottom y label straddles the x axis line there) and is outside the band searched.
- Such an axis bounds the plot as one with tick marks does (plotBounds), and its line and far end are clip
  candidates.
- The app's axes panel starts it as categories and says the positions are label centers.
- The positions are the labels' ink centers: bench3's are 205.0 and 478.5 px against lanes at 206.5 and 479.5 (a
  "1" isn't centered on its advance), cells1's 431.5 and 660.0 under bars centered near 430 and 661.

Not this case, and left: hair1 (its baseline is broken by the dots sitting on it, and no line is found; it has two
tick marks), boxdots1 (no y axis; box edges for lines), beeswarm-boxplot (no axis line; names on a gray band under
the plot, one in red), protrusion3 (one box, no label at all).

Reading the labels (asked 2026-11-21; nothing tried yet). Tesseract.js would do it in the browser, at the price of
the first runtime dependency and a download of its engine and language data; a built-in text detector isn't
available across browsers. A reader of our own for tick labels looks feasible without either: the alphabet is
small (digits, point, minus, sometimes % or an exponent), the glyph boxes are already found, and the axis checks
the reading, since evenly spaced ticks must fit a linear or log scale (two right labels fix it, a misread shows as a
tick off the fit, as typed values are flagged now). 23 truth files already list their tick values (bench1–3 and
about 20 real figures, with log scales and decimals), so its accuracy can be measured from the start. Open:
superscript exponents (phase1's 10¹), rotated y labels, units and thousands separators.

Results: the ten images gain an x axis of two label positions, and no axis of the other 108 changes. Every fitted
dot and estimate of the 118 is the same, by digest, the plot's new bottom side included, and the bench rows of
bench3 and cells1 are unchanged; cells1's lane truth now pairs with the label positions (431.5 and 660 px) where it
took the lanes from the fitted dots. 233 tests pass. In the app bench3 dots-swarm shows the x axis with a mark at
each label, starts it as categories, and fits the same 204 dots, with no console errors. Tests: a synthetic graph
with a labels-only axis (synth.js AxisSpec.labelsOnly), a title under the labels and a y label at the corner; one
word, no word, and a labeled line away from the corner give no axis; bench3 (three renderings) and cells1.

**Reading tick labels: a reader in core, measured; not yet in the app** (2026-11-21)

labels.js reads an axis's tick labels as numbers (`readAxisLabels`), and nothing calls it yet: this step is the
reader and its measurement. It is a reader of our own, with no dependency: a label is cut into glyphs, the glyphs are
digits, points, commas and minus signs by their size and place, digits are told apart by correlation with templates,
and the axis checks the result.

- **Finding the label again.** A label's box from axes.js is where to look, not what to read (cells1's "100" boxed
  as "00", autocor1's "1.0" as "0", JMP's "60" boxed with two letters of the rotated title). The label is the line
  of text through the box, cut into glyphs at empty columns and into words at gaps over 0.45 of the digits' height,
  and the word at the tick (under an x tick; nearest the line beside a y axis, short of its tick marks). Touching
  digits (a "4" and a "0" at 13 px share a gray column) are cut at the lightest column near even divisions.
- **One line per axis** (`labelOptions`). The labels are found twice: each from its own box, and then on the line
  the cleanly found ones share (one row below an x axis; one height and one offset from the tick beside a y axis).
  A glyph reaching more than 0.15 of the height beyond that line is foreign and ends a word. This is what reads
  JMP's y axes: the plot title's "p" hangs through "90" and the rotated axis title sits 6 px from "60", taller than
  it. Before it the JMP y axes read 5 of 7 labels and the two others as "???" and "??".
- **Kinds by geometry** (`glyphKinds`): tall glyphs are digits; a point is small and on the baseline; a comma's tail
  ends below the label's own digits' feet (measured from them, not from the axis's line, which can be a pixel off:
  fibers-distance's "3.7" read "3,7"); a minus is flat at mid-height; a dash after the number is a tick mark and is
  dropped. Thousands separators are removed ("2,000"); anything else makes the label unread, never guessed.
- **Unit signs** (`unitSigns`): a glyph that is no digit and is alike (correlation 0.85) at the start or end of
  every label of an axis is left out: wages2025's "$320" is 320, droplet1's "75%" is 75 (in percent, the axis's own
  unit, where its CSV has fractions).
- **The scale check** (`consistentValues`): among linear and log scales through each pair of read values, the one
  most read values agree with (within 2% of the least tick gap, or 1 px), refitted by least squares to those. It is
  believed with at least three agreeing and at most a third as many read values against it. Ticks that are unread
  or disagree get the scale's value, rounded to the roundest number within the tolerance (89.9083 from a tick half
  a pixel off is 90). So one misread among several is corrected (dog-ages's "80" was read "60" by the first
  templates), and without a scale nothing is offered: two labels, names, a repeated value.
- **Templates** (digit-templates.js, generated). First try: the mean shape of each digit in bench1–3's own labels,
  one font (JMP's Helvetica). On the real figures it misread Arial's "8" as "6" (dog-ages, step1, score2, comment1)
  and didn't know other fonts' "1", "3", "5", "7" (adults1, wages2025, landing1's bold). Replaced by shapes drawn by
  Chrome (tools/digit-fonts.html, run by tools/make-digit-templates.mjs) in 18 fonts of this Mac, each the mean over
  nine sizes and four subpixel places, cut out by the reader's own segmentation. No template comes from a test image
  now. Leaving each font out in turn, the other 17 tell all 180 of its digits. The floor for a digit's correlation
  is 0.7: of 1,282 labels that their scale confirms, the least sure digit scores 0.72 and the rest 0.82 or more;
  letters reach 0.6 (with a floor of 0.6 names read as "?5?1?").

Measured with `npm run bench:labels` on the 93 images whose truth lists tick values (181 axes, 1,218 labels):

| set | images | axes | labels read right | axes solved | wrong | unchecked (right) | none |
|---|---|---|---|---|---|---|---|
| bench1 | 32 | 64 | 448 of 480 | 64 | 0 | 0 | 0 |
| bench2 | 32 | 64 | 448 of 480 | 64 | 0 | 0 | 0 |
| bench3 | 9 | 18 | 72 of 81 | 9 | 0 | 9 (9) | 0 |
| testgraphs | 2 | 3 | 19 of 19 | 2 | 0 | 1 (1) | 0 |
| bench-nature | 14 | 24 | 98 of 121 | 18 | 0 | 2 (1) | 4 |
| bench-plos | 4 | 8 | 29 of 37 | 5 | 0 | 0 | 3 |
| all | 93 | 181 | 1,114 of 1,218 (91.5%) | 162 (89.5%) | 0 | 12 (11) | 7 |

Solved: a scale found and every labeled tick's value right. Wrong: a scale found with a wrong value, the outcome that
matters, and there is none. The first run, with one font and no shared line, rounding or disagreement rule, read 70%
of the labels it covered, and its scale gave a wrong value on 73 of 108 axes: unrounded values such as 89.9083 on
every JMP y axis, and hair3's doses taken for a log scale.

The 104 labels not read one by one: JMP's "90" under the title's "p" in all 73 renderings (the scale gives it), 23
names on Nature's category axes, and 8 on bench-plos (landing1's two names, grader1's six). The 19 axes not solved:
- unchecked, every label read as a number and no scale to check it: bench3's x axes (two labels, "1" and "2"),
  escape1 x (1 and 4), pathogen y (64 down to 1, and 0, on log(1 + y): neither linear nor log), hair3 x (doses 0, 5,
  20, 50, 100, 200 as category names; read as written, where its CSV has 1 to 6);
- none: names (cells1, hair2, protrusion1, protrusion2, landing1), and grader1's two axes, read from gridlines,
  whose label boxes are not its labels (an axes.js matter).

On the 45 real figures (83 axes with labels, most without tick truth): a scale on 61, unchecked 6, none 16. In 59 of
the 61 every label was read and agrees. delta1's y axis has two false ticks (error-bar caps) that get the scale's
unround values 154.3 and 145; kite2's "2" sits 13 px off its axis's even spacing, so the scale says 1.9 there and
the tick is flagged as disagreeing. The 25 of these axes with truth are right; the other 36 give round, evenly
stepped sequences that match the labels as read but were not each compared with the image by eye.

Cost: 0.58 s for all 118 images (5 ms each, 45 ms at most, autocor1's 47 px digits), against 4.3 s for detectAxes.

Not done, and known:
- The app doesn't use it. Next: offer a confirmed scale's values in the axes panel as proposals the user can
  accept or overtype, marking the ticks whose own label disagrees or wasn't read, with the scale type.
- Exponents: phase1's and primates' "10" with a raised power reads as 10, 10, 10 (the raised digit is foreign), so
  those log axes get nothing. Common enough to be worth reading next.
- Two-label axes have nothing to check them. Offering unchecked readings would have been right on the 12 with
  truth (hair3's doses are right as written) and wrong on the exponent axes above.
- Decimal commas, rotated labels, dates. Fonts with old-style figures (Georgia). matplotlib's DejaVu Sans isn't on
  this Mac; Verdana, its near relation, stands in, and no local figure tests it.

Tests (test/labels.test.js, 16): the templates' leave-one-font-out; kinds by geometry, with the point-versus-comma
regression from fibers-distance; text drawn from the templates at 12 to 28 px with minus, comma and point, every
font, a box cut short, touching digits; the scale check (an unread and a misread tick, log, decimals, and the cases
that must give no scale); synthetic axes with thousands, unit signs, a neighbor's glyphs beside and through a label,
names; bench-plos (step1, score2, droplet1, landing1) and bench1's medium-dots (PNG and JPEG). Removing the shared
line, the unit signs, the splitting, the rounding, the disagreement rule or the baseline rule each fails a test.

**Tick values read into the Axes panel** (2026-11-21)

The app now reads the tick labels when an image is opened (labels.js `readTickLabels`, right after detectAxes, on
the main thread: 5 ms an image, 45 at most) and an axis whose labels a scale confirms starts with that scale's
values and type, so data coordinates are there with nothing typed.

- **What is offered.** Only a confirmed scale (at least three labels on one linear or log scale): its value at
  every tick, labeled or not. An axis without one starts empty as before (two labels, names, exponents), or as
  categories when it has no tick marks.
- **Where each value is from**, a column beside it: "read" (its own label, which fits the scale), "scale" (its label
  wasn't read, or it has none: bench3's minor ticks, JMP's "90" under the title), or "label: 2" in the warning color
  when its label reads as another number that the scale contradicts (kite2's bottom tick, 13 px off the others'
  spacing: the value is the scale's 1.9, and the summary warns). The tooltip says the same in a sentence. The summary
  counts them ("Read from the labels: 6 of 7 fit a linear scale; the scale gives the other 7 ticks"), and the section
  heading says "x linear, read".
- **The value at a disagreeing tick is the scale's, not the label's**: of the three ways it happens, a misread label
  and a false tick (delta1's error-bar caps: 154.3 and 145) are right that way, and a tick truly off its axis's
  spacing (kite2) is right neither way and is flagged both ways.
- **No provenance is stored.** An entry is still type and texts; a tick is marked while its value equals the
  reading's (`tickStatus`), so typing over one unmarks it and typing the reading's own value is the same as reading
  it. An entry that is just the reading isn't written to localStorage (it is read again next time, by the reader as
  it then is); anything the user changed is remembered and wins, a cleared axis included. A button per axis: Clear
  while the values are the reading's, Use read values otherwise.
- The label crop in each row is the tick's box joined with the glyphs read, so a box that axes.js cut short (cells1's
  "00") shows the whole label. A unit sign the reading leaves out stays in view where the box has it (wages2025's
  "$320" beside 320): the values are in the unit the axis states.

Checked in the app (the built-in browser, images dropped on the view): bench1 medium-dots (x 8 of 8 read; y 6 read
and "90" from the scale), kite2 (the "label: 2" warning), wages2025 (y log, 320 down to 10, "$" left out; x left
empty), bench3 dots-swarm (x categories; y 7 labels and 6 minor ticks), hair3 (x left empty), delta1. Typing over a
value, Use read values, Clear, and reopening the image after each behave as described; no console errors. The
About text and the README's steps say values are read where they can be checked.

Tests (test/axes-panel.test.js, 4): the entry a reading gives, each tick status and its loss on typing over, the
starting entries (read, empty, categories), and bench-plos step1 from the image to data units with nothing typed.

Left: the exported table and the readout don't say whether an axis's values were read or typed; the panel does.
Unchecked readings (two-label axes, hair3's doses) are not shown anywhere yet.

**Exponent labels read** (2026-11-21)

A number with a raised exponent is now read as its power: "10" with a raised 3 is 1000, shown as the text "10^3".
Three local figures label log axes that way (phase1 x, primates1 x, primates2 x and y); before, each label read as
"10" and the axis got nothing.

Measured first (phase1's digits are 53 px tall, primates' 28): an exponent's digits are 0.65–0.68 of the digits'
height H, start 0.36–0.39 H above the digits' top and end 0.25–0.32 H below it, 0.14–0.18 H after the "0". A negative
exponent's minus sign is 3 px thick at the digits' top itself, and in primates2's x labels it touches the "2" after
it.

- **Finding the exponent** (labels.js `labelGlyphs`, the new `raised` glyphs of a found label). After the number's
  last glyph, the columns with ink in a band from 0.6 H above the digits' top to 0.6 H below it and none lower in the
  line are cut into glyphs from the band's own rows (the line's rows show a raised "1" without its flag, and can miss
  a minus sign altogether). The run ends at a word's gap, at a glyph of the line, or at one that reaches above the
  band (a title's descender: the search now looks one row beyond the band to tell). It is an exponent only if one of
  its glyphs is at least 0.45 H tall; otherwise the label reads exactly as before (a degree sign, a mark at the top).
  A minus sign touching its digit is cut off as the run's first columns where they are flat and at mid-height (a
  "1"'s flag is flat too, higher up).
- **Reading it** (`glyphsRead`). The exponent is read as a line of its own, with its own digit height, by the same
  kinds and templates. The value is an integer to an integer power, or the label is unread. Powers of ten are taken
  from their decimal text: `10 ** -5` is 0.000009999999999999999 in JavaScript.
- **The label's box and tick** (axes.js `mergeWords`). primates1 showed a second problem upstream: "10" was boxed
  without its raised 2, the box's center was nearer the minor tick of 90 (7.7 px before the tick of 100) than its
  own, and the three labels, 174 and 159 px apart, fit no scale. An exponent now joins its number's word: a shorter
  box right after a taller one (gap at most 0.3 of its height), at most 0.8 as tall, starting between 0.6 above and
  0.25 below its top and ending below its top and at most 0.6 down. The first form of the rule, a band alone, merged
  three of autocor1's x labels into one word and changed three other axes (Laser x, cantons y, pathogen y). With the
  limits on gap, height and start only pathogen's y axis still changed: its "32", right of a part of the rotated
  title and wholly above it, joined the title and its tick lost its label. Requiring the exponent to end below the
  number's top keeps them apart.

Results:

| axis | before | now |
|---|---|---|
| phase1 x | 10, 10, 10, 10: no scale | 10, 100, 1000, 10000, log, 4 of 4 |
| primates1 x | 10, 10, 10 at the wrong ticks: no scale | 100, 1000, 10000, log, 3 of 3; the 20 minor ticks 70 to 20000 from the scale, all round |
| primates2 y | 10, 10, 10, "10?": no scale | 100, 10, 1, 0.1, log, 4 of 4; the 31 minor ticks 200 to 0.04 from the scale |
| primates2 x | unread | "10^-2" and "10^-1" at their own ticks: two labels, so no scale and nothing offered |

Nothing else moved. Of the 229 labeled axes of the 118 local images, the other 225 read the same texts and scales
(grader1's unread second x label is now "?^?" where it was "?"). The label benchmark is identical: 1,114 of 1,218
labels, 162 of 181 axes solved, 0 wrong (none of the four exponent axes has tick truth; their values were compared
with the images by eye). detectAxes gives the same lines, ticks and label boxes on all 118 images except the label
boxes of those four axes, and the three figures' fits are unchanged (bench rows equal to RESULTS.md's, which was not
regenerated). Reading the labels of all 118 images takes 0.68 s, from 0.64.

In the app (the built-in browser, images dropped on the view): primates1 opens as "x log, read · y linear, read",
with 100, 1000 and 10000 marked read and the minor ticks marked scale, and its 27 dots get data values with nothing
typed (a dot at x 328.7 px is 89.2, just under the tick of 90); primates2 as "x: 13 ticks, no values · y log,
read"; phase1 with x log. No console errors.

Tests (6 new in test/labels.test.js, 2 in test/axes.test.js): powers at 14 to 28 px (negative, two digits, base 2,
a minus before the number, a box without the exponent); a touching minus cut off, and a "1" with a flag left whole;
a small raised mark, an exponent that is no number and a stroke from above are no exponents; log axes labeled with
powers; minor ticks beside labeled ones, in drawn text and in blocks; a label above a taller glyph; the three
Nature figures. Each rule removed in turn fails a test: the join in axes.js, its end-below-the-top condition, the
cut of a touching minus, its mid-height condition, the least height, the look above the band, the reach limit, and
the decimal text of powers of ten.

Not done, and known:
- primates2's x axis has two labels and 11 minor ticks at log positions between them: the minor ticks could check a
  two-label reading on a log axis. This is the nearest next step for unchecked readings.
- A mantissa with a power ("2 × 10³") and R's "1e+05" stay unread (the "×" and the "e" are no number's glyphs).
- An exponent of two digits that touch is not cut apart and stays unread.
- Only Nature's layout of exponents was measured. The limits leave room for exponents set lower (matplotlib's), but
  no local figure tests that.

**Two labels on a log axis, checked by its minor ticks** (2026-11-21)

Two read values always agree with the scale through them, so until now an axis with two labels got nothing. A log
axis offers another check: its minor ticks stand at 2 to 9 times each power of ten, bunched toward the next power,
and only the right scale puts them on those values (labels.js `logByMinorTicks`, called by `consistentValues` when
three labels don't settle it).

The rule: a log scale through two read values is believed when
- the unread ticks it puts on a multiple of a power of ten show at least 3 of the multiples 2 to 9
  (`MINOR_MIN_MULTIPLES`),
- the unread ticks it puts on none are at most a third of those (a false tick, as `SCALE_MAX_DISAGREE` allows one
  label in four off a scale), and no more than that share of the read values is off it (so two labels means both),
- and its tolerance, 1 px or 2% of the least tick gap, is at most 0.02 of a decade (`MINOR_MAX_TOL`): 9 and 10 are
  0.046 apart.

The scale is then fitted to the labels and those ticks together, and every tick gets its value as before.

What a wrong reading does to the minor ticks, which is why three multiples are asked for:

| the two labels read as | where the ticks of 2 … 9 land | on a multiple |
|---|---|---|
| right | 2, 3, …, 9 | all 8 |
| backwards (0.1, 0.01 for 0.01, 0.1) | 5, 3.33, 2.5, 2, 1.67, 1.43, 1.25, 1.11 | 2 (the ticks of 2 and 5) |
| two decades for one | 4, 9, 16, 25, … | 2 |
| three decades for one | 8, 27, 64, … | 1 |
| leading digit wrong in both (7, 70 for 1, 10) | 14, 21, 28, … | 0 |
| leading digit 2 in both | 4, 6, 8, 10, 12, … | 4 of 8, and 4 off: refused by the share off |

Not told apart: both labels off by the same power of ten (0.1 and 1 for 0.01 and 0.1), which moves every tick to
another multiple. That takes two misreadings that agree, as three labels off by one factor would on a linear axis;
the panel says so beside the values. Also refused, by design: minor ticks at 2 and 5 only (backwards they land on 5
and 2), an axis with decade ticks alone, and a decade shorter than 50 px. A linear axis with two labels stays
unchecked: evenly spaced ticks fit every linear scale.

Results: primates2's x axis, the one local axis of this kind, now reads 0.009, 0.01, 0.02, …, 0.1, 0.2, 0.3 on a
log scale from its labels "10^-2" and "10^-1", with all 11 unlabeled ticks on a multiple (its decade is 426.6 px,
the 13 ticks within 0.03 px of the fitted scale); the two labels were compared with the image by eye. The other 228
labeled axes of the 118 local images read the same texts and scales, among them the two-label axes that must stay
unchecked (bench3's categories, escape1's 1 and 4, the economist's 2016 and 25). The label benchmark is identical
(1,114 of 1,218 labels, 162 of 181 axes, 0 wrong; primates2 has no tick truth). Reading all 118 images takes 0.61 s.

In the app primates2 opens as "x log, read · y log, read" and its 15 dots get data values with nothing typed. The x
axis's summary says what checked it: "Two labels alone prove no scale: this one is checked by 11 ticks without
labels, which it puts on 2 to 9 times a power of ten. Both labels off by the same power of ten would pass too." The
About text and the axes note name the two-label case. `TickValues` has a new `minor`, the number of confirming
ticks (0 when three labels agree).

Tests (2 new in test/labels.test.js, and primates2's x axis in the bench-nature test): the scale and its values
from two labels and eleven minor ticks, a false tick among them, `minor` 0 with three labels; no scale from labels
read backwards, three decades for one, a wrong leading digit (7 and 2), a third label off the scale, minor ticks
at 2 and 5 only, a 40 px decade, as many ticks off the multiples as on, and a linear axis of whole numbers labeled
at its ends. Each condition removed or loosened in turn fails a test (one or two multiples enough, any tolerance,
ticks off ignored, read values off ignored, a power of ten counted as a multiple).

**Scientific notation read: "1e+05" and "2 × 10³"** (2026-11-21)

Two more ways figures write large and small numbers, both unread until now (the "e", the "+" and the "×" were no
number's glyphs): the power of ten after an "e", as R writes it in base graphics and ggplot2 ("1e+05", "1e-03"),
JMP ("1e+5") and Excel ("1.E+05"), and a number times a power of ten ("2 × 10³"). No local figure has either, so
the evidence is a new set of R's own renderings, `bench-r/` (21 images drawn by `bench-r/make.R`, each with a CSV
of its points and tick values; R 4.6.0 on macOS, text in Helvetica, digits 9 to 28 px tall).

What the reader did with them before: "1e+05" read "1??05"; "2 × 10³" read "2?10^3" where the sign stood close, and
where its spaces were wider than a word's gaps the label was the "10³" alone, read as 1000 for 8000 (no scale took
it, four labels being one value).

The rules (labels.js):
- **Signs by template.** digit-templates.js now has the shapes of e, E, +, × and the x typed for × in the same 18
  fonts, drawn by Chrome like the digits (the digit templates are byte for byte the same). A glyph is the sign it
  is most like at 0.75 or more (`SIGN_MIN_SCORE`) and more than it is like any digit; e, + and × must be 0.45 to
  0.9 of the digits' height, E as tall as a digit. In bench-r's 160 labels read: e scores 0.87–0.97 at 0.68–0.78
  of the height, + 0.90–0.99 at 0.60–0.78, × 0.83–0.99 at 0.53–0.65, E 0.95–0.97.
- **The number's grammar.** A plain number may end in e or E, an optional sign and digits, and is then what
  `Number` makes of it ("1e-07", "2.5e+03", "2.E+05"). Before a raised exponent, "number × 10" is that number times
  the power (text "2×10^3"). Anything else with a sign in it stays unread: "1e", "e5", "2+3", "+5", "2×10" without
  an exponent, "2×5^3".
- **A multiplication sign keeps its word together.** Glyphs up to 0.9 of the digits' height apart are one word when
  one of them is the sign (`TIMES_MAX_GAP`; 0.45 ends any other word). Measured gaps beside it: 0.30–0.65, in
  plotmath's spacing and with typed spaces.
- **A minus sign touching its neighbor is cut off.** At R's default device size (digits 9 px) the "e" of "1e-01"
  runs into the minus, and at 96 dpi (12 px) the minus into the "0". The glyph's flat first columns at mid-height
  are a minus when what is left is more of a digit than the whole; its flat last columns when what is left is an
  "e".

Two bugs found on the way, both with regression tests from Chrome's own pixels:
- The first version of the minus cut took the tip of a 4's bar for a minus where the bar starts well left of the
  diagonal: Gill Sans's "1454" at 17 px read "1-154". Hence "more of a digit than the whole": a 4 is less of a 4
  without its tip.
- The exponent reader committed on 2026-11-21 had the same flaw: Arial's "10" with a raised 4 at 16 px read
  "10^-?". It has the same condition now.
A check of "4" alone, "1454" and "10⁴" in 20 fonts at 18 sizes in Chrome (15 of the template fonts plus Georgia,
Futura, Optima, Palatino and Symbol) gives the same readings as the committed reader everywhere else.

What a score of 0.75 does not exclude, measured on every letter and common symbol of 16 fonts at four sizes: as an
"e", a "c" reaches 0.85, an "o" 0.76; as an "E", an "F" 0.83 and an "L" 0.81; as a plus, "÷" 0.89 and "±" 0.78; as
"×", an "X" 0.80. The guard is the grammar and the scale check, as for a digit and a letter that look alike ("l"
scores 0.99 as a 1): "1c5" is no label a figure has, and three labels must fit one scale.

Results on bench-r (`npm run bench:labels -- --set bench-r`): 191 of 209 labels read right, 38 of 42 axes solved,
0 wrong, 1 unchecked, 3 none. Every "e" axis is solved at every size, the default 72 dpi included; so are the "×"
axes from 96 dpi up, with a typed × and x, decimals before the sign, and ggplot2's parsed labels. Not solved:
- log-e-turned y: R's default turns the y labels along the axis, and turned text isn't read (none).
- times-r72 x (none): at 9 px two of the three plotmath exponents touch the zero under them, the pair is one glyph
  and the label's line is taken from it. The same leaves one label unread, with the scale still found, on
  times-log x (96 dpi) and gg-times-bw x. An exponent that touches its base is not cut from it.
- times-r72 y (none): a typed "³" at 9 px is 4 px tall, under an exponent's least height: "8×10?".
- small-e-r72 y (unchecked): plain decimals at 9 px, "0.00025" read "000025" in two of six labels. Not scientific
  notation: a decimal point lost between digits.

Nothing else moved. The other six sets of the label benchmark are identical (1,114 of 1,218 labels, 162 of 181 axes
solved, 0 wrong; with bench-r 1,305 of 1,427 and 200 of 223). Of the 229 labeled axes of the 118 local images, all
read the same values and scales; five unread axes of category names show an "e", "E" or "+" in their text where
they showed "?" (protrusion1, escape2, landing1, myeloid1, violin-beeswarm). Of 2,760 digits read in those images,
none is more like a sign than like its digit (the nearest by 0.09). Reading the labels of all 118 images takes
0.62 s, as before.

In the app (page check, images dropped on the view): log-e-default opens as "x log, read · y log, read" with 0.001
to 100000 on x, five read and four from the scale, and times-r96 as "x linear, read · y linear, read" with 0 to
600000 and 2000 to 8000; 40 dots found of 40 in both; no console errors. The About text names the forms read.

Tests (10 new in test/labels.test.js): the signs' templates told by the other fonts', and no digit more like a
sign; "e" numbers at 12 to 28 px in R's, JMP's and Excel's forms; "×" numbers spaced, unspaced, with decimals, a
minus and a typed x; the number kept across the spaces; signs that make no number; a low glyph that is no sign; a
touching minus, drawn and in R's own "1e-03" and "1e-01", with R's "1e+05" whole; the 4s of Gill Sans and Arial;
digits lower than their line; axes of both forms; and bench-r's images. Each rule removed or loosened in turn
fails a test (15 mutations: either cut, either cut's condition, the join, the join for any glyph, the score, the
height limits, the sign-over-digit comparison either way, the base 10, the mantissa, the typed x, the "e" grammar).

Not done, and known:
- An exponent touching its base (above): the nearest next step, since plotmath's and ggplot2's "10⁵" at small
  sizes would fail the same way.
- A minus drawn half inside its "e", so that less than a quarter of the digits' height shows, is read as no minus
  ("1e03"). No renderer sets text so (R's touch at 9 px, and the whole minus shows); found by drawing it.
- "e", minus and digit all three touching stay unread.
- ggplot2's default theme (gray panel, white grid lines, no axis line): the axes are not found, so nothing is read.
  This is axes.js's, and older than this change; it is why bench-r has no such image.
- Other fonts than Helvetica are covered by templates and drawn tests only: bench-r is R on macOS. matplotlib
  (DejaVu Sans) and Excel's own renderings are not measured.
- A leading plus ("+5") and a centered dot for the sign ("2·10³") are not read.

**An exponent touching its base, cut from it** (2026-11-21)

At small sizes plotmath's exponent follows its base with no empty column between them: "10" with a raised 5 at
9 px digits is a "1" and one glyph of the 0 and the 5, taller than the digits. The label's line was taken from that
glyph, the other digits were too low to be digits, and nothing read: 4 labels of bench-r (two on times-r72 x,
which lost its scale, one each on times-log x and gg-times-bw x).

The rule (labels.js `cutRaised`, in `labelGlyphs`): given the digits' line, a glyph that rises above it and whose
last columns have no ink below an exponent's rows is cut before those columns, when what is left then keeps to the
line (a tall letter's stem doesn't). The left part is a digit of the line, its rows the line's, and the cut-off
part is found again as an exponent standing apart is. Without a line from the axis's other labels, the label's own
line came from its tallest glyph: when the one glyph that starts highest is cut this way against the line of the
others, on their baseline, the line is theirs.

The base's rows are the line's because an exponent may hang over the base's last columns: STIX's "10" with a
raised 9 a pixel closer than touching read "16^9" (the 9's bowl over the 0 makes a 6) until the base was kept to
its line. That case is a regression test from Chrome's pixels.

Results: all four labels read, and times-r72 x has its scale. bench-r is now 195 of 209 labels and 39 of 42 axes
solved, 0 wrong (from 191 and 38); the other six sets of the label benchmark are identical (1,114 of 1,218; 162 of
181; 0 wrong), and so are the texts and scales of all 229 labeled axes of the 118 local images. Reading them takes
0.67 s (0.62 before).

A wider check, since bench-r has four such labels: "10" with a raised power (2 to 9, -3, 12) drawn by Chrome in 16
fonts at 13 sizes (12 to 40 px type), the exponent 0.66 of the size and a third of the digits' height above them,
at four distances from the base. Of 2,080 labels each:

| gap, × the type size | right, before → now | unread | wrong |
|---|---|---|---|
| 0.04 (apart) | 1,970 → 1,978 | 100 → 92 | 10 → 10 |
| 0 (touching) | 1,553 → 1,848 | 524 → 227 | 3 → 5 |
| -0.04 (a pixel closer) | 812 → 1,692 | 1,265 → 378 | 3 → 10 |
| -0.08 | 261 → 1,306 | 1,818 → 760 | 1 → 14 |

The wrong readings are exponent digits misread, a factor of ten or more each: with or without the cut about one
label in 200 to 400 at realistic distances, all at 12 to 20 px type (exponent digits 6 to 9 px tall) and all with a
template score of 0.81 or less, where about four in five of the right ones score more. They were there before for exponents
standing apart (the first row); the cut adds its own where the exponent overlaps the base and loses its first
columns (a 4 without its tip reads 1, a 6 reads 5). The axis's scale is the guard, as for every reading. A higher
score asked of exponent digits would trade most of them for unread labels, times-r72's among them (0.73–0.77):
not done.

Tests (3 new in test/labels.test.js, and two assertions on bench-r): R's own "4 × 10⁵" at 9 px; drawn powers of
one and two digits touching, after a number and a "×", and set lower; STIX's 9 over the 0; a touching raised minus
left alone; a tall letter with a flag and a slash between digits left whole, and a 7; an axis whose labels share a
line. Each condition removed fails a test (7 mutations: the rise above the line, the base keeping to it, the line
from the other glyphs, its baseline, which glyph is the highest, the base's rows, the cut itself).

Not done, and known:
- A raised minus sign touching the base (it doesn't rise above the line as a digit does), and two exponent digits
  touching each other, stay unread.
- An exponent set so low that it rises less above the digits than a foreign glyph must (15% of the label's height)
  is not cut.
- A typed "³" at 9 px, turned labels and ggplot2's gray theme are as before.

**Axes without axis lines: the edges of a filled panel (ggplot2's default theme)** (2026-11-21)

ggplot2's default theme draws no axis lines: the plot is a gray panel (230, 230, 230) on a white page, the
gridlines are white, and the tick marks, 2 px wide and 4 or 5 long at 110 dpi, leave the panel's bottom and left
edges into the page. detectAxes found nothing to hang ticks on, since an axis was a dark line. What it found
instead were the gridlines' antialiased rims (white is a background color, but 246 beside 230 is neither), as light
"gridlines" standing in for ticks, with scraps of dots for labels. Five such plots are now in bench-r (gg-gray at
72, 110 and 300 dpi, and with labels in each scientific notation).

The rule (axes.js `panelEdges`, used by `detectAxes`):
- A panel is the extent of a background color other than the page's, the page being the background at the image's
  border (in these plots the panel is the commoner color). It is kept when it is a rectangle: its first and last
  rows and columns are of its color at least 0.8 of the way across (`PANEL_EDGE_MIN_FILL`; gridlines cross them, a
  few percent in all). Dots of one color, bands and ellipses are backgrounds by their area too, 20 of the 118 local
  images have one, and their first row is a sliver.
- Its four edges are lines of no thickness, tried as axis lines when no dark line of that direction is an axis,
  and before gridlines stand in. Dark lines keep their precedence where both would do (a panel inside axis lines:
  the lines and their ticks leave its edges too).
- Along an edge a tick is no wider than it is long (a line's thickness measures a tick's width elsewhere), and its
  darkness is read from its second pixel: the first is the edge's own row, panel, page and stroke blended (194 at
  72 dpi, where the tick is 146).
Everything after that is as for an axis line: labels, the plot's bounds, the clip edges.

Results. All ten axes of the five plots are found, at the panel's edges, with every tick and its label, at 72, 110
and 300 dpi (ticks 3, 5 and 12 px long). The label benchmark on bench-r: 26 images, 228 of 245 labels read right,
47 of 52 axes solved, 0 wrong (21 images, 195 of 209, 39 of 42 before); eight of the ten new axes are solved, and
gg-gray-small's two are not, their digits 6 px tall ("50" read "?0"). The other six sets are identical (1,114 of
1,218; 162 of 181; 0 wrong). detectAxes gives the same lines, ticks and label boxes as before on all 118 local
images and bench-r's other 21. It takes 4% longer over those 144 images (5.14 s from 4.96).

The dots, checked once through a copy of the benchmark with bench-r registered (R's data placed through the
detected ticks): 40 of 40 found on gg-gray, gg-gray-small, gg-gray-log-e and gg-gray-times, precision and recall
1.000, RMSE 0.11 to 0.21 px. In the app gg-gray opens as "x linear, read · y linear, read" with 25, 50, 75 and 10
to 40, the axes drawn along the panel's edges, and 40 dots found.

Tests (4 new in test/axes.test.js): a drawn panel plot with blended edge rows and light first tick pixels; the
edges of a rectangle, and none for an ellipse, for a band fanning out to full height at its ends, for the same
turned, and for a page of one color; no axis from a panel without tick marks, and dark axis lines before a panel's
edges; bench-r's three sizes. Each rule removed fails a test (8 mutations: the fill, for rows and for columns
alone, the page by the border, a tick's width, its second pixel, the order, the edges themselves).

Not done, and known:
- gg-gray-hi (300 dpi): the app finds the 40 dots, and beside them a second family, dark gray and translucent with
  no radius, which is the labels' text; the benchmark's fit of every family stops on it ("The dot radius is
  unknown"). Fixed in the next entry. bench-r is not in the dot benchmark yet.
- The app's hint about JPEG-like noise shows on these plots: the gridlines' rims are background pixels slightly
  off.
- A panel without tick marks (seaborn's darkgrid, ggplot2 with `axis.ticks` blank) gets no axis: its white
  gridlines are not ink, so they can't stand in. Several panels in one image (facets) are taken as one extent.
- Text 6 px tall (ggplot2 at 72 dpi) is under what the label reader reads.

**A family with no ink inside the plot is no family; and three more details of a panel's edges** (2026-11-21)

The false family on gg-gray-hi, looked into. It is not the labels' text alone: the tick labels' gray (60 at their
cores) and the tick marks' (38) lie on the gray line from the panel's 230, at steps that fit two levels of a
translucent family exactly (α 0.87, full color 35), with 54 and 99 flat pixels. The text is thick enough at 300 dpi
not to be thin, and seven level-2 pieces are too few for the nesting rule to judge (20). It shows on the gray theme
only: on a white page the gridlines' gray is the commonest color in that direction and takes the line, with text
and ticks beyond its one level (theme_bw, theme_classic, theme_minimal and base R at 300 dpi: no such family); at
200 dpi the text has too few flat pixels.

Its ink is the axes' own, all of it beyond them. So the rule is about where a family is, not what its colors fit
(extract.js `insidePlot`): with the plot's bounds, a family whose ink lies less than 1% inside them is left out of
the scan's result (`PLOT_MIN_INK`), its ink being the pixels its count image gives half a dot or more
(`INK_MIN_COUNT`: the false family's image also gave 0.1 or less to 8,000 pixels of gridline rims inside the plot).
Nothing is left out when the axes bound nothing, or when no family would remain. The app passes the bounds to the
scan worker, and extractAll finds the axes before the scan.

Measured on all 191 families the scan finds in the 144 images (the 118 local ones and bench-r): the false family
has 0.000 of its 3,390 ink pixels inside the plot; every other family 0.287 or more (the lowest are black and
near-black dots, whose color is the text's too: molar1 0.287, droplet1 0.314, Laser 0.396).

A second 300 dpi plot, 7 in square, still failed, for want of a y axis, and showed three things about a panel's
edges that the 5 by 4 in one had not:
- Its y ticks stand 2 px off the panel, and the edge's own pixel between is white, the end of the gridline at the
  tick. A tick at an edge may now begin up to 0.3% of the image's width out, at least 3 px
  (`EDGE_TICK_GAP_FRACTION`), and its length is from the edge.
- With the ticks found, the labels were taken from inside the panel: the gridlines' rims, between the panel's
  color and their own, are ink, and made a "label" beside every tick, five to the page's four. An edge has its
  labels on its ticks' side only.
- The x axis's first tick sits at the panel's corner, 2 px left of it. Seen in part, from 2 px beyond the edge's
  end, it was placed 2 px off, and from the y edge it was a fifth y tick. Ticks are now looked for as far beyond
  an edge's ends as they may stand off it and are wide: the tick is seen whole, at 139.2 where the scale's
  spacing puts it, and from the y edge it is wider than long.
On bench-r's five gray plots this moved five tick positions by 0.1 or 0.2 px and nothing else.

Results. gg-gray-hi and the 7 in plot: one family each, 40 dots of 40. extractAll on the other gray plots: 40 of 40
as before. The label benchmark is unchanged (bench-r 228 of 245 labels, 47 of 52 axes, 0 wrong; the other sets
identical). detectAxes gives the same result as before this entry on every image but those five. The dot
benchmark's 144 rows are identical to RESULTS.md's apart from times, with no failure: no family of its images is
left out (RESULTS.md was not regenerated).

Tests (2 new in test/extract.test.js, 1 in test/axes.test.js): a drawn graph with three dots of another color under
its axis, left out with the bounds, kept without them and with bounds that would leave nothing; gg-gray-hi through
extractAll; and a drawn panel with its ticks apart, a label missing on the page, rims inside and a tick at the
corner. Each rule removed fails a test (10 mutations: the filter, its two thresholds, the empty case, the bounds in
extractAll; the gap, the length from the edge, the other side, the reach beyond the ends, the second pixel).

Not done, and known:
- The translucent rules still accept text and ticks as two levels where both lie inside the plot's bounds or the
  axes bound nothing (labels above or right of a plot whose axes are at its left and bottom are inside).
- bench-r is in the label benchmark only.

**bench-r in the dot benchmark** (2026-11-21)

bench-r's 26 plots are now a set of the dot benchmark too (`npm run bench -- --set bench-r`; `CSV_SETS` in
tools/bench.mjs, and a table in RESULTS.md). Each plot has 40 opaque dots whose values R wrote beside it; the
benchmark places them through the detected ticks, as it does a figure's published values.

Result: every plot gives 40 found of 40, precision and recall 1.000, no dot flagged, RMSE 0.02 to 0.16 px (the
largest on ggplot2's gray theme at 72 and 110 dpi, 0.15 and 0.16; 0.04 at 300 dpi), at radii from 3.1 px (R's
default device) to 14.1. These are easy plots for the fit, sparse and opaque: the set is there for the axes and
the labels, and now says that what follows from them, the plot's bounds, the mask and the families, holds too.
log-e-turned, whose turned y labels are not read, and gg-gray-small, whose digits are too small, are placed like
the rest: the truth goes through the ticks' positions and the header's values, not the labels as read.

RESULTS.md has the new table, with a note per plot: 170 rows from 144, no failures, the other sets' rows as they
were apart from times. It was built from the benchmark run set by set and merged, because a truth file being
written in testgraphs (song-cents.csv, of 2026-11-21, with pixel positions and a y tick header that doesn't parse)
stops a whole run; that file is left as it is, and has no row. testgraphs' ten images were fitted one at a time
for the same reason, so their times in this table are each image's own, shorter than the header's "10 at a time"
says. `npm run results` writes the same table in one run once that file reads.

**song-cents, looked into** (2026-11-22)

A local test image (testgraphs/song-cents.png, 2360 × 1010, a screenshot of an SVG chart): 325 opaque dots of
radius 11.8 px in six fills with white strokes (mid blue 92, dark blue 65, orange 74, light blue 30, navy 18, and 46
hollow rings in gray), under a solid and a dotted black line; light horizontal gridlines with gray labels at their
left ends and no tick marks, year labels under the bottom gridline, and a dashed orange reference line at 7 with
its own label. Its CSV holds the dots' SVG coordinates and fills.

What was found, in the order asked:

1. **"Find dots failed: undefined".** Not the image: the fit of all six families runs (44 s in headless Chrome,
   359 dots). The message is what the app said when the fit's worker script could not be fetched. A worker that
   fails to load gives a bare event with no message; the scan worker is kept between requests, so a page whose
   server has stopped still scans and measures picked colors, and the fit's worker, started anew for every Find
   dots, then fails. Reproduced by pointing the fit's worker at a missing file. Whether that is what happened is
   not known (the local server on port 8003 was restarted that day, and source files were being changed under
   it). Now the app says what a silent worker failure means (main.js `workerFailure`),
   for the scan too.
2. **Truth.** The CSV's SVG coordinates map to image px by one scale and a shift: px = 2.35596 × svg − 2.98 in x,
   2.35599 × svg − 4.01 in y, found by voting over scales (every truth and found dot of one fill votes for a shift)
   and refined by least squares on 270 matched dots, rms 0.05 px. testgraphs/truth/song-cents.json has the 325 dots
   in px by fill, so the image is scored as hand truth is. The CSV itself is no data-unit truth; the benchmark
   reports it as an error row.
3. **Families missed by the scan: two of six.**
   - Light blue (144, 180, 226) lies 0.69 of the way from the white page to the mid blue (93, 146, 205), where a
     rim of the mid blue lies too, and was dismissed as that blend (opaque.js `isBlend`). **Fixed**: a blend
     cluster with 0.3 or more of its pixels interior (`BLEND_DOT_MIN_INTERIOR`; this one 0.53, rims 0.00) is read
     as any cluster is and is a family if it reads as dots; a band in a tint, which has insides too, does not.
     Not a cluster within a found family's own color or its outline's: the first version made a second family of
     the rim between a fill and its outline, whose cluster takes in fill pixels (interior 0.73), which the test of
     outlined dots caught.
   - Navy (15, 39, 66) is, to 0.3 units, the labels' gray (88) plus 0.45 of the mid blue's offset from the page,
     and `throughLayer` takes it for the mid blue seen through a gray layer. A layer should be an area (solid at a
     few px), not text; with that the navy is a family. **Not applied here, done in the next entry**: once the navy is a family, the hollow
     rings are lost, because the ring search leaves out every direction of darkening within 18° of a found
     family's, and the navy's is 5° from the rings' gray. The two changes belong together: the ring search should
     leave out ink that is a family's own (its fill's rim, its outline, its stroke) rather than a whole direction.
4. **Axes: none usable.** x is not found; y has 5 of its 7 gridlines, with scraps for labels. Three things this
   layout needs: gridlines found under the dots that break them (20 and 10 have no run of 30% of the width);
   labels beyond a gridline's end and centered on it (the gridline rule wants them above or below the line, as the
   economist chart has them); and an axis from labels alone under the bottom gridline, which the rule for label
   axes takes only at a dark axis line. The dashed orange line at 7 is colored, so it is no gridline, and its label
   stands 0.4 of a spacing from the 5 line, outside the third that a gridline's label may be off it. Not done.

Scores against the new truth (dots matched within half a radius): with the fix, 277 found of 325, precision 0.978,
recall 0.834, RMSE 0.05 px (before it, without the light blue's dots: about 250 found). The 6 extras are zeros of
the y labels read as hollow rings, which the plot's bounds would mask if the axes were found. Of the 54 missed, 18
are the navy; the other 36 each have another dot's center within a diameter, opaque dots overlapped by others.
With the navy picked by hand its fit gives 83 dots for 18: 65 along the black lines, whose color is within its
reach.

The dot benchmark with the fix: the 170 rows of the other images are identical apart from times; RESULTS.md is
regenerated from that one full run (171 rows and the CSV's error row), so the times are again all of 10 at a time.

Tests (1 new in test/opaque.test.js): drawn dots in a tint of another family's color are a second family, and a
band in that tint under the dots is not; without the rule, or with it taking any cluster with insides, the test
fails, and without the exception for a family's own colors the test of outlined dots does.

**song-cents: the navy family, and a ring search that leaves out a family's ink, not its direction** (2026-11-22)

The two changes the last entry left together:

- **A layer is an area** (opaque.js `throughLayer`, `LAYER_MIN_HALF_WIDTH` 4 px by scan.js `solidRegion`). "A found
  family's dots seen through a translucent layer of color P" now asks that P be solid at 4 px, as a band is. The
  labels' gray, 88, strokes a few px wide, is no layer, so the navy dots (that gray plus 0.45 of the mid blue's
  offset, to 0.3 units) are a family.
- **The ring search leaves out a family's own ink** (outline.js `findHollowRings`, which now takes the found
  families' colors, not their directions). It skipped every direction of darkening within 18° of a found family's:
  hair1's black outlines and step1's blue rims are those families' own, not rings. The navy's direction is 9° from
  the gray of the hollow rings, so finding the navy lost all 46. Now a direction within 18° of a family's is that
  family's only when at least half its thin ink (`HOLLOW_OWN_SHARE`) lies within 3 px (`HOLLOW_OWN_REACH`) of
  pixels of the family's color: the rim of its fill, its outline, its stroke. The rings are nowhere near the navy
  dots, and are found.

Results. song-cents: all six families from the scan, 303 found of 325, precision 0.954, recall 0.889 (277, 0.978,
0.834 before). The navy family gives 26 dots for 18: all 18, and 8 on the black lines,
every one flagged hidden (picked by hand as a lone color it gave 83; found by the scan it is measured against the
other families' inks). The extras are those 8 and the 6 label zeros read as rings; the 36 missed are overlapped
dots, as before. The dot benchmark: the other 170 rows identical apart from times; the summed row times went from
1,416 s to 1,486 s in one run each: song-cents, which now fits one more family, is 18 s of the 70, and the
dilation per family color is new work on every image. RESULTS.md is regenerated from that run.

Tests: mid blue and navy dots among strokes of the gray are two families (test/opaque.test.js); gray rings are
found beside navy dots 9° off their direction, rings in a family's own color are not rings, and pale hairlines
within reach of every ring pixel, darkening the page another way, don't hide them (test/outline.test.js). Each
condition removed fails one (5 mutations: the layer's area, the share at 0 and at 2, the reach, the direction).

Not done, and known:
- The 8 navy dots on the black lines: the solid and dotted lines are near enough the navy's color where they cross
  other dots. They are flagged, not removed.
- The axes of this layout (last entry, item 4), and with them the label zeros read as rings.
- Rings of one family drawn mostly within 3 px of another family's dots of nearly the same direction of darkening
  would be left out as that family's ink. No image has that.

**song-cents and grader1: axes that are gridlines with their labels off their ends** (2026-11-22)

The layout of song-cents and of ggplot2's theme_minimal (bench-plos grader1): no axis lines and no tick marks; light
gridlines under the dots; each gridline's label off its start, level with it, where an axis's label would be; and on
song-cents the years in a row under the zero line, with no vertical gridlines. Before, song-cents had no x axis and
5 of its 7 gridlines with scraps for labels, and grader1's gridlines had their labels only by accident. Five rules
in axes.js:

- **A light line in pieces** (`brokenLine`). A line was the longest run of ink in its row, 30% of the width; the 20
  and 10 lines of song-cents have no such run under their dots. A row is now a line too when its pieces of light
  ink, each at least 2% of the width (`BROKEN_MIN_SEGMENT`), add up to 30% of it and fill half the span from the
  first to the last (`BROKEN_MIN_COVER`). A piece is flat: through 80% of its pixels (`BROKEN_MIN_FLAT`) the light
  ink across the line starts and ends at the same rows, to a pixel, and is no thicker than a line. The pieces of one
  line share those edges; shorter flat pieces with them (8 px, `BROKEN_MIN_END`) extend the line to its ends. Rows
  within 2 px of a dark line's run (`DARK_EDGE_REACH`) are its antialiased edge and are left alone.
- **A label off a gridline's end** (`gridLabels`). A gridline's label was a word above or below the line near
  either end (economist). A word wholly beyond the line's end, with at least 2 px of itself on either side of the
  line, is one too, as near as its middle is to the line. Ink joined to the line (a ring on it) is left out of the
  search.
- **The line through the gridlines' ends** (`gridEndLine`). When at least `AXIS_MIN_LABELED_TICKS` gridlines have
  labels and every label lies off the same end, the ends stand in for the axis line (thickness 0, `edge`), as a
  panel's edge does.
- **Labels under the last gridline** (`labelsUnder`). With such a y axis and no x axis, the row of words nearest
  below the last gridline, their middles within its extent, gives the x positions (`source: 'label'`), as
  `labelAxis` takes them at an axis line.
- **The plot's bounds** (`plotBounds`) count labeled gridlines when the axis has that end line, so ink outside is
  not a dot family (the label zeros, read as hollow rings before).

What the first version broke, each fixed and pinned by a test: a dark line's light edges became lines of their own
(the dark-run exclusion); rows through packed light gray dots became gridlines (nicest language: the flat test, in
place of a fixed clearance that also split thick gridlines); a glyph's edge in a line's row took grader1's label
(pieces must share the line's edges); a ring on a gridline merged with it into one blob as tall as the search
(ink on the line is left out); a gridline's stub before a dot was its label (short end pieces); rows of
beeswarm-boxplot's light band between its dots (the thickness of the ink across); in the synthetic beeswarm
preset, two junk "gridlines" 2 px apart took slivers of a dot for labels and the plot ended there, recall 1.000 to
0.986 (a label reaches both sides of its line); and in an opaque test's scene, 11 junk "gridlines" along a slanted
light curve got an x axis from the dots under the last, which masked every dot (labels are looked for under a
gridline axis only when it has its end line, that is, labels of its own).

Results.
- Axes of all 145 local images, compared before and after: they change for song-cents (y: 7 gridlines from 5, all
  labeled, 30 to 0; x: 7 labels, 1960 to 2020; the plot bounded at left 143 and bottom 905.4), grader1 (both axes:
  every gridline labeled, 0, 15, 30, 45, bounds left 96 and bottom 1154) and the economist chart's y labels (one
  box 5 px wider). The dashed orange line at 7 on song-cents is no gridline, as intended.
  27 images' line lists change (296 lines to 324): gridlines that had a partial run now span their pieces, and a
  few new light lines, none of which changes an axis.
- Label benchmark: testgraphs 26 of 26 labels and 3 of 4 axes (19 of 19, 2 of 3: song-cents's x axis is new and
  solved; its y axis is not paired with the CSV's header, which lists 8 ticks with the reference line's 7);
  bench-plos 37 of 39 labels and 7 of 8 axes (29 of 37, 5 of 8: grader1's two axes); all sets 1,357 of 1,472 labels,
  212 of 234 axes, 0 wrong (1,342 of 1,463, 209 of 233).
- Dot benchmark: song-cents 297 found of 325, precision 0.973, recall 0.889 (303, 0.954, 0.889): the 6 label zeros
  are gone, and the 8 extras left are the navy dots on the black lines. The other 170 rows are identical apart from
  times. In the app song-cents shows both axes read and 297 dots, no console errors.
- Time: `detectAxes` over the 145 images 5.3 s to 6.4 s (about 8 ms more per image: every row and column with 30%
  light ink is walked for pieces).

Tests (test/axes.test.js, 6 new; grader1's values in test/labels.test.js): a synthetic chart of this layout with
dots over the gridlines, a ring on one, a stub, a title and a colored reference line; gridlines without labels, or
with ink beside their ends, or with labels along them, have no end line and no labels under them; rows through
light gray dots; a striped ellipse, a slanted band, a thick bar and two short rules are no lines, and a dark line
under dots keeps its run; grader1 and song-cents themselves. 13 mutations, one per condition, each fail a test, and
the thickness and same-end conditions fail a synthetic one without the local images.

Not done, and known:
- The y axis of song-cents in the label benchmark (the CSV's header lists the 7), and the CSV's "data" row, which
  is in SVG coordinates.
- The reference line's label, 7, is not a tick: correct for the scale, and the app does not show it.
- An axis with a single labeled gridline has no end line (`AXIS_MIN_LABELED_TICKS`, 2), and so no plot bound.

**song-cents: the CSV truth in data units** (2026-11-22)

testgraphs/song-cents.csv (local) held the chart's SVG coordinates, so the dot benchmark gave an error row for it
and only the by-hand JSON scored the image. It is now a truth file like the others: x and cents in data units, the
fill as the group, linear scales, and the table's year and the SVG coordinates kept as extra columns.

- The user's export of the page's table (testgraphs/song-cents-source.csv, 325 rows of year and cents) pairs with
  the 325 SVG circles one to one: svgy = 386.002 − 12.067 × cents to the 0.1 the SVG is written in (residual at most
  0.05 svg units), so y is the table's cents.
- x is not the year. The page dodges the dots of one year sideways into lanes 0.189 years apart (2.60 svg units,
  6.1 px; the dots' radius is 11.8 px): svgx = −26881.231 + 13.754 × (year + lane × 0.189), lanes −2 to 2, residual
  at most 0.05 svg units. It is a fixed dodge, not random jitter: in all 66 years the lanes, left to right, are the
  year's rows in the table's order (62 years have 5 rows, 3 have 4, 1 has 3). The truth's x is where each dot is
  drawn, year + lane × 0.189, so it is scored by position ("exact"), not by cell.

Results. Dot benchmark, the data row, placed through the detected ticks: 297 found of 325, precision 0.973, recall
0.889, the same dots as the by-hand row, RMSE 0.49 px against its 0.04. The difference is the x axis: its ticks are
the middles of the year labels, which lie 0.08 to 0.76 px left of the years' positions by the SVG map (the y
gridlines are within 0.12 px). That is 0.02 years at most, and it is what an axis of labels alone can give.
Label benchmark: both axes of song-cents solved, 7 of 7 labels each (testgraphs 33 of 33 labels, 4 of 5 axes; the y
axis was not paired before, when the header listed the reference line's 7). RESULTS.md has the new row, its time
from a run of this image alone.

**song-cents: the white stroke around the dots** (2026-11-22, open question 10)

The dots of song-cents have a white stroke, 1 SVG unit, 2.36 px. On a dot alone it shows nowhere (white on a white
page). Where a dot lies over another it cuts an arc of the page's color through the one below, often through its
center. The opaque model has no stroke, so a dot placed where the cut dot is pays for every pixel of that arc, and
the fit leaves it out. Looked into first: the 36 missed dots are not hidden (by a count that includes same-color
neighbors, about two thirds of each one's fill shows), and with the thin white gaps between dot inks ignored by the
fit, in a throwaway script, 27 of them were found.

What is built (src/core/stroke.js, `pageStroke`; layers.js `buildMask` option `strokes`; extract.js `maskOptions`
and `extractAll`; the app's mask options and a note in the family's panel):

- **Reading the stroke, before the fit.** Each opaque family's ink (coverage 0.5 or more) falls into pieces. A piece
  with a disc's area (0.85 to 1.12 of π r²) and a disc's box is a whole dot: one on top, bounded all round by its own
  stroke. A smaller piece is a cut dot, unless it is a speck (under 0.08 of a dot) or has no pixel 2 px inside it (a
  dash or a line in the family's color). Outward from each whole dot's edge, a direction for each px of its circle,
  where a cut piece's ink lies within reach (half a radius, 4 to 8 px) with the page's color between, the width of
  the page is measured, the edges' pixels counted by their coverage. The dots have a stroke when there are 40 such
  gaps or more (about an arc and a half) and 0.75 of them are within 0.75 px of their median, which is the stroke's
  width.
- **Why whole dots and cut pieces.** Dots packed side by side (a beeswarm) have a ring of the page round each of
  them, about one width for small dots, and no stroke: whole dots beside whole dots tell nothing. A piece cut by
  the whole dot's own stroke lies at one distance from it all along; without a stroke a dot is cut only by ink that
  touches it, and a cut piece beside a whole dot merely lies near it, at any distance, the gap widening along it.
- **What the fit does with it.** The pixels that aren't dot ink, in gaps no wider than the stroke plus 1.75 px
  between dot inks with the page's color in them (across in a row, a column or a diagonal), are masked, with no
  margin. Any family's ink bounds a gap, a sliver's too: what shows of a dot under a stroke may be no more. Other ink
  between two dots (a line) is no gap.
- **The page** is the image's most common background, not a family's own ground: hair1's pink fills lie on their
  black outlines, 3.5 px wide and one width all round, and were "stroked" for a while.
- Only opaque families are read: a translucent family's overlaps show in its levels, and rings are fitted in their
  drawing order. The hollow rings of song-cents are untouched.

Tried and dropped: runs of directions round a whole dot told as an arc (one width) or a near miss (growing by
r θ²): too few runs go far enough round to tell (song-cents 11 to 15 arcs of 164 whole dots, and 5 "near misses"),
and small packed dots pass. Gaps by the first ink pixel instead of by coverage: ±0.5 px of noise, a peak 1.5 px
wide. Joining what a stroke parts in the mask's opening for thin marks: 3 more false navy dots, nothing gained.
Limits on mixed pixels along a radius and on gaps narrower than the stroke: no image or test needed them.

Results.
- The 145 local images: one stroke, song-cents (760 gaps, 0.83 of one width, 2.37 px against the SVG's 2.36). The
  two other images with 40 gaps or more are a beeswarm over a box (bench3 dots-swarm-box: 400, 0.43) and cantons
  (59, 0.53); nicest language has 20 (0.75). comment1, whose near-duplicates an earlier entry put down to a thin white
  outline, has none that shows: its overlapping dots are one piece of ink, with no page between.
- song-cents: 323 found of 325, precision 0.978, recall 0.972 (297, 0.973, 0.889): 27 of the 36 missed dots are
  found, and the dots that aren't there go from 8 to 7, all navy on the black lines and flagged. Position RMSE
  0.09 px against the by-hand truth (0.04): two of the dots found under others are 1.2 and 0.7 px off; the median
  error is 0.025 px, as before. Still missed, 9: 5 hollow rings and 4 dots with little showing.
- The other 170 rows of the dot benchmark are identical apart from times. `pageStroke` takes 20 ms an image (2.9 s
  over the 145; song-cents 0.19 s).
- In the app song-cents gives 323 dots, and the family's notes say the dots have a stroke and how wide.
- Times. The first full run with the stroke read had row times 3% up at the median; two later runs, with the
  machine busy (load average 6 to 8), 14% and 16% up on every row. One image alone takes the same with the code
  and without (cantons at one job: 5.41 s twice, against 5.42 and 5.49 s). RESULTS.md keeps the other 170 rows of
  the run before, whose numbers are the same, and takes song-cents's two rows from the last run (51 and 52 s, on
  the busy machine).
- The reach of the mask, measured on song-cents: at the stroke plus 1.25, 1.75 and 2.75 px the fit finds 316, 316
  and 315 of 325 with position RMSE 0.13, 0.09 and 0.24 px.

Tests (test/stroke.test.js, 6): synthetic piles of stroked dots in two colors (width 2.5 px read within 0.25; 58
of 60 dots found, 29 without the mask); no stroke for the same piles unstroked, a packed swarm, dark outlines, one
stroked pair, near misses beside dots cut by bars, and dashes in the dots' color; the mask beside a sliver and not
on a line between dots; a part of song-cents against its truth (66 of 72, 57 without). 14 mutations, one per
condition, each fail a test.

Not done, and known:
- The scan did not find a family in two synthetic scenes made for these tests: orange dots with none alone, in
  stroked piles (found when the same piles had no strokes), and orange dots nearly all cut by blue ones, without
  strokes. song-cents has whole dots of every fill. Looked into in the next entry.
- Translucent dots with a white edge (score1's teal, α 0.72: 38 gaps, 0.95 of one width when read as if opaque;
  wages2025's rims) are not read, and their fits are unchanged.
- The stroke is not drawn by the model: the Model and Residual views show the dots without it, the arcs masked.
- Fewer than 40 gaps (one or two overlapping pairs in an image) are no stroke, and those cut dots stay missed.

**The scan and a family with no dot alone: two causes** (2026-11-22)

The two synthetic scenes of the last entry, in which the scan found the blue family and not the orange. They fail
for different reasons, and neither is about strokes as such.

1. **Stroked piles, every dot alone blue.** The orange has whole dots (those on top of their piles) and none
   isolated: each has blue ink within the margin an isolated dot must have clear. Its radius then comes from the
   depths of its pieces, and most pieces are crescents under the strokes of the dots over them: 4.97 px for 10. The
   scan of larger radii by Hough runs to 1.6 times that, 7.95, and finds no center. The scan around the radii of the
   pieces themselves (componentRadii: 10.5, the whole dots) was left out, because a piece's scan was skipped
   whenever it began within the first one's range, and 0.75 × 10.5 = 7.88. Fixed in `dotCenters`: around each
   piece's radius, the radii the first scan did not reach are tried. Orange: r 10.96 from the scan, 10.67 by the
   fit, 29 of its 30 dots found.
2. **A family drawn first, nearly every dot partly under another's** (no strokes: 14 orange dots, 12 of them a third
   under a blue one 10.8 px away, 2 alone). The radius is right (10.0, from the two alone). The dot test asks that
   the disc of 0.75 r at a center be the family's own fill to 0.85 (a round glyph is hollow), and a third of each
   such disc is blue: 2 centers, coverage 0.21, no family. Now `measureOpaque` also counts the centers of dots
   partly under other ink (`dotsUnder`, `coverageUnder`): half the edge seen as before, what the other ink doesn't
   hold of the disc filled to 0.85, and half the whole disc the family's own (`DOT_MIN_SEEN`). They count in one
   place, `siblingFills`: a color that failed the dot test, at a found family's radius. Orange: 12 such centers,
   coverage 1.00 with them, a family; 14 of 14 found, and the 27 blue.

Tried first for (2): letting such centers pass the dot test everywhere. It moved bench1/ellipse (100 dots found
either way; 2 flagged → 15): the dots' color over the page and over the ellipse are two opaque candidates of 56
centers each in the first scan, the one listed first decides which background is the region under the dots, and
with 16 more centers for the dots that straddle the ellipse's edge the order turned over (the region the ellipse,
1 level measured for 2, 15 dots flagged deep). Hence only at a found family's radius, where nothing found before
changes. That tie on ellipse is fragile in itself; not touched.

Also seen: the scenes as test/stroke.test.js draws them (4 × 4 samples a pixel) have edge pixels at exactly half
coverage, and the measure's ink is coverage over 0.5: measuring the orange at 223, 105, 59 and at the cluster's
mean, 0.4 units off, gave radii 11.26 and 4.97. An artifact of that drawing, not of real renderings; the new tests
draw with render.js.

Results: the three scenes find both families and their dots (stroked piles 29 of 30 orange and 27 of 30 blue; the
same piles without strokes 30 and 27; cut orange 14 of 14 and 27 of 27). The dot benchmark: all 172 rows identical
apart from times, so RESULTS.md stands. In the app song-cents gives 323 dots as before.

Tests (test/opaque.test.js, 2 new): the cut family (its counts with and without the dots under other ink, both
families' dots, and no family from dots with under half their disc showing, nor more centers from dots with a
bite of the page out of them, with or without a line of other ink through the rest); the stroked piles (the
family, its radius within 1.2 px before the fit and 0.8 after, at least 28 of 30 dots). 7 mutations, one per condition,
each fail one.

Not done, and known:
- The radius of a family with no isolated dot comes out high by up to a pixel from the scan (10.96 for 10), and
  the fit's refit leaves 0.3 to 0.7.
- A family with fewer than three dots at least half in view, or one at no found family's radius, is still none.

**The view: dot marks on dots of their own color** (2026-11-22)

A found dot's mark (circle and center cross, 1.5 CSS px, blue) is hard to see on a blue dot, and a flagged one's red
on an orange dot: song-cents has both. Two ways tried, on one zoomed view of it (304%):

- **A lighter shade of the mark's color where the dot's is near it** (the user's first choice). The color under
  each mark read from the image as shown (the mean in the dot's middle), the mark's own color kept when it stood
  out by lightness (25 units of CIELAB L*) or by color (ΔE 60), else a lighter shade, or a darker one over a dot too
  light for that. The cross showed. The circle did not: it lies half on the page, and a pale blue line on white is
  as hard to see as a blue one on blue. And the rule gave neighbors different shades: song-cents's mid blue (L* 59)
  is too light for the lighter shade, so its dots got navy marks, except those with a black line across their
  middle, which got pale ones. ΔE alone is no guide here: the blue mark is 43 and 41 from the two blues, a
  difference a thin line doesn't carry. Dropped.
- **A white border round every mark** (the second). Each mark is stroked first in white, 1 CSS px wider on either
  side (`MARK_CASING`), then in its color; every border before any mark, so a mark's border doesn't break the
  marks it crosses. On the page the white doesn't show; on a dot it parts mark and fill. The marks keep the
  legend's colors, and nothing is read from the image. Kept, for every mark and the ring around near-duplicates.

Checked in the app: song-cents zoomed in and at the fitted view, dog-ages (95 translucent dots, many overlapping)
at the fitted view; no console errors. No test: view.js draws on a canvas.

**Fifteen more PLOS ONE figures** (2026-11-22)

The user added fifteen figures to `bench-plos/` (fourteen with hand truth, saved from the app after correcting its
fit; funnel has none), with notes on each and three general ones. Credits are in `bench-plos/README.md` (Crossref;
funnel is CC0, the rest CC BY). First run, before any change:

| Image | Dots | Found | P | R | What goes wrong |
|---|---|---|---|---|---|
| alloys | 25 | 25 | 1.000 | 1.000 | the top y label isn't found (the scale gives its value) |
| biomarker | 83 | 112 | 0.348 | 0.470 | zero line read as the x axis, so half the plot masked; radius 4.5 for 6.0 |
| cattle | 325 | 300 | 1.000 | 0.923 | the red family (25 dots) not found |
| corneal | 121 | – | – | – | no family: hollow rings and black dots |
| crimson | 180 | 189 | 0.952 | 1.000 | x label 10² read as 70; y labels, clipped, not read |
| expression | 46 | 46 | 0.978 | 0.978 | y labels with no ticks, off the gridlines: no y axis |
| funnel | – | – | – | – | rotated y labels not read (dots right, by eye) |
| gait | 17 | 17 | 1.000 | 1.000 | outline not noticed |
| lung-tumor | 473 | – | – | – | no family with exact colors (found with noisy matching); border lines as gridlines |
| mpo | 45 | – | – | – | opaque dots over gray boxes read as translucent, no radius; a bracket as the x axis |
| richness | 48 | 170 | 0.282 | 1.000 | 122 dots in the confidence band of the dots' color |
| stiffness | 305 | 39 | 0.846 | 0.108 | outlines masked; no axes |
| synthetic | 16 | 13 | 0.923 | 0.750 | two families of 2 dots each not found |
| tsne | 257 | 252 | 0.972 | 0.953 | crescents under white edges (translucent families) |
| viability | 236 | 252 | 0.937 | 1.000 | the tick labels' zeros as a family of rings; no axes (labels only) |

Done in this round: the three general notes and biomarker. The rest are open, each its own piece of work.

*The general notes.*

- **A category axis's ticks are numbered 1, 2, …** (`axes-panel.js defaultEntry`): they were placeholders in empty
  boxes. They are the categories' names until typed over, and values at once if the axis's type is set to linear,
  which gives each dot a position between the categories (a beeswarm's jitter) instead of the nearest one's name.
  A remembered entry of unnamed categories is numbered when loaded.
- **A family is described by the α in effect.** The mask and the fit already go by it (opaque from 1), but the
  list and the Fit line went by how the family was measured: an opaque one with α typed to 0.5 still said opaque.
  Now "translucent, α 0.50 (typed)", or "opaque (α typed)" for a translucent one set to 1, and the Fit line says
  how it was measured and how it is fitted. The estimate isn't measured again: an opaque family fitted as translucent
  has no overlap levels, a translucent one fitted as opaque keeps its own reading of the image. Measuring again as
  the other kind is what mpo needs, and is not done.
- **Ticks can be added and removed by hand.** Each axis has an Add tick button: then a click in the image puts a
  tick at the click's x (or y), a line under the pointer showing where; Esc ends it. Each tick's row has a button to
  remove it. An axis that wasn't found gets the same button, so two clicks and two typed values give a scale
  (stiffness, viability, expression's y). Once the ticks are changed the entry lists every tick's position
  (`ticks`), remembered with the values; detected ticks keep their label crops and read values, placed ones are
  marked "placed" and drawn as dashed lines across the image, and "Detected ticks" brings the detected set back.
  The changed ticks are the scale's only: the mask and the plot's bounds stay the detected axis's. A placed tick
  is where it was clicked, to 0.1 px, not snapped to a tick mark or gridline: zoom in to place it.

*Biomarker* (83 dots: 112 found, P 0.348, R 0.470 → 92 found, P 0.902, R 1.000). Three causes, in order:

1. **The zero line was the x axis.** The dots standing on it are strokes off the line, as ticks are (10 of them),
   and the dots above those were their labels; the labels' side, above the line, was outside the plot, and every dot
   there was masked. Now a line through the middle of the plot (the other axis has labeled ticks on both sides of it)
   is a reference line when another line of its direction lies at an end of the other axis (`axes.js detectAxes`,
   `reference`); axes that cross in the middle with nothing at the plot's edges stay axes. With the zero line out,
   the frame's bottom is the x axis by its two labels, once a frame's side counts as dark where it is as dark as the
   other axis (the gray frame's sides have mean luminance 175 and 181, either side of the threshold of 180):
   `FRAME_LUMINANCE_TOL`. Of the 160 local images, only biomarker's axes changed. Recall 0.470 → 0.976, 216 found.
2. **The radius was 4.5 for 6.0**, so the fit put two or three dots in each. The dots are blurred, with edges about
   4 px wide. No dot counted as whole, for two reasons. Beyond 2 px from the dots' own color (`NEAR_FAMILY`) a soft
   edge passes through the frame's gray, which is other ink: every dot had a ring of pixels read as the frame's,
   and the pixels nearer in were read against that ring as their background. And a blurred disc fails the test of a
   whole dot's second moments (1.07 against 1.03): blur spreads the moments and not the area. So the radius came
   from the distance transform.
3. Now a plain opaque family with no whole dot is **read a second time as blurred dots** (`opaque.js softGround`):
   the near zone goes on outward along each dot's own falling slope (other ink is flat along itself), the family's
   background is read beyond the edges, and a blurred disc counts as whole when its pixels at half coverage or more
   are a disc of the same area (`shape.js isolatedDots`, `blurred`). The second reading is taken only if it shows
   whole dots. Radius 5.98, five whole dots, 92 found; the 9 extras are second dots in the touching rows (at radius
   6.6 with blur 1.6, typed, 83 are found for 83, so the measured blur, 1.1, is low).

Tried first and dropped, each for what it moved elsewhere (full benchmark, 183 rows):

- Other ink in a blend of the dots' color and the page counted only where it is flat: phase1's gray fills are such a
  blend and their edges aren't flat (19 black dots for 16), and boxdots1 and song-cents moved.
- The near zone grown along soft edges for every family: violin-beeswarm 722 found for 712 (702 there), cattle 298
  for 300, tsne 250 for 252.
- Blurred discs counted as whole dots everywhere: round specks of text passed and outvoted the dots (adults1: no
  family found), and fibers-distance, expression and boxdots1 moved.

As it is, the full benchmark changes in biomarker's row alone (182 rows the same). 307 tests pass, with regression
tests for the zero line (and an x axis across the middle of the y axis, which stays), a blurred dot as a whole
one (and one in a faint halo, which isn't), and blurred black dots in a gray frame; each fails with its fix taken
out. Limit: at a blur of 2 px on dots of 6 the same synthetic scene finds no family at all.

*Open, from the notes, in the order I would take them:*

1. **stiffness** (39 of 305): outlines masked. The largest miss, and gait's unnoticed outline and corneal's hollow
   rings are the same area (outlined families).
2. **richness** (122 extra) and **mpo**: a filled shape in the dots' own color under them (a band, boxes).
3. **corneal**, **lung-tumor**, **cattle**'s red, **synthetic**'s pairs: families the scan misses.
4. **Axes and labels**: stiffness and viability (none found), lung-tumor (border lines as gridlines), mpo (a
   bracket as an axis), crimson (10² as 70, clipped labels), funnel (rotated labels), alloys (a label not found),
   expression (labels with no ticks).
5. **tsne**: the white edge around translucent dots (the stroke reading is for opaque families).
6. **viability**: tick labels' zeros as rings.

**stiffness: outlines in the mask** (2026-11-22)

305 blue dots in thin dark outlines, a dense cloud with a regression line: 39 found, P 0.846, R 0.108. The family
was found and set up to be fitted by its rings, but every ring was in the mask (the user's observation): only the
fills' middles were left, and a ring fit has nothing to go by there. Two causes, both in the ring reading
(`opaque.js familyRings`), which reads each pixel as a mix of page, fill and outline and leaves it unknown otherwise:

1. **The outline's color was not the outline's.** The stroke is a pixel or two wide and blurred, so no pixel shows
   it alone; the scan takes its darkest pixels, 16, 57, 83: black mixed with a third of the fill (49, 179, 253).
   A ring's inner side lies on the line from the fill to that navy. Its outer side is black mixed with the page,
   which is gray (134, 134, 133 halfway), not the blue-gray a mix of navy and page would be: it reads as navy
   less some fill, a negative weight, off the triangle. Now the reading works out the color beyond the one seen
   (`outlineItself`): over the ring pixels that are part page and no fill, the share of fill they lack, at its
   ninth decile (0.32 here, which puts the color at black to within 4 units; the median, 0.24, because most outer
   pixels of a blurred ring have some fill in them too, left the ring model at rms 0.057 where black gives 0.019).
   Only from a share of 0.1 (with black given as the outline the same measure is 0.05). Too far is the safe
   side: any mix of page, fill and outline stays inside the triangle of a color beyond the outline's. With this
   alone, 242 found.
2. **The rings reach farther than 4 px from the fill** (`OUTLINE_REACH`, which keeps other ink in the outline's
   color, text and axes, from being read as rings): here 6 px from the fill's color to the page. The reach now goes
   on down a ring's falling tail to its faint end (`downhill`, the growth biomarker's soft edges use), but only
   where the reach's edge runs through the rings, so that the tails add as many pixels as that edge has, or more
   (stiffness 2.55 times; degradation1 0.16 and 0.38). Grown always, the edges of degradation1's gray error bars
   joined its gray rings and no ring was whole. With the faint end left out, the last pixels of each tail were
   unknown, the mask (dilated by a pixel) took the rims, and the rings' radius read 9.6 for 8.8.

Now 264 found, P 0.996, R 0.862, radius 8.8 (ring from 6.6 to 8.9 px, 12 whole rings); in the app the same, no
console errors. Of the 42 missed, 32 have another dot within 9 px, about a radius: they show a crescent of
outline at most, and an outline farther than the reach from any fill's color is still unknown. 10 lie on the
regression line, which is masked (4 are both). The full benchmark changes in stiffness's row alone (182 rows the
same), and the scan takes as long as before (39.8 s for 38.0). 309 tests pass: a ring reading on a synthetic scene
(a thin black outline drawn and blurred, given its darkest color), and the reading and ring model on the image
itself; each fails with the color correction or the tails taken out, and degradation1's test fails with the tails
grown always.

Found on the way, not fixed:

- **A crop of the image is no family.** Three crops (380 × 300 below the line with 11 dots alone; two with the line)
  scan as "No dot family found": the blue's plain measure has no dot (radius 10 to 12 from the distance transform),
  and nothing reads the outline. The whole image's blue is found only through its outline (its plain measure:
  27 dots, coverage 0.11). What the whole image has that a crop lacks isn't diagnosed. A synthetic scene drawn like
  it (discs with a black ring, blurred) is no family either, which is why the fit of such a scene isn't tested
  end to end.
- Stiffness's axes: none found, not looked at. Ticks can be placed by hand.
- The user's other outline notes (gait: the outline not noticed, and Find working badly with it set by hand;
  corneal) aren't tested against this change. A color picked by hand as the outline is a darkest pixel too, so the
  correction may help there.

**richness and mpo: a band and boxes in the dots' own colors** (2026-11-22)

*richness* (48 dots: 170 found, P 0.282 → 57 found, P 0.842, R 1.000). Translucent blue dots with outlines over a
regression line and its confidence band. The band is exactly one dot's color (216, 222, 245: the line's blue at the
dots' α), and the line is the dots' full color, so both read as the family's own levels and the fit filled them
with dots. Color can't tell them apart. Shape was considered and dropped: the band is narrower than a dot in its
middle (21 px on either side of the line, dots 26 px across), dots sit across its edges in its own color, and dots
on it cut it into pieces shorter than any run long enough to tell it from a dot.

The user's suggestion decides it: these dots have outlines and a band has none. After the fit, each dot of a
translucent outlined family is checked for its ring (`fit.js ringless`): the step in the residual from just inside
the ring to the rim, in 12 sectors round the dot, each as a share of the ring the model draws (1 as drawn, 0 none),
and the median over sectors. Whatever lies under the whole dot (the band) is in both terms and drops out; marks
crossing a rim (the line, another dot's ring) spoil a few sectors, which the median passes over. On richness's first
fit the 48 dots read 0.43 to 2.29 (median 1.02) and the 122 others −2.1 to 1.4 (median −0.19). Dots under 0.5 go, and
the rest are read again until none does. Measured there:

| Variant | Found | Of the 48 |
|---|---|---|
| as it is (threshold 0.5, median, ties spared) | 57 | 48 |
| threshold 0.35 | 108 | 48 |
| threshold 0.65 | 53 | 47 |
| mean over sectors instead of the median | 59 | 45 |
| dots almost at a dot whose ring shows judged too | 54 | 47 |

- Tried first: the ring's share by all rim pixels at once (the 122 others up to 3.8: the line under a rim is far
  darker than any ring), and the share of rim pixels that show the ring (a dot on the band has the band under its
  rim too, and none shows). The step from inside to rim is the one that doesn't see what is under the whole dot.
- A dot within 0.35 r of one whose ring shows stays: it may be a second dot under the same ring. bench2
  stroked-dots lost one of three dots within 4 px of each other without that (204 → 203, its ring 0.18).
- Not for rings narrower than two blurs (the step is predicted without the blur), nor dots placed by hand.
- Nothing is born after the check, so what the removed dots explained stays in the residual: the Residual view
  shows the band and the line.
- The fit still spends its time on the band (27 s, 275 candidates) before the check removes them.
- The 9 extras are each within 9 px of a real dot, 6 of them within 2.5 px and flagged as close pairs.

*mpo* (45 black dots over gray-filled box plots: "radius unknown" → 49 found, P 0.918, R 1.000). Three causes:

1. **Black was a page color for the axes.** `backgroundColors` takes any color over 5% of the image as a
   background (a panel); mpo's black (dots, lines 7 px thick, bold labels) is 5.6%, so the black axis lines were no
   ink: the y axis showed one tick and the top bracket, with two stars for labels, was the x axis, with the plot
   "above" it. Now, for the axes, a background differs from the page by less than 128 in luminance
   (`PANEL_MAX_CONTRAST`): a panel is a tint of the page. Both axes are found, the y axis read 1000 to 300. Of the
   160 local images only mpo's axes change. The scan's own list of backgrounds is as it was (31 images have a
   second one, many of them dot colors, and the fits were tuned with them).
2. **Black was never tried as a family.** The page and the boxes' grays, 160 and 127, are the levels 0 to 2 of a
   gray of 107 at α 0.64, a translucent "family" with no dot; a cluster on a translucent family's color line is
   taken for one of its levels, and black is on that line. But levels approach the family's full color and never
   pass it, and black is 185 units beyond that gray: the line now ends at the full color plus a quarter of its
   distance from the background (`LEVEL_LINE_BEYOND`). Black is then an opaque family, radius 16.7.
3. **The grays stayed a family**, with the black dots' radius as a sibling's. A translucent family with no dot
   alone whose first level is a solid area at the opaque dots' radius is what those dots lie on: dropped
   (`style.js findCandidates`), as `withoutSolidBands` does with the translucent families' own radii. The boxes
   are then regions under the dots.

The whiskers, box outlines, medians and brackets are thin marks in the dots' color and masked; the stars aren't
found as dots. The 4 extras are each within 6 px of a real dot, 3 of them flagged as close pairs.

Full benchmark: richness's and mpo's rows alone change (183 others the same, 184 rows, 2 failures: corneal and
lung-tumor). 314 tests pass. New: the band scene (synthetic, 36 dots over a band of one dot's color and a line),
richness and mpo themselves, bench2 stroked-dots' ties, and black axes under black marks; each fails with its fix
taken out. In the app: richness 57 dots, mpo 49, no console errors.

**Families the scan missed: cattle, corneal, lung-tumor** (2026-11-22)

*cattle* (325 dots: 300 found → 326, R 1.000). The red family, 25 dots, was never tried: a color cluster needed a
thousandth of the image (`CLUSTER_MIN_FRACTION`), 4339 px of cattle's 2174 × 1996, and red has 3074 at its color.
The plain measure of red is a family at once (24 centers, 18 whole dots, coverage 0.88). The floor now stops
growing at 1500 px (`CLUSTER_MAX_MIN_PIXELS`): a family's size goes with its dots, not with the picture. Full
benchmark: cattle's row alone, and the time in all is the same (2073 s and 2029 s over the rows; cattle itself 50 s
→ 59 s, with a third family to fit).

*corneal* (63 black dots and 58 hollow black rings: no family → 66 found, the 63 dots among them).

- The black dots were no family for want of strong arcs: a blurred dot's edge, 3 or 4 px wide, votes for its center
  from every radius across it, and none of the 62 centers had 0.75 of its edge at one radius (at radius 7.5, four).
  It has 25 whole isolated dots of radius 8.22 to 8.31, which the test of a whole dot (aspect, second moments) is
  stricter about than an arc: ten or more now stand in for the strong arcs (`DOT_MANY_ISOLATED`), with a tenth of
  the color's pixels in the dots as before (the labels are bold: coverage 0.28).
- The rings are not found, for two reasons. The hollow-ring search leaves a found family's color alone (its own
  edges read as rings). And with black allowed, the rings fit neither reading it has: as thin translucent rings,
  with ink over 0.8 masked as text, pieces of them are masked and the rest fits at rms 0.109 (limit 0.06); as
  opaque black rings, unmasked, at 0.142 (limit 0.1), their cores only 0.71 of black. They are dark gray rings 3 px
  thick with soft edges. Open: a third reading (unmasked, the ring's darkness fitted), and two families of one
  color, solid and hollow, in one fit.

*lung-tumor* (473 small dots: no family). Diagnosed, not fixed. The dots are 13 px across and blurred: a light blue
fill (173, 193, 214) about 5 px across inside a darker ring of its hue (132, 162, 194), fading to white. With
exact colors there is nothing flat to read levels from. With noisy matching a family is read at 80, 122, 169 and
α 0.92, whose first level is the ring's color: the fills are then half a level, no dot is whole, and there is no
radius (counting blurred discs as whole ones, as for biomarker, finds none either). It needs a reading of small
ringed translucent dots that doesn't start from flat pixels.

*synthetic*'s two families of 2 dots each are not looked at: a family needs three dot centers by every rule.

314 tests before these, 317 with: a cluster in a 4.4 Mpx image, `isDotFamily` with whole dots for arcs, and corneal's
black dots; each fails with its fix taken out.

**corneal: open circles beside filled dots of one color** (2026-11-22)

The 58 hollow rings (121 dots in all: 66 found → 128, R 1.000, P 0.945). They are open circles as GraphPad Prism
draws them beside filled ones: the same black, walls 5 px thick of a radius of 9.5, soft-edged. Two paths existed
for rings and neither takes them:

- **The search for hollow rings** (`outline.js findHollowRings`) leaves a found family's color alone, since the
  rims of its dots read as rings. With black let in, it finds 14 isolated "rings", ten of them glyphs of the bold
  labels, and reads the real ones badly: against the fill around each pixel (a closing 5 px long, for outlines
  1.5 to 3 px wide) a 5 px wall is uneven, darker at the top and bottom than on the diagonals; pieces at 0.8 of
  black or more are masked as text; rms 0.109 against a limit of 0.06, and as opaque black rings, unmasked, 0.142
  against 0.1. With a 9 px closing the walls read evenly, but then the filled dots' rims are the "rings" (every
  isolated one in the plot is a filled dot), and a ring with ink over half its box (0.61 for these) isn't taken as isolated.
- **Fills in the page's color inside an outline** (`pageFilledFamily`, tumour1's kind): it finds 53 centers, but
  reads the outline's color 2 px beyond the 3.9 px hole, where the soft wall is still gray (184), and wants an
  outline at most 0.4 of the fill's radius (here 1.0). With the color read farther out and darker and the width
  let through, the family is found (46 to 48 centers), and the fit then keeps 10: the holes, 8 px across, are thin
  marks in the "dots' color" (the page's) to the mask, and the walls are other ink.

So a reading of its own (`thickRings`), tried for a direction of darkening that is a plain opaque family's:

- each pixel is read against the page along that direction, not against the fill around it;
- the family's solid ink (what an opening by 0.45 of its dots' radius keeps: dots, not walls or strokes) is
  unknown, with 3 px around it for the dots' soft edges;
- isolated rings may have ink over up to 0.7 of their box (a disc: 0.79), and count only inside the plot, when
  the axes bound it (`opts.inside`, from the scan's bounds): glyphs of corneal's x labels are rings of 13 to 19 px
  to it, and took the median radius from the three real isolated ones (10 px);
- the rings' darkness is fitted with nothing masked, and they must fit the annulus to 0.25 of it (corneal: 0.148
  at 0.66, ragged rings from a small figure enlarged);
- and the rings found must be thick-walled, inner radius at most 0.7 of the outer: without that the reading finds
  thin rings again along a solid family's direction (song-cents's 46 gray rings as a second family of 42, P 0.978
  → 0.871; the o's of viability's red legend text as red rings).

Corneal: a ring from 5.5 to 8.9 px at 0.66 of black, from 3 isolated rings; fitted by rings in drawing order as
other hollow rings are: all 58, and no ring on a filled dot. The 7 extras are 3 second dots in filled dots and 4 on
the two zeros of "574.500", a label inside the plot that the text mask doesn't take. Full benchmark: corneal's row
alone (184 the same). 319 tests pass, with a synthetic scene of 30 open circles and 30 filled dots in one black
(all found, none more), corneal itself, and viability's legend text; each of the five parts above fails a test when
taken out. In the app: 128 dots in two families ("opaque" and "hollow rings"), no console errors.

Limits: three isolated rings are needed, inside the plot; the walls must be 0.3 of the radius or more (thinner
open circles in a solid family's color are still left to that family); and a label's zeros inside the plot are
rings to it, as viability's are to the search by thin ink.

**Axes and labels of the new PLOS figures** (2026-11-22)

Four things fixed, on six images; the rest of the notes on axes are listed at the end.

1. **Both axes with labels and no tick marks** (stiffness, as SPSS draws them: black lines and numbers; viability,
   as Excel does: light gray lines, 191). An axis of labels alone was found only from the other axis, found by
   its ticks; with ticks on neither, no axes. Now, when neither is found, every vertical line is tried as the y axis:
   the x axis that `labelAxis` gives from it must give it back (`cornerLabelAxes`). Three things it needed:
   - the lines are strokes on the page, the same color on both sides: the row where ggplot2's panel ends is a light
     "line" with labels beyond it too, and labels alone beside a panel stay without an axis, as before;
   - ink that touches the line is no label (1 px clear at least): viability's dots at zero lie across its x axis,
     and their lower halves were the row nearest the line;
   - the labels are read whole. They were looked for from the other axis to the line's end, and a label at either
     end was cut there, centered up to 10 px inward: viability's 250 at 248, its 120 and 0 at 119 and 1, stiffness's
     100 at 102. Now words are found a label's reach beyond both ends and kept when centered within them.
   Stiffness: 100 to 400 on both axes; viability: 0 to 250 and 120 to 0, every value read or given by the scale
   exactly. Viability's family of hollow rings (the labels' zeros, 16 extras) is gone with it, its ink now beyond
   the plot: 252 found → 236, P 0.937 → 1.000.
2. **A gridline in pieces, and the panel's border** (lung-tumor, gait). A gridline that dots or a curve cross is
   found in rows whose runs differ, so as two or three lines a pixel apart, each with the gridline's label
   (lung-tumor's 5 three times, gait's 0.4 twice): lines whose rows touch are one (`oneLineEach`). And the
   outermost lines at the ends of the gridlines that cross them are the panel's border, unless they stand where the
   others' spacing puts a gridline (gait's border is its gridlines of 0 and 1.6): left out (`withoutFrame`).
   Lung-tumor: 9 y ticks → 5 (10 to −10) and 8 x ticks → 6 (7 to 12), every label read, and the labels now lie off
   the gridlines' ends, which bounds the plot. Gait: 10 x ticks → 9.
3. **Labels written along the y axis** (funnel, as Stata writes them; bench-r log-e-turned, base R's default). When
   a y axis's labels fit no scale upright, they are read on the image turned a quarter clockwise, as the labels of
   an x axis, and their boxes turned back (`labels.js readTurned`). Funnel: "0", ".05", ".1", ".15" (the reader
   already took a number without its leading zero). log-e-turned read "1?+06": the e of R's turned labels, 7 px
   tall, scores 0.74999 against the 0.75 a sign needed; `SIGN_MIN_SCORE` is 0.72 now, and the label benchmark
   differs in that axis alone. The label benchmark now reads labels as the app does (`readTickLabels`).
4. Of the 160 local images the axes change on these six and no other. Label benchmark: 1372 of 1479 labels read
   right (92.8%, from 92.2%), 214 of 235 axes solved (from 213), none wrong. Full benchmark: viability's row alone
   (184 the same). 323 tests pass: the synthetic corner (with dots across the x axis line), stiffness and viability,
   lung-tumor and gait, funnel and log-e-turned; nine parts taken out each fail one. In the app: viability
   "x linear, read · y linear, read", 236 dots; funnel both axes read.

Still open from the notes on axes:

- **crimson**: read now (see "crimson: labels set by hand", 2026-11-23). The ticks of 10² fit neither axis's scale.
- **alloys**: the top y label (12.5) isn't attached to its tick; the scale gives its value. Not looked at.
- **expression**: two y labels with no ticks, not level with the gridlines: by hand (Add tick).
- **biomarker**: the y axis misses four of its ticks (blurred); its scale is right from the three labels read.
- **corneal**: the dotted reference line is a tick of the y axis, unlabeled, at its true value (574.5).
- Dots lying across an axis line just off its middle can still be taken for its ticks (seen in a synthetic scene:
  three dots 1.5 px above an unticked x axis became its ticks). Biomarker's rule covers a line across the plot only.

**tsne: crescents under a blurred white edge** (2026-11-22)

257 dots in five colors, each with a white edge: cyan, green and blue read as translucent (α 0.80), gray (125
dots) and brown as opaque. 252 found, 12 missed (by family, as the benchmark counts; pooled over the families, 246
of 257 matched and 11 missed). Eight of the missed are gray dots under other gray dots, the crescents of the user's
note; three of the six "extra" dots were second dots put on the dot above such a crescent.

The white edge is already read as a stroke around the gray dots (`stroke.js pageStroke`: 3.1 px wide, 120 gaps,
0.78 of one width), and its pixels should be masked where a dot lies over another. They weren't: a pixel is the
page's only within a quarter of the way from the page to the dots' color (at most 40 units), and this image is
soft: between two gray dots of 154 the stroke reads 204 to 238, a quarter of the way at its very middle, if that.
So few gaps showed "the page" and the rest stayed in the fit, where a dot placed under them pays for every pixel.

Finding the stroke keeps its strict test (it was set on 145 images). Masking its gaps now takes a pixel within half
the way to the dots' color as the page showing (`MASK_PAGE_SHARE`): only once a stroke is found, and only in gaps
between dot inks no wider than the stroke. On tsne, by that share:

| Share | Found | Matched (pooled) | Gray dots missed |
|---|---|---|---|
| 0.25 | 255 | 248 | 6 |
| 0.5 | 256 | 253 | 1 |
| 0.6 | 256 | 253 | 1 |

The benchmark's row: 256 found, P 0.984, R 0.981 (was 252, 0.972, 0.953); nothing else changes (184 rows the same,
song-cents among them), 324 tests pass. The regression test is the stroke test's piles in one gray under a blur of
1.2 px: 40 of 60 found with the gaps masked and 24 without, as with no stroke read at all (at a blur of 1.0 both
find 49: the gaps still reach the page's color). In the app: 256 dots, axes read, no console errors.

What is left on tsne:

- One gray dot 5.4 px from the dot over it: a sliver.
- Three dots that the truth has as second blue dots at one place. The color there is not blue over blue (40, 124,
  184, as at the two blue pairs that are found) but 55, 125, 174 at two of them and 57, 119, 166 at the third:
  blue at α 0.8 over gray and over brown. So each is a gray or a brown dot lying exactly under a blue one, showing
  through at a fifth. An opaque family isn't read through a translucent dot of another, and these aren't found.
- Tried: reading the gray as translucent, which it is (its overlaps are 132 for 154, α 0.8 of a gray of 127). Its
  levels aren't found because 547 flat pixels of 253, a tint of the page, stand as a first level ahead of them;
  with that tint dropped the gray is a fourth translucent family, the dots under blue ones would be its, but it
  then takes the brown dots too (150 gray found for 125, two on each brown dot): 280 found for 257. Dropped; it
  needs the brown read as translucent with it, or kept out of the mix.

**lung-tumor: small blurred dots with a ring inside the rim, under a band** (2026-11-22)

473 dots in the hand truth (the figure's title says n = 502): no family → 501 found, all 473 of the truth among
them. The dots are matplotlib's translucent markers: a fill at α 0.44, and the edge stroked in the same color at the
same alpha, half of it over the fill. So a dot is one layer deep in the middle, two in a ring, and one again in a
band outside the ring. A blurred disc with such a ring, fitted to the pooled profile of 43 lone dots, leaves 1.3 of
255 rms (radius 6.8, σ 0.6, the ring from 3.6 to 5.1 px, 0.29 more opaque; α(1 − α) is 0.25); with the ring made to
reach the rim, as the model had it, 4.2. Five things stood between the image and its dots:

1. **Level 1 was left out as thin.** An opening by 1.5 px keeps 0.239 of the fill's color (the limit is 0.25): the
   fills are 6 px across, and more of the color is in the dots' soft rims. Over all local images only one other
   first level is thin, microplastics1's border bar (0.233), which the rule is for; the share can't tell them apart.
   Whole dots can: a first level thin as a whole is a fill all the same where ten whole dots show on the levels
   with it (`levels.js THIN_FILL_MIN_DOTS`; lung-tumor has 43, the bar none). With level 1 back: α 0.44 from 807
   flat pixels at level 1 and 8 at level 2.
2. **No dot was whole.** A piece deeper than one dot is a dot if all of that is in its outer part, beyond 0.6 of
   the radius; here it starts at 0.42 to 0.50. Now a third (`shape.js RING_MIN_RADIUS_FRACTION`). That alone let two
   pieces of expression's red dots through when read with noisy matching (opaque dots at "α 0.96", their outer
   halves a level deeper), which gave that scan two families with a radius and the win: 1033 dots for 46. So a piece
   deeper in its outer part is a whole dot only where a ring is measured there (`estimateStyle`, and the radius
   `unmix.js` reads later).
3. **The ring's outer radius.** `style.ring` has `outer` (× the radius, 1 by default), the renderer and the fit's
   ring check use it, and `measureRing` searches both radii with the image's blur (every candidate disc rendered
   and blurred once, the pairs from their products). Lung-tumor: 0.51 to 0.76, 0.25 more opaque. It moves the rings
   read before: bench1 and bench2 stroked-dots 0.82 to 0.96 (was 0.80 to the rim; all dots found as before),
   richness 0.81 to 0.94 at 0.16 more opaque (α(1 − α) is 0.17: matplotlib again).
4. **The band was masked with its 22 dots.** Over or under is voted by the flat dot pixels in the region, 50 needed;
   dots this small have one or two each, 24 in all, every one for a layer over the dots. Now 20
   (`background.js MODE_MIN_VOTES`), with a floor on what the layer lets through (`S_MIN` 0.1): at 20, crimson's
   dark gray of 35 with 28 pixels of 23 to 26 in it was "a layer with s = 0.04" and was divided out.
5. **The band's waist wasn't divided out.** Where the band is 10 px across with the regression line through it, its
   color shows only in strips 2 px wide, which are no seeds; the 8 dots there took the page's label, stayed as drawn,
   off the family's line, and were masked. A dot pixel under a layer lies on the line from the layer's color along
   c − B, and one beside it on the line from B; here the two lines are 26 units apart. Where a pixel is within 4 of
   one and more than 8 off the other, its own color now labels it (`labelByColor`), and such a pixel is divided out
   even at the region's edge. Not where the lines coincide: mpo's boxes and dots are all gray, and with every color
   on one line the boxes' lighter edges went to the page (56 found for 45).

| Step | Found | Matched of 473 |
|---|---|---|
| 1 to 3 (the family, its radius and ring) | 478 | 451 |
| 4 (the band divided out) | 496 | 465 |
| 5 (its waist too) | 501 | 473 |

**The truth is short.** At 26 of the 28 "extra" dots the count image shows about one layer more than the truth's
dots put there, and what the fit's dots put there (at 530, 308: 2.36 shown, 1.68 by the truth, 2.44 by the fit):
they are dots lying on others, which a hand truth misses. 501 found for the title's 502. Against those 473 the
benchmark's precision was 0.944 (R 1.000, RMSE 0.60 px). The user then saved the truth from the fit: 501 dots, so
the row now reads 501 for 501 at RMSE 0.00 and checks the count, not the positions.

Full benchmark: 19 rows change, 167 the same. Better: lung-tumor; the bands over dots (bench2 smoother-ci 201 → 204
found, R 0.985 → 1.000; its JPEG 195 → 199, R 0.951 → 0.966; bench1's JPEG P 0.990 → 1.000, R 0.970 → 0.980, RMSE
0.29 → 0.09; bench2 group-smoother-ci P 0.995 → 1.000), from the labels by color; bench2 gradient-dots-stroke.jpg
(220 → 215, P 0.877 → 0.902) and grader1 (46 → 45), which moved with steps 2 to 4 (not told apart). Worse by one
dot: richness, 57 → 58 for 48 (four extras on the right gone, a pile of four on the regression line kept as ties of
one another). The rest move in
RMSE, flags or radius only (stroked-dots' radius 10.1 → 10.3). 332 tests pass (324 before): the renderer's ring,
`measureRing`, a ringed whole dot, the synthetic of such dots, the no-ring rule, a band's waist, a dark region, and
lung-tumor itself; each fails with its fix taken out, mpo's test with the coincident lines let through. In the
app: 501 dots, axes read, no console errors.

Open: the one dot short of the title's 502; richness's pile on its line; and the family still hangs on 8 flat
pixels two layers deep, though every ring is two layers deep and could give level 2 where no two fills overlap.

**richness: the regression line masked, and the band taken off the dots left on it** (2026-11-23)

48 dots in the hand truth: 58 found → 51, the 48 among them. The other three are second dots: see "The truth is
short" below. Of the 10 too many, 4 were a pile on the regression line, 3 lay under real dots on the band, and 3 are
those second dots. Three causes and what each needed:

1. **The regression line wasn't masked.** A line in the family's full color is masked where its pixels are
   saturated, within 8 units of c (`layers.js SATURATION_TOL`). Here c comes from two levels at α 0.22, (76, 110,
   211), and the line is (58, 94, 204): 25 units past it on the family's line, with its first two rows, resampled,
   10 and 22 units past that. Nothing of the line was deep, the fit read it as 14 dots deep (275 candidates, 54 s)
   and left a pile of four on it. An error in the levels grows as 1 / α² in c, so a pixel up to
   `PAST_FULL` / α² (1.5: 30 units here, 8 at α 0.5 or more) past c on the line is saturated too, and ink farther
   past it beside a thin mark is the mark's edge (`THIN_EDGE_REACH`, 2 px). Without a limit, every pixel past c: bench2
   gradient-dots-stroke lost dots to its darker strokes (png 207 → 199 found, R 0.971 → 0.951; jpg R 0.951 → 0.907).
   With the line masked the fit starts from 48 candidates, the 48 dots, and takes 11 s.
2. **A dot under a real dot on the band stayed.** Once the ring check has taken the band's dots out, the band is
   missing from the model, and a dot left on it within 0.35 r of a ringed one stayed as a possible tie under that
   ring. A second measure tells them: the layers the model lacks inside the dot less those it lacks just outside
   its rim, plus one (`fit.js fillShown`, by sectors, the median). A real dot reads 1 whatever is under it, a dot
   that is the band 0. Where the ring check removed anything, dots under 0.5 go one at a time, the one with the
   least ring first (the dots it lies on read no better until it has gone). Richness, at the first version: the 51
   dots 0.88 to 1.67, the four others −0.00 to 0.20. Bench2 stroked-dots' 204 dots, where nothing goes: 0.78 to
   1.18; lung-tumor's 428 readable: 421 within 0.8 to 1.2, the least 0.48 (its ring check removes nothing, so the
   fill check doesn't run).
3. **The dots left on the band lay off their places.** A dot on a band the model lacks moves to cover more of it.
   On synthetic scenes with exact positions (36 dots and two ties over a band of one dot's color and a line, `test/
   extract.test.js bandScene`, six seeds): RMSE up to 0.62 px and single dots up to 3.7 px off; with the dots beside
   a removed one placed again, up to 4.1 px and a dot lost; with all placed again, up to three lost. So the band is
   divided out of the fit's image first (`divideUnder`: the layers the model lacks, averaged over 7 × 7 px and
   rounded, so that a ring a little off its place averages out), then all dots are placed again: RMSE 0.021 to
   0.048 px, the worst dot 0.18 px off, over 28 seeds.

The synthetic scenes also showed the ring check losing real dots, which richness's 48 don't show:

- A ring over a band the model lacks reads low: in t, what is under the dot counts by what the model lets through,
  less on the ring than inside it (a single dot 1 − α of its ring at best, two in one place 0.53 at α 0.22). The
  share is now read against the model scaled by what the image lets through just inside the ring (`ringShown`'s k).
  At α 0.45, 8 of 10 seeds lost a tie's second dot without it, 1 of 10 with it.
- Read in layers instead, as the fill is, a nearly opaque ring is many layers: grader1 (α 0.93, ring 0.07) lost 19
  of 45 dots. Hence k, in transmittance. Grader1 has a test now.
- Two ringless dots almost in one place wait until the others have gone, whose rings across theirs spoil the
  reading (a tie on the line: 0.34 and 0.41 among the band's dots, 1.72 for the one left); then the lesser goes and
  the other is read again.
- Tried and dropped: keeping the second of such a pair when their shares sum to 0.5 (no seed needs it once k is
  in), and removing only the least-ringed of all overlapping dots each round (13 of 14 seeds at α 0.45, against 9 of
  10: one fixed, another broken).

28 seeds at α 0.22: 38 of 38 found in each, nothing more. At α 0.45: 9 of 10.

**The truth is short.** At three of the 48 the image is two layers deep inside the dot and none outside it, off
the band: (468, 460), (489, 460) and the right one of the pair at (586, 338). Measured from the count image by
sectors against the truth's own dots, inside less outside: 2.00, 2.00 and 1.97 there, 0.96 to 1.11 at the other 45.
The first two are visibly darker in the figure. The fit has two dots at each: against the 48, the benchmark's
precision was 0.941 (R 1.000, RMSE 0.43 px). The user then saved the truth from the fit: 51 dots, so the row now
reads 51 for 51 at RMSE 0.00 and checks the count, not the positions. (PLOS papers seldom share their data, so
these figures have no truth but the image; the Nature ones often do.)

Full benchmark: richness's row alone changes (184 the same; lung-tumor's is the truth's 501): 58 → 51 found,
P 0.828 → 0.941, R 1.000, flagged 15 → 8, 54 s → 10.5 s at 10 jobs. RMSE against the hand truth 0.39 → 0.43 px:
four dots, two on the band and two in the pile at (585, 338), are 0.7 to 1.8 px from the truth's places; at
(402, 459), the farthest, the ring in the image puts the dot between the two by eye, 0.7 px from the truth's place
and 1.1 from the fit's. 335 tests pass (332 before): a line 27 units past the full color
read (layers), the band scene with ties at four seeds and α 0.22 and 0.45 (extract), grader1, and richness with 51
dots and its line masked. Each fails with its fix taken out: the saturation past c, the mark's edge, the fill check,
the least ring first, the pairs' wait, the division, k. In the app: 51 dots, 8 flagged, the three ties shown as
double rings, axes read, no console errors.

Open: at α 0.45 one synthetic seed in ten still loses a dot to the
ring check, read among the band's dots; the fit still fills the band with dots first (88 born) and removes them.

**crimson: labels set by hand** (2026-11-23)

The labels and tick marks of crimson's axes were placed by hand (the user's reading, and the pixels agree): the y
axis's marks are 42, 30, 25 and 42 px long, each label 6 px from its own mark, and the x axis's "10²" stands 5 px
lower than the other three. Read before: x "10?", 10³, 10⁴, 10⁵ (a log scale through three, 70 at the first); y
10⁵, "10^?", "10^?", 10² (no scale). The note that the y labels were cut off at the image's left edge was wrong:
they were cut by the search.

1. **Labels cut at the longest mark's reach.** `axes.js findLabels` looks for labels beyond the longest tick mark,
   and `labels.js labelOptions` kept the reader from going nearer the axis than that. The exponents of the labels
   at the two short marks were cut through at x = 278 (13 and 9 of their 20 px left). Now a label at a mark shorter
   than the longest has its box searched again as far as its own mark (`uncutLabel`), taking in the words that reach
   into the box, and the reader's limit is the longest mark's reach or the box's end, whichever is nearer the axis.
   - Tried first, and dropped: searching each stretch of the axis beyond the nearest tick's own mark from the start.
     On delta1 an error bar's caps on the y axis line are two "ticks" of 11 px; the dot between them, beyond their
     reach, became the nearest "label", and the axis's own labels were left out as too far: no y axis.
2. **A label off the shared line.** The reader finds each label twice, the second time on the line the axis's
   labels share. On the others' line (5 px higher) the 2 of "10²", an exponent set low (to 0.52 of the digits'
   height below their top), ended below an exponent's rows (0.6) and was a glyph of the number: "10?". Now a label
   found cleanly on its own, its digits within a pixel of the shared height, keeps its own line
   (`OWN_LINE_TOL`). That also solves bench-r's small-e-r72 y axis: the shared line of a y axis is placed by each
   tick's position, a row off at two of six ticks, and the labels' point, one pixel square in the last row, was
   left out ("000025").

Crimson now reads all eight labels. Both scales are log through 10³, 10⁴ and 10⁵; the ticks of 10² fit neither (316
px from 10³ on the x axis where a decade is 271, 248 on the y axis where it is 285), so each gets its scale's value,
70 and 140, with "label: 10^2" beside it in the app. Tick placement is the figure's own and left alone.

Of the 160 local images the axes and readings change on these and no other: crimson; small-e-r72's y axis (solved);
delta1, where "150", boxed at a false tick (an error bar's cap), now reads and is contradicted by the scale (the
values are the same); hair2 and tumour1, whose x labels are "+" and "−": a "+" alone now reads "1" (score 0.73) where
it read "?". Two alike give no scale, so nothing is offered, but a lone "+" should read as a sign: open.
Label benchmark: 1374 of 1479 labels read right (92.9%, from 92.8%), 215 of 235 axes solved (from 214), none wrong.
Crimson's row in the benchmark is the same (189 found for 180); no other image's label boxes moved.
338 tests pass (335 before): a synthetic y axis with marks of two lengths and labels 4 px from their own, a
synthetic x axis with low exponents and one label 2 px down, crimson itself, and small-e-r72's y axis. Each fails
with its fix taken out: the box searched again, the reader's limit, the own line. In the app: crimson "x log,
read · y log, read", no console errors.

**crimson: bold text as thick as the dots** (2026-11-23)

Crimson's 9 extras, by the user's reading and the fit's: six are second dots on true ones (three black, three red,
0.1 to 0.6 px from the truth's, each flagged close), possible doubles that opaque dots don't settle and that stay
for the user to judge; three, unflagged (margin 41), are the crossings of the t's in "Productive" and "Latent".

The text is bold, black as the black dots, and was no text to `annotations.js`: a glyph's stroke had to be under
0.6 of the dots' radius (`TEXT_MAX_STROKE_RADII`, set against adults1's chains of touching dots), and these
measure 8 to 10 px beside a radius of 9.4. The two numbers aren't measured alike, though: a piece's inscribed
radius takes in its antialiased edge, and crimson's whole dots measure 11.2 to 11.7 by it. Against that, the text's
strokes are 0.70 to 0.88 of a dot, and a piece that holds a whole dot (adults1's chains) is 1 or more.

So where the image shows whole dots (three or more round, filled pieces about the family's radius), a glyph's
stroke may be up to 0.9 of their median thickness (`TEXT_MAX_STROKE_DOTS`), or 0.6 of the radius as before,
whichever is more; without whole dots, as before. Ten lines in `annotationBoxes`; the other tests of a line of text
(thin for its height, on a baseline, glyphs that vary) are as they were.

Full benchmark: crimson's row alone changes (185 the same): 189 → 186 found for 180, P 0.952 → 0.968, R 1.000.
Tests: a synthetic bold word (strokes 13 px beside dots of radius 10, with whole dots and without) and crimson's two
words, with no box over any of its 180 dots; each fails with the rule taken out, and the adults1 scene fails with
the share at 1.2 or the limit dropped.
340 tests pass (338 before). In the app: 186 dots, 14 flagged, the six possible doubles shown as double rings, none on
the text, no console errors.
