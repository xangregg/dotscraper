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
