# bench-r2: R's scatterplots with translucent, overlapping, many and colored dots, fitted lines and bands

Scatterplots drawn by R for this project, the way figures commonly draw them, to benchmark the finding of dots:
translucent dots, dots overlapping each other, hundreds or thousands of dots, dots in several colors, and dots with
fitted lines and curves and their confidence regions drawn under or over them.
Run with `npm run bench -- --set bench-r2` (the dots, placed through the detected ticks and scored by color group)
and `npm run bench:labels -- --set bench-r2` (the axes and their labels).
`bench-r/` is the other R set, about tick labels in scientific notation and ggplot2's gray theme, 40 dots each.

Nothing here comes from a publication: `make.R` draws every image from random points and writes, beside each, a CSV
of the points in data units (x, y and a group when the dots have colors) with the tick values in header lines.
`Rscript make.R` in this directory draws them again (R 4.x with ggplot2; the committed images are R 4.6.0's and
ggplot2 4.0.3's on macOS, text in Helvetica).

| images | dots | what they show |
|---|---|---|
| `gg-alpha` | 300 | one color at alpha 0.35, `theme_bw()` |
| `gg-alpha-groups` | 450 | three colors at alpha 0.5 overlapping each other, ggplot2's gray theme with a legend |
| `base-alpha` | 400 | base R, `rgb(0.8, 0.1, 0.1, 0.3)` |
| `gg-many` | 1000 | small opaque dots, clumped in the middle, `theme_classic()` |
| `gg-overlap` | 250 | large opaque dots along a line, most of them overlapping |
| `gg-dense-alpha` | 2000 | black at alpha 0.15, at 150 dpi |
| `gg-colors` | 240 | four opaque colors, `theme_classic()` |
| `base-colors` | 150 | base R, three opaque colors with a legend inside the plot |
| `gg-outlined` | 150 | shape 21: three fill colors outlined in black |
| `gg-lm` | 80 | `geom_smooth(method = "lm")`: a blue line and a gray band drawn over the dots |
| `gg-loess` | 120 | a loess curve and its band over black dots, ggplot2's gray theme |
| `gg-loess-groups` | 140 | two colors at alpha 0.6, each with its loess curve and a band in its own color |
| `gg-lm-under` | 100 | the band and line drawn first, the dots over them in the line's color |
| `gg-ellipse` | 180 | three colors, each with a 95% ellipse filled at alpha 0.15 and outlined in its color, over the dots |
| `base-lm` | 150 | base R, translucent dots, a band filled over them, the fitted line and dashed confidence limits |
