# bench-r: R's tick labels in scientific notation, and ggplot2's gray theme

Scatterplots drawn by R for this project, to benchmark and test the finding of axes without axis lines (ggplot2's
default theme) and the reading of tick labels written the two ways figures write large and small numbers: with the
power of ten after an "e" (`1e+05`, R's own form in base graphics and ggplot2; JMP's `1e+5`; Excel's `1.E+05`) and as
a number times a power of ten (`2 × 10³`, by plotmath or typed).
Run with `npm run bench:labels -- --set bench-r` (the axes and their labels) and `npm run bench -- --set bench-r`
(the dots: each plot has 40, placed through the detected ticks).

Nothing here comes from a publication: `make.R` draws every image from random points and writes, beside each, a CSV
of the points in data units with the tick values in header lines. `Rscript make.R` in this directory draws them again
(R 4.x with ggplot2 and scales; the committed images are R 4.6.0's on macOS, text in Helvetica).

| images | what they show |
|---|---|
| `log-e`, `log-e-hi`, `log-e-default` | base R, both axes log, `1e-03` to `1e+06`, at 96 and 170 dpi and at the device's default size (digits 9 px tall, where a minus sign touches its neighbor) |
| `log-e-turned` | the same with the y labels turned along the axis, R's default: Dot Scraper doesn't read turned text |
| `linear-e` | a linear axis from `0e+00` to `6e+05` |
| `small-e-r72`, `-r120`, `-r220` | p-values on a log axis, `1e-07` to `1e-03`, and a linear axis in plain decimals |
| `excel-E`, `jmp-e` | the forms of Excel (`2.E+05`, `2.0E+03`) and JMP (`1e-7`, `2e+3`), set by R |
| `gg-linear-e`, `gg-log-e`, `gg-log-e-bw` | ggplot2 with `theme_bw()` and `theme_classic()` |
| `times-r72`, `-r96`, `-r144`, `-r220` | `2 × 10⁵` by plotmath under the plot, `2 × 10³` typed beside it |
| `times-log` | a log axis labeled 1 and 3 times each power of ten |
| `times-x` | decimals before the sign (`1.5 × 10⁵`), and the letter x typed for it |
| `gg-times`, `gg-times-bw` | ggplot2 with parsed labels |
| `gg-gray`, `gg-gray-small`, `gg-gray-hi` | ggplot2's default theme at 110, 72 and 300 dpi: a gray panel with white grid lines and no axis lines, the tick marks at its edges |
| `gg-gray-log-e`, `gg-gray-times` | the default theme with labels in each notation |
