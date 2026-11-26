# bench-r2: scatterplots drawn by R as figures commonly draw them: translucent and overlapping dots, many dots, dots
# in several colors, and dots with fitted lines and curves and their confidence regions. Each image gets a CSV of its
# points in data units (x, y and a group when there are colors) with the tick values in header lines, which the
# benchmark places in the image through the detected ticks (npm run bench -- --set bench-r2).
#
# Usage, from this directory: Rscript make.R   (R 4.x with ggplot2; written with R 4.6.0 and ggplot2 4.0.3 on macOS,
# whose quartz device sets the text in Helvetica)

suppressMessages(library(ggplot2))
set.seed(11)

num <- function(v) paste(trimws(formatC(v, format = "g", digits = 15)), collapse = ", ")
fmt <- function(v) trimws(formatC(v, format = "g", digits = 10))

# the CSV beside an image: tick values (those inside the plot), scales, and the points, with their groups if any
truth <- function(name, x, y, xticks, yticks, group = NULL) {
    rows <- if (is.null(group)) paste(fmt(x), fmt(y), sep = ",") else paste(fmt(x), fmt(y), group, sep = ",")
    lines <- c(
        paste("# x ticks:", num(xticks)), paste("# y ticks:", num(yticks)),
        "# x scale: linear", "# y scale: linear",
        "# source: drawn by bench-r2/make.R", if (is.null(group)) "x,y" else "x,y,group", rows
    )
    writeLines(lines, paste0(name, ".csv"))
}

# a ggplot of data frame d (columns x, y, and g for a group): the tick values are its scales' breaks
gg <- function(name, p, d, width = 6, height = 4.6, dpi = 110) {
    ggsave(paste0(name, ".png"), p, width = width, height = height, dpi = dpi)
    params <- ggplot_build(p)$layout$panel_params[[1]]
    breaks <- function(b) b[!is.na(b)]
    truth(name, d$x, d$y, breaks(params$x$breaks), breaks(params$y$breaks), if ("g" %in% names(d)) d$g else NULL)
}

# of the values `at`, those inside the open plot
inside <- function(at, side) {
    usr <- par("usr")[if (side == 1) 1:2 else 3:4]
    at[at >= usr[1] & at <= usr[2]]
}

# a base-graphics plot of data frame d: `draw` plots it
base <- function(name, d, draw, width = 660, height = 506, res = 96) {
    png(paste0(name, ".png"), width = width, height = height, res = res, type = "quartz")
    par(mar = c(4, 4.5, 1, 1))
    draw()
    truth(name, d$x, d$y, inside(axTicks(1), 1), inside(axTicks(2), 2), if ("g" %in% names(d)) d$g else NULL)
    invisible(dev.off())
}

# a linear trend with noise, and a curved one
linear <- function(n, slope = 0.6, sd = 6) {
    x <- runif(n, 0, 100)
    data.frame(x = x, y = 20 + slope * x + rnorm(n, 0, sd))
}
curved <- function(n, sd = 0.35) {
    x <- runif(n, 0, 10)
    data.frame(x = x, y = sin(x / 1.6) + 0.15 * x + rnorm(n, 0, sd))
}
groups <- function(k, n, f) do.call(rbind, lapply(seq_len(k), function(i) cbind(f(n, i), g = LETTERS[i])))
HUE3 <- c("#F8766D", "#00BA38", "#619CFF") # ggplot2's own palette for three groups

# --- translucent dots ---

d <- linear(300, sd = 9)
gg("gg-alpha", ggplot(d, aes(x, y)) + geom_point(colour = "#1f4e9c", alpha = 0.35, size = 2.5) + theme_bw(), d)
d <- groups(3, 150, function(n, i) {
    x <- rnorm(n, 30 + 15 * i, 9)
    data.frame(x = x, y = 10 + (0.3 + 0.2 * i) * x + rnorm(n, 0, 6))
})
gg("gg-alpha-groups", ggplot(d, aes(x, y, colour = g)) + geom_point(alpha = 0.5, size = 2.5), d)
d <- linear(400, sd = 10)
base("base-alpha", d, function() plot(d$x, d$y, pch = 19, col = rgb(0.8, 0.1, 0.1, 0.3), xlab = "x", ylab = "y", las = 1))

# --- many dots, opaque and overlapping ---

d <- data.frame(x = rnorm(1000, 50, 15), y = rnorm(1000, 50, 15))
gg("gg-many", ggplot(d, aes(x, y)) + geom_point(colour = "#333333", size = 1.5) + theme_classic(), d)
d <- linear(250, sd = 4)
gg("gg-overlap", ggplot(d, aes(x, y)) + geom_point(colour = "#2b8cbe", size = 3.5) + theme_bw(), d)
d <- data.frame(x = rnorm(2000, 0, 1), y = rnorm(2000, 0, 1))
d$y <- 0.6 * d$x + 0.8 * d$y
gg("gg-dense-alpha", ggplot(d, aes(x, y)) + geom_point(alpha = 0.15, size = 1.8) + theme_bw(), d, dpi = 150)

# --- dots in several colors ---

d <- groups(4, 60, function(n, i) data.frame(x = runif(n, 0, 100), y = runif(n, 0, 100)))
gg("gg-colors", ggplot(d, aes(x, y, colour = g)) + geom_point(size = 2.5) + theme_classic(), d)
d <- groups(3, 50, function(n, i) data.frame(x = rnorm(n, 20 * i, 10), y = rnorm(n, 10 * i, 8)))
base("base-colors", d, function() {
    plot(d$x, d$y, pch = 19, col = HUE3[match(d$g, LETTERS)], xlab = "x", ylab = "y", las = 1)
    legend("topleft", legend = LETTERS[1:3], pch = 19, col = HUE3, bty = "n")
})
d <- groups(3, 50, function(n, i) data.frame(x = rnorm(n, 20 * i, 10), y = rnorm(n, 10 * i, 8)))
gg("gg-outlined", ggplot(d, aes(x, y, fill = g)) + geom_point(shape = 21, colour = "black", size = 3) + theme_bw(), d)

# --- fitted lines and curves, with confidence regions ---

d <- linear(80)
gg("gg-lm", ggplot(d, aes(x, y)) + geom_point(colour = "#1b9e77", size = 2.5) + geom_smooth(method = "lm", formula = y ~ x) + theme_classic(), d)
d <- curved(120)
gg("gg-loess", ggplot(d, aes(x, y)) + geom_point(size = 2.5) + geom_smooth(method = "loess", formula = y ~ x), d)
d <- groups(2, 70, function(n, i) {
    x <- runif(n, 0, 10)
    data.frame(x = x, y = (1 + i) * sin(x / 2.2) + 0.3 * i * x + rnorm(n, 0, 0.5))
})
gg("gg-loess-groups", ggplot(d, aes(x, y, colour = g, fill = g)) + geom_point(alpha = 0.6, size = 2.5) +
    geom_smooth(method = "loess", formula = y ~ x, alpha = 0.2) + theme_bw(), d)
# the band drawn first, the dots over it
d <- linear(100, sd = 8)
gg("gg-lm-under", ggplot(d, aes(x, y)) + geom_smooth(method = "lm", formula = y ~ x, colour = "#d95f02", fill = "#fdd0a2") +
    geom_point(colour = "#d95f02", size = 2.5) + theme_bw(), d)
d <- groups(3, 60, function(n, i) {
    x <- rnorm(n, 25 * i, 8)
    data.frame(x = x, y = 30 + (2 - i) * 0.5 * (x - 25 * i) + 15 * i + rnorm(n, 0, 6))
})
gg("gg-ellipse", ggplot(d, aes(x, y, colour = g)) + geom_point(size = 2.5) + stat_ellipse(aes(fill = g), geom = "polygon", alpha = 0.15) +
    stat_ellipse() + theme_classic(), d)
d <- linear(150, sd = 8)
fit <- lm(y ~ x, data = d)
grid <- data.frame(x = seq(0, 100, length.out = 200))
band <- predict(fit, grid, interval = "confidence")
base("base-lm", d, function() {
    plot(d$x, d$y, pch = 19, col = rgb(0.2, 0.2, 0.7, 0.5), xlab = "x", ylab = "y", las = 1)
    polygon(c(grid$x, rev(grid$x)), c(band[, "lwr"], rev(band[, "upr"])), col = rgb(0.2, 0.2, 0.7, 0.15), border = NA)
    lines(grid$x, band[, "fit"], col = rgb(0.2, 0.2, 0.7), lwd = 2)
    lines(grid$x, band[, "lwr"], lty = 2)
    lines(grid$x, band[, "upr"], lty = 2)
})
