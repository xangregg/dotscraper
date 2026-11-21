# bench-r: scatterplots drawn by R whose tick labels are in scientific notation, the two ways figures write it:
# "1e+05" (R's own, in base graphics and ggplot2; JMP's "1e+5"; Excel's "1.E+05") and "2 × 10³" (plotmath, or typed).
# Each image gets a CSV of its points in data units with the tick values in header lines, as tools/label-bench.mjs
# reads them (npm run bench:labels -- --set bench-r).
#
# Usage, from this directory: Rscript make.R   (R 4.x with ggplot2 and scales; written with R 4.6.0 on macOS, whose
# quartz device sets the text in Helvetica)

suppressMessages(library(ggplot2))
set.seed(7)
N <- 40
BLUE <- "#1f77b4"

num <- function(v) paste(trimws(formatC(v, format = "g", digits = 15)), collapse = ", ")

# the CSV beside an image: tick values (those inside the plot), scales, and the points
truth <- function(name, x, y, xticks, yticks, xlog = FALSE, ylog = FALSE) {
    lines <- c(
        paste("# x ticks:", num(xticks)), paste("# y ticks:", num(yticks)),
        paste("# x scale:", if (xlog) "log" else "linear"), paste("# y scale:", if (ylog) "log" else "linear"),
        "# source: drawn by bench-r/make.R", "x,y", paste(trimws(formatC(x, format = "g", digits = 10)), trimws(formatC(y, format = "g", digits = 10)), sep = ",")
    )
    writeLines(lines, paste0(name, ".csv"))
}

# of the values `at`, those inside the open plot (par("usr") is in log10 units on a log axis)
inside <- function(at, side) {
    usr <- par("usr")[if (side == 1) 1:2 else 3:4]
    v <- if (par(if (side == 1) "xlog" else "ylog")) log10(at) else at
    at[v >= usr[1] & v <= usr[2]]
}

# a base-graphics plot: `draw` plots and returns the tick values it gave the axes, list(x = , y = )
base <- function(name, x, y, draw, width = 640, height = 520, res = 96, mar = c(4, 6, 1, 1), xlog = FALSE, ylog = FALSE) {
    png(paste0(name, ".png"), width = width, height = height, res = res, type = "quartz")
    par(mar = mar)
    ticks <- draw()
    truth(name, x, y, inside(ticks$x, 1), inside(ticks$y, 2), xlog, ylog)
    invisible(dev.off())
}

# a ggplot: the tick values are its scales' breaks
gg <- function(name, p, x, y, width = 6, height = 4.6, dpi = 110, xlog = FALSE, ylog = FALSE) {
    ggsave(paste0(name, ".png"), p, width = width, height = height, dpi = dpi)
    params <- ggplot_build(p)$layout$panel_params[[1]]
    breaks <- function(b, log) {
        b <- b[!is.na(b)]
        if (log) 10^b else b
    }
    truth(name, x, y, breaks(params$x$breaks, xlog), breaks(params$y$breaks, ylog), xlog, ylog)
}

# --- "1e+05": a number with its power of ten after an "e" ---

x <- 10^runif(N, -3.5, 5.5)
y <- 10^runif(N, -1.5, 6.5)
loglog <- function(las) function() {
    plot(x, y, log = "xy", pch = 19, col = BLUE, las = las, ylab = "")
    list(x = axTicks(1), y = axTicks(2))
}
# R's device as it comes (480 × 480 at 72 dpi, 12 pt: digits 9 px tall), labels upright
base("log-e-default", x, y, loglog(1), width = 480, height = 480, res = 72, xlog = TRUE, ylog = TRUE)
# … and with the y labels turned along the axis, R's default: dotscraper doesn't read turned text
base("log-e-turned", x, y, loglog(0), width = 480, height = 480, res = 72, xlog = TRUE, ylog = TRUE)
base("log-e", x, y, loglog(1), xlog = TRUE, ylog = TRUE)
base("log-e-hi", x, y, loglog(1), width = 1100, height = 900, res = 170, xlog = TRUE, ylog = TRUE)

u <- runif(N, 0, 6.2e5)
v <- runif(N, 0, 9e3)
base("linear-e", u, v, function() {
    plot(u, v, pch = 19, col = "#d62728", las = 1, ylab = "")
    list(x = axTicks(1), y = axTicks(2))
})

# small values: p-values on a log axis, and a linear axis R writes in full
p <- 10^runif(N, -7.8, -2.2)
q <- runif(N, 0, 2.6e-4)
for (res in c(72, 120, 220)) {
    base(sprintf("small-e-r%d", res), p, q, function() {
        plot(p, q, log = "x", pch = 19, col = BLUE, las = 1, ylab = "")
        list(x = axTicks(1), y = axTicks(2))
    }, width = round(6.6 * res), height = round(5.4 * res), res = res, xlog = TRUE)
}

# as other programs write it: Excel's "2.E+05" and "2.0E+03", and JMP's "1e-7" and "2e+3"
atu <- c(0, 2e5, 4e5, 6e5)
atv <- c(0, 2e3, 4e3, 6e3, 8e3)
base("excel-E", u, v, function() {
    plot(u, v, pch = 19, col = "#9467bd", xlab = "u", ylab = "", axes = FALSE)
    axis(1, at = atu, labels = c("0.E+00", "2.E+05", "4.E+05", "6.E+05"))
    axis(2, at = atv, labels = c("0.0E+00", "2.0E+03", "4.0E+03", "6.0E+03", "8.0E+03"), las = 1)
    box()
    list(x = atu, y = atv)
}, width = 700)
atp <- 10^(-8:-2)
base("jmp-e", p, v, function() {
    plot(p, v, log = "x", pch = 19, col = "#8c564b", xlab = "p", ylab = "", axes = FALSE)
    axis(1, at = atp, labels = sprintf("1e-%d", 8:2))
    axis(2, at = atv, labels = c("0", "2e+3", "4e+3", "6e+3", "8e+3"), las = 1)
    box()
    list(x = atp, y = atv)
}, width = 700, xlog = TRUE)

gg("gg-linear-e", ggplot(data.frame(u, w = v * 1e5), aes(u, w)) + geom_point(colour = "#9467bd", size = 2.5) + theme_bw(), u, v * 1e5)
gg("gg-log-e", ggplot(data.frame(x, y), aes(x, y)) + geom_point(colour = "#2ca02c", size = 2.5) + scale_x_log10(breaks = 10^(-3:5)) +
    scale_y_log10(breaks = 10^(-1:6)) + theme_classic(), x, y, width = 7, dpi = 150, xlog = TRUE, ylog = TRUE)
gg("gg-log-e-bw", ggplot(data.frame(x, y), aes(x, y)) + geom_point(colour = "#2ca02c", size = 2.5) + scale_x_log10() + scale_y_log10() + theme_bw(),
    x, y, xlog = TRUE, ylog = TRUE)

# --- "2 × 10³": a number times a power of ten ---

# plotmath below (the sign from the Symbol font, on the baseline), typed text beside (Helvetica's "×" and "³")
for (res in c(72, 96, 144, 220)) {
    base(sprintf("times-r%d", res), u, v, function() {
        plot(u, v, pch = 19, col = "#ff7f0e", xlab = "u", ylab = "", axes = FALSE)
        axis(1, at = atu, labels = expression(0, 2 %*% 10^5, 4 %*% 10^5, 6 %*% 10^5))
        axis(2, at = atv, labels = c("0", "2 × 10³", "4 × 10³", "6 × 10³", "8 × 10³"), las = 1)
        box()
        list(x = atu, y = atv)
    }, width = round(6.6 * res), height = round(5.4 * res), res = res)
}
# a log axis with 1 and 3 times each power
atl <- c(1e2, 3e2, 1e3, 3e3, 1e4, 3e4)
xl <- 10^runif(N, 2, 4.5)
base("times-log", xl, v, function() {
    plot(xl, v, log = "x", pch = 19, col = "#8c564b", xlab = "x", ylab = "", axes = FALSE)
    axis(1, at = atl, labels = expression(1 %*% 10^2, 3 %*% 10^2, 1 %*% 10^3, 3 %*% 10^3, 1 %*% 10^4, 3 %*% 10^4))
    axis(2, at = atv, las = 1)
    box()
    list(x = atl, y = atv)
}, width = 760, xlog = TRUE)
# decimals before the sign, and the letter x typed for it
atd <- c(0, 1.5e5, 3e5, 4.5e5, 6e5)
base("times-x", u, v, function() {
    plot(u, v, pch = 19, col = "#2ca02c", xlab = "u", ylab = "", axes = FALSE)
    axis(1, at = atd, labels = expression(0, 1.5 %*% 10^5, 3 %*% 10^5, 4.5 %*% 10^5, 6 %*% 10^5))
    axis(2, at = atv, labels = c("0", "2 x 10³", "4 x 10³", "6 x 10³", "8 x 10³"), las = 1)
    box()
    list(x = atd, y = atv)
}, width = 700)

parsed <- function(text) scales::label_parse()(text)
times <- ggplot(data.frame(u, v), aes(u, v)) + geom_point(colour = "#17becf", size = 2.5) +
    scale_x_continuous(breaks = atu, labels = parsed(c("0", "2 %*% 10^5", "4 %*% 10^5", "6 %*% 10^5"))) +
    scale_y_continuous(breaks = atv, labels = parsed(c("0", "2 %*% 10^3", "4 %*% 10^3", "6 %*% 10^3", "8 %*% 10^3")))
gg("gg-times-bw", times + theme_bw(), u, v)
gg("gg-times", times + theme_classic(), u, v, dpi = 150)

# --- ggplot2's default theme: a gray panel with white grid lines and no axis lines, the ticks at its edges ---

a <- runif(N, 2, 98)
b <- runif(N, 3, 47)
gray <- ggplot(data.frame(a, b), aes(a, b)) + geom_point(colour = "#d62728", size = 2.5)
gg("gg-gray", gray, a, b)
gg("gg-gray-small", gray, a, b, dpi = 72)
# ggsave's own resolution, 300 dpi
gg("gg-gray-hi", gray, a, b, width = 5, height = 4, dpi = 300)
gg("gg-gray-log-e", ggplot(data.frame(x, y), aes(x, y)) + geom_point(colour = "#2ca02c", size = 2.5) + scale_x_log10() + scale_y_log10(),
    x, y, xlog = TRUE, ylog = TRUE)
gg("gg-gray-times", times, u, v)
