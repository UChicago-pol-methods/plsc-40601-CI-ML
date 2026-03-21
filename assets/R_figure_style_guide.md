# R Figure Generation Style Guide

## Core principles

- Accessibility first: use colorblind-friendly palettes and strong contrast
- Publication ready: export figures at consistent size and resolution
- Clean design: minimize chart junk and keep labels readable
- Reproducibility: define themes, palettes, and seeds up front

## Recommended packages

```r
library(ggplot2)
library(viridis)
library(hrbrthemes)
library(ggthemes)
library(RColorBrewer)
library(ggdist)
library(corrplot)
```

## Default palettes

### Categorical palette

```r
cbPalette <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
)
```

### Extended palette

```r
cbPalette_extended <- c(
  "#125A56", "#00767B", "#238F9D", "#42A7C6",
  "#60BCE9", "#9DCCEF", "#C6DBED", "#DEE6E7",
  "#F0E6B2", "#F9D576", "#FFB954", "#FD9A44",
  "#F57634", "#E94C1F", "#D11807", "#A01813"
)
```

### Continuous scales

```r
scale_fill_viridis(option = "magma", begin = 0.4, end = 0.9)
scale_color_viridis(option = "viridis", begin = 0.2, end = 0.8)
```

## Theme setup

Set a global theme near the top of each script.

```r
theme_set(theme_minimal())
theme_update(
  plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
  text = element_text(size = 12),
  panel.background = element_rect(fill = "white"),
  plot.background = element_rect(fill = "white", color = "white"),
  panel.grid.minor = element_blank()
)
```

If a project needs a single reusable theme, define it explicitly:

```r
custom_theme <- theme(
  plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
  axis.title = element_text(size = 12),
  axis.text = element_text(size = 10),
  legend.text = element_text(size = 10),
  legend.title = element_text(size = 11, face = "bold"),
  panel.grid.major = element_line(color = "grey90", size = 0.25),
  panel.grid.minor = element_blank(),
  panel.background = element_rect(fill = "white"),
  plot.margin = margin(t = 20, r = 20, b = 20, l = 20),
  axis.ticks.length = unit(5, "pt"),
  axis.ticks = element_line(size = 0.5),
  legend.position = "bottom",
  legend.background = element_rect(fill = "white", color = NA)
)
```

## Common patterns

### Bar chart

```r
ggplot(data, aes(x = category, y = value, fill = group)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  scale_fill_manual(values = cbPalette) +
  labs(
    title = "Clear, descriptive title",
    x = "Category",
    y = "Value",
    fill = "Group"
  ) +
  custom_theme
```

### Scatter plot with uncertainty

```r
ggplot(data, aes(x = x_var, y = y_var, color = group)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
  scale_color_manual(values = cbPalette[c(2, 3)]) +
  custom_theme
```

### Interval or uncertainty plot

```r
ggplot(data, aes(x = treatment, y = estimate)) +
  stat_gradientinterval(
    aes(ydist = distributional::dist_normal(estimate, std_error)),
    width = 0.6,
    point_size = 2,
    interval_alpha = 0.8
  ) +
  scale_fill_viridis(option = "viridis", begin = 0.3, end = 0.7) +
  custom_theme
```

## Export settings

Standard export:

```r
ggsave(
  "figures/figure_name.png",
  plot = p,
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)
```

Useful variants:

```r
ggsave("figure.png", width = 10, height = 6, dpi = 300)
ggsave("figure.pdf", width = 10, height = 6)
ggsave("figure_presentation.png", width = 12, height = 7, dpi = 150)
```

## Accessibility checklist

- [ ] Use a colorblind-friendly palette
- [ ] Make titles and axis labels explicit
- [ ] Keep text large enough to read in slides or print
- [ ] Use shapes or line types when color alone is not enough
- [ ] Add alt text in R Markdown when figures are rendered there
- [ ] Set and report a seed for any randomized visualization pipeline

## R Markdown chunk defaults

```r
{r fig-name, fig.width=8, fig.height=6, fig.cap="Caption here", fig.alt="Alt text here", dpi=300}
# plotting code
```
