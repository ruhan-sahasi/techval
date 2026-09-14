/*
 * Section renderer: propensity.
 *
 * Layout carries the argument, so the figures are drawn in a fixed order rather
 * than the snapshot's (which is sorted by key): the sample tiles, then the
 * verdict and what moves it, then the coefficients that will not hold still,
 * then the cycle and the calibration, then the top of the list and the list
 * itself. Five figures need marks the kit does not draw and are drawn here
 * with TV.svg, TV.scale, TV.tooltip and TV.tableView:
 *
 *   auc_by_fold,       dumbbells whose gap label sits beside the dot it
 *   recovered_deals,   describes, on the rows the collector chose; calibration
 *   calibration        adds hollow thin buckets and a row count per bucket;
 *   coefficient_signs  a sign grid with feature names in the mono face at full
 *                      length, rows grouped by whether their sign changes, and
 *                      colour classes on fixed breaks rather than a linear bound;
 *   target_list        the ranked names as a table, which is how a list is read.
 *
 * Each keeps the spec shape of the kit chart its kind names, so the kit's
 * default loop would still draw it truthfully.
 *
 * Every number drawn is in the snapshot; this file formats and places them.
 */
(function (TV) {
  "use strict";

  var el = TV.el;
  var svg = TV.svg;
  var GAP = 2;
  var HIT_MIN = 24;

  var PAIRS = [
    ["auc_by_fold", "recovered_deals"],
    ["coefficient_signs"],
    ["base_rate_by_year", "calibration"],
    ["precision_at_k", "recall_at_k"],
    ["target_list"],
  ];
  var WIDE = { coefficient_signs: true, target_list: true };

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  /* A chart container redrawn when its width changes, as the kit's own charts are. */
  function frame(body, kind, draw) {
    var wrap = el("div", { class: "tv-chart tv-chart--" + kind });
    body.appendChild(wrap);
    var last = -1;
    function render(force) {
      var width = Math.floor(wrap.clientWidth);
      if (!width) return;
      if (!force && width === last) return;
      last = width;
      TV.clear(wrap);
      draw(wrap, Math.max(width, 260));
    }
    render(true);
    if (typeof ResizeObserver !== "undefined") {
      var pending = false;
      new ResizeObserver(function () {
        if (pending) return;
        pending = true;
        window.requestAnimationFrame(function () {
          pending = false;
          render(false);
        });
      }).observe(wrap);
    }
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () {
        render(true);
      });
    }
    return wrap;
  }

  var monoCtx = null;

  /* TV.measure reads the sans face; identifiers here are set in mono. */
  function measureMono(text, size) {
    if (!monoCtx) monoCtx = document.createElement("canvas").getContext("2d");
    monoCtx.font = (size || 12) + "px " + (TV.token("--font-mono") || "monospace");
    return Math.ceil(monoCtx.measureText(String(text)).width * 1.04) + 1;
  }

  function markGroup(parent, hit, marks, label) {
    var g = svg("g", { class: "tv-markg", tabindex: "0", "aria-label": label });
    g.appendChild(svg("rect", Object.assign({ class: "tv-hit" }, hit)));
    marks.forEach(function (m) {
      g.appendChild(m);
    });
    parent.appendChild(g);
    return g;
  }

  /* A caution under a figure: a warning glyph, not the refusal chip, since the figure stands. */
  function caution(handle, text) {
    handle.notes.appendChild(
      el("div", { class: "tv-note", role: "note" }, TV.chip.icon("inside_noise"), el("p", null, text))
    );
  }

  function card(grid, data, id) {
    var f = data.figures[id];
    var prov = (data.provenance || []).filter(function (p) {
      return p && p.figure === id;
    })[0];
    return TV.figure(grid, {
      id: id,
      anchor: "fig-" + data.id + "-" + id,
      title: f.title,
      subtitle: f.subtitle,
      provenance: prov,
      wide: !!WIDE[id],
    });
  }

  function kitChart(handle, f) {
    var chart = TV.charts[f.kind];
    if (!chart) {
      handle.addNote({ what: handle.title, why: "The chart kit has no chart named " + JSON.stringify(String(f.kind)) + "." });
      return;
    }
    chart(handle.body, Object.assign({ title: f.title }, f.data || {}));
    /* An explicit empty legend means the bars are labelled; drop the empty list it leaves. */
    if (handle.legend && !handle.legend.querySelector("li")) TV.clear(handle.legend);
  }

  /* coefficient_signs ------------------------------------------------------ */

  function signClass(v, breaks) {
    if (!isNum(v)) return null;
    if (v === 0) return 0;
    var a = Math.abs(v);
    var k = a < breaks[0] ? 1 : a < breaks[1] ? 2 : 3;
    return v > 0 ? k : -k;
  }

  function signFill(k) {
    if (k === null) return "var(--missing)";
    if (k === 0) return "var(--div-0)";
    return k > 0 ? "var(--div-pos-" + k + ")" : "var(--div-neg-" + -k + ")";
  }

  function signGrid(handle, spec) {
    var rows = spec.rows || [];
    var cols = spec.cols || [];
    var values = spec.values || [];
    var breaks = spec.breaks || [0.1, 0.5];
    var groups = spec.groups || [{ label: "", count: rows.length }];

    TV.tableView(handle, {
      columns: [{ key: "feature", label: spec.rowHeader || "Feature", mono: true }]
        .concat(
          cols.map(function (c, j) {
            return { key: "c" + j, label: c, align: "right", format: spec.format };
          })
        )
        .concat([
          { key: "mean", label: "Mean", align: "right", format: "signed:2" },
          { key: "sd", label: "SD", align: "right", format: "num:2" },
          { key: "flips", label: "Changes sign" },
        ]),
      rows: rows.map(function (r, i) {
        var out = { feature: r, mean: (spec.mean || [])[i], sd: (spec.sd || [])[i], flips: (spec.flips || [])[i] ? "yes" : "no" };
        cols.forEach(function (_, j) {
          out["c" + j] = values[i] ? values[i][j] : null;
        });
        return out;
      }),
    });

    frame(handle.body, "heat", function (wrap, W) {
      var labelW = Math.max.apply(
        null,
        [40].concat(
          rows.map(function (r) {
            return measureMono(r, 12);
          })
        )
      );
      var annotIndex = rows.indexOf(spec.annotate);
      var annotText =
        annotIndex >= 0
          ? "mean " + TV.format("signed:2", spec.mean[annotIndex]) + ", sd " + TV.format("num:2", spec.sd[annotIndex])
          : "";
      var annotW = annotText ? TV.measure(annotText, 11, 500) + 12 : 0;
      var left = labelW + 12;
      var cellW = Math.min(88, Math.max(HIT_MIN, (W - left - annotW - 4) / Math.max(1, cols.length)));
      if (annotText && left + cellW * cols.length + annotW > W) {
        annotText = "";
        cellW = Math.min(88, Math.max(HIT_MIN, (W - left - 4) / Math.max(1, cols.length)));
      }
      var cellH = HIT_MIN;
      var headH = 22;
      var groupH = 24;
      var H = headH + groups.length * groupH + rows.length * cellH + 4;
      var gridW = cellW * cols.length;
      var root = svg("svg", {
        class: "tv-svg",
        width: Math.max(W, left + gridW + 4),
        height: H,
        viewBox: "0 0 " + Math.max(W, left + gridW + 4) + " " + H,
        role: "img",
        "aria-label": spec.title || "Coefficient signs",
      });

      cols.forEach(function (c, j) {
        root.appendChild(
          svg("text", { class: "tv-tick", x: left + cellW * j + cellW / 2, y: headH - 8, "text-anchor": "middle" }, String(c))
        );
      });

      var marks = svg("g");
      var y = headH;
      var i = 0;
      groups.forEach(function (group) {
        if (group.label) {
          marks.appendChild(
            svg("text", { class: "tv-tick", x: 0, y: y + groupH - 8, style: { fontWeight: "500" } }, group.label)
          );
          marks.appendChild(
            svg("line", { class: "tv-axisline", x1: 0, x2: left + gridW, y1: crisp(y + groupH - 2), y2: crisp(y + groupH - 2) })
          );
        }
        y += groupH;
        for (var n = 0; n < group.count && i < rows.length; n++, i++) {
          drawRow(marks, i, y, left, cellW, cellH, annotText, root);
          y += cellH;
        }
      });
      root.appendChild(marks);
      wrap.appendChild(root);

      /* The scale: six classes, since a coefficient of exactly zero does not occur. */
      var classes = [-3, -2, -1, 1, 2, 3];
      wrap.appendChild(
        el(
          "div",
          { class: "tv-scale" },
          el(
            "div",
            { class: "tv-scale__ramp" },
            el(
              "span",
              { class: "tv-scale__title" },
              (spec.scaleLabel || "Value") + ", classes break at " + TV.format("num:1", breaks[0]) + " and " + TV.format("num:1", breaks[1])
            ),
            el(
              "div",
              { class: "tv-scale__swatches", "aria-hidden": "true" },
              classes.map(function (k) {
                return el("span", { class: "tv-scale__swatch", style: { background: signFill(k) } });
              })
            ),
            el(
              "div",
              { class: "tv-scale__ticks" },
              el("span", null, TV.format("signed:1", -breaks[1]) + " or less"),
              el("span", null, "0"),
              el("span", null, TV.format("signed:1", breaks[1]) + " or more")
            )
          )
        )
      );
    });

    function drawRow(parent, i, top, left, cellW, cellH, annotText) {
      var feature = rows[i];
      var cy = top + cellH / 2;
      parent.appendChild(
        svg("text", { class: "tv-label", x: 0, y: cy, dy: "0.35em", style: { fontFamily: "var(--font-mono)" } }, feature)
      );
      cols.forEach(function (c, j) {
        var v = values[i] ? values[i][j] : null;
        var k = signClass(v, breaks);
        var x = left + cellW * j;
        var cell = svg("rect", {
          class: "tv-mark",
          x: x + GAP / 2,
          y: top + GAP / 2,
          width: Math.max(0, cellW - GAP),
          height: Math.max(0, cellH - GAP),
          rx: 2,
          style: { fill: signFill(k) },
        });
        var g = markGroup(parent, { x: x, y: top, width: cellW, height: cellH }, [cell], feature + ", " + c + ": " + TV.format(spec.format, v));
        TV.tooltip.attach(g, function () {
          return {
            title: feature + " · " + c,
            rows: [
              { label: spec.valueLabel || "Value", value: TV.format(spec.format, v), color: signFill(k), shape: "rect" },
              { label: "Mean across fits", value: TV.format("signed:2", spec.mean[i]) },
              { label: "SD across fits", value: TV.format("num:2", spec.sd[i]) },
            ],
          };
        });
      });
      if (annotText && feature === spec.annotate) {
        parent.appendChild(
          svg("text", { class: "tv-value", x: left + cellW * cols.length + 10, y: cy, dy: "0.35em" }, annotText)
        );
      }
    }
  }

  /* Dumbbells: auc_by_fold, recovered_deals, calibration ------------------- */

  function legendList(items) {
    return el(
      "ul",
      { class: "tv-legend" },
      items.map(function (item) {
        return el(
          "li",
          { class: "tv-legend__item" },
          el("span", { class: "tv-key tv-key--dot", style: item.style, "aria-hidden": "true" }),
          el("span", null, item.label)
        );
      })
    );
  }

  /* A gap in probability, as points: +30.5 pts. */
  function points(v) {
    return isNum(v) ? TV.fmt.signed(v * 100, 1) + " pts" : "n/a";
  }

  function gapText(spec, v) {
    return spec.gapFormat === "points" ? points(v) : TV.format(spec.gapFormat || "signed:3", v);
  }

  /* Tick labels with only the decimals the step needs, in the chart's unit. */
  function tickLabels(fmt, ticks) {
    var name = String(fmt || "num").split(":")[0];
    var scaleBy = name === "pct" ? 100 : 1;
    var dp = 0;
    ticks.forEach(function (t) {
      var text = String(parseFloat((t * scaleBy).toPrecision(12)));
      var dot = text.indexOf(".");
      if (dot >= 0) dp = Math.max(dp, text.length - dot - 1);
    });
    return function (v) {
      return name === "pct" ? TV.fmt.pct(v, dp) : TV.fmt.num(v, dp);
    };
  }

  function wrapLabel(text, maxWidth) {
    var words = String(text).split(/\s+/);
    var lines = [];
    var line = "";
    words.forEach(function (word) {
      var next = line ? line + " " + word : word;
      if (!line || TV.measure(next, 12) <= maxWidth) line = next;
      else {
        lines.push(line);
        line = word;
      }
    });
    if (line) lines.push(line);
    if (lines.length > 2) lines = [lines[0], lines.slice(1).join(" ")];
    return lines;
  }

  /*
   * Two series per row on one axis, joined by a hairline. The row's gap (first
   * series less second, computed by the collector) is labelled only on rows
   * the collector flagged, beside the rightmost dot. Rows flagged thin are
   * drawn hollow with muted labels, and rows carrying a count get a count
   * column when the spec asks for one.
   */
  function dumbbell(handle, spec) {
    var rows = spec.rows || [];
    var series = spec.series || [];
    var a = series[0];
    var b = series[1];
    var fmt = spec.format || "num:3";
    var anyThin = rows.some(function (r) {
      return r.thin;
    });

    TV.clear(handle.legend);
    var keys = series.map(function (s) {
      return { label: s.name, style: { "--key": TV.color(s.role) } };
    });
    if (anyThin) {
      keys.push({
        label: "Fewer than " + spec.thin_below + " rows",
        style: { background: "transparent", boxShadow: "inset 0 0 0 1.5px var(--ink-3)" },
      });
    }
    handle.legend.appendChild(legendList(keys));

    var columns = [
      { key: "label", label: spec.labelHeader || "Row" },
      { key: "a", label: a.name, align: "right", format: spec.tableFormat || fmt },
      { key: "b", label: b.name, align: "right", format: spec.tableFormat || fmt },
      {
        key: "gap",
        label: spec.gapLabel || "Difference",
        align: "right",
        format: function (v) {
          return gapText(spec, v);
        },
      },
    ];
    if (spec.showCounts) {
      columns.push({ key: "n", label: "Rows", align: "right", format: "int" });
      if (anyThin) columns.push({ key: "thin", label: "Thin" });
    }
    TV.tableView(handle, {
      columns: columns,
      rows: rows.map(function (r) {
        return {
          label: r.label,
          a: r.values[a.key],
          b: r.values[b.key],
          gap: r.gap,
          n: r.n,
          thin: r.thin ? "yes" : "no",
        };
      }),
    });

    frame(handle.body, "dot", function (wrap, W) {
      var maxLabel = Math.min(Math.max(90, W * 0.36), 230);
      var wrapped = rows.map(function (r) {
        return wrapLabel(r.label, maxLabel);
      });
      var labelW = Math.max.apply(
        null,
        [40].concat(
          wrapped.map(function (lines) {
            return Math.max.apply(
              null,
              lines.map(function (l) {
                return TV.measure(l, 12);
              })
            );
          })
        )
      );
      var countText = rows.map(function (r) {
        return TV.fmt.int(r.n) + (r.thin ? ", thin" : "");
      });
      var countW = spec.showCounts
        ? Math.max.apply(
            null,
            [TV.measure("Rows", 11)].concat(
              countText.map(function (t) {
                return TV.measure(t, 11);
              })
            )
          ) + 20
        : 0;
      var refs = (spec.reference || []).filter(function (r) {
        return r && isNum(r.value);
      });
      var gapW = Math.max.apply(
        null,
        [0].concat(
          rows.map(function (r) {
            return r.labelled ? TV.measure(gapText(spec, r.gap), 11, 500) + 12 : 0;
          })
        )
      );
      var m = {
        top: spec.showCounts || refs.length ? 24 : 8,
        right: countW + gapW + 8,
        bottom: 26,
        left: labelW + 16,
      };
      var heights = wrapped.map(function (lines) {
        return lines.length > 1 ? 40 : 28;
      });
      var plotH = heights.reduce(function (s, h) {
        return s + h;
      }, 0);
      var H = m.top + plotH + m.bottom;
      var all = [];
      rows.forEach(function (r) {
        all.push(r.values[a.key], r.values[b.key]);
      });
      refs.forEach(function (r) {
        all.push(r.value);
      });
      all = all.filter(isNum);
      var dom = spec.domain || [Math.min.apply(null, all), Math.max.apply(null, all)];
      /* The domain is niced to the same tick count it is labelled with, or a narrow domain keeps one tick. */
      var tickCount = Math.max(2, Math.floor((W - m.left - m.right) / 70));
      var x = TV.scale.linear(dom, [m.left + 6, W - m.right], { nice: !spec.domain, ticks: tickCount });
      var root = svg("svg", {
        class: "tv-svg",
        width: W,
        height: H,
        viewBox: "0 0 " + W + " " + H,
        role: "img",
        "aria-label": spec.title || "Dumbbell chart",
      });

      var grid = svg("g");
      var ticks = x.ticks(tickCount);
      var tickText = tickLabels(fmt, ticks);
      ticks.forEach(function (t) {
        var px = crisp(x(t));
        grid.appendChild(svg("line", { class: "tv-gridline", x1: px, x2: px, y1: m.top, y2: m.top + plotH }));
        grid.appendChild(svg("text", { class: "tv-tick", x: x(t), y: m.top + plotH + 16, "text-anchor": "middle" }, tickText(t)));
      });
      if (spec.showCounts) {
        grid.appendChild(svg("text", { class: "tv-tick", x: W, y: m.top - 8, "text-anchor": "end" }, "Rows"));
      }
      refs.forEach(function (ref) {
        var px = crisp(x(ref.value));
        grid.appendChild(svg("line", { class: "tv-ref", x1: px, x2: px, y1: m.top - 4, y2: m.top + plotH }));
        grid.appendChild(
          svg("text", { class: "tv-ref-label", x: px, y: m.top - 10, "text-anchor": "middle" }, ref.label || TV.format(fmt, ref.value))
        );
      });
      root.appendChild(grid);

      var marks = svg("g");
      var labels = svg("g", { class: "tv-values" });
      var top = m.top;
      rows.forEach(function (r, i) {
        var rowH = heights[i];
        var cy = top + rowH / 2;
        var va = r.values[a.key];
        var vb = r.values[b.key];
        var muted = r.thin ? { fill: "var(--ink-3)" } : null;
        var text = svg("text", { class: "tv-label", x: labelW + 2, y: cy, "text-anchor": "end", style: muted });
        wrapped[i].forEach(function (line, k) {
          var dy = k === 0 ? (wrapped[i].length > 1 ? "-0.25em" : "0.35em") : "1.2em";
          text.appendChild(svg("tspan", { x: labelW + 2, dy: dy }, line));
        });
        marks.appendChild(text);
        if (spec.showCounts) {
          marks.appendChild(svg("text", { class: "tv-tick", x: W, y: cy, dy: "0.35em", "text-anchor": "end", style: muted }, countText[i]));
        }
        var parts = [];
        if (isNum(va) && isNum(vb)) {
          parts.push(svg("line", { class: "tv-hairline", x1: x(Math.min(va, vb)), x2: x(Math.max(va, vb)), y1: crisp(cy), y2: crisp(cy) }));
        }
        /* The first series is drawn last and a half pixel larger, so it wins a tie. */
        [
          [vb, b.role, 4],
          [va, a.role, 4.5],
        ].forEach(function (d) {
          if (!isNum(d[0])) return;
          var paint = TV.color(d[1]);
          parts.push(
            svg("circle", {
              class: r.thin ? "tv-mark" : "tv-mark tv-dot",
              cx: x(d[0]),
              cy: cy,
              r: r.thin ? d[2] - 0.75 : d[2],
              style: r.thin ? { fill: "var(--surface)", stroke: paint, strokeWidth: "1.5" } : { fill: paint },
            })
          );
        });
        var g = markGroup(
          marks,
          { x: 0, y: top, width: W, height: rowH },
          parts,
          r.label + ": " + a.name + " " + TV.format(fmt, va) + ", " + b.name + " " + TV.format(fmt, vb)
        );
        if (r.labelled && isNum(va) && isNum(vb)) {
          labels.appendChild(
            svg("text", { class: "tv-value", x: x(Math.max(va, vb)) + 9, y: cy, dy: "0.35em" }, gapText(spec, r.gap))
          );
        }
        TV.tooltip.attach(g, function () {
          var out = [
            { label: a.name, value: TV.format(spec.tableFormat || fmt, va), color: a.role, shape: "dot" },
            { label: b.name, value: TV.format(spec.tableFormat || fmt, vb), color: b.role, shape: "dot" },
            { label: spec.gapLabel || "Difference", value: gapText(spec, r.gap) },
          ];
          if (isNum(r.n)) out.push({ label: "Rows", value: TV.fmt.int(r.n) + (r.thin ? ", thin" : "") });
          return { title: r.label, rows: out };
        });
        top += rowH;
      });
      root.appendChild(marks);
      root.appendChild(labels);
      wrap.appendChild(root);
    });
  }

  /* target_list ------------------------------------------------------------ */

  function targetTable(handle, spec) {
    TV.tableView(
      { table: handle.body },
      {
        caption: handle.title,
        columns: [
          { key: "rank", label: "Rank", align: "right" },
          { key: "ticker", label: "Ticker", mono: true },
          { key: "name", label: "Company" },
          { key: "sub_vertical", label: "Sub-vertical" },
          { key: "probability", label: "Stated probability", align: "right", format: spec.format || "pct:1" },
          { key: "drivers", label: "Largest contributions", mono: true },
        ],
        rows: spec.table || [],
      }
    );
  }

  /* The section ------------------------------------------------------------ */

  var CUSTOM = {
    auc_by_fold: dumbbell,
    recovered_deals: dumbbell,
    coefficient_signs: signGrid,
    calibration: dumbbell,
    target_list: targetTable,
  };

  TV.sections.register("propensity", function (root, data) {
    var figures = (data && data.figures) || {};
    var handles = {};

    var sample = figures.sample;
    if (sample) {
      var set = el(
        "div",
        { class: "tv-tileset", "data-figure-id": "sample" },
        sample.title ? el("h3", { class: "tv-tileset__title" }, sample.title) : null
      );
      root.appendChild(set);
      TV.charts.tiles(set, sample.data || {});
      handles.sample = {
        root: set,
        body: set,
        title: sample.title || "",
        addNote: function (r) {
          set.appendChild(TV.refusalNote(r));
        },
      };
    }

    var placed = { sample: true };
    var grid = el("div", { class: "tv-grid" });
    root.appendChild(grid);

    function draw(id) {
      var f = figures[id];
      placed[id] = true;
      if (!f) return;
      var handle = card(grid, data, id);
      handles[id] = handle;
      if (CUSTOM[id]) CUSTOM[id](handle, Object.assign({ title: f.title }, f.data || {}));
      else kitChart(handle, f);
      ((f.data && f.data.notes) || []).forEach(function (text) {
        caution(handle, text);
      });
    }

    PAIRS.forEach(function (pair) {
      pair.forEach(draw);
    });
    /* A figure this layout does not know is still drawn, after the rest. */
    Object.keys(figures)
      .sort()
      .forEach(function (id) {
        if (!placed[id]) draw(id);
      });
    if (!grid.firstChild) root.removeChild(grid);
    return handles;
  });
})(window.TV);
