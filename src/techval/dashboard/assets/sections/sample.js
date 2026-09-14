/*
 * Section renderer: sample.
 *
 * Three figures, in the order the argument runs: the counts each model is
 * quoted on against the counts that limit it, how far back the record reaches,
 * and the limits no amount of care downstream fixes.
 *
 *   counts   a ladder per model on a log scale, drawn here because the kit's
 *            dot chart is linear and these counts span three orders of
 *            magnitude. Each model is a group; each row names its unit, since
 *            the unit changes from row to row. A thin rule runs from each
 *            smaller count back to the model's largest, so the drop is read as
 *            a distance. The limiting count is orange, the rest grey. A count
 *            the collector refused keeps its row, marked refused, and its note
 *            hangs under the figure.
 *   record   the kit's hbar, with the collector's table in place of the kit's
 *            own so the dates and sources behind each span are one click away.
 *   limits   a numbered list, with a table of every number it quotes and the
 *            field each was read from.
 *
 * The data keeps the shape of the kit chart each kind names where there is one,
 * so the default loop would still draw it truthfully. Every number drawn is in
 * the snapshot; this file formats and places them.
 */
(function (TV) {
  "use strict";

  var el = TV.el;
  var svg = TV.svg;
  var HIT_MIN = 24;

  var ORDER = ["counts", "record", "limits"];

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

  function provenanceFor(data, id) {
    return ((data && data.provenance) || []).filter(function (p) {
      return p && p.figure === id;
    })[0];
  }

  function card(grid, data, id, f) {
    return TV.figure(grid, {
      id: id,
      anchor: "fig-" + (data.id || "sample") + "-" + id,
      title: f.title,
      subtitle: f.subtitle,
      provenance: provenanceFor(data, id),
      wide: !!(f.wide || (f.data && f.data.wide)),
    });
  }

  function wrapLines(text, maxWidth, size, weight) {
    var words = String(text || "").split(/\s+/).filter(Boolean);
    var lines = [];
    var line = "";
    words.forEach(function (word) {
      var next = line ? line + " " + word : word;
      if (!line || TV.measure(next, size, weight) <= maxWidth) {
        line = next;
      } else {
        lines.push(line);
        line = word;
      }
    });
    if (line) lines.push(line);
    return lines.length ? lines : [""];
  }

  function textLines(parent, lines, x, y, attrs, lineHeight) {
    var node = svg("text", Object.assign({ x: x, y: y - ((lines.length - 1) * lineHeight) / 2 }, attrs));
    lines.forEach(function (line, i) {
      node.appendChild(svg("tspan", { x: x, dy: i === 0 ? "0.35em" : lineHeight }, line));
    });
    parent.appendChild(node);
    return node;
  }

  /* counts ----------------------------------------------------------------- */

  /* Powers of ten either side of the counts, so every tick is a round order of magnitude. */
  function logDomain(values) {
    var lo = Math.min.apply(null, values);
    var hi = Math.max.apply(null, values);
    var k0 = Math.floor(Math.log10(lo));
    var k1 = Math.ceil(Math.log10(hi));
    if (k1 === k0) k1 += 1;
    return [k0, k1];
  }

  function countsChart(handle, spec) {
    var rows = (spec.rows || []).filter(Boolean);
    var groups = (spec.groups || []).filter(Boolean);
    var fmt = spec.format || "num:0";

    if (spec.table) TV.tableView(handle, spec.table);
    TV.legend(
      handle.legend,
      (spec.legend || []).map(function (item) {
        return { label: item.label, color: item.role, shape: "dot" };
      })
    );

    var values = rows
      .map(function (r) {
        return r.values ? r.values.count : null;
      })
      .filter(function (v) {
        return isNum(v) && v > 0;
      });
    if (!values.length) {
      handle.body.appendChild(el("p", { class: "tv-empty" }, "No count could be read, so there is nothing to draw."));
      return;
    }

    var nominalOf = {};
    rows.forEach(function (r) {
      if (r.step === "nominal" && r.values && isNum(r.values.count)) nominalOf[r.group] = r.values.count;
    });

    frame(handle.body, "dot", function (wrap, W) {
      var exps = logDomain(values);
      var narrow = W < 560;
      var labelMax = narrow ? Math.max(110, W * 0.42) : Math.min(300, W * 0.34);
      var layout = [];
      groups.forEach(function (g) {
        layout.push({ type: "group", group: g, height: 30 });
        rows
          .filter(function (r) {
            return r.group === g.key;
          })
          .forEach(function (r) {
            var lines = wrapLines(r.label, labelMax, 12);
            layout.push({ type: "row", row: r, lines: lines, height: Math.max(HIT_MIN + 2, lines.length * 14 + 10) });
          });
      });
      var labelW = Math.min(
        labelMax,
        Math.max.apply(
          null,
          [60].concat(
            layout
              .filter(function (item) {
                return item.type === "row";
              })
              .map(function (item) {
                return Math.max.apply(
                  null,
                  item.lines.map(function (line) {
                    return TV.measure(line, 12);
                  })
                );
              })
          )
        )
      );
      var valueW = TV.measure("10,000", 11, 500) + 10;
      var m = { top: 4, right: 12, bottom: 44, left: labelW + 16 + valueW };
      var plotH = layout.reduce(function (sum, item) {
        return sum + item.height;
      }, 0);
      var H = Math.round(m.top + plotH + m.bottom);
      var lx = TV.scale.linear([exps[0], exps[1]], [m.left, W - m.right - valueW]);
      var x = function (v) {
        return lx(Math.log10(v));
      };

      var root = svg("svg", {
        class: "tv-svg",
        width: W,
        height: H,
        viewBox: "0 0 " + W + " " + H,
        role: "img",
        "aria-label": spec.title || "Counts by model on a log scale",
      });

      var grid = svg("g");
      for (var k = exps[0]; k <= exps[1]; k++) {
        var v = Math.pow(10, k);
        var px = crisp(x(v));
        grid.appendChild(svg("line", { class: "tv-gridline", x1: px, x2: px, y1: m.top, y2: m.top + plotH }));
        grid.appendChild(svg("text", { class: "tv-tick", x: px, y: m.top + plotH + 16, "text-anchor": "middle" }, TV.format("num:0", v)));
      }
      grid.appendChild(
        svg("text", { class: "tv-tick", x: lx(exps[0]), y: m.top + plotH + 34, "text-anchor": "start" }, "Count, log scale: each gridline is ten times the one before")
      );
      root.appendChild(grid);

      var marks = svg("g");
      var values2 = svg("g", { class: "tv-values" });
      var y = m.top;
      layout.forEach(function (item, i) {
        var cy = y + item.height / 2;
        if (item.type === "group") {
          var g = item.group;
          if (i > 0) marks.appendChild(svg("line", { class: "tv-gridline", x1: 0, x2: W, y1: crisp(y), y2: crisp(y) }));
          marks.appendChild(svg("text", { class: "tv-label", x: 0, y: cy + 2, dy: "0.35em", style: { fontWeight: "var(--fw-semibold)" } }, g.label));
          if (g.ratioText) {
            marks.appendChild(svg("text", { class: "tv-tick", x: W - m.right, y: cy + 2, dy: "0.35em", "text-anchor": "end" }, g.ratioText));
          }
        } else {
          var r = item.row;
          var count = r.values ? r.values.count : null;
          var nominal = nominalOf[r.group];
          textLines(marks, item.lines, labelW, cy, { class: "tv-label", "text-anchor": "end" }, 14);
          var parts = [];
          if (isNum(count) && count > 0) {
            var cx = x(count);
            if (r.step !== "nominal" && isNum(nominal) && nominal > count) {
              parts.push(svg("line", { class: "tv-connector", x1: cx, x2: x(nominal), y1: crisp(cy), y2: crisp(cy) }));
            }
            parts.push(svg("circle", { class: "tv-dot tv-mark", cx: cx, cy: cy, r: 4, style: { fill: TV.color(r.role || "baseline") } }));
            var label = TV.format(fmt, count);
            var right = r.step === "nominal";
            values2.appendChild(
              svg(
                "text",
                {
                  class: "tv-value",
                  x: right ? cx + 8 : cx - 8,
                  y: cy,
                  dy: "0.35em",
                  "text-anchor": right ? "start" : "end",
                  style: r.step === "limiting" ? { fill: "var(--ink-1)" } : null,
                },
                label
              )
            );
          } else {
            values2.appendChild(svg("text", { class: "tv-value", x: m.left, y: cy, dy: "0.35em", "text-anchor": "start" }, "Refused"));
          }
          var group = groups.filter(function (gg) {
            return gg.key === r.group;
          })[0];
          var hit = svg("g", {
            class: "tv-markg",
            tabindex: "0",
            "aria-label": (group ? group.label + ", " : "") + r.label + ": " + (isNum(count) ? TV.format("auto", count) : "refused"),
          });
          hit.appendChild(svg("rect", { class: "tv-hit", x: 0, y: y, width: W, height: item.height }));
          parts.forEach(function (p) {
            hit.appendChild(p);
          });
          marks.appendChild(hit);
          TV.tooltip.attach(hit, function () {
            var out = [
              {
                label: r.label,
                value: isNum(count) ? TV.format("auto", count) : "refused",
                color: isNum(count) ? r.role || "baseline" : null,
                shape: "dot",
              },
            ];
            if (isNum(count) && isNum(nominal) && r.step !== "nominal") {
              out.push({ label: "Largest count over this", value: TV.format("num:1", nominal / count) });
            }
            if (r.step === "limiting" && group && group.why) out.push({ label: group.why, value: "" });
            out.push({ label: "Read from " + r.source, value: "" });
            return { title: group ? group.label : r.label, rows: out };
          });
        }
        y += item.height;
      });
      root.appendChild(marks);
      root.appendChild(values2);
      wrap.appendChild(root);
    });

    /* Why each limiting count limits, in words under the chart, since a tooltip is not the only way in. */
    var reasons = groups.filter(function (g) {
      return g.why;
    });
    if (reasons.length) {
      handle.body.appendChild(
        el(
          "dl",
          {
            class: "tv-sample-why",
            style: {
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))",
              gap: "var(--space-2) var(--space-5)",
              margin: "var(--space-4) 0 0",
              fontSize: "var(--fs-small)",
              color: "var(--ink-2)",
            },
          },
          reasons.map(function (g) {
            return el(
              "div",
              null,
              el("dt", { style: { color: "var(--ink-1)", fontWeight: "var(--fw-medium)" } }, g.label),
              el("dd", { style: { margin: "0" } }, g.why + ".")
            );
          })
        )
      );
    }
  }

  /* record ----------------------------------------------------------------- */

  function recordChart(handle, f) {
    var spec = Object.assign({ title: f.title }, f.data || {});
    TV.charts.hbar(handle.body, spec);
    if (spec.table) {
      TV.clear(handle.table);
      TV.tableView(handle, spec.table);
    }
    if (handle.legend && !handle.legend.querySelector("li")) TV.clear(handle.legend);
  }

  /* limits ----------------------------------------------------------------- */

  function limitsList(handle, spec) {
    var items = (spec.items || []).filter(Boolean);
    if (spec.table && (spec.table.rows || []).length) TV.tableView(handle, spec.table);
    var list = el("ol", {
      class: "tv-sample-limits",
      style: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 340px), 1fr))",
        gap: "var(--space-4) var(--space-5)",
        margin: "0",
        padding: "0",
        listStyle: "none",
      },
    });
    items.forEach(function (item, i) {
      list.appendChild(
        el(
          "li",
          { style: { display: "grid", gridTemplateColumns: "auto 1fr", gap: "var(--space-3)", alignItems: "baseline" } },
          el("span", { class: "tv-mono", "aria-hidden": "true", style: { color: "var(--ink-3)", fontSize: "var(--fs-small)" } }, String(i + 1)),
          el(
            "div",
            null,
            el("p", { style: { margin: "0", fontWeight: "var(--fw-semibold)", color: "var(--ink-1)" } }, item.what),
            el("p", { style: { margin: "2px 0 0", color: "var(--ink-2)", fontSize: "var(--fs-small)", lineHeight: "var(--lh-body)" } }, item.why)
          )
        )
      );
    });
    handle.body.appendChild(list);
  }

  /* the section ------------------------------------------------------------ */

  TV.sections.register("sample", function (root, data) {
    var figures = (data && data.figures) || {};
    var handles = {};
    var grid = el("div", { class: "tv-grid" });
    root.appendChild(grid);

    ORDER.forEach(function (id) {
      var f = figures[id];
      if (!f) return;
      var handle = card(grid, data, id, f);
      handles[id] = handle;
      if (id === "counts") {
        countsChart(handle, Object.assign({ title: f.title }, f.data || {}));
        /* A refused count names its row; its note belongs under this figure. */
        ((f.data && f.data.rows) || []).forEach(function (r) {
          if (r && r.refusal) handles[r.refusal] = handle;
        });
      } else if (id === "record") {
        recordChart(handle, f);
        ((f.data && f.data.refusals) || []).forEach(function (name) {
          handles[name] = handle;
        });
      } else {
        limitsList(handle, f.data || {});
      }
    });

    /* Anything a later collector adds still reaches the page. */
    var rest = {};
    Object.keys(figures).forEach(function (id) {
      if (ORDER.indexOf(id) < 0) rest[id] = figures[id];
    });
    if (Object.keys(rest).length) {
      var more = TV.sections.figures(root, Object.assign({}, data, { figures: rest }));
      Object.keys(more).forEach(function (id) {
        handles[id] = more[id];
      });
    }
    return handles;
  });
})(window.TV);
