/*
 * Section renderer: sample.
 *
 * Three figures, in the order the argument runs: the counts each model is
 * quoted on against the counts that limit it, how far back the record reaches,
 * and the limits no amount of care downstream fixes. The kit's figure loop
 * draws the cards, their order, anchors, source lines and refusal routing.
 *
 *   counts   a ladder per model on a log scale. The kit's dot chart is linear
 *            and these counts span three orders of magnitude, so the ladder is
 *            laid out here, but on the kit's frame, log scale, axis, wrapped
 *            labels, mark groups and tooltip. Each model is a group; each row
 *            names its unit, since the unit changes from row to row. A thin rule
 *            runs from each smaller count back to the model's largest, so the
 *            drop is read as a distance. The limiting count is orange, the rest
 *            grey. A count the collector refused keeps its row, marked refused,
 *            and its note hangs under the figure.
 *   record   the kit's hbar as it stands: the collector's fuller table replaces
 *            the chart's own, so the dates and sources behind each span are one
 *            click away, and its refusals are routed by the kit.
 *   limits   a numbered list, which the kit has no kind for, with a table of
 *            every number it quotes and the field each was read from.
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

  /* counts ----------------------------------------------------------------- */

  function countsChart(handle, f) {
    var spec = Object.assign({ title: f.title }, f.data || {});
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

    TV.frame(handle.body, "dot", function (wrap, W) {
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
            var wrapped = TV.wrapText(r.label, labelMax, 12);
            layout.push({
              type: "row",
              row: r,
              lines: wrapped.lines,
              width: wrapped.width,
              height: Math.max(HIT_MIN + 2, wrapped.lines.length * 14 + 10),
            });
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
                return item.width;
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
      var x = TV.scale.log(Math.min.apply(null, values), Math.max.apply(null, values), [m.left, W - m.right - valueW]);
      /* One gridline per power of ten: ask for no more ticks than the domain has decades. */
      var decades = Math.round(Math.log10(x.domain[1] / x.domain[0]));
      var ticks = x.ticks(decades + 1);
      var tenfold = ticks.every(function (t, i) {
        return i === 0 || Math.abs(t / ticks[i - 1] - 10) < 1e-6;
      });

      var root = TV.chartSvg(W, H, { ariaLabel: spec.title || "Counts by model on a log scale" }, "Dot");

      var grid = svg("g");
      TV.axis.x(grid, x, m.top, m.top + plotH, "num:0", decades + 1);
      /* The note sits under the axis where the axis starts, and slides left when the chart is too narrow to hold it there. */
      var axisNote = tenfold && ticks.length > 1 ? "Count, log scale: each gridline is ten times the one before" : "Count, log scale";
      var noteX = Math.max(0, Math.min(x.range[0], W - TV.measure(axisNote, 11) - 2));
      grid.appendChild(
        svg("text", { class: "tv-tick", x: noteX, y: m.top + plotH + 34, "text-anchor": "start" }, axisNote)
      );
      root.appendChild(grid);

      var marks = svg("g");
      var labels = svg("g", { class: "tv-values" });
      var y = m.top;
      layout.forEach(function (item, i) {
        var cy = y + item.height / 2;
        if (item.type === "group") {
          var g = item.group;
          if (i > 0) marks.appendChild(svg("line", { class: "tv-gridline", x1: 0, x2: W, y1: TV.crisp(y), y2: TV.crisp(y) }));
          marks.appendChild(svg("text", { class: "tv-label", x: 0, y: cy + 2, dy: "0.35em", style: { fontWeight: "var(--fw-semibold)" } }, g.label));
          if (g.ratioText) {
            marks.appendChild(svg("text", { class: "tv-tick", x: W - m.right, y: cy + 2, dy: "0.35em", "text-anchor": "end" }, g.ratioText));
          }
        } else {
          var r = item.row;
          var count = r.values ? r.values.count : null;
          var nominal = nominalOf[r.group];
          TV.textBlock(marks, item.lines, labelW, cy, { class: "tv-label", "text-anchor": "end" }, 14);
          var parts = [];
          if (isNum(count) && count > 0) {
            var cx = x(count);
            if (r.step !== "nominal" && isNum(nominal) && nominal > count) {
              parts.push(svg("line", { class: "tv-connector", x1: cx, x2: x(nominal), y1: TV.crisp(cy), y2: TV.crisp(cy) }));
            }
            parts.push(svg("circle", { class: "tv-dot tv-mark", cx: cx, cy: cy, r: 4, style: { fill: TV.color(r.role || "baseline") } }));
            var right = r.step === "nominal";
            labels.appendChild(
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
                TV.format(fmt, count)
              )
            );
          } else {
            labels.appendChild(svg("text", { class: "tv-value", x: m.left, y: cy, dy: "0.35em", "text-anchor": "start" }, "Refused"));
          }
          var group = groups.filter(function (gg) {
            return gg.key === r.group;
          })[0];
          var hit = TV.markGroup(
            marks,
            { x: 0, y: y, width: W, height: item.height },
            parts,
            (group ? group.label + ", " : "") + r.label + ": " + (isNum(count) ? TV.format("auto", count) : "refused")
          );
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
            return { title: group ? group.label : r.label, rows: out };
          });
        }
        y += item.height;
      });
      root.appendChild(marks);
      root.appendChild(labels);
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

  /* limits ----------------------------------------------------------------- */

  function limitsList(handle, f) {
    var spec = f.data || {};
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
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: {
        counts: { draw: countsChart },
        limits: { draw: limitsList },
      },
      /* A refused count names its row; its note belongs under the ladder. */
      refusals: function (id, figure) {
        if (id !== "counts") return [];
        return ((figure.data && figure.data.rows) || [])
          .map(function (r) {
            return r && r.refusal;
          })
          .filter(Boolean);
      },
    });
  });
})(window.TV);
