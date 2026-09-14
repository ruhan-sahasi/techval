/*
 * Section renderer: encoder.
 *
 * The figures are drawn in reading order rather than snapshot key order: the
 * diagnostics tiles, every method on one axis, the paired lift, warm against
 * cold start, the tower ablation and the truncation sweep. Most figures are
 * ordinary kit charts. Two are not, and both are drawn here:
 *
 *   whiskers    a dot with an interval either side, for the paired lift (one
 *               standard error) and the tower ablation (one fold standard
 *               deviation). The kit's dot chart has no interval, and on both
 *               figures the interval is the finding: whether it crosses zero.
 *               The figure's data is still a valid kit dot spec, so a page
 *               without this renderer draws the dots and loses only the whiskers.
 *   table       the truncation sweep draws three lines and tables every method,
 *               which the kit's line chart cannot do without a note saying it
 *               drew fewer series than it listed.
 *
 * Nothing here computes a figure. Every value, interval end and direct label
 * arrives in the snapshot; this file positions and formats them.
 */
(function (TV) {
  "use strict";

  var el = TV.el;
  var svg = TV.svg;

  var ORDER = ["ndcg_by_method", "paired_lift", "warm_cold", "tower_ablation", "truncation"];
  var ROW = 32;
  var GROUP_ROW = 26;

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  /* Tick labels carry the decimals the step needs, in the figure's own format family. */
  function tickText(format, ticks) {
    var name = String(format || "num").split(":")[0];
    var fn = TV.fmt[name] || TV.fmt.num;
    var dp = 0;
    ticks.forEach(function (t) {
      var text = String(parseFloat(Number(t).toPrecision(12)));
      var dot = text.indexOf(".");
      if (dot >= 0 && text.indexOf("e") < 0) dp = Math.max(dp, text.length - dot - 1);
    });
    return function (v) {
      return fn(v, Math.min(dp, 6));
    };
  }

  /* A chart container that redraws when its width changes, as the kit's charts do. */
  function frame(body, kind, draw) {
    var wrap = el("div", { class: "tv-chart tv-chart--" + kind });
    body.appendChild(wrap);
    var last = -1;
    function render(force) {
      var width = Math.floor(wrap.clientWidth);
      if (!width || (!force && width === last)) return;
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

  /* whiskers --------------------------------------------------------------- */

  function whiskers(handle, spec) {
    var rows = (spec.rows || []).filter(Boolean);
    var key = spec.series && spec.series[0] ? spec.series[0].key : "value";
    if (spec.table) TV.tableView(handle, spec.table);
    if (!rows.length) {
      handle.body.appendChild(el("p", { class: "tv-empty" }, "No values to draw."));
      return;
    }
    function valueOf(r) {
      return r.values ? r.values[key] : null;
    }

    frame(handle.body, "whiskers", function (wrap, W) {
      var lines = [];
      var lastGroup = null;
      rows.forEach(function (r) {
        if (r.group && r.group !== lastGroup) {
          lines.push({ group: r.group });
          lastGroup = r.group;
        }
        lines.push({ row: r });
      });

      var labelW = Math.min(
        W * 0.4,
        Math.max.apply(
          null,
          [40].concat(
            rows.map(function (r) {
              return TV.measure(r.label, 12);
            })
          )
        )
      );
      var textW = Math.max.apply(
        null,
        [0].concat(
          rows.map(function (r) {
            return r.text ? TV.measure(r.text, 11, 500) : 0;
          })
        )
      );
      var refs = (spec.reference || []).filter(function (r) {
        return r && isNum(r.value);
      });
      var m = { top: refs.length ? 22 : 8, right: textW + 14, bottom: 26, left: labelW + 16 };
      var plotH = lines.reduce(function (h, line) {
        return h + (line.group ? GROUP_ROW : ROW);
      }, 0);
      var H = Math.round(m.top + plotH + m.bottom);

      var all = [];
      rows.forEach(function (r) {
        all.push(valueOf(r), r.lo, r.hi);
      });
      refs.forEach(function (r) {
        all.push(r.value);
      });
      var vals = all.filter(isNum);
      var lo = Math.min.apply(null, spec.zero ? [0].concat(vals) : vals);
      var hi = Math.max.apply(null, spec.zero ? [0].concat(vals) : vals);
      var x = TV.scale.linear([lo, hi], [m.left, W - m.right], { nice: true });

      var root = svg("svg", {
        class: "tv-svg",
        width: W,
        height: H,
        viewBox: "0 0 " + W + " " + H,
        role: "img",
        "aria-label": spec.title || "Interval chart",
      });

      var grid = svg("g");
      var ticks = x.ticks(Math.max(2, Math.floor((W - m.left - m.right) / 64)));
      var fmtTick = tickText(spec.format, ticks);
      var lastRight = -Infinity;
      ticks.forEach(function (t) {
        var px = crisp(x(t));
        grid.appendChild(svg("line", { class: "tv-gridline", x1: px, x2: px, y1: m.top, y2: m.top + plotH }));
        var label = fmtTick(t);
        var w = TV.measure(label, 11);
        if (px - w / 2 < lastRight + 6) return;
        lastRight = px + w / 2;
        grid.appendChild(svg("text", { class: "tv-tick", x: px, y: m.top + plotH + 16, "text-anchor": "middle" }, label));
      });
      root.appendChild(grid);

      var marks = svg("g");
      var labels = svg("g", { class: "tv-values" });
      var y = m.top;
      lines.forEach(function (line) {
        if (line.group) {
          marks.appendChild(
            svg("text", { class: "tv-label", x: 0, y: y + GROUP_ROW / 2 + 2, dy: "0.35em", style: { fontWeight: 500, fill: "var(--ink-2)" } }, line.group)
          );
          y += GROUP_ROW;
          return;
        }
        var r = line.row;
        var cy = y + ROW / 2;
        var v = valueOf(r);
        var fill = TV.color(r.role || (spec.series && spec.series[0] && spec.series[0].role) || "model");
        var parts = [];
        if (isNum(r.lo) && isNum(r.hi)) {
          var x0 = x(r.lo);
          var x1 = x(r.hi);
          parts.push(svg("line", { class: "tv-line tv-mark", x1: x0, x2: x1, y1: cy, y2: cy, style: { stroke: fill } }));
          parts.push(svg("line", { class: "tv-line tv-mark", x1: x0, x2: x0, y1: cy - 5, y2: cy + 5, style: { stroke: fill } }));
          parts.push(svg("line", { class: "tv-line tv-mark", x1: x1, x2: x1, y1: cy - 5, y2: cy + 5, style: { stroke: fill } }));
        }
        if (isNum(v)) {
          parts.push(svg("circle", { class: "tv-mark tv-dot", cx: x(v), cy: cy, r: 4.5, style: { fill: fill } }));
        }
        var g = svg("g", { class: "tv-markg", tabindex: "0", "aria-label": r.label + ": " + TV.format(spec.format, v) + (r.text ? ", " + r.text : "") });
        g.appendChild(svg("rect", { class: "tv-hit", x: 0, y: y, width: W, height: ROW }));
        parts.forEach(function (p) {
          g.appendChild(p);
        });
        marks.appendChild(g);
        marks.appendChild(
          svg("text", { class: "tv-label", x: labelW + 4, y: cy, dy: "0.35em", "text-anchor": "end" }, r.label)
        );
        if (r.text) {
          var end = Math.max(isNum(r.hi) ? r.hi : v, isNum(v) ? v : -Infinity);
          labels.appendChild(svg("text", { class: "tv-value", x: x(end) + 8, y: cy, dy: "0.35em" }, r.text));
        }
        TV.tooltip.attach(g, function () {
          var out = [{ label: spec.valueLabel || "Value", value: TV.format(spec.format, v), color: fill, shape: "dot" }];
          if (isNum(r.lo) && isNum(r.hi)) {
            out.push({
              label: spec.intervalLabel || "Interval",
              value: TV.format(spec.format, r.lo) + " to " + TV.format(spec.format, r.hi),
            });
          }
          (r.tip || []).forEach(function (t) {
            out.push({ label: t.label, value: TV.format(t.format, t.value) });
          });
          return { title: r.group ? r.label + ", " + r.group : r.label, rows: out };
        });
        y += ROW;
      });
      root.appendChild(marks);

      var refLayer = svg("g");
      refs.forEach(function (ref) {
        var px = crisp(x(ref.value));
        refLayer.appendChild(svg("line", { class: "tv-ref", x1: px, x2: px, y1: m.top - 4, y2: m.top + plotH }));
        if (ref.label) {
          var w = TV.measure(ref.label, 11, 500);
          var lx = Math.max(m.left + w / 2, Math.min(W - m.right - w / 2, px));
          refLayer.appendChild(svg("text", { class: "tv-ref-label", x: lx, y: m.top - 10, "text-anchor": "middle" }, ref.label));
        }
      });
      root.appendChild(refLayer);
      root.appendChild(labels);
      wrap.appendChild(root);
    });
  }

  /* A limit of the input, stated under the figure. Not a refusal, so no refusal chip. */
  function limitNote(handle, text) {
    handle.notes.appendChild(el("div", { class: "tv-note", role: "note" }, el("p", null, text)));
  }

  function provenanceFor(data, id) {
    return ((data && data.provenance) || []).filter(function (p) {
      return p && p.figure === id;
    })[0];
  }

  TV.sections.register("encoder", function (root, data) {
    var figures = (data && data.figures) || {};
    var handles = {};
    var ids = Object.keys(figures);
    if (!ids.length) return handles;

    ids
      .filter(function (id) {
        return figures[id] && figures[id].kind === "tiles";
      })
      .forEach(function (id) {
        var f = figures[id];
        var set = el("div", { class: "tv-tileset", "data-figure-id": id }, f.title ? el("h3", { class: "tv-tileset__title" }, f.title) : null);
        root.appendChild(set);
        TV.charts.tiles(set, f.data || {});
        handles[id] = {
          root: set,
          body: set,
          title: f.title || "",
          addNote: function (r) {
            set.appendChild(TV.refusalNote(r));
          },
        };
      });

    var cards = ORDER.filter(function (id) {
      return figures[id] && figures[id].kind !== "tiles";
    }).concat(
      ids.filter(function (id) {
        return ORDER.indexOf(id) < 0 && figures[id] && figures[id].kind !== "tiles";
      })
    );
    if (!cards.length) return handles;
    var grid = el("div", { class: "tv-grid" });
    root.appendChild(grid);

    cards.forEach(function (id) {
      var f = figures[id];
      var spec = Object.assign({ title: f.title }, f.data || {});
      var handle = TV.figure(grid, {
        id: id,
        anchor: "fig-" + data.id + "-" + id,
        title: f.title,
        subtitle: f.subtitle,
        provenance: provenanceFor(data, id),
        wide: !!spec.wide,
      });
      handles[id] = handle;
      if (spec.whiskers) {
        whiskers(handle, spec);
      } else if (TV.charts[f.kind]) {
        TV.charts[f.kind](handle.body, spec);
        if (spec.table) {
          TV.clear(handle.table);
          TV.tableView(handle, spec.table);
        }
      } else {
        handle.addNote({ what: id, why: "The chart kit has no chart named " + JSON.stringify(String(f.kind)) + "." });
      }
      if (spec.note) limitNote(handle, spec.note);
    });
    return handles;
  });
})(window.TV);
