/*
 * Section renderer: datalayer.
 *
 * Five checks on the data underneath the models, each drawn as its own group so
 * a tile row sits over the charts that explain it: the debt ladders, the stock
 * splits beside Nvidia's closes, the sub-vertical taxonomy, the peer labels, and
 * ticker resolution. The snapshot sorts figure keys, so the order is set here.
 * A figure the list does not name is still drawn, after the named ones.
 *
 * One figure needs marks the kit does not draw and is drawn here with TV.svg,
 * TV.scale, TV.tooltip and TV.tableView:
 *
 *   nvda_closes   a line on a log scale, so a four-for-one or ten-for-one step
 *                 would read as the same size at any price, with the split
 *                 windows shaded and the largest single-day move labelled.
 *
 * It keeps the spec shape of the kit's line chart, so the default loop would
 * still draw it truthfully, on a linear axis.
 *
 * A refusal that belongs to a figure is named in that figure's data.near, and the
 * handle is returned under the refusal's name too, so app.js hangs the note
 * under the figure rather than at the foot of the section.
 *
 * Every number drawn is in the snapshot; this file formats and places them.
 */
(function (TV) {
  "use strict";

  var el = TV.el;
  var svg = TV.svg;
  var HIT_MIN = 24;

  var GROUPS = [
    ["debt_reach"],
    ["split_counts", "split_oracle", "nvda_closes"],
    ["taxonomy_by_vertical", "taxonomy_how"],
    ["peer_label_counts", "peer_groups_lost", "peer_label_funnel"],
    ["ticker_reach"],
  ];
  var LOCAL = { nvda_closes: closesChart };

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
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

  /* Tiles on the page plane, as the kit's default loop draws them. */
  function tileset(root, id, f) {
    var set = el(
      "div",
      { class: "tv-tileset", "data-figure-id": id },
      f.title ? el("h3", { class: "tv-tileset__title" }, f.title) : null
    );
    root.appendChild(set);
    TV.charts.tiles(set, f.data || {});
    return {
      root: set,
      body: set,
      title: f.title || "",
      addNote: function (r) {
        set.appendChild(TV.refusalNote(r));
      },
    };
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
      wide: !!(f.wide || (f.data && f.data.wide)),
    });
  }

  function drawCard(handle, id, f) {
    if (LOCAL[id]) return LOCAL[id](handle, f.data || {}, f);
    var chart = TV.charts[f.kind];
    if (!chart) {
      handle.addNote({ what: handle.title, why: "The chart kit has no chart named " + JSON.stringify(String(f.kind)) + "." });
      return;
    }
    chart(handle.body, Object.assign({ title: f.title }, f.data || {}));
  }

  /* nvda_closes ------------------------------------------------------------ */

  function parseDay(text) {
    var t = Date.parse(String(text) + "T00:00:00Z");
    return isFinite(t) ? t : NaN;
  }

  /* 1, 2, 5 times a power of ten, between two positive values. */
  function logTicks(lo, hi, maxCount) {
    var out = [];
    var k0 = Math.floor(Math.log10(lo));
    var k1 = Math.ceil(Math.log10(hi));
    [[1, 2, 5], [1]].some(function (mults) {
      out = [];
      for (var k = k0; k <= k1; k++) {
        mults.forEach(function (m) {
          var v = m * Math.pow(10, k);
          if (v >= lo * 0.999 && v <= hi * 1.001) out.push(parseFloat(v.toPrecision(6)));
        });
      }
      return out.length <= maxCount;
    });
    return out;
  }

  function niceLogBound(v, up) {
    var k = Math.floor(Math.log10(v));
    var steps = [1, 2, 5, 10];
    var base = Math.pow(10, k);
    if (up) {
      for (var i = 0; i < steps.length; i++) if (steps[i] * base >= v) return steps[i] * base;
    } else {
      for (var j = steps.length - 1; j >= 0; j--) if (steps[j] * base <= v) return steps[j] * base;
    }
    return v;
  }

  function priceText(v) {
    return TV.format(v >= 100 ? "num:0" : v >= 1 ? "num:2" : "num:3", v);
  }

  function tickText(v) {
    return TV.format(v >= 1 ? "num:0" : v >= 0.1 ? "num:1" : "num:2", v);
  }

  function closesChart(handle, spec) {
    var series = (spec.series || [])[0] || { name: "Close", role: "model", values: [] };
    var role = series.role || "model";
    var pts = (series.values || [])
      .filter(function (p) {
        return p && isNum(p.y) && p.y > 0 && isFinite(parseDay(p.x));
      })
      .map(function (p) {
        return { day: String(p.x), t: parseDay(p.x), y: p.y };
      })
      .sort(function (a, b) {
        return a.t - b.t;
      });
    var windows = (spec.windows || []).filter(function (w) {
      return w && isFinite(parseDay(w.from)) && isFinite(parseDay(w.to));
    });
    var mark = spec.mark && isNum(spec.mark.y) && spec.mark.y > 0 && isFinite(parseDay(spec.mark.x)) ? spec.mark : null;
    var valueName = spec.valueLabel || series.name || "Close";

    TV.tableView(handle, {
      caption: handle.title + ", closes",
      columns: [
        { key: "day", label: (spec.x && spec.x.label) || "Date", mono: true },
        { key: "y", label: valueName, align: "right", format: spec.format },
      ],
      rows: pts,
    });
    if (windows.length) {
      TV.tableView(handle, {
        caption: handle.title + ", split windows",
        columns: [
          { key: "label", label: "Split window" },
          { key: "from", label: "From", mono: true },
          { key: "to", label: "To", mono: true },
          { key: "worst_day", label: "Worst day", align: "right", format: "pct:1" },
          { key: "worst_date", label: "On", mono: true },
          { key: "unadjusted_move", label: "Unrestated step", align: "right", format: "pct:0" },
        ],
        rows: windows,
      });
    }
    if (pts.length < 2) {
      handle.body.appendChild(el("p", { class: "tv-empty" }, "No values to draw."));
      return;
    }
    var legend = [{ label: series.name || "Close", color: role, shape: "line" }];
    if (windows.length) {
      legend.push({ label: "Split window, from the last filing on the old share basis to the first on the new", color: "--wash-strong", shape: "rect" });
    }
    TV.legend(handle.legend, legend);

    frame(handle.body, "line", function (wrap, W) {
      var H = 300;
      var ys = pts.map(function (p) {
        return p.y;
      });
      var lo = niceLogBound(Math.min.apply(null, ys), false);
      var hi = niceLogBound(Math.max.apply(null, ys), true);
      var m = { top: 34, right: 16, bottom: 30, left: 0 };
      var plotH = H - m.top - m.bottom;
      var ticks = logTicks(lo, hi, Math.max(3, Math.floor(plotH / 22)));
      m.left =
        Math.max.apply(
          null,
          [0].concat(
            ticks.map(function (t) {
              return TV.measure(tickText(t), 11);
            })
          )
        ) + 14;
      var yLog = TV.scale.linear([Math.log10(lo), Math.log10(hi)], [H - m.bottom, m.top]);
      var y = function (v) {
        return yLog(Math.log10(v));
      };
      var x = TV.scale.linear([pts[0].t, pts[pts.length - 1].t], [m.left + 4, W - m.right]);

      var root = svg("svg", {
        class: "tv-svg",
        width: W,
        height: H,
        viewBox: "0 0 " + W + " " + H,
        role: "img",
        "aria-label": handle.title || "Line chart",
      });

      var grid = svg("g");
      ticks.forEach(function (t) {
        var py = crisp(y(t));
        grid.appendChild(svg("line", { class: "tv-gridline", x1: m.left, x2: W - m.right, y1: py, y2: py }));
        grid.appendChild(svg("text", { class: "tv-tick", x: m.left - 8, y: py, dy: "0.35em", "text-anchor": "end" }, tickText(t)));
      });
      grid.appendChild(svg("text", { class: "tv-tick", x: m.left - 8, y: m.top - 22, "text-anchor": "start" }, "USD, log scale"));

      /* x ticks on the first day of a year, thinned to fit */
      var y0 = new Date(pts[0].t).getUTCFullYear();
      var y1 = new Date(pts[pts.length - 1].t).getUTCFullYear();
      var years = [];
      for (var yr = y0; yr <= y1 + 1; yr++) {
        var t = Date.UTC(yr, 0, 1);
        if (t >= pts[0].t && t <= pts[pts.length - 1].t) years.push({ yr: yr, t: t });
      }
      var labelW = TV.measure("2020", 11) + 12;
      var spacing = years.length > 1 ? x(years[1].t) - x(years[0].t) : Infinity;
      var every = Math.max(1, Math.ceil(labelW / spacing));
      years.forEach(function (d, i) {
        if (i % every !== 0) return;
        var px = x(d.t);
        grid.appendChild(svg("text", { class: "tv-tick", x: px, y: H - m.bottom + 16, "text-anchor": "middle" }, String(d.yr)));
      });
      root.appendChild(grid);

      /* split windows, shaded, with their names stacked above the plot */
      var shade = svg("g");
      var lastRight = -Infinity;
      windows.forEach(function (w) {
        var x0 = clamp(x(parseDay(w.from)), m.left, W - m.right);
        var x1 = clamp(x(parseDay(w.to)), m.left, W - m.right);
        if (x1 - x0 < 1) x1 = x0 + 1;
        shade.appendChild(svg("rect", { x: x0, y: m.top, width: x1 - x0, height: plotH, style: { fill: "var(--wash-strong)" } }));
        var text = w.label || "Split window";
        var tw = TV.measure(text, 11, 500);
        var cx = clamp((x0 + x1) / 2, m.left + tw / 2, W - m.right - tw / 2);
        var row = cx - tw / 2 < lastRight + 6 ? 1 : 0;
        if (!row) lastRight = cx + tw / 2;
        shade.appendChild(svg("text", { class: "tv-ref-label", x: cx, y: m.top - 8 - row * 13, "text-anchor": "middle" }, text));
      });
      root.appendChild(shade);

      root.appendChild(svg("line", { class: "tv-axisline", x1: m.left, x2: W - m.right, y1: crisp(H - m.bottom), y2: crisp(H - m.bottom) }));

      var d = pts
        .map(function (p, i) {
          return (i ? "L" : "M") + x(p.t).toFixed(1) + "," + y(p.y).toFixed(1);
        })
        .join("");
      root.appendChild(svg("path", { class: "tv-line", d: d, style: { stroke: TV.color(role) } }));
      var lastPt = pts[pts.length - 1];
      root.appendChild(svg("circle", { class: "tv-dot", cx: x(lastPt.t), cy: y(lastPt.y), r: 4, style: { fill: TV.color(role) } }));

      if (mark) {
        var mx = x(parseDay(mark.x));
        var my = y(mark.y);
        var labels = svg("g", { class: "tv-values" });
        var text = mark.label || TV.format("signed:1", mark.move);
        var tw = TV.measure(text, 11, 500);
        /* Beside the dot and under the line where there is room, since a price
           series mostly climbs away from its lows; above it otherwise. */
        var below = H - m.bottom - my >= 22;
        var right = mx + 8 + tw <= W - m.right;
        labels.appendChild(
          svg(
            "text",
            {
              class: "tv-value",
              x: right ? mx + 8 : mx - 8,
              y: below ? my + 16 : my - 10,
              "text-anchor": right ? "start" : "end",
            },
            text
          )
        );
        root.appendChild(svg("circle", { class: "tv-dot", cx: mx, cy: my, r: 4, style: { fill: TV.color(role) } }));
        root.appendChild(labels);
      }

      /* Crosshair: the pointer finds the nearest week; arrows move it. */
      var hover = svg("g", { hidden: true });
      var hair = svg("line", { class: "tv-crosshair", x1: 0, x2: 0, y1: m.top, y2: H - m.bottom });
      var hoverDot = svg("circle", { class: "tv-dot", r: 4, style: { fill: TV.color(role) } });
      hover.appendChild(hair);
      hover.appendChild(hoverDot);
      root.appendChild(hover);
      var plot = svg("g", {
        class: "tv-plot",
        tabindex: "0",
        "aria-label": (handle.title || "Line chart") + ". Use the arrow keys to read each week.",
      });
      plot.appendChild(svg("rect", { class: "tv-hit", x: m.left, y: m.top, width: Math.max(0, W - m.left - m.right), height: Math.max(HIT_MIN, plotH) }));
      root.appendChild(plot);

      var current = -1;
      function content(i) {
        var p = pts[i];
        var rows = [{ label: series.name || "Close", value: priceText(p.y), color: role, shape: "line" }];
        windows.forEach(function (w) {
          if (p.t > parseDay(w.from) && p.t <= parseDay(w.to)) {
            rows.push({ label: w.label || "Split window", value: "worst day " + TV.format("pct:1", w.worst_day) });
          }
        });
        if (mark && p.day === String(mark.x)) rows.push({ label: "Move that day", value: TV.format("signed:1", mark.move * 100) + "%" });
        return { title: p.day, rows: rows };
      }
      function showAt(i, clientX, clientY) {
        current = clamp(i, 0, pts.length - 1);
        var p = pts[current];
        var px = x(p.t);
        hair.setAttribute("x1", crisp(px));
        hair.setAttribute("x2", crisp(px));
        hoverDot.setAttribute("cx", px);
        hoverDot.setAttribute("cy", y(p.y));
        hover.removeAttribute("hidden");
        if (!isNum(clientX)) {
          var box = root.getBoundingClientRect();
          clientX = box.left + px;
          clientY = box.top + y(p.y);
        }
        TV.tooltip.show(content(current), clientX, clientY, plot);
      }
      function nearest(clientX) {
        var box = root.getBoundingClientRect();
        var t = x.invert(clientX - box.left);
        var best = 0;
        var bestD = Infinity;
        pts.forEach(function (p, i) {
          var dd = Math.abs(p.t - t);
          if (dd < bestD) {
            bestD = dd;
            best = i;
          }
        });
        return best;
      }
      function hide() {
        hover.setAttribute("hidden", "");
        TV.tooltip.hide(plot);
      }
      plot.addEventListener("pointermove", function (ev) {
        showAt(nearest(ev.clientX), ev.clientX, ev.clientY);
      });
      plot.addEventListener("pointerleave", hide);
      plot.addEventListener("focus", function () {
        showAt(current < 0 ? pts.length - 1 : current);
      });
      plot.addEventListener("blur", hide);
      plot.addEventListener("keydown", function (ev) {
        var next = current;
        if (ev.key === "ArrowRight") next = current + 1;
        else if (ev.key === "ArrowLeft") next = current - 1;
        else if (ev.key === "Home") next = 0;
        else if (ev.key === "End") next = pts.length - 1;
        else if (ev.key === "Escape") return hide();
        else return;
        ev.preventDefault();
        showAt(next);
      });
      wrap.appendChild(root);
    });
  }

  /* The section ------------------------------------------------------------ */

  TV.sections.register("datalayer", function (root, data) {
    var figures = (data && data.figures) || {};
    var named = {};
    var groups = GROUPS.map(function (ids) {
      return ids.filter(function (id) {
        named[id] = true;
        return Object.prototype.hasOwnProperty.call(figures, id);
      });
    });
    var rest = Object.keys(figures).filter(function (id) {
      return !named[id];
    });
    if (rest.length) groups.push(rest);

    var handles = {};
    groups.forEach(function (ids) {
      if (!ids.length) return;
      var tiles = ids.filter(function (id) {
        return figures[id] && figures[id].kind === "tiles";
      });
      var cards = ids.filter(function (id) {
        return tiles.indexOf(id) < 0;
      });
      tiles.forEach(function (id) {
        handles[id] = tileset(root, id, figures[id]);
      });
      if (cards.length) {
        var grid = el("div", { class: "tv-grid" });
        root.appendChild(grid);
        cards.forEach(function (id) {
          var handle = card(grid, data, id);
          handles[id] = handle;
          drawCard(handle, id, figures[id]);
        });
      }
      ids.forEach(function (id) {
        var near = (figures[id].data && figures[id].data.near) || [];
        near.forEach(function (what) {
          if (typeof what === "string" && what && !handles[what]) handles[what] = handles[id];
        });
      });
    });
    return handles;
  });
})(window.TV);
