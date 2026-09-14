/*
 * Section renderer: tmt.
 *
 * Three reads, laid out as three groups because they answer three questions:
 * what acquirers paid, what one conglomerate's parts earn, and which operating
 * figures a filing really states. The snapshot sorts figure keys, so the order
 * is set here.
 *
 * The kit draws the tiles, the bars and both waterfalls. Three figures are drawn
 * locally with TV.svg, TV.scale, TV.tooltip and TV.tableView:
 *
 *   premia   a dumbbell whose gap, in points, is written beside every row and
 *            marked where it passes the flag. The kit's dot chart labels one
 *            value on one row and has no way to mark a row.
 *   deals    tables, which the kit has no figure for, built in the kit's own
 *   kpis     table classes with cells that can hold a refused chip.
 *
 * Refusals attach to the figure they concern. A figure's data may carry
 * `refusals`, the `what` of each section refusal that belongs under it, and the
 * handle map returned to app.js carries those names as extra keys, so app.js
 * hangs each note on its card. Anything unnamed stays a section note.
 *
 * Every number arrives computed, the gaps in points included. This file only
 * formats and draws.
 */
(function (TV) {
  "use strict";

  var el = TV.el;
  var svg = TV.svg;

  var GROUPS = [
    { title: "Precedent transactions", tiles: ["precedent_tiles"], cards: ["premia", "ev_revenue"], wide: ["deals"] },
    { title: "Segments and the sum of the parts", tiles: ["segment_margins"], cards: ["segment_revenue", "segment_operating_income"] },
    { title: "Operating metrics", wide: ["kpis"] },
  ];

  var ACRONYMS = { rpo: "RPO", arr: "ARR", arpu: "ARPU" };

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  function provenanceFor(data, id) {
    return (data.provenance || []).filter(function (p) {
      return p && p.figure === id;
    })[0];
  }

  function card(parent, data, id, wide) {
    var f = data.figures[id];
    return TV.figure(parent, {
      id: id,
      anchor: "fig-" + data.id + "-" + id,
      title: f.title,
      subtitle: f.subtitle,
      provenance: provenanceFor(data, id),
      wide: !!wide,
    });
  }

  function refusedCell() {
    return TV.chip("refused", "Refused", { compact: true });
  }

  function muted(text) {
    return el("span", { class: "tv-muted" }, text);
  }

  function stack(first, second) {
    return el("div", null, el("div", null, first), second ? el("div", { class: "tv-muted" }, second) : null);
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
    return wrap;
  }

  /* A table in the kit's own table classes, with cells that may be nodes. */
  function table(handle, caption, columns, rows) {
    var head = el(
      "tr",
      null,
      columns.map(function (c) {
        return el("th", { scope: "col", class: c.num ? "tv-num" : null }, c.label);
      })
    );
    var body = el(
      "tbody",
      null,
      rows.map(function (cells) {
        return el(
          "tr",
          null,
          cells.map(function (cell, i) {
            var attrs = {
              class: columns[i].num ? "tv-num" : null,
              style: columns[i].nowrap ? { whiteSpace: "nowrap" } : null,
            };
            return i === 0 ? el("th", Object.assign({ scope: "row" }, attrs), cell) : el("td", attrs, cell);
          })
        );
      })
    );
    var node = el(
      "table",
      { class: "tv-table" },
      el("caption", { class: "tv-visually-hidden" }, caption),
      el("thead", null, head),
      body
    );
    handle.body.appendChild(
      el("div", { class: "tv-table-wrap", tabindex: "0", role: "region", "aria-label": caption }, node)
    );
  }

  /* premia: a dumbbell with every gap written and the flagged ones marked --- */

  function gapText(points, flagged) {
    if (!isNum(points)) return "n/a";
    return TV.fmt.signed(points, 1) + " pts" + (flagged ? ", flagged" : "");
  }

  function tickFormat(ticks) {
    var step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) * 100 : 10;
    var dp = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step) - 1e-9));
    return function (v) {
      return TV.fmt.pct(v, dp);
    };
  }

  function premia(handle, f) {
    var d = f.data || {};
    var rows = d.rows || [];
    var series = d.series || [];
    var fmt = d.format || "pct:1";
    if (series.length !== 2) {
      TV.charts.dot(handle.body, Object.assign({ title: f.title }, d));
      return;
    }
    var a = series[0];
    var b = series[1];
    var threshold = isNum(d.leakPoints) ? TV.fmt.num(d.leakPoints, 0) : null;

    TV.tableView(handle, {
      columns: [
        { key: "label", label: d.labelHeader || "Target" },
        { key: "acquirer", label: "Acquirer" },
        { key: "a", label: a.name, align: "right", format: fmt },
        { key: "b", label: b.name, align: "right", format: fmt },
        { key: "gap", label: "Gap, points", align: "right", format: "signed:1" },
        { key: "flag", label: threshold ? "Above " + threshold + " points" : "Flagged" },
      ],
      rows: rows.map(function (r) {
        var v = r.values || {};
        return { label: r.label, acquirer: r.acquirer || "n/a", a: v[a.key], b: v[b.key], gap: r.gap_points, flag: r.flagged ? "Flagged" : "No" };
      }),
    });
    if (!rows.length) return;
    TV.legend(handle.legend, [
      { label: a.name, color: a.role, shape: "dot" },
      { label: b.name, color: b.role, shape: "dot" },
    ]);

    frame(handle.body, "dot", function (wrap, W) {
      var values = [];
      rows.forEach(function (r) {
        var v = r.values || {};
        if (isNum(v[a.key])) values.push(v[a.key]);
        if (isNum(v[b.key])) values.push(v[b.key]);
      });
      var lo = Math.min.apply(null, [0].concat(values));
      var hi = Math.max.apply(null, [0].concat(values));
      var labelW = Math.max.apply(
        null,
        [40].concat(
          rows.map(function (r) {
            return TV.measure(r.label, 12);
          })
        )
      );
      var gapW = Math.max.apply(
        null,
        [0].concat(
          rows.map(function (r) {
            return TV.measure(gapText(r.gap_points, r.flagged), 11, r.flagged ? 600 : 400);
          })
        )
      );
      var m = { top: 8, right: gapW + 20, bottom: 26, left: labelW + 16 };
      var rowH = 36;
      var plotH = rowH * rows.length;
      var H = Math.round(m.top + plotH + m.bottom);
      var x = TV.scale.linear([lo, hi], [m.left + 6, W - m.right - 6], { nice: true });
      var root = svg("svg", {
        class: "tv-svg",
        width: W,
        height: H,
        viewBox: "0 0 " + W + " " + H,
        role: "img",
        "aria-label": f.title || "Premia by deal",
      });

      var grid = svg("g");
      var ticks = x.ticks(Math.max(2, Math.floor((W - m.left - m.right) / 90)));
      var tickText = tickFormat(ticks);
      ticks.forEach(function (t) {
        var px = crisp(x(t));
        grid.appendChild(svg("line", { class: "tv-gridline", x1: px, x2: px, y1: m.top, y2: m.top + plotH }));
        grid.appendChild(svg("text", { class: "tv-tick", x: px, y: m.top + plotH + 16, "text-anchor": "middle" }, tickText(t)));
      });
      var zero = crisp(x(0));
      grid.appendChild(svg("line", { class: "tv-axisline", x1: zero, x2: zero, y1: m.top, y2: m.top + plotH }));
      root.appendChild(grid);

      var marks = svg("g");
      rows.forEach(function (r, i) {
        var v = r.values || {};
        var va = v[a.key];
        var vb = v[b.key];
        var cy = m.top + rowH * i + rowH / 2;
        var g = svg("g", {
          class: "tv-markg",
          tabindex: "0",
          "aria-label":
            r.label + ": " + a.name + " " + TV.format(fmt, va) + ", " + b.name + " " + TV.format(fmt, vb) + ", gap " + gapText(r.gap_points, r.flagged),
        });
        g.appendChild(svg("rect", { class: "tv-hit", x: 0, y: m.top + rowH * i, width: W, height: rowH }));
        g.appendChild(svg("text", { class: "tv-label", x: m.left - 10, y: cy, dy: "0.35em", "text-anchor": "end" }, r.label));
        if (isNum(va) && isNum(vb)) {
          g.appendChild(
            svg("line", { class: "tv-hairline", x1: x(Math.min(va, vb)), x2: x(Math.max(va, vb)), y1: crisp(cy), y2: crisp(cy) })
          );
        }
        [[vb, b], [va, a]].forEach(function (pair) {
          if (isNum(pair[0])) {
            g.appendChild(svg("circle", { class: "tv-mark tv-dot", cx: x(pair[0]), cy: cy, r: 4.5, style: { fill: TV.color(pair[1].role) } }));
          }
        });
        g.appendChild(
          svg(
            "text",
            {
              class: r.flagged ? "tv-value" : "tv-tick",
              x: W - m.right + 14,
              y: cy,
              dy: "0.35em",
              "text-anchor": "start",
              style: r.flagged ? { fontWeight: "600" } : null,
            },
            gapText(r.gap_points, r.flagged)
          )
        );
        marks.appendChild(g);
        TV.tooltip.attach(g, function () {
          var out = [
            { label: a.name, value: TV.format(fmt, va), color: a.role, shape: "dot" },
            { label: b.name, value: TV.format(fmt, vb), color: b.role, shape: "dot" },
            { label: "Gap", value: gapText(r.gap_points, r.flagged) },
          ];
          if (r.acquirer) out.push({ label: "Acquirer", value: r.acquirer });
          if (r.announced) out.push({ label: "Announced", value: String(r.announced) });
          return { title: r.label, rows: out };
        });
      });
      root.appendChild(marks);
      wrap.appendChild(root);
    });
  }

  function evRevenue(handle, f) {
    var median = f.data.median;
    var spec = Object.assign({ title: f.title }, f.data, {
      rows: (f.data.rows || []).map(function (r) {
        return Object.assign({}, r, {
          note: [r.acquirer, r.status, r.announced ? "announced " + r.announced : null].filter(Boolean).join(", "),
        });
      }),
      reference: isNum(median) ? [{ value: median, label: "Median " + TV.fmt.mult(median, 1) }] : [],
    });
    TV.charts.hbar(handle.body, spec);
  }

  function kitChart(handle, f) {
    var chart = TV.charts[f.kind];
    if (!chart) {
      handle.addNote({ what: f.title, why: "The chart kit has no chart named " + JSON.stringify(String(f.kind)) + "." });
      return;
    }
    chart(handle.body, Object.assign({ title: f.title }, f.data || {}));
  }

  var DRAW = {
    premia: premia,
    ev_revenue: evRevenue,
  };

  /* Tables ---------------------------------------------------------------- */

  function money(v) {
    return isNum(v) ? TV.fmt.num(v, 2) : null;
  }

  function orRefused(text) {
    return text === null ? refusedCell() : text;
  }

  function deals(handle, f) {
    var columns = [
      { label: "Target" },
      { label: "Acquirer" },
      { label: "Announced", nowrap: true },
      { label: "Status" },
      { label: "Offer, $", num: true },
      { label: "Premium, last close", num: true },
      { label: "Premium, 30-day mean", num: true },
      { label: "EV/Revenue", num: true },
    ];
    var rows = (f.data.rows || []).map(function (r) {
      return [
        stack(r.ticker, r.target),
        stack(r.acquirer || "not identified", r.consideration ? r.consideration + " consideration" : null),
        r.announced || "n/a",
        stack(r.status, r.closed ? "closed " + r.closed : null),
        orRefused(money(r.offer_price)),
        orRefused(isNum(r.premium_1d) ? TV.fmt.pct(r.premium_1d, 1) : null),
        orRefused(isNum(r.premium_30d) ? TV.fmt.pct(r.premium_30d, 1) : null),
        orRefused(isNum(r.ev_revenue) ? TV.fmt.mult(r.ev_revenue, 1) : null),
      ];
    });
    table(handle, f.title, columns, rows);
  }

  function metricName(name) {
    return String(name)
      .split("_")
      .map(function (w, i) {
        if (ACRONYMS[w]) return ACRONYMS[w];
        return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w;
      })
      .join(" ");
  }

  function kpiValue(v, unit) {
    if (!isNum(v)) return "n/a";
    if (unit === "usd_mm") return TV.fmt.mm(v, 1);
    if (unit === "ratio") return TV.fmt.pct(v, 0);
    if (unit === "count") return TV.fmt.int(v);
    var per = /^usd_per_(\w+)$/.exec(String(unit));
    if (per) return "$" + TV.fmt.num(v, 2) + (per[1] === "period" ? "" : " a " + per[1]);
    return TV.fmt.auto(v);
  }

  function kpis(handle, f) {
    var columns = [
      { label: "Filer" },
      { label: "Metric" },
      { label: "Value", num: true },
      { label: "Period end", nowrap: true },
      { label: "Source", nowrap: true },
      { label: "Tag or phrase" },
    ];
    var rows = (f.data.rows || []).map(function (r) {
      return [
        r.ticker,
        metricName(r.metric),
        r.refused ? refusedCell() : kpiValue(r.value, r.unit),
        r.period_stated ? r.period_end || "n/a" : stack(muted("not stated"), r.read_from),
        r.evidence,
        el(
          "div",
          null,
          el("div", { class: "tv-mono" }, r.tag_or_phrase),
          r.refused && r.reason ? el("div", { class: "tv-muted" }, r.reason) : null
        ),
      ];
    });
    table(handle, f.title, columns, rows);
    var sources = f.data.sources || [];
    if (sources.length) {
      handle.body.appendChild(
        el(
          "ul",
          { class: "tv-figure__subtitle", style: { margin: "0", paddingLeft: "1.2em" } },
          sources.map(function (s) {
            var read = "instance " + s.instance_accession + ", filed " + s.instance_filed;
            if (s.text_accession) read += "; Item 7 of 10-K " + s.text_accession;
            var looked = isNum(s.missing) ? "; " + TV.fmt.int(s.missing) + " more metrics looked for and not disclosed" : "";
            return el("li", null, s.ticker + ": " + read + looked);
          })
        )
      );
    }
  }

  var TABLES = { deals: deals, kpis: kpis };

  /* Layout ---------------------------------------------------------------- */

  function draw(handle, id, f) {
    if (TABLES[id] && f.kind === "table") TABLES[id](handle, f);
    else if (DRAW[id]) DRAW[id](handle, f);
    else kitChart(handle, f);
  }

  function remember(handles, id, handle, f) {
    handles[id] = handle;
    ((f.data && f.data.refusals) || []).forEach(function (what) {
      handles[what] = handle;
    });
  }

  function tileset(block, id, f) {
    var set = el(
      "div",
      { class: "tv-tileset", "data-figure-id": id },
      f.title ? el("h3", { class: "tv-tileset__title" }, f.title) : null,
      f.subtitle ? el("p", { class: "tv-figure__subtitle" }, f.subtitle) : null
    );
    block.appendChild(set);
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

  TV.sections.register("tmt", function (root, data) {
    var figures = (data && data.figures) || {};
    var handles = {};
    var placed = [];
    GROUPS.forEach(function (group) {
      placed = placed.concat(group.tiles || [], group.cards || [], group.wide || []);
    });
    /* A figure this layout does not name is still drawn, after the groups. */
    var rest = Object.keys(figures).filter(function (id) {
      return placed.indexOf(id) < 0;
    });
    GROUPS.concat(rest.length ? [{ title: "Other figures", cards: rest }] : []).forEach(function (group) {
      var ids = [].concat(group.tiles || [], group.cards || [], group.wide || []).filter(function (id) {
        return figures[id];
      });
      if (!ids.length) return;
      var block = el("div", { class: "tv-section__body" }, el("h3", { class: "tv-eyebrow" }, group.title));
      root.appendChild(block);

      (group.tiles || []).forEach(function (id) {
        if (!figures[id]) return;
        remember(handles, id, tileset(block, id, figures[id]), figures[id]);
      });

      var cards = (group.cards || []).filter(function (id) {
        return figures[id];
      });
      var wide = (group.wide || []).filter(function (id) {
        return figures[id];
      });
      if (cards.length || wide.length) {
        var grid = el("div", { class: "tv-grid" });
        block.appendChild(grid);
        cards.forEach(function (id) {
          var handle = card(grid, data, id, cards.length === 1);
          draw(handle, id, figures[id]);
          remember(handles, id, handle, figures[id]);
        });
        wide.forEach(function (id) {
          var handle = card(grid, data, id, true);
          draw(handle, id, figures[id]);
          remember(handles, id, handle, figures[id]);
        });
      }
    });
    return handles;
  });
})(window.TV);
