/*
 * Section renderer: overview, the scoreboard at the top of the page.
 *
 * Three blocks, drawn by the kit's figure loop in a fixed order. First one tile
 * per model section, in page order, each linking to its section: the verdict
 * chip, the score, the metric and n, the baseline score and the lift, then the
 * baseline's name. Below them, every other section's state as the kit's table
 * figure, beside the tiles that say how the page was collected, drawn in a card.
 *
 * The model tiles are laid out here, because the kit's tiles cannot share row
 * tracks. Each tile spans six rows of the grid as a subgrid, so every score sits
 * on one line and every chip on another however long a baseline's name runs,
 * and the column count is chosen so no tile is left alone in a row. The grid is
 * drawn in the kit's frame, which redraws it when its width changes.
 *
 * Every number drawn is in the snapshot. The scoreboard, the section table and
 * the assumptions come from this section's figures. The commit, the date, the
 * fixture count, the provenance rows and their seconds are read from the
 * snapshot as it is drawn, because the scoreboard's cached result is keyed
 * without timings and could otherwise carry an earlier run's.
 */
(function (TV) {
  "use strict";

  var el = TV.el;
  var ORDER = [["scoreboard"], ["sections", "collection"]];
  var TILE_MIN = 184;
  var GUTTER = 24;
  var TRACKS = 6;

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  /* Decimals that follow the magnitude, as the headline strip in app.js formats. */
  function dp(v) {
    var a = Math.abs(v);
    return a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 4;
  }

  function num(v) {
    return isNum(v) ? TV.fmt.num(v, dp(v)) : "n/a";
  }

  function signed(v) {
    return isNum(v) ? TV.fmt.signed(v, dp(v)) : "n/a";
  }

  /* A section title inside a sentence: "the peer encoder", "the M&A propensity". */
  function nameOf(title) {
    var rest = String(title || "");
    if (/^the /i.test(rest)) rest = rest.slice(4);
    if (/^.[a-z]/.test(rest)) {
      rest = rest.charAt(0).toLowerCase() + rest.slice(1);
    }
    return "the " + rest;
  }

  /*
   * The most columns that fit, short of any count that would leave one tile
   * alone on the last row: five tiles go 5, then 3 and 2, then one per row.
   */
  function columnsFor(width, count) {
    var fit = Math.max(1, Math.floor((width + GUTTER) / (TILE_MIN + GUTTER)));
    for (var c = Math.min(fit, count); c > 1; c--) {
      if (count % c !== 1) return c;
    }
    return 1;
  }

  /* scoreboard ------------------------------------------------------------- */

  function small(text, extra) {
    return el("p", { class: "tv-tile__sub", style: extra || null }, text);
  }

  function noScore() {
    return el("p", { class: "tv-tile__value", style: { color: "var(--ink-3)", fontSize: "var(--fs-lead)", fontWeight: "var(--fw-medium)", alignSelf: "end" } }, "No score");
  }

  function pair(items) {
    return el(
      "div",
      { style: { display: "flex", flexWrap: "wrap", gap: "2px var(--space-4)", fontSize: "var(--fs-small)" } },
      items.map(function (item) {
        return el(
          "span",
          { style: { whiteSpace: "nowrap" } },
          el("span", { style: { color: "var(--ink-3)", marginRight: "var(--space-1)" } }, item[0]),
          el("span", { style: { color: "var(--ink-1)", fontWeight: "var(--fw-medium)", fontVariantNumeric: "tabular-nums" } }, item[1])
        );
      })
    );
  }

  /* One model tile: six rows, whatever state the model is in. */
  function modelTile(t) {
    var label = el("p", { class: "tv-tile__label", style: { color: "var(--ink-1)", fontSize: "var(--fs-body)" } }, t.label);
    var parts;
    if (t.state === "scored") {
      var chip = TV.chip.known(t.verdict_status) ? TV.chip(t.verdict_status) : TV.chip("refused", "No verdict");
      parts = [
        label,
        el("div", null, chip),
        el("p", { class: "tv-tile__value", style: { fontVariantNumeric: "tabular-nums" } }, num(t.score)),
        small(t.metric + (t.higher_is_better === false ? ", lower is better" : "") + (isNum(t.n) ? ", n = " + TV.fmt.int(t.n) : "")),
        pair([
          ["Baseline", num(t.baseline_score)],
          ["Lift", signed(t.lift)],
        ]),
        small("Baseline: " + (t.baseline_name || "unnamed"), { color: "var(--ink-3)" }),
      ];
    } else if (t.state === "refused") {
      var reason = t.reason || {};
      parts = [
        label,
        el("div", null, TV.chip("refused")),
        noScore(),
        el(
          "div",
          { style: { gridRow: "span 3", display: "grid", alignContent: "start", gap: "2px" } },
          reason.what && reason.what !== t.label
            ? el("p", { class: "tv-tile__sub", style: { color: "var(--ink-1)", fontWeight: "var(--fw-medium)" } }, reason.what)
            : null,
          el(
            "p",
            {
              class: "tv-tile__sub",
              title: reason.why || null,
              style: { display: "-webkit-box", WebkitBoxOrient: "vertical", WebkitLineClamp: "5", overflow: "hidden" },
            },
            reason.why || "No reason was recorded."
          )
        ),
      ];
    } else {
      parts = [
        label,
        el("div", null, el("span", { class: "tv-rail__state", style: { fontSize: "var(--fs-small)" } }, t.state === "no_headline" ? "No headline" : "Not collected")),
        noScore(),
        el("div", { style: { gridRow: "span 3" } }, small(t.sub || "")),
      ];
    }
    return el(
      "a",
      {
        class: "tv-tile",
        href: t.href || "#" + t.section,
        "data-section": t.section,
        "data-state": t.state,
        "aria-label": t.label + ": " + (t.state === "scored" ? TV.chip.label(t.verdict_status) + ", " + t.metric + " " + num(t.score) + " against " + num(t.baseline_score) : t.value),
        style: { display: "grid", gridTemplateRows: "subgrid", rowGap: "6px", alignContent: "stretch" },
      },
      parts
    );
  }

  /*
   * The tiles on a grid whose rows each tile spans as a subgrid, with a spacer
   * track between rows of tiles, in the kit's frame so the columns follow the width.
   */
  function tileGrid(body, tiles) {
    return TV.frame(body, "tiles", function (wrap, width) {
      var cols = columnsFor(width, tiles.length);
      var rows = Math.ceil(tiles.length / cols);
      var template = [];
      for (var r = 0; r < rows; r++) {
        for (var t = 0; t < TRACKS; t++) template.push("auto");
        if (r < rows - 1) template.push("var(--space-5)");
      }
      var grid = el("div", {
        class: "tv-overview-grid",
        style: {
          display: "grid",
          columnGap: GUTTER + "px",
          rowGap: "0",
          gridTemplateColumns: "repeat(" + cols + ", minmax(0, 1fr))",
          gridTemplateRows: template.join(" "),
        },
      });
      tiles.forEach(function (tile, i) {
        var node = modelTile(tile);
        node.style.gridColumn = String((i % cols) + 1);
        node.style.gridRow = Math.floor(i / cols) * (TRACKS + 1) + 1 + " / span " + TRACKS;
        grid.appendChild(node);
      });
      wrap.appendChild(grid);
    });
  }

  function toggleTable(root, view, table) {
    var btn = el("button", { class: "tv-btn", type: "button" }, "Show data");
    btn.addEventListener("click", function () {
      var showTable = table.hidden;
      table.hidden = !showTable;
      view.hidden = showTable;
      btn.textContent = showTable ? "Show scores" : "Show data";
    });
    root.appendChild(el("div", null, btn));
  }

  /*
   * The scoreboard sits on the page plane, as tiles do, but it leads the page,
   * so its heading is set as a figure title rather than the small label a set
   * of tiles carries. The kit draws no heading for it (see the override below).
   */
  function drawScoreboard(handle, f) {
    var tiles = ((f.data && f.data.tiles) || []).filter(Boolean);
    handle.root.id = "fig-overview-scoreboard";
    handle.title = f.title || "";
    handle.body.appendChild(
      el(
        "div",
        { class: "tv-figure__head" },
        el("h3", { class: "tv-figure__title" }, f.title || ""),
        f.subtitle ? el("p", { class: "tv-figure__subtitle" }, f.subtitle) : null
      )
    );
    var view = el("div");
    handle.body.appendChild(view);
    tileGrid(view, tiles);
    var table = el("div", { hidden: true });
    handle.body.appendChild(table);
    TV.tableView({ table: table, title: f.title }, {
      caption: f.title,
      columns: [
        { key: "label", label: "Model" },
        { key: "verdict", label: "Verdict" },
        { key: "metric", label: "Metric" },
        { key: "score", label: "Score", align: "right" },
        { key: "baseline_name", label: "Baseline" },
        { key: "baseline_score", label: "Baseline score", align: "right" },
        { key: "lift", label: "Lift", align: "right" },
        { key: "n", label: "n", align: "right", format: "int" },
      ],
      rows: tiles.map(function (t) {
        var scored = t.state === "scored";
        return {
          label: t.label,
          verdict: scored ? TV.chip.label(t.verdict_status) : t.value,
          metric: scored ? t.metric : "",
          score: scored ? num(t.score) : "",
          baseline_name: scored ? t.baseline_name : "",
          baseline_score: scored ? num(t.baseline_score) : "",
          lift: scored ? signed(t.lift) : "",
          n: scored ? t.n : "",
        };
      }),
    });
    toggleTable(handle.body, view, table);
  }

  /* sections --------------------------------------------------------------- */

  function statusCell(row) {
    if (row.status === "refused") return TV.chip("refused", null, { compact: true });
    if (row.status === "ok" && row.verdict_status && TV.chip.known(row.verdict_status)) {
      return TV.chip(row.verdict_status, null, { compact: true });
    }
    if (row.status === "ok") return el("span", { class: "tv-muted" }, "Collected");
    if (row.status === "not_built") return el("span", { style: { color: "var(--ink-3)" } }, "Not collected");
    return el("span", { class: "tv-muted" }, String(row.status));
  }

  /* A count, faint at zero; a refusal count that is not zero is set strong. */
  function count(v, strong) {
    var style = strong && v ? { color: "var(--ink-1)", fontWeight: "var(--fw-semibold)" } : v ? null : { color: "var(--ink-3)" };
    return el("span", { style: style }, TV.fmt.int(v));
  }

  function total(v) {
    return el("strong", { style: { fontWeight: "var(--fw-semibold)" } }, TV.fmt.int(v));
  }

  function drawSections(handle, f) {
    var d = f.data || {};
    var rows = (d.table || []).filter(Boolean);
    var totals = d.totals || {};
    TV.charts.table(handle.body, {
      caption: f.title,
      columns: [
        { key: "section", label: "Section" },
        { key: "state", label: "State" },
        { key: "figures", label: "Figures", align: "right" },
        { key: "refusals", label: "Refusals", align: "right" },
      ],
      rows: rows
        .map(function (r) {
          return {
            section: el("a", { href: r.href || "#" + r.id, style: { textDecoration: "none" } }, r.title),
            state: statusCell(r),
            figures: count(r.figures, false),
            refusals: count(r.refusals, true),
          };
        })
        .concat([
          {
            section: "All " + (totals.sections || rows.length),
            state: { text: (totals.collected || 0) + " collected", muted: true },
            figures: total(totals.figures),
            refusals: total(totals.refusals),
          },
        ]),
    });
  }

  /* collection ------------------------------------------------------------- */

  function readCollection(snapshot) {
    var sections = (snapshot && snapshot.sections) || {};
    var rows = 0;
    var seconds = 0;
    var top = null;
    Object.keys(sections)
      .sort()
      .forEach(function (id) {
        var s = sections[id] || {};
        var own = 0;
        (s.provenance || []).forEach(function (p) {
          rows += 1;
          if (p && isNum(p.seconds)) own += p.seconds;
        });
        seconds += own;
        if (!top || own > top.seconds) top = { title: s.title || id, seconds: own };
      });
    return {
      commit: (snapshot && snapshot.techval_commit) || "n/a",
      collected: (snapshot && snapshot.collected_at) || "n/a",
      fixtures: Object.keys((snapshot && snapshot.fixtures) || {}).length,
      rows: rows,
      seconds: seconds,
      top: top,
    };
  }

  function topShare(c) {
    if (!c.top || !(c.top.seconds > 0) || !(c.seconds > 0)) return "";
    var n = TV.fmt.num(c.top.seconds, 0);
    return c.top.seconds * 2 > c.seconds
      ? " Most of it is " + nameOf(c.top.title) + "'s " + n + "."
      : " The largest share is " + nameOf(c.top.title) + "'s " + n + ".";
  }

  TV.overviewCollection = readCollection;

  /* The collector's tiles, after the ones read from the snapshot as it is drawn. */
  function collectionData(d, snapshot) {
    var c = readCollection(snapshot);
    var read = [
      { label: "Commit", value: c.commit, sub: "The techval commit every section ran at" },
      { label: "Collected", value: c.collected, sub: "The date this snapshot is stamped with" },
      {
        label: "Fixtures digested",
        value: c.fixtures,
        format: "int",
        sub: c.fixtures ? "Each file's sha256 is recorded, so a changed byte shows" : "No fixture digest is recorded",
      },
      { label: "Provenance rows", value: c.rows, format: "int", sub: "Each names the entry point behind a figure and the inputs it read" },
      {
        label: "Seconds of computation",
        value: TV.fmt.num(c.seconds, c.seconds >= 100 ? 0 : 1),
        sub: "An upper bound, since a block timed for two figures counts twice." + topShare(c),
      },
    ];
    return Object.assign({}, d, { tiles: read.concat((d && d.tiles) || []) });
  }

  TV.sections.register("overview", function (root, data, snapshot) {
    var scoreboard = (data && data.figures && data.figures.scoreboard) || {};
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: {
        /* No heading from the kit: drawScoreboard sets its own, from the snapshot's figure. */
        scoreboard: {
          title: null,
          subtitle: null,
          draw: function (handle) {
            drawScoreboard(handle, scoreboard);
          },
        },
        sections: { draw: drawSections },
        collection: {
          card: true,
          data: function (d) {
            return collectionData(d, snapshot);
          },
        },
      },
    });
  });
})(window.TV);
