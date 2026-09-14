/*
 * Section renderer: overview, the scoreboard at the top of the page.
 *
 * Three blocks. First one tile per model section, in page order, each linking
 * to its section: the verdict chip, the score, the metric and n, the baseline
 * score and the lift, then the baseline's name. The tiles share row tracks, so
 * every score sits on one line and every chip on another however long a
 * baseline's name runs, and the column count is chosen so no tile is left
 * alone in a row. Below them, every other section's state as a table beside
 * the tiles that say how the page was collected.
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
  var FIGURES = ["scoreboard", "sections", "collection"];
  var TILE_MIN = 184;
  var GUTTER = 24;

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

  function provenanceFor(data, id) {
    return (data.provenance || []).filter(function (p) {
      return p && p.figure === id;
    })[0];
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

  /*
   * Lay tiles on a grid whose rows each tile spans as a subgrid, with a spacer
   * track between rows of tiles. Placement is recomputed when the width changes.
   */
  function tileGrid(parent, tiles, tracks) {
    var grid = el("div", { class: "tv-overview-grid", style: { display: "grid", columnGap: GUTTER + "px", rowGap: "0" } });
    tiles.forEach(function (tile) {
      tile.style.display = "grid";
      tile.style.gridTemplateRows = "subgrid";
      tile.style.rowGap = "6px";
      tile.style.alignContent = "stretch";
      grid.appendChild(tile);
    });
    parent.appendChild(grid);
    var last = -1;
    function place() {
      var width = Math.floor(grid.clientWidth);
      if (!width || width === last) return;
      last = width;
      var cols = columnsFor(width, tiles.length);
      var rows = Math.ceil(tiles.length / cols);
      var template = [];
      for (var r = 0; r < rows; r++) {
        for (var t = 0; t < tracks; t++) template.push("auto");
        if (r < rows - 1) template.push("var(--space-5)");
      }
      grid.style.gridTemplateColumns = "repeat(" + cols + ", minmax(0, 1fr))";
      grid.style.gridTemplateRows = template.join(" ");
      tiles.forEach(function (tile, i) {
        var row = Math.floor(i / cols);
        tile.style.gridColumn = String((i % cols) + 1);
        tile.style.gridRow = row * (tracks + 1) + 1 + " / span " + tracks;
      });
    }
    place();
    if (typeof ResizeObserver !== "undefined") {
      var pending = false;
      new ResizeObserver(function () {
        if (pending) return;
        pending = true;
        window.requestAnimationFrame(function () {
          pending = false;
          place();
        });
      }).observe(grid);
    }
    return grid;
  }

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
      },
      parts
    );
  }

  function toggleTable(root, view, table) {
    var btn = el("button", { class: "tv-btn", type: "button" }, "View table");
    btn.addEventListener("click", function () {
      var showTable = table.hidden;
      table.hidden = !showTable;
      view.hidden = showTable;
      btn.textContent = showTable ? "View tiles" : "View table";
    });
    root.appendChild(el("div", null, btn));
  }

  function tileset(root, id, f) {
    var set = el(
      "div",
      { class: "tv-tileset", "data-figure-id": id, id: "fig-overview-" + id },
      el("h3", { class: "tv-tileset__title", style: { color: "var(--ink-1)", fontSize: "var(--fs-h3)", fontWeight: "var(--fw-semibold)" } }, f.title || ""),
      f.subtitle ? el("p", { class: "tv-figure__subtitle", style: { marginTop: "-8px" } }, f.subtitle) : null
    );
    root.appendChild(set);
    return {
      root: set,
      body: set,
      title: f.title || "",
      addNote: function (r) {
        set.appendChild(TV.refusalNote(r));
      },
    };
  }

  /* scoreboard ------------------------------------------------------------- */

  function drawScoreboard(root, data) {
    var f = data.figures.scoreboard;
    var handle = tileset(root, "scoreboard", f);
    var tiles = ((f.data && f.data.tiles) || []).filter(Boolean);
    var view = el("div");
    handle.body.appendChild(view);
    tileGrid(view, tiles.map(modelTile), 6);
    var table = el("div", { hidden: true });
    handle.body.appendChild(table);
    TV.tableView({ table: table }, {
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
    return handle;
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

  function count(v, strong) {
    return el(
      "td",
      { class: "tv-num", style: strong && v ? { color: "var(--ink-1)", fontWeight: "var(--fw-semibold)" } : v ? null : { color: "var(--ink-3)" } },
      TV.fmt.int(v)
    );
  }

  function drawSections(grid, data) {
    var f = data.figures.sections;
    var d = f.data || {};
    var rows = (d.table || []).filter(Boolean);
    var totals = d.totals || {};
    var handle = TV.figure(grid, {
      id: "sections",
      anchor: "fig-overview-sections",
      title: f.title,
      subtitle: f.subtitle,
      provenance: provenanceFor(data, "sections"),
    });
    var cell = { paddingTop: "5px", paddingBottom: "5px", verticalAlign: "middle" };
    var table = el(
      "table",
      { class: "tv-table" },
      el("caption", { class: "tv-visually-hidden" }, f.title),
      el(
        "thead",
        null,
        el(
          "tr",
          null,
          el("th", { scope: "col" }, "Section"),
          el("th", { scope: "col" }, "State"),
          el("th", { scope: "col", class: "tv-num" }, "Figures"),
          el("th", { scope: "col", class: "tv-num" }, "Refusals")
        )
      ),
      el(
        "tbody",
        null,
        rows.map(function (r) {
          return el(
            "tr",
            null,
            el("th", { scope: "row", style: cell }, el("a", { href: r.href || "#" + r.id, style: { textDecoration: "none" } }, r.title)),
            el("td", { style: cell }, statusCell(r)),
            count(r.figures, false),
            count(r.refusals, true)
          );
        })
      ),
      el(
        "tfoot",
        null,
        el(
          "tr",
          null,
          el("th", { scope: "row", style: { borderBottom: "0" } }, "All " + (totals.sections || rows.length)),
          el("td", { class: "tv-muted", style: { borderBottom: "0" } }, (totals.collected || 0) + " collected"),
          el("td", { class: "tv-num", style: { borderBottom: "0", fontWeight: "var(--fw-semibold)" } }, TV.fmt.int(totals.figures)),
          el("td", { class: "tv-num", style: { borderBottom: "0", fontWeight: "var(--fw-semibold)" } }, TV.fmt.int(totals.refusals))
        )
      )
    );
    handle.body.appendChild(el("div", { class: "tv-table-wrap", tabindex: "0", role: "region", "aria-label": f.title }, table));
    return handle;
  }

  /* collection ------------------------------------------------------------- */

  function plainTile(label, value, sub, mono) {
    return el(
      "div",
      { class: "tv-tile" },
      el("p", { class: "tv-tile__label" }, label),
      el("p", { class: "tv-tile__value", style: mono ? { fontFamily: "var(--font-mono)", fontSize: "22px", fontWeight: "var(--fw-medium)" } : { fontVariantNumeric: "tabular-nums" } }, value),
      el("p", { class: "tv-tile__sub" }, sub || "")
    );
  }

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

  function drawCollection(grid, data, snapshot) {
    var f = data.figures.collection;
    var c = readCollection(snapshot);
    var holder = el("div", { style: { display: "grid", gap: "var(--space-3)", alignContent: "start", minWidth: "0" } });
    grid.appendChild(holder);
    var handle = tileset(holder, "collection", f);
    var secondsText = TV.fmt.num(c.seconds, c.seconds >= 100 ? 0 : 1);
    var tiles = [
      plainTile("Commit", c.commit, "The techval commit every section ran at", true),
      plainTile("Collected", c.collected, "The date this snapshot is stamped with", true),
      plainTile(
        "Fixtures digested",
        TV.fmt.int(c.fixtures),
        c.fixtures ? "Each file's sha256 is recorded, so a changed byte shows" : "No fixture digest is recorded"
      ),
      plainTile("Provenance rows", TV.fmt.int(c.rows), "Each names the entry point behind a figure and the inputs it read"),
      plainTile(
        "Seconds of computation",
        secondsText,
        "An upper bound, since a block timed for two figures counts twice." + topShare(c)
      ),
    ];
    ((f.data && f.data.tiles) || []).forEach(function (t) {
      tiles.push(plainTile(t.label, isNum(t.value) ? TV.format(t.format, t.value) : String(t.value), t.sub));
    });
    tileGrid(handle.body, tiles, 3);
    return handle;
  }

  TV.sections.register("overview", function (root, data, snapshot) {
    var figures = data.figures || {};
    var handles = {};
    if (figures.scoreboard) handles.scoreboard = drawScoreboard(root, data);
    if (figures.sections || figures.collection) {
      var grid = el("div", { class: "tv-grid" });
      root.appendChild(grid);
      if (figures.sections) handles.sections = drawSections(grid, data);
      if (figures.collection) handles.collection = drawCollection(grid, data, snapshot);
    }
    /* Anything a later collector adds still reaches the page. */
    var rest = {};
    Object.keys(figures).forEach(function (id) {
      if (FIGURES.indexOf(id) < 0) rest[id] = figures[id];
    });
    var more = TV.sections.figures(root, { id: data.id, figures: rest, provenance: data.provenance || [] });
    Object.keys(more).forEach(function (id) {
      handles[id] = more[id];
    });
    return handles;
  });
})(window.TV);
