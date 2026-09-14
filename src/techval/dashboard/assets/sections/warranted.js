/*
 * Section renderer: warranted.
 *
 * The section argues in two halves, so the layout is fixed here and does not
 * follow the snapshot's key order (which is alphabetical): the deflation beside
 * the model against its baselines, then the folds beside the two noise tests,
 * then the screen and the re-rating at full width, then the panel tiles.
 *
 * Two things the kit's default loop does not do are done here with kit calls.
 * A figure whose data carries "table" gets that table in place of the chart's
 * own, so a value that would otherwise live only in a tooltip (a screen row's
 * traded and warranted multiples) is reachable without hovering. And the role
 * "model-muted" is drawn as a tint of the model colour, so the pooled score can
 * sit behind the differenced one while both stay the model's colour; its
 * figure passes legend false because the row labels already name both bars.
 */
(function (TV) {
  "use strict";

  var el = TV.el;

  var ORDER = ["deflation", "baselines", "folds", "noise", "screen", "rerating", "panel"];

  var ROLE_COLOURS = {
    "model-muted": "color-mix(in oklab, var(--c-model) 38%, var(--surface))",
  };

  function provenanceFor(data, id) {
    return ((data && data.provenance) || []).filter(function (p) {
      return p && p.figure === id;
    })[0];
  }

  function withColours(spec) {
    if (!Array.isArray(spec.rows)) return spec;
    spec.rows = spec.rows.map(function (row) {
      return row && ROLE_COLOURS[row.role] ? Object.assign({}, row, { role: ROLE_COLOURS[row.role] }) : row;
    });
    return spec;
  }

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

  TV.sections.register("warranted", function (root, data) {
    var figures = (data && data.figures) || {};
    var ids = ORDER.filter(function (id) {
      return figures[id];
    }).concat(
      Object.keys(figures).filter(function (id) {
        return ORDER.indexOf(id) < 0;
      })
    );
    var handles = {};
    var grid = null;

    ids.forEach(function (id) {
      var f = figures[id] || {};
      if (f.kind === "tiles") {
        handles[id] = tileset(root, id, f);
        grid = null;
        return;
      }
      if (!grid) {
        grid = el("div", { class: "tv-grid" });
        root.appendChild(grid);
      }
      var spec = withColours(Object.assign({ title: f.title }, f.data || {}));
      var handle = TV.figure(grid, {
        id: id,
        anchor: "fig-" + (data.id || "warranted") + "-" + id,
        title: f.title,
        subtitle: f.subtitle,
        provenance: provenanceFor(data, id),
        wide: !!(f.wide || spec.wide),
      });
      handles[id] = handle;
      var chart = TV.charts[f.kind];
      if (!chart) {
        handle.addNote({ what: id, why: "The chart kit has no chart named " + JSON.stringify(String(f.kind)) + "." });
        return;
      }
      chart(handle.body, spec);
      if (spec.legend === false) TV.clear(handle.legend);
      if (spec.table) {
        TV.clear(handle.table);
        TV.tableView(handle, spec.table);
      }
    });
    return handles;
  });
})(window.TV);
