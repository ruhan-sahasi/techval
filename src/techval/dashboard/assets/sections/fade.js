/*
 * Section renderer: fade.
 *
 * The layout reads as the argument does. First the model against its baselines
 * and the fitted curve against the typed schedule, side by side. Then one real
 * company: Datadog's two growth paths beside what each path is worth, so the
 * values sit next to the paths that produce them. Then the filing depth that
 * limits all of it.
 *
 * Charts are drawn by the kit's default loop, one group at a time. Two things
 * the loop does not do are done here: the values card, which holds the revenue
 * tiles and, when the snapshot has them, the per-share tiles in one card beside
 * the paths; and the survivorship note under the curve, which is a limit of the
 * figure and not a refusal. Every string and number comes from the snapshot.
 */
(function (TV) {
  "use strict";

  var el = TV.el;
  var ORDER = [["error_by_horizon", "fade_curve"], ["ddog_paths"], ["depth"]];

  function subset(data, ids) {
    var figures = {};
    ids.forEach(function (id) {
      if (data.figures && data.figures[id]) figures[id] = data.figures[id];
    });
    return { id: data.id, figures: figures, provenance: data.provenance || [] };
  }

  function provenanceFor(data, id) {
    return (data.provenance || []).filter(function (p) {
      return p && p.figure === id;
    })[0];
  }

  /* A stated limit of a figure. Styled as a note, and not labelled a refusal. */
  function limitNote(note) {
    return el(
      "div",
      { class: "tv-note", role: "note" },
      el("p", null, el("strong", null, note.what), note.why ? ". " + note.why : "")
    );
  }

  function valuesCard(grid, data, handles) {
    var figures = data.figures || {};
    var revenue = figures.ddog_revenue;
    if (!revenue) return;
    var card = TV.figure(grid, {
      id: "ddog_revenue",
      anchor: "fig-" + data.id + "-ddog_revenue",
      title: revenue.title,
      subtitle: revenue.subtitle,
      provenance: provenanceFor(data, "ddog_revenue"),
    });
    TV.charts.tiles(card.body, revenue.data || {});
    handles.ddog_revenue = card;

    var dcf = figures.ddog_dcf;
    if (dcf) {
      var block = el(
        "div",
        { class: "tv-fade-dcf", "data-figure-id": "ddog_dcf", style: { display: "grid", gap: "var(--space-3)", marginTop: "var(--space-5)" } },
        el(
          "div",
          { class: "tv-figure__head" },
          /* layout.css resets margins on h1 to h3 only, and this heading sits inside a card's h3. */
          el("h4", { class: "tv-figure__title", style: { margin: "0" } }, dcf.title),
          dcf.subtitle ? el("p", { class: "tv-figure__subtitle" }, dcf.subtitle) : null
        )
      );
      card.body.appendChild(block);
      TV.charts.tiles(block, dcf.data || {});
      handles.ddog_dcf = { root: block, body: block, title: dcf.title || "", addNote: card.addNote };
    } else if (revenue.data && revenue.data.refusalSlot) {
      /* The refusal of the per-share values names this slot, so app.js hangs it here. */
      handles.ddog_dcf = { root: card.root, body: card.body, title: revenue.data.refusalSlot, addNote: card.addNote };
    }
  }

  TV.sections.register("fade", function (root, data) {
    var figures = data.figures || {};
    var handles = {};
    var drawn = {};

    function draw(ids) {
      var got = TV.sections.figures(root, subset(data, ids));
      Object.keys(got).forEach(function (id) {
        handles[id] = got[id];
        drawn[id] = true;
      });
      return got;
    }

    draw(ORDER[0]);
    var curve = handles.fade_curve;
    if (curve && curve.notes && figures.fade_curve.note) {
      curve.notes.appendChild(limitNote(figures.fade_curve.note));
    }

    var paths = draw(ORDER[1]).ddog_paths;
    var grid = paths && paths.root.parentNode;
    if (!grid && figures.ddog_revenue) {
      grid = el("div", { class: "tv-grid" });
      root.appendChild(grid);
    }
    if (grid) valuesCard(grid, data, handles);
    drawn.ddog_revenue = drawn.ddog_dcf = true;

    draw(ORDER[2]);

    /* Anything a later collector adds still reaches the page. */
    draw(
      Object.keys(figures).filter(function (id) {
        return !drawn[id];
      })
    );
    return handles;
  });
})(window.TV);
