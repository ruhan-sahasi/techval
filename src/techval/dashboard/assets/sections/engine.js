/*
 * Section renderer: engine.
 *
 * The exhibits read in the order an analyst builds them, so the order is set
 * here; the snapshot's keys are sorted and carry none. The cost of capital
 * comes first, led by the valuation date, then the football field, the bridge
 * and the sensitivity grid, then the simulation and APV. Any figure not named
 * below follows, so nothing collected is dropped.
 *
 * Drawing is the kit's, with one option set: the APV title is about the row with
 * no tax shield, so that row carries the direct label, the gap between the two
 * values, rather than the row with the widest gap.
 *
 * The one addition is refusal routing: a refusal about a
 * row inside a figure (the exit-multiple terminal value, say) is named for the
 * row, and the figure lists those names under "refused", so the kit files the
 * figure's handle under each name and app.js hangs the note under the card it
 * belongs to.
 */
(function (TV) {
  "use strict";

  var ORDER = ["cost_of_capital", "football", "bridge", "sensitivity", "montecarlo", "apv"];

  TV.sections.register("engine", function (root, data) {
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: {
        apv: { data: { labels: { rows: [0], text: "gap" } } },
      },
      refusals: function (id, figure) {
        return figure.refused;
      },
    });
  });
})(window.TV);
