/*
 * Section renderer: engine.
 *
 * The exhibits read in the order an analyst builds them, so the order is set
 * here; the snapshot's keys are sorted and carry none. The cost of capital
 * comes first, led by the valuation date, then the football field, the bridge
 * and the sensitivity grid, then the simulation and APV. Any figure not named
 * below follows in snapshot order, so nothing collected is dropped.
 *
 * Drawing is the kit's default loop. The one addition is refusal routing: a
 * refusal about a row inside a figure (the exit-multiple terminal value, say)
 * is named for the row, and the figure lists those names under "refused", so
 * each name is added as another key for that figure's handle and app.js hangs
 * the note under the card it belongs to.
 */
(function (TV) {
  "use strict";

  var ORDER = ["cost_of_capital", "football", "bridge", "sensitivity", "montecarlo", "apv"];

  TV.sections.register("engine", function (root, data) {
    var figures = (data && data.figures) || {};
    var ordered = {};
    ORDER.concat(Object.keys(figures)).forEach(function (id) {
      if (figures[id] && !Object.prototype.hasOwnProperty.call(ordered, id)) ordered[id] = figures[id];
    });
    var handles = TV.sections.figures(root, Object.assign({}, data, { figures: ordered }));
    Object.keys(ordered).forEach(function (id) {
      var names = ordered[id].refused || [];
      names.forEach(function (name) {
        if (handles[id] && !handles[name]) handles[name] = handles[id];
      });
    });
    return handles;
  });
})(window.TV);
