/*
 * Section renderer: signal.
 *
 * The snapshot stores figures with sorted keys, so the default loop would draw
 * them alphabetically. Order carries the argument here: the evidence tiles,
 * then the two t-statistics on one axis, then the coefficient series that
 * produced them, then why the naive standard error is too small and what the
 * buckets earned. So this renderer puts the figures in that order and hands
 * them to the kit's default loop, which draws each by its kind. A figure the
 * list does not name is still drawn, after the named ones.
 *
 * It returns the figure handles so app.js can hang refusal notes under the
 * figures they name.
 */
(function (TV) {
  "use strict";

  var ORDER = ["evidence", "t_statistics", "ic_by_date", "ic_autocorrelation", "bucket_returns"];

  TV.sections.register("signal", function (root, data) {
    var figures = (data && data.figures) || {};
    var ordered = {};
    ORDER.forEach(function (id) {
      if (Object.prototype.hasOwnProperty.call(figures, id)) ordered[id] = figures[id];
    });
    Object.keys(figures).forEach(function (id) {
      if (!Object.prototype.hasOwnProperty.call(ordered, id)) ordered[id] = figures[id];
    });
    return TV.sections.figures(root, Object.assign({}, data, { figures: ordered }));
  });
})(window.TV);
