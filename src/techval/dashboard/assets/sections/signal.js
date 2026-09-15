/*
 * Section renderer: signal.
 *
 * Order carries the argument here: the evidence tiles, then the two
 * t-statistics on one axis, then the coefficient series that produced them,
 * then why the naive standard error is too small and what the buckets earned.
 * The snapshot stores figures with sorted keys, so the order is set here and
 * the kit draws each figure by its kind. A figure the list does not name is
 * still drawn, after the named ones.
 *
 * One kit option is set: the autocorrelation title is about adjacent dates,
 * which is the first row, so that row carries the direct label rather than the
 * row with the widest gap.
 *
 * It returns the figure handles so app.js can hang refusal notes under the
 * figures they name.
 */
(function (TV) {
  "use strict";

  var ORDER = ["evidence", "t_statistics", "ic_by_date", "ic_autocorrelation", "bucket_returns"];

  TV.sections.register("signal", function (root, data) {
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: {
        ic_autocorrelation: { data: { labels: { rows: [0] } } },
      },
    });
  });
})(window.TV);
