/*
 * Section renderer: encoder.
 *
 * The figures are drawn in reading order rather than snapshot key order: the
 * diagnostics tiles, every method on one axis, the paired lift, warm against
 * cold start, the tower ablation and the truncation sweep.
 *
 * Every figure is a kit chart. The paired lift and the tower ablation are dot
 * charts whose rows carry an interval (lo and hi, one standard error and one
 * fold standard deviation), a direct label (text), extra tooltip rows (tip) and,
 * for the ablation, a group per fold cut; on both the interval is the finding,
 * whether it crosses zero. The truncation sweep draws three lines and tables
 * every method through the figure's fuller table, and the ablation's note on
 * its empty features is the kit's caution under the figure.
 *
 * Nothing here computes a figure. Every value, interval end and direct label
 * arrives in the snapshot.
 */
(function (TV) {
  "use strict";

  var ORDER = ["diagnostics", "ndcg_by_method", "paired_lift", "warm_cold", "tower_ablation", "truncation"];

  TV.sections.register("encoder", function (root, data) {
    return TV.sections.figures(root, data, { order: ORDER });
  });
})(window.TV);
