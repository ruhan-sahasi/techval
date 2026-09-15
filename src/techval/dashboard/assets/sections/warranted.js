/*
 * Section renderer: warranted.
 *
 * The section argues in two halves, so the layout is fixed here and does not
 * follow the snapshot's key order (which is alphabetical): the deflation beside
 * the model against its baselines, then the folds beside the two noise tests,
 * then the screen and the re-rating at full width, then the panel tiles.
 *
 * Everything else the figures ask for is the kit's: a figure whose data carries
 * "table" gets that fuller table in place of the chart's own, so a screen row's
 * traded and warranted multiples are reachable without hovering; the role
 * "model-muted" is the kit's tint of the model colour, so the pooled score can
 * sit behind the differenced one while both stay the model's; and "legend"
 * false leaves the deflation bars to their row labels.
 */
(function (TV) {
  "use strict";

  var ORDER = ["deflation", "baselines", "folds", "noise", "screen", "rerating", "panel"];

  TV.sections.register("warranted", function (root, data) {
    return TV.sections.figures(root, data, { order: ORDER });
  });
})(window.TV);
