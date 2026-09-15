/*
 * Section renderer: warranted.
 *
 * The section argues in two halves, so the layout is fixed here and does not
 * follow the snapshot's key order (which is alphabetical). First the audit of
 * the panel's enterprise values, because every figure after it is fitted on
 * the panel it audits: its counts as tiles in one wide card, with the filers
 * it found drawn as a part of the same card. Then the deflation beside the
 * model against its baselines, the folds beside the two noise tests, the
 * screen and the re-rating at full width, and the panel tiles.
 *
 * The screen's refusal is hung under the audit card rather than at the foot of
 * the section, since the audit is the reason for it.
 *
 * Everything else the figures ask for is the kit's: a figure whose data carries
 * "table" gets that fuller table in place of the chart's own, so a screen row's
 * traded and warranted multiples are reachable without hovering; the role
 * "model-muted" is the kit's tint of the model colour, so the pooled score can
 * sit behind the differenced one while both stay the model's; "legend" false
 * leaves the deflation bars to their row labels; and a figure's "notes" are
 * the kit's cautions.
 */
(function (TV) {
  "use strict";

  var ORDER = [["audit"], ["deflation", "baselines", "folds", "noise", "screen", "rerating", "panel"]];

  TV.sections.register("warranted", function (root, data) {
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: { audit: { card: true } },
      parts: { audit_filers: "audit" },
      refusals: function (id) {
        return id === "audit" ? ["screen"] : [];
      },
    });
  });
})(window.TV);
