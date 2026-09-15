/*
 * Section renderer: fade.
 *
 * The layout reads as the argument does. First the model against its baselines
 * and the fitted curve against the typed schedule, side by side. Then one real
 * company: Datadog's two growth paths beside what each path is worth, so the
 * values sit next to the paths that produce them. Then the filing depth that
 * limits all of it.
 *
 * The kit draws all of it. The revenue tiles are drawn in a card beside the
 * paths, and the per-share tiles, when the snapshot has them, are a part of that
 * card under their own heading, with their table and entry point joining the
 * card's. When the per-share values are refused, the refusal names the slot
 * they would have filled (refusalSlot), so the note hangs under the same card.
 * The survivorship note under the curve is the kit's caution: a limit of the
 * figure, not a refusal. Every string and number comes from the snapshot.
 */
(function (TV) {
  "use strict";

  var ORDER = [["error_by_horizon", "fade_curve"], ["ddog_paths", "ddog_revenue"], ["depth"]];

  TV.sections.register("fade", function (root, data) {
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: { ddog_revenue: { card: true } },
      parts: { ddog_dcf: "ddog_revenue" },
      refusals: function (id, figure) {
        var slot = figure.data && figure.data.refusalSlot;
        return typeof slot === "string" ? [slot] : [];
      },
    });
  });
})(window.TV);
