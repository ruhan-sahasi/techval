/*
 * Section renderer: overview.
 *
 * A stub until the section is designed. It draws the section's figures with
 * the kit's default loop, one card per figure in snapshot order, and returns
 * the figure handles so app.js can hang refusal notes under the figures they
 * name. app.js draws the eyebrow, title, takeaway, headline, states and
 * provenance around whatever this returns.
 */
(function (TV) {
  "use strict";

  TV.sections.register("overview", function (root, data) {
    return TV.sections.figures(root, data);
  });
})(window.TV);
