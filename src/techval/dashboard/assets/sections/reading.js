/*
 * Section renderer: reading.
 *
 * Four figures, in the order the argument runs:
 *   - every fact as each reader read it;
 *   - the errors by kind;
 *   - how often the search put the answer in front of the readers;
 *   - what the Claude readings were recorded from.
 * Each is a kit kind as it stands (table, hbar, tiles), so the kit's figure
 * loop draws them with their source lines and refusal routing.
 */
(function (TV) {
  "use strict";

  var ORDER = ["readings", "errors", "retrieval", "recording"];

  TV.sections.register("reading", function (root, data) {
    return TV.sections.figures(root, data, { order: ORDER });
  });
})(window.TV);
