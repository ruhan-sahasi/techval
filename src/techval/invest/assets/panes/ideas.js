/*
 * Ideas: the screen's cheap and rich, and the first thing on the pane is the
 * measured fact that this signal lost. The tables are a description of today,
 * not a forecast, and the pane never shows a name the book already holds.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var IV = global.IV;
  var el = TV.el;

  function table(host, title, rows, call) {
    var fig = TV.figure(host, {
      title: title,
      subtitle: "Against the warranted line on " + rows.as_of + ", holdings excluded, sorted by within-date z",
      id: "iv-ideas-" + call,
    });
    TV.tableView(fig, {
      caption: title,
      columns: [
        { key: "ticker", label: "Ticker" },
        { key: "sub_vertical", label: "Sub-vertical" },
        { key: "traded", label: "Traded", align: "right", format: "mult:1" },
        { key: "warranted", label: "Warranted", align: "right", format: "mult:1" },
        { key: "residual_log", label: "Residual, log", align: "right", format: "signed:2" },
        { key: "z", label: "z", align: "right", format: "signed:1" },
        { key: "call", label: "Call" },
      ],
      rows: rows.list.map(function (idea) {
        return {
          ticker: idea.ticker,
          sub_vertical: (idea.sub_vertical || "").replace(/_/g, " "),
          traded: idea.traded,
          warranted: idea.warranted,
          residual_log: idea.residual_log,
          z: idea.z,
          call: { chip: call, chipText: call },
        };
      }),
    });
    if (fig.table) fig.table.hidden = false;
    if (fig.toggle) fig.toggle.hidden = true;
  }

  function render(host, snapshot) {
    var ideas = snapshot.ideas;
    host.appendChild(el("div", { class: "iv-banner" }, ideas.verdict));
    table(host, "Cheapest to the warranted line", { list: ideas.cheap, as_of: ideas.as_of }, "cheap");
    table(host, "Richest to the warranted line", { list: ideas.rich, as_of: ideas.as_of }, "rich");
    host.appendChild(
      IV.note(
        "The residual describes where a name sits against its fundamentals, out of sample, on the recorded panel's " +
          "latest date. The panel predates the debt-ladder fix, and the results dashboard's warranted section carries " +
          "that audit in full."
      )
    );
  }

  IV.panes.ideas = { title: "Ideas", render: render };
})(window);
