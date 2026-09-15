/*
 * Section renderer: datalayer.
 *
 * Five checks on the data underneath the models, each drawn as its own group so
 * a tile row sits over the charts that explain it: the debt ladders, the stock
 * splits beside Nvidia's closes, the sub-vertical taxonomy, the peer labels, and
 * ticker resolution. The snapshot sorts figure keys, so the order is set here.
 * A figure the list does not name is still drawn, after the named ones.
 *
 * The kit draws every figure. Nvidia's closes are its line chart with the
 * collector's fields read as options: dates along x, a log scale so a four-for-one
 * or ten-for-one step would read as the same size at any price, the split
 * windows shaded and named, and the largest single-day move labelled. The table
 * view lists the closes and the split windows.
 *
 * A refusal that belongs to a figure is named in that figure's data.near, and
 * the kit files the figure's handle under the refusal's name too, so app.js
 * hangs the note under the figure rather than at the foot of the section.
 *
 * Every number drawn is in the snapshot; this file formats and places them.
 */
(function (TV) {
  "use strict";

  var GROUPS = [
    ["debt_reach"],
    ["split_counts", "split_oracle", "nvda_closes"],
    ["taxonomy_by_vertical", "taxonomy_how"],
    ["peer_label_counts", "peer_groups_lost", "peer_label_funnel"],
    ["ticker_reach"],
  ];

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function closes(d) {
    var series = (d.series || [])[0];
    if (!series) return d;
    var windows = (d.windows || []).filter(Boolean);
    var mark = d.mark && isNum(d.mark.y) ? d.mark : null;
    var valueName = d.valueLabel || series.name || "Close";
    return Object.assign({}, d, {
      x: Object.assign({}, d.x, { type: "date" }),
      yScale: "log",
      yTitle: "USD, log scale",
      height: 300,
      endLabels: false,
      shade: windows.map(function (w) {
        var name = w.label || "Split window";
        return { from: w.from, to: w.to, label: name, tip: { label: name, value: "worst day " + TV.format("pct:1", w.worst_day) } };
      }),
      shadeLegend: windows.length ? "Split window, from the last filing on the old share basis to the first on the new" : null,
      points: mark
        ? [
            {
              x: mark.x,
              y: mark.y,
              label: mark.label || TV.format("signed:1", mark.move),
              tip: isNum(mark.move) ? { label: "Move that day", value: TV.format("signed:1", mark.move * 100) + "%" } : null,
            },
          ]
        : [],
      table: [
        {
          caption: "Closes",
          columns: [
            { key: "x", label: (d.x && d.x.label) || "Date", mono: true },
            { key: "y", label: valueName, align: "right", format: d.format },
          ],
          rows: (series.values || []).filter(Boolean),
        },
      ].concat(
        windows.length
          ? [
              {
                caption: "Split windows",
                columns: [
                  { key: "label", label: "Split window" },
                  { key: "from", label: "From", mono: true },
                  { key: "to", label: "To", mono: true },
                  { key: "worst_day", label: "Worst day", align: "right", format: "pct:1" },
                  { key: "worst_date", label: "On", mono: true },
                  { key: "unadjusted_move", label: "Unrestated step", align: "right", format: "pct:0" },
                ],
                rows: windows,
              },
            ]
          : []
      ),
    });
  }

  TV.sections.register("datalayer", function (root, data) {
    return TV.sections.figures(root, data, {
      order: GROUPS,
      figures: { nvda_closes: { data: closes } },
      refusals: function (id, figure) {
        return figure.data && figure.data.near;
      },
    });
  });
})(window.TV);
