/*
 * Section renderer: propensity.
 *
 * Layout carries the argument, so the figures are drawn in a fixed order rather
 * than the snapshot's (which is sorted by key): the sample tiles, then the
 * verdict and what moves it, then the coefficients that will not hold still,
 * then the cycle and the calibration, then the top of the list and the list
 * itself.
 *
 * The kit draws every figure. What this file adds is the reading of the
 * collector's fields as kit options, and nothing is computed:
 *
 *   auc_by_fold,       dumbbells whose gap label sits beside the pair it
 *   recovered_deals,   describes, on the rows the collector flagged; calibration
 *   calibration        draws buckets under the row floor hollow and writes each
 *                      bucket's row count in a column of its own;
 *   coefficient_signs  a heat grid with feature names in the mono face at full
 *                      length, rows grouped by whether their sign changes and
 *                      colour classes on the collector's fixed breaks;
 *   base_rate_by_year  one context series, so it takes the baseline colour, not
 *                      the orange that means the named comparison;
 *   target_list        the ranked names as a table, which is how a list is read.
 *
 * Notes under a figure are the kit's cautions: the figure stands, but fragile.
 */
(function (TV) {
  "use strict";

  var ORDER = [
    ["sample"],
    ["auc_by_fold", "recovered_deals"],
    ["coefficient_signs"],
    ["base_rate_by_year", "calibration"],
    ["precision_at_k", "recall_at_k"],
    ["target_list"],
  ];

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function isDumbbell(d) {
    return Array.isArray(d.series) && d.series.length === 2 && Array.isArray(d.rows);
  }

  /*
   * The collector's gap, labelled beside the pair on the rows it flagged. Row
   * labels get 36% of the width, the share this section's own dumbbell gave
   * them, so a fold's date range keeps to one line.
   */
  function gapLabels(d) {
    if (!isDumbbell(d)) return d;
    return Object.assign({ labelWidth: 0.36 }, d, { labels: { text: "gap", rows: "flagged" } });
  }

  function calibration(d) {
    if (!isDumbbell(d)) return d;
    var a = d.series[0];
    var b = d.series[1];
    var fmt = d.tableFormat || d.format;
    var anyThin = d.rows.some(function (r) {
      return r && r.thin;
    });
    var out = Object.assign({}, gapLabels(d), {
      rows: d.rows.map(function (r) {
        return Object.assign({}, r, {
          hollow: !!r.thin,
          aside: d.showCounts && isNum(r.n) ? TV.fmt.int(r.n) + (r.thin ? ", thin" : "") : null,
        });
      }),
      hollowLegend: anyThin && isNum(d.thin_below) ? "Fewer than " + TV.fmt.int(d.thin_below) + " rows" : null,
      asideHeader: d.showCounts ? "Rows" : null,
      asideAlign: "end",
    });
    var columns = [
      { key: "label", label: d.labelHeader || "Row" },
      { key: "a", label: a.name, align: "right", format: fmt },
      { key: "b", label: b.name, align: "right", format: fmt },
      { key: "gap", label: d.gapLabel || "Difference", align: "right", format: d.gapFormat || "signed:3" },
    ];
    if (d.showCounts) {
      columns.push({ key: "n", label: "Rows", align: "right", format: "int" });
      if (anyThin) columns.push({ key: "thin", label: "Thin" });
    }
    out.table = {
      columns: columns,
      rows: d.rows.map(function (r) {
        var v = r.values || {};
        return { label: r.label, a: v[a.key], b: v[b.key], gap: r.gap, n: r.n, thin: r.thin ? "yes" : "no" };
      }),
    };
    return out;
  }

  function signGrid(d) {
    var rows = d.rows || [];
    var mean = d.mean || [];
    var sd = d.sd || [];
    var flips = d.flips || [];
    var cols = d.cols || [];
    var values = d.values || [];
    return Object.assign({}, d, {
      mono: true,
      labelAlign: "start",
      cellMax: 88,
      cellHeight: 24,
      cellLabels: false,
      rowNotes: rows.map(function (r, i) {
        return r === d.annotate ? "mean " + TV.format("signed:2", mean[i]) + ", sd " + TV.format("num:2", sd[i]) : null;
      }),
      rowTips: rows.map(function (r, i) {
        return [
          { label: "Mean across fits", value: mean[i], format: "signed:2" },
          { label: "SD across fits", value: sd[i], format: "num:2" },
        ];
      }),
      table: {
        columns: [{ key: "feature", label: d.rowHeader || "Feature", mono: true }]
          .concat(
            cols.map(function (c, j) {
              return { key: "c" + j, label: c, align: "right", format: d.format };
            })
          )
          .concat([
            { key: "mean", label: "Mean", align: "right", format: "signed:2" },
            { key: "sd", label: "SD", align: "right", format: "num:2" },
            { key: "flips", label: "Changes sign" },
          ]),
        rows: rows.map(function (r, i) {
          var out = { feature: r, mean: mean[i], sd: sd[i], flips: flips[i] ? "yes" : "no" };
          cols.forEach(function (_, j) {
            out["c" + j] = values[i] ? values[i][j] : null;
          });
          return out;
        }),
      },
    });
  }

  /* A single context series: the baseline's grey, since orange names the one comparison. */
  function contextSeries(d) {
    return Object.assign({}, d, {
      rows: (d.rows || []).map(function (r) {
        return r && r.role === "alt" ? Object.assign({}, r, { role: "baseline" }) : r;
      }),
    });
  }

  function targetTable(d) {
    if (!Array.isArray(d.table)) return d;
    return Object.assign({}, d, {
      columns: [
        { key: "rank", label: "Rank", align: "right" },
        { key: "ticker", label: "Ticker", mono: true },
        { key: "name", label: "Company" },
        { key: "sub_vertical", label: "Sub-vertical" },
        { key: "probability", label: "Stated probability", align: "right", format: d.format || "pct:1" },
        { key: "drivers", label: "Largest contributions", mono: true },
      ],
      rows: d.table,
      table: null,
    });
  }

  TV.sections.register("propensity", function (root, data) {
    var figures = (data && data.figures) || {};
    var target = figures.target_list;
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: {
        auc_by_fold: { data: gapLabels },
        recovered_deals: { data: gapLabels },
        calibration: { data: calibration },
        coefficient_signs: { wide: true, data: signGrid },
        base_rate_by_year: { data: contextSeries },
        target_list: target && target.data && Array.isArray(target.data.table) ? { wide: true, kind: "table", data: targetTable } : null,
      },
    });
  });
})(window.TV);
