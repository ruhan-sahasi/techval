/*
 * Section renderer: tmt.
 *
 * Three reads, laid out as three titled groups because they answer three
 * questions: what acquirers paid, what one conglomerate's parts earn, and which
 * operating figures a filing really states. The snapshot sorts figure keys, so
 * the order is set here.
 *
 * The kit draws every figure. What this file does is format the collector's
 * rows into kit options:
 *
 *   premia   a dumbbell whose gap, in points, is written beside every row in a
 *            column of its own and set bold where it passes the flag;
 *   ev_revenue  bars with the median as a reference rule and each deal's
 *            acquirer and status in the tooltip;
 *   deals    tables (TABLES below), with a refused chip in a cell the module
 *   kpis     declined to fill and a muted second line where a cell has one.
 *
 * Refusals attach to the figure they concern. A figure's data may carry
 * `refusals`, the `what` of each section refusal that belongs under it, and the
 * kit files the figure's handle under those names, so app.js hangs each note on
 * its card. Anything unnamed stays a section note.
 *
 * Every number arrives computed, the gaps in points included. This file only
 * formats.
 */
(function (TV) {
  "use strict";

  var el = TV.el;

  var ORDER = [
    { title: "Precedent transactions", ids: ["precedent_tiles", "premia", "ev_revenue", "deals"] },
    { title: "Segments and the sum of the parts", ids: ["segment_margins", "segment_revenue", "segment_operating_income"] },
    { title: "Operating metrics", ids: ["kpis"] },
  ];

  var ACRONYMS = { rpo: "RPO", arr: "ARR", arpu: "ARPU" };
  var REFUSED = { chip: "refused", chipText: "Refused" };

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  /* premia ----------------------------------------------------------------- */

  /* The gap in points. Which rows are flagged is said once, in the column heading, and shown by the strong figure. */
  function gapText(points) {
    if (!isNum(points)) return "n/a";
    return TV.fmt.signed(points, 1) + " pts";
  }

  function premia(d) {
    var series = d.series || [];
    if (series.length !== 2 || !Array.isArray(d.rows)) return d;
    var a = series[0];
    var b = series[1];
    var fmt = d.format || "pct:1";
    var threshold = isNum(d.leakPoints) ? TV.fmt.num(d.leakPoints, 0) : null;
    return Object.assign({}, d, {
      labels: "none",
      zeroLine: true,
      rowHeight: 36,
      asideHeader: threshold ? "Gap, points; bold moves more than " + threshold : "Gap, points",
      asideAlign: "end",
      rows: d.rows.map(function (r) {
        var tip = [];
        if (r.acquirer) tip.push({ label: "Acquirer", value: r.acquirer });
        if (r.announced) tip.push({ label: "Announced", value: String(r.announced) });
        return Object.assign({}, r, { aside: gapText(r.gap_points), asideStrong: !!r.flagged, tip: tip });
      }),
      table: {
        columns: [
          { key: "label", label: d.labelHeader || "Target" },
          { key: "acquirer", label: "Acquirer" },
          { key: "a", label: a.name, align: "right", format: fmt },
          { key: "b", label: b.name, align: "right", format: fmt },
          { key: "gap", label: "Gap, points", align: "right", format: "signed:1" },
          { key: "flag", label: threshold ? "Above " + threshold + " points" : "Flagged" },
        ],
        rows: d.rows.map(function (r) {
          var v = r.values || {};
          return { label: r.label, acquirer: r.acquirer || "n/a", a: v[a.key], b: v[b.key], gap: r.gap_points, flag: r.flagged ? "Flagged" : "No" };
        }),
      },
    });
  }

  function evRevenue(d) {
    var median = d.median;
    return Object.assign({}, d, {
      rows: (d.rows || []).map(function (r) {
        return Object.assign({}, r, {
          note: [r.acquirer, r.status, r.announced ? "announced " + r.announced : null].filter(Boolean).join(", "),
        });
      }),
      reference: isNum(median) ? [{ value: median, label: "Median " + TV.fmt.mult(median, 1) }] : [],
    });
  }

  /* Tables ------------------------------------------------------------------ */

  function money(v) {
    return isNum(v) ? TV.fmt.num(v, 2) : REFUSED;
  }

  function deals(d) {
    return Object.assign({}, d, {
      columns: [
        { key: "target", label: "Target" },
        { key: "acquirer", label: "Acquirer" },
        { key: "announced", label: "Announced", nowrap: true },
        { key: "status", label: "Status" },
        { key: "offer", label: "Offer, $", align: "right" },
        { key: "premium_1d", label: "Premium, last close", align: "right" },
        { key: "premium_30d", label: "Premium, 30-day mean", align: "right" },
        { key: "ev_revenue", label: "EV/Revenue", align: "right" },
      ],
      rows: (d.rows || []).map(function (r) {
        return {
          target: { text: r.ticker, sub: r.target },
          acquirer: { text: r.acquirer || "not identified", sub: r.consideration ? r.consideration + " consideration" : null },
          announced: r.announced || "n/a",
          status: { text: r.status, sub: r.closed ? "closed " + r.closed : null },
          offer: money(r.offer_price),
          premium_1d: isNum(r.premium_1d) ? TV.fmt.pct(r.premium_1d, 1) : REFUSED,
          premium_30d: isNum(r.premium_30d) ? TV.fmt.pct(r.premium_30d, 1) : REFUSED,
          ev_revenue: isNum(r.ev_revenue) ? TV.fmt.mult(r.ev_revenue, 1) : REFUSED,
        };
      }),
    });
  }

  function metricName(name) {
    return String(name)
      .split("_")
      .map(function (w, i) {
        if (ACRONYMS[w]) return ACRONYMS[w];
        return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w;
      })
      .join(" ");
  }

  function kpiValue(v, unit) {
    if (!isNum(v)) return "n/a";
    if (unit === "usd_mm") return TV.fmt.mm(v, 1);
    if (unit === "ratio") return TV.fmt.pct(v, 0);
    if (unit === "count") return TV.fmt.int(v);
    var per = /^usd_per_(\w+)$/.exec(String(unit));
    if (per) return "$" + TV.fmt.num(v, 2) + (per[1] === "period" ? "" : " a " + per[1]);
    return TV.fmt.auto(v);
  }

  function kpis(d) {
    return Object.assign({}, d, {
      columns: [
        { key: "ticker", label: "Filer" },
        { key: "metric", label: "Metric" },
        { key: "value", label: "Value", align: "right" },
        { key: "period", label: "Period end", nowrap: true },
        { key: "evidence", label: "Source", nowrap: true },
        { key: "tag", label: "Tag or phrase" },
      ],
      rows: (d.rows || []).map(function (r) {
        return {
          ticker: r.ticker,
          metric: metricName(r.metric),
          value: r.refused ? REFUSED : kpiValue(r.value, r.unit),
          period: r.period_stated ? r.period_end || "n/a" : { text: "not stated", muted: true, sub: r.read_from },
          evidence: r.evidence,
          tag: { text: r.tag_or_phrase, mono: true, sub: r.refused && r.reason ? r.reason : null },
        };
      }),
    });
  }

  /* Which filings the KPI table read, under the table. */
  function kpiSources(handle, figure) {
    var sources = (figure.data && figure.data.sources) || [];
    if (!sources.length) return;
    handle.body.appendChild(
      el(
        "ul",
        { class: "tv-figure__subtitle", style: { margin: "0", paddingLeft: "1.2em" } },
        sources.map(function (s) {
          var read = "instance " + s.instance_accession + ", filed " + s.instance_filed;
          if (s.text_accession) read += "; Item 7 of 10-K " + s.text_accession;
          var looked = isNum(s.missing) ? "; " + TV.fmt.int(s.missing) + " more metrics looked for and not disclosed" : "";
          return el("li", null, s.ticker + ": " + read + looked);
        })
      )
    );
  }

  var TABLES = { deals: deals, kpis: kpis };

  /* A table figure's rows are formatted here; any other kind is left to the kit as it came. */
  function tableData(id) {
    return function (d, figure) {
      return figure.kind === "table" ? TABLES[id](d) : d;
    };
  }

  TV.sections.register("tmt", function (root, data) {
    return TV.sections.figures(root, data, {
      order: ORDER,
      figures: {
        premia: { data: premia },
        ev_revenue: { data: evRevenue },
        deals: { wide: true, data: tableData("deals") },
        kpis: { wide: true, data: tableData("kpis"), after: kpiSources },
      },
    });
  });
})(window.TV);
