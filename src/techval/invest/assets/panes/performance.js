/*
 * Performance: what the money did and which positions did it. Contribution is
 * dollars, not percentages, because dollars are what compound; the lot table
 * is the cost basis a tax return would want.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var IV = global.IV;
  var el = TV.el;

  function render(host, snapshot) {
    var perf = snapshot.performance;
    var over = snapshot.overview;

    var vs = over.twr_pct - over.benchmark_twr_pct;
    host.appendChild(
      el(
        "div",
        { class: "iv-tiles" },
        IV.tile("Time-weighted return", IV.fmt.signedPct(over.twr_pct, 1), "since " + snapshot.meta.first_date, { signed: over.twr_pct }),
        IV.tile(perf.benchmark + " over the same days", IV.fmt.signedPct(over.benchmark_twr_pct, 1), "same calendar, same flows stripped", { signed: over.benchmark_twr_pct }),
        IV.tile("Gap", TV.fmt.points(vs, 1), "portfolio less " + perf.benchmark, { signed: vs }),
        IV.tile("Realized", IV.fmt.signedMoney(perf.realized_total), "closed lots, FIFO basis", { signed: perf.realized_total }),
        IV.tile("Dividends", IV.fmt.money(perf.dividends_total, 0), "received to cash")
      )
    );

    /* Risk beside return, the book against the benchmark on the same days. */
    var risk = perf.risk;
    var pct = function (v, dp) {
      return v === null || v === undefined ? "n/a" : TV.fmt.pct(v, dp === undefined ? 1 : dp);
    };
    var riskCard = IV.card(
      "Return and risk",
      "Time-weighted, so deposits are neither return nor drawdown. Annualized only over a year or more."
    );
    var riskTable = el(
      "table",
      { class: "tv-table" },
      el("caption", { class: "tv-visually-hidden" }, "Return and risk"),
      el(
        "thead",
        null,
        el(
          "tr",
          null,
          ["", "Annualized", "Volatility", "Max drawdown", "Peak", "Trough"].map(function (label, i) {
            return el("th", { scope: "col", class: i >= 1 && i <= 3 ? "tv-num" : null }, label);
          })
        )
      ),
      el(
        "tbody",
        null,
        [
          [snapshot.meta.name, risk.portfolio],
          [perf.benchmark, risk.benchmark],
        ].map(function (pair) {
          var r = pair[1];
          return el(
            "tr",
            null,
            el("th", { scope: "row", style: { fontWeight: "650" } }, pair[0]),
            el("td", { class: IV.signClass(r.annualized, "tv-num") || "tv-num" }, r.annualized === null ? "under a year" : IV.fmt.signedPct(r.annualized, 1)),
            el("td", { class: "tv-num" }, pct(r.volatility)),
            el("td", { class: IV.signClass(r.max_drawdown, "tv-num") || "tv-num" }, r.max_drawdown ? TV.fmt.pct(r.max_drawdown, 1) : "none"),
            el("td", null, r.drawdown_peak || ""),
            el("td", null, r.drawdown_trough || "")
          );
        })
      )
    );
    riskCard.appendChild(el("div", { class: "tv-table-wrap", tabindex: "0", role: "region", "aria-label": "Return and risk" }, riskTable));
    host.appendChild(riskCard);

    /* Underwater: how far each line sits below its own running peak, every day. */
    function underwater(path) {
      var peak = -Infinity;
      return path.map(function (g) {
        peak = Math.max(peak, g);
        return g / peak - 1;
      });
    }
    if (perf.dates.length > 1) {
      var mine = underwater(perf.growth);
      var theirs = underwater(perf.benchmark_growth);
      var water = TV.figure(host, {
        title: "Below the last high",
        subtitle: "Each line's distance under its own running peak, time-weighted. The deepest point is the max drawdown above.",
        id: "iv-drawdown",
      });
      TV.charts.line(water.body, {
        x: { type: "date", label: "Date" },
        format: "pct:0",
        series: [
          {
            name: snapshot.meta.name,
            role: "model",
            values: perf.dates.map(function (d, i) {
              return { x: d, y: mine[i] };
            }),
          },
          {
            name: perf.benchmark,
            role: "baseline",
            values: perf.dates.map(function (d, i) {
              return { x: d, y: theirs[i] };
            }),
          },
        ],
      });
    }

    if (!perf.contributions.length) {
      var empty = IV.card("No positions yet", "Contribution and lots start with the first buy.");
      empty.appendChild(IV.note("The return tiles above are already honest: cash earns the book 0.0% however the benchmark moves."));
      host.appendChild(empty);
      return;
    }

    /* Contribution by position, dollars, positive right and negative left. */
    var fig = TV.figure(host, {
      title: "What each position added, in dollars",
      subtitle: "Unrealized on FIFO cost, plus realized, plus dividends. The order is the answer.",
      id: "iv-contribution",
    });
    TV.charts.hbar(fig.body, {
      format: "compact",
      legend: false,
      rows: perf.contributions.map(function (c) {
        return {
          label: c.symbol,
          value: c.total,
          role: c.total >= 0 ? "pos" : "neg",
          note: IV.fmt.signedMoney(c.unrealized) + " unrealized, " + IV.fmt.signedMoney(c.realized) + " realized, " + IV.fmt.signedMoney(c.dividends) + " dividends",
        };
      }),
      table: {
        caption: "Contribution by position",
        columns: [
          { key: "symbol", label: "Position" },
          { key: "unrealized", label: "Unrealized", align: "right", format: "signed:0" },
          { key: "realized", label: "Realized", align: "right", format: "signed:0" },
          { key: "dividends", label: "Dividends", align: "right", format: "signed:0" },
          { key: "total", label: "Total", align: "right", format: "signed:0" },
        ],
        rows: perf.contributions,
      },
    });

    /* The lots, oldest first, each against today's price. */
    var lots = IV.card("Open lots", perf.lots.length + " lots, FIFO. Split-adjusted where a split arrived.");
    var table = el(
      "table",
      { class: "tv-table" },
      el("caption", { class: "tv-visually-hidden" }, "Open lots"),
      el(
        "thead",
        null,
        el(
          "tr",
          null,
          ["Position", "Opened", "Shares", "Cost/share", "Price", "Unrealized"].map(function (label, i) {
            return el("th", { scope: "col", class: i >= 2 ? "tv-num" : null }, label);
          })
        )
      ),
      el(
        "tbody",
        null,
        perf.lots.map(function (lot) {
          return el(
            "tr",
            null,
            el("th", { scope: "row", style: { fontWeight: "650" } }, lot.symbol),
            el("td", null, lot.opened),
            el("td", { class: "tv-num" }, TV.fmt.num(lot.shares, Number.isInteger(lot.shares) ? 0 : 4)),
            el("td", { class: "tv-num" }, IV.fmt.money(lot.cost_per_share, 2)),
            el("td", { class: "tv-num" }, IV.fmt.money(lot.price, 2)),
            el("td", { class: IV.signClass(lot.unrealized, "tv-num") || "tv-num" }, IV.fmt.signedMoney(lot.unrealized))
          );
        })
      )
    );
    lots.appendChild(el("div", { class: "tv-table-wrap", tabindex: "0", role: "region", "aria-label": "Open lots" }, table));
    host.appendChild(lots);
  }

  IV.panes.performance = { title: "Performance", render: render };
})(window);
