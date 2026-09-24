/*
 * Holdings: the dense sortable book. Click a numeric header to sort; signed
 * cells wear sign colours; the engine's call sits beside each covered name as
 * a chip and the uncovered say what they are instead.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var IV = global.IV;
  var el = TV.el;

  var COLUMNS = [
    { key: "symbol", label: "Position" },
    { key: "shares", label: "Shares", num: true },
    { key: "price", label: "Price", num: true },
    { key: "day_pct", label: "Day", num: true, signed: true },
    { key: "value", label: "Value", num: true },
    { key: "weight", label: "Weight", num: true },
    { key: "cost", label: "Cost", num: true },
    { key: "unrealized", label: "Unreal $", num: true, signed: true },
    { key: "unrealized_pct", label: "Unreal %", num: true, signed: true },
    { key: "realized", label: "Realized", num: true, signed: true },
    { key: "engine", label: "Engine" },
  ];

  function cellText(row, column) {
    var v = row[column.key];
    switch (column.key) {
      case "shares":
        return TV.fmt.num(v, Number.isInteger(v) ? 0 : 4);
      case "price":
        return IV.fmt.money(v, 2);
      case "day_pct":
        return IV.fmt.signedPct(v);
      case "value":
      case "cost":
        return IV.fmt.money(v, 0);
      case "weight":
        return TV.fmt.pct(v, 1);
      case "unrealized":
      case "realized":
        return IV.fmt.signedMoney(v);
      case "unrealized_pct":
        return v === null ? "n/a" : IV.fmt.signedPct(v, 1);
      default:
        return v === null || v === undefined ? "" : String(v);
    }
  }

  function engineCell(row) {
    if (row.engine) {
      return TV.chip(row.engine.call, row.engine.call + " " + TV.fmt.pct(Math.abs(row.engine.residual_log), 0), { compact: true });
    }
    return el("span", { class: "tv-muted" }, row.kind === "stock" ? "no model" : row.kind);
  }

  function render(host, snapshot) {
    var rows = snapshot.positions;
    var over = snapshot.overview;

    var head = el(
      "tr",
      null,
      COLUMNS.map(function (c) {
        var attrs = { scope: "col", class: c.num ? "tv-num" : null };
        if (c.num) attrs["data-sort"] = "";
        return el("th", attrs, c.label);
      })
    );

    var body = el(
      "tbody",
      null,
      rows.map(function (row) {
        return el(
          "tr",
          null,
          COLUMNS.map(function (c, i) {
            var cls = c.num ? "tv-num" : null;
            if (c.signed) cls = IV.signClass(row[c.key], cls) || cls;
            var content = c.key === "engine" ? engineCell(row) : cellText(row, c);
            if (c.key === "symbol") {
              content = [el("span", { style: { fontWeight: "650" } }, row.symbol), el("span", { class: "tv-muted" }, " " + row.kind)];
            }
            var attrs = { class: cls };
            if (c.num) attrs["data-value"] = typeof row[c.key] === "number" ? String(row[c.key]) : "";
            return i === 0 ? el("th", Object.assign({ scope: "row" }, attrs), content) : el("td", attrs, content);
          })
        );
      })
    );

    var table = el(
      "table",
      { class: "tv-table" },
      el("caption", { class: "tv-visually-hidden" }, "Holdings"),
      el("thead", null, head),
      body
    );
    var wrap = el("div", { class: "tv-table-wrap iv-book", tabindex: "0", role: "region", "aria-label": "Holdings, sortable table" }, table);
    IV.sortable(wrap);

    var card = IV.card(
      "The book",
      rows.length + " positions and " + IV.fmt.money(over.cash, 0) + " cash. Click a numeric header to sort."
    );
    card.appendChild(wrap);
    card.appendChild(
      IV.note(
        "Cheap and rich are the warranted-multiple residual at " +
          "the recorded panel's latest date, and carry that model's verdict on the engine pane. " +
          "ETFs and crypto are priced and weighed, never valued from filings."
      )
    );
    host.appendChild(card);
  }

  IV.panes.holdings = { title: "Holdings", render: render };
})(window);
