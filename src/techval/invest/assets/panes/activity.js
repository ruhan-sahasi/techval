/*
 * Activity: the ledger itself, newest first. The page never edits it; the
 * file is the record and this pane is the receipt.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var IV = global.IV;
  var el = TV.el;

  var LABELS = {
    buy: "Buy",
    sell: "Sell",
    deposit: "Deposit",
    withdraw: "Withdraw",
    dividend: "Dividend",
    split: "Split",
  };

  function amount(row) {
    if (row.type === "buy" || row.type === "sell") {
      var total = row.shares * row.price;
      return (row.type === "buy" ? "−" : "+") + IV.fmt.money(total, 0);
    }
    if (row.type === "deposit") return "+" + IV.fmt.money(row.amount, 0);
    if (row.type === "withdraw") return "−" + IV.fmt.money(row.amount, 0);
    if (row.type === "dividend") return "+" + IV.fmt.money(row.amount, 2);
    if (row.type === "split") return TV.fmt.num(row.ratio, 0) + ":1";
    return "";
  }

  function detail(row) {
    if (row.type === "buy" || row.type === "sell") {
      return TV.fmt.num(row.shares, Number.isInteger(row.shares) ? 0 : 4) + " sh at " + IV.fmt.money(row.price, 2);
    }
    if (row.type === "split") return "share count times " + TV.fmt.num(row.ratio, 0);
    return "";
  }

  function render(host, snapshot) {
    var rows = snapshot.activity;
    var card = IV.card("The ledger", rows.length + " transactions, newest first. Edit portfolio.yaml to change the record.");
    var table = el(
      "table",
      { class: "tv-table" },
      el("caption", { class: "tv-visually-hidden" }, "Transactions"),
      el(
        "thead",
        null,
        el(
          "tr",
          null,
          ["Date", "Type", "Symbol", "Detail", "Amount"].map(function (label, i) {
            return el("th", { scope: "col", class: i === 4 ? "tv-num" : null }, label);
          })
        )
      ),
      el(
        "tbody",
        null,
        rows.map(function (row) {
          var flow = row.type === "buy" ? -1 : row.type === "withdraw" ? -1 : row.type === "split" ? 0 : 1;
          return el(
            "tr",
            null,
            el("th", { scope: "row" }, row.date),
            el("td", null, LABELS[row.type] || row.type),
            el("td", { style: { fontWeight: "650" } }, row.symbol || ""),
            el("td", { class: "tv-muted" }, detail(row)),
            el("td", { class: (flow > 0 ? "iv-pos " : flow < 0 ? "iv-neg " : "") + "tv-num" }, amount(row))
          );
        })
      )
    );
    card.appendChild(el("div", { class: "tv-table-wrap", tabindex: "0", role: "region", "aria-label": "Transactions" }, table));
    host.appendChild(card);
  }

  IV.panes.activity = { title: "Activity", render: render };
})(window);
