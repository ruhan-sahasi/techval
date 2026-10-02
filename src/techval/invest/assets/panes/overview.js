/*
 * Overview: the hero number, growth against the benchmark, and the tiles.
 * Reads overview, performance and hygiene; computes nothing itself.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var IV = global.IV;
  var el = TV.el;

  function render(host, snapshot) {
    var over = snapshot.overview;
    var perf = snapshot.performance;

    /* The hero card: total value, today, and since the first deposit. */
    var value = el("div", { class: "iv-hero__value" }, "");
    IV.countUp(value, over.value, function (v) {
      return IV.fmt.money(v, 0);
    }, "hero");
    var dayClass = IV.signClass(over.day_abs) || "";
    var sinceClass = IV.signClass(over.twr_pct) || "";
    var hero = el(
      "section",
      { class: "iv-card" },
      el(
        "div",
        { class: "iv-hero" },
        el("span", { class: "iv-hero__label" }, "Total value"),
        value,
        el(
          "div",
          { class: "iv-hero__delta" },
          el("span", { class: dayClass }, IV.fmt.signedMoney(over.day_abs) + " (" + IV.fmt.signedPct(over.day_pct) + ") on " + snapshot.meta.prices_as_of),
          el("span", { class: sinceClass }, IV.fmt.signedPct(over.twr_pct) + " time-weighted since " + snapshot.meta.first_date),
          el("span", null, IV.fmt.money(over.cash, 0) + " cash")
        )
      )
    );

    /* The tile row. Coverage is the featured tile: it is the honest one. */
    var vs = over.twr_pct - over.benchmark_twr_pct;
    var tiles = el(
      "div",
      { class: "iv-tiles" },
      IV.tile("Last session", IV.fmt.signedPct(over.day_pct), IV.fmt.signedMoney(over.day_abs) + " on " + snapshot.meta.prices_as_of, { signed: over.day_pct }),
      IV.tile("vs " + snapshot.meta.benchmark, TV.fmt.points(vs, 1), "time-weighted, since inception", { signed: vs }),
      IV.tile(
        "Top 5 weight",
        TV.fmt.pct(over.top5_share, 0),
        over.n_positions ? over.n_positions + " positions and cash" : "cash only, so far"
      ),
      IV.tile(
        "Engine coverage",
        TV.fmt.pct(over.covered_value_share, 0),
        over.cheap + " cheap · " + over.rich + " rich · " + over.uncovered + " unvalued",
        { feature: true }
      )
    );

    /* Growth of one dollar against the benchmark, with a brokerage-style range. */
    var fig = TV.figure(host, {
      title: "Growth of $1 against " + perf.benchmark,
      subtitle: "Time-weighted, so deposits and withdrawals move neither line. Both lines start at $1 on the range's first session.",
      id: "iv-growth",
    });
    var buttons = {};
    var bar = el("div", { class: "iv-range", role: "group", "aria-label": "Chart range" });
    IV.range.KEYS.forEach(function (key) {
      var b = el("button", { class: "iv-range__btn", type: "button", "aria-pressed": "false" }, key);
      b.addEventListener("click", function () {
        draw(key);
      });
      buttons[key] = b;
      bar.appendChild(b);
    });
    fig.root.firstChild.appendChild(bar);

    function draw(key) {
      IV.range.current = key;
      IV.pref.set("range", key);
      Object.keys(buttons).forEach(function (k) {
        buttons[k].setAttribute("aria-pressed", k === key ? "true" : "false");
      });
      var i = IV.range.startIndex(perf.dates, key);
      var dates = perf.dates.slice(i);
      var mine = IV.range.rebase(perf.growth, i);
      var theirs = IV.range.rebase(perf.benchmark_growth, i);
      /* A redraw replaces the chart, its legend and its table view together. */
      TV.clear(fig.body);
      TV.clear(fig.legend);
      TV.clear(fig.table);
      TV.charts.line(fig.body, {
        x: { type: "date", label: "Date" },
        format: "num:2",
        series: [
          {
            name: snapshot.meta.name,
            role: "model",
            values: dates.map(function (d, j) {
              return { x: d, y: mine[j] };
            }),
          },
          {
            name: perf.benchmark,
            role: "baseline",
            values: dates.map(function (d, j) {
              return { x: d, y: theirs[j] };
            }),
          },
        ],
      });
    }
    var remembered = IV.pref.get("range", "ALL");
    draw(IV.range.current || (IV.range.KEYS.indexOf(remembered) >= 0 ? remembered : "ALL"));

    /* Movers: the day's largest absolute moves. */
    var movers = IV.card("Biggest movers", "The three largest moves in dollars, session of " + snapshot.meta.prices_as_of);
    if (!over.movers.length) {
      movers.appendChild(IV.note("Nothing moved: the book holds no positions yet."));
    }
    over.movers.forEach(function (m) {
      movers.appendChild(
        el(
          "div",
          { class: "iv-hero__delta" },
          el("span", { style: { minWidth: "56px", fontWeight: "650" } }, m.symbol),
          el("span", { class: IV.signClass(m.day_abs) }, IV.fmt.signedMoney(m.day_abs)),
          el("span", { class: IV.signClass(m.day_pct) }, IV.fmt.signedPct(m.day_pct))
        )
      );
    });

    var fresh = IV.freshness(snapshot);
    if (fresh) host.appendChild(fresh);
    host.appendChild(hero);
    host.appendChild(tiles);
    /* fig is already on host; move it after the tiles. */
    host.appendChild(fig.root);
    host.appendChild(movers);
  }

  IV.panes.overview = { title: "Overview", render: render };
})(window);
