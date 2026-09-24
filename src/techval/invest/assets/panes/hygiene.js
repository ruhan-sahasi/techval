/*
 * Hygiene: concentration, exposure, drift and coverage. Arithmetic on the
 * owner's own book and targets; the only judgement is the honest bucketing.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var IV = global.IV;
  var el = TV.el;

  function render(host, snapshot) {
    var hygiene = snapshot.hygiene;

    host.appendChild(
      el(
        "div",
        { class: "iv-tiles" },
        IV.tile("Top five weight", TV.fmt.pct(hygiene.top5_share, 0), "of the whole book, cash included"),
        IV.tile("Engine coverage", TV.fmt.pct(hygiene.coverage.covered_value_share, 0), hygiene.coverage.covered_positions + " of " + hygiene.coverage.positions + " positions valued", { feature: true }),
        IV.tile("Buckets held", String(Object.keys(hygiene.exposure).length), "sub-verticals, funds, crypto, cash")
      )
    );

    var weights = TV.figure(host, {
      title: "Where the money sits",
      subtitle: "Every position and cash, largest first",
      id: "iv-weights",
    });
    TV.charts.hbar(weights.body, {
      format: "pct:1",
      legend: false,
      rows: snapshot.hygiene.weights.map(function (w) {
        return {
          label: w.symbol,
          value: w.weight,
          role: w.kind === "cash" ? "total" : w.kind === "stock" ? "model" : "alt",
          note: IV.fmt.money(w.value, 0) + " as " + w.kind,
        };
      }),
    });

    var exposure = TV.figure(host, {
      title: "Exposure by bucket",
      subtitle: "TMT names take the engine's own sub-verticals; the rest say what they are",
      id: "iv-exposure",
    });
    var buckets = Object.keys(hygiene.exposure).sort(function (a, b) {
      return hygiene.exposure[b] - hygiene.exposure[a];
    });
    TV.charts.hbar(exposure.body, {
      format: "pct:1",
      legend: false,
      rows: buckets.map(function (bucket) {
        return { label: bucket, value: hygiene.exposure[bucket], role: "model" };
      }),
    });

    if (hygiene.drift.length) {
      var drift = TV.figure(host, {
        title: "Drift against your own targets",
        subtitle: "Weight held less weight targeted, in points; the targets are yours, not the engine's",
        id: "iv-drift",
      });
      TV.charts.hbar(drift.body, {
        format: "points",
        legend: false,
        rows: hygiene.drift.map(function (g) {
          return {
            label: g.symbol,
            value: g.gap,
            role: g.gap >= 0 ? "pos" : "neg",
            note: TV.fmt.pct(g.weight, 1) + " held against " + TV.fmt.pct(g.target, 1) + " targeted",
          };
        }),
      });
      drift.body.appendChild(
        IV.note("A positive bar is overweight. The arithmetic to close a gap is a trade only you can decide to make.")
      );
    }
  }

  IV.panes.hygiene = { title: "Hygiene", render: render };
})(window);
