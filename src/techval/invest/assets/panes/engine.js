/*
 * Engine read: what the models say about each holding, one card per name.
 * Every figure carries the verdict its model earned, and a name the engine
 * cannot value gets its refusals as footnotes, never a blank.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var IV = global.IV;
  var el = TV.el;

  function stat(label, value, sub, signed) {
    return el(
      "div",
      { class: "iv-hero" },
      el("span", { class: "iv-tile__label" }, label),
      el("span", { class: IV.signClass(signed, "iv-tile__value") || "iv-tile__value" }, value),
      sub ? el("span", { class: "iv-tile__sub" }, sub) : null
    );
  }

  function verticalLabel(read) {
    return read.sub_vertical ? read.sub_vertical.replace(/_/g, " ") : "outside TMT";
  }

  function symbolCard(host, read) {
    var fig = TV.figure(host, {
      title: read.symbol + ", " + verticalLabel(read),
      id: "iv-engine-" + read.symbol,
      wide: true,
    });

    if (read.warranted) {
      /* The line chart owns fig.legend, so the call sits in the figure's head. */
      fig.root.firstChild.appendChild(
        el(
          "div",
          { style: { marginTop: "6px" } },
          TV.chip(read.warranted.call, read.warranted.call + " by " + TV.fmt.pct(Math.abs(read.warranted.residual_log), 0) + " in log points")
        )
      );
    }

    var stats = el("div", { class: "iv-tiles" });
    if (read.dcf) {
      stats.appendChild(
        stat("Engine DCF", IV.fmt.money(read.dcf.per_share, 2), "per share, WACC " + TV.fmt.pct(read.dcf.wacc, 1))
      );
      stats.appendChild(stat("Market price", IV.fmt.money(read.dcf.price, 2), "at the snapshot's close"));
      stats.appendChild(
        stat("DCF against price", IV.fmt.signedPct(read.dcf.gap_pct, 1), "the engine's base case, not a target", read.dcf.gap_pct)
      );
    }
    if (read.warranted) {
      stats.appendChild(
        stat("Traded EV/Revenue", TV.fmt.mult(read.warranted.traded), "on " + read.warranted.as_of)
      );
      stats.appendChild(
        stat("Warranted EV/Revenue", TV.fmt.mult(read.warranted.warranted), "fitted from fundamentals, out of sample")
      );
      stats.appendChild(
        stat("Residual", TV.fmt.signed(read.warranted.z, 1) + " sd", "within-date z of the log residual", -read.warranted.z)
      );
    }
    if (stats.firstChild) fig.body.appendChild(stats);

    if (read.fade) {
      var years = read.fade.years;
      fig.body.appendChild(
        el(
          "div",
          { class: "iv-note", style: { marginTop: "12px", fontWeight: "500" } },
          "Revenue growth path: fitted against the engine's straight line, trailing " +
            (read.fade.trailing === null ? "n/a" : TV.fmt.pct(read.fade.trailing, 1))
        )
      );
      TV.charts.line(fig.body, {
        format: "pct:0",
        x: { label: "Year" },
        series: [
          {
            name: "Fitted fade",
            role: "model",
            values: years.map(function (y, i) {
              return { x: y, y: read.fade.fitted[i], lo: read.fade.lower[i], hi: read.fade.upper[i] };
            }),
          },
          {
            name: "Typed straight line",
            role: "baseline",
            values: years.map(function (y, i) {
              return { x: y, y: read.fade.assumed[i] };
            }),
          },
        ],
      });
      fig.body.appendChild(
        IV.note(
          "Years " + read.fade.basis.filter(function (b) { return b === "fitted"; }).length +
            " and earlier are fitted; the rest fall back to the typed line. " + read.fade.verdict
        )
      );
    }
    if (read.warranted) {
      fig.body.appendChild(IV.note(read.warranted.verdict));
    }
    (read.refusals || []).forEach(function (refusal) {
      fig.addNote(refusal);
    });
  }

  function render(host, snapshot) {
    var reads = Object.keys(snapshot.engine).map(function (k) {
      return snapshot.engine[k];
    });
    var covered = reads.filter(function (r) {
      return r.covered;
    });
    var uncovered = reads.filter(function (r) {
      return !r.covered;
    });

    host.appendChild(
      el(
        "div",
        { class: "iv-banner" },
        "Nothing here is advice. Each figure sits beside the baseline its model was scored against, " +
          "and the one signal tested on forward returns lost; the Ideas pane carries that verdict in full."
      )
    );

    covered
      .sort(function (a, b) {
        return a.symbol < b.symbol ? -1 : 1;
      })
      .forEach(function (read) {
        symbolCard(host, read);
      });

    if (uncovered.length) {
      var fig = TV.figure(host, {
        title: "Priced, never valued",
        subtitle: "What the engine will not pretend to value from filings",
        id: "iv-engine-uncovered",
      });
      uncovered.forEach(function (read) {
        (read.refusals || []).forEach(function (refusal) {
          fig.addNote(refusal);
        });
      });
    }
  }

  IV.panes.engine = { title: "Engine read", render: render };
})(window);
