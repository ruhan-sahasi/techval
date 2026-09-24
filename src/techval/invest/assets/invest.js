/*
 * The investing app shell: window.IV.
 *
 * Reads the embedded snapshot, builds the rail and the top bar, and routes
 * between panes by location hash. Panes register themselves on IV.panes and
 * draw with the chart kit; the shell owns theme, motion and the helpers they
 * share. Every node is built with TV.el or TV.svg; nothing here hands a
 * string to the HTML parser.
 */

(function (global) {
  "use strict";

  var TV = global.TV;
  var el = TV.el;
  var svg = TV.svg;

  var SNAPSHOT_ID = "iv-snapshot";
  var SCHEMA = 1;

  var IV = { panes: {} };
  global.IV = IV;

  /*
   * The app opens dark unless the system asks for light; ?theme= still forces
   * either, so a headless screenshot can pin the theme. Stamped at once so the
   * first paint is already right.
   */
  function stampTheme() {
    var match = /[?&]theme=(dark|light)(?:&|#|$)/.exec(global.location ? global.location.search : "");
    if (match) {
      document.documentElement.setAttribute("data-theme", match[1]);
      return;
    }
    var lighter = global.matchMedia && global.matchMedia("(prefers-color-scheme: light)").matches;
    document.documentElement.setAttribute("data-theme", lighter ? "light" : "dark");
  }

  stampTheme();

  function motionAllowed() {
    return !!(global.matchMedia && global.matchMedia("(prefers-reduced-motion: no-preference)").matches);
  }

  /* Formatting the kit does not carry: dollars. ---------------------------- */

  function money(v, dp) {
    if (typeof v !== "number" || !isFinite(v)) return "n/a";
    var body = TV.fmt.num(Math.abs(v), typeof dp === "number" ? dp : v < 100 && v > -100 ? 2 : 0);
    return (v < 0 ? "−$" : "$") + body;
  }

  function signedMoney(v, dp) {
    if (typeof v !== "number" || !isFinite(v)) return "n/a";
    if (v === 0) return "$0";
    return (v > 0 ? "+$" : "−$") + TV.fmt.num(Math.abs(v), typeof dp === "number" ? dp : 0);
  }

  function signedPct(v, dp) {
    if (typeof v !== "number" || !isFinite(v)) return "n/a";
    return TV.fmt.signed(v * 100, typeof dp === "number" ? dp : 2) + "%";
  }

  IV.fmt = { money: money, signedMoney: signedMoney, signedPct: signedPct };

  /* The sign class the tables use; zero and non-numbers stay plain. */
  IV.signClass = function (v, base) {
    var cls = TV.signedClass(v);
    return cls ? (base ? base + " " : "") + (cls === "tv-pos" ? "iv-pos" : "iv-neg") : base || null;
  };

  /*
   * Count up once on first draw. Under reduced motion, or on any redraw, the
   * final text lands at once; the tween never carries meaning.
   */
  var counted = {};

  IV.countUp = function (node, value, format, key) {
    var done = function () {
      node.textContent = format(value);
    };
    if (!motionAllowed() || (key && counted[key])) return done();
    if (key) counted[key] = true;
    var start = null;
    var from = value * 0.75;
    function step(now) {
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / 600);
      var eased = 1 - Math.pow(1 - t, 3);
      node.textContent = format(from + (value - from) * eased);
      if (t < 1) global.requestAnimationFrame(step);
    }
    global.requestAnimationFrame(step);
  };

  /* A dense sortable table: click a header, the rows re-order in place. ----- */

  IV.sortable = function (wrap) {
    var table = wrap.querySelector("table");
    if (!table) return;
    var body = table.tBodies[0];
    var headers = Array.prototype.slice.call(table.tHead.rows[0].cells);
    headers.forEach(function (th, index) {
      if (!th.hasAttribute("data-sort")) return;
      th.setAttribute("aria-sort", "none");
      th.addEventListener("click", function () {
        var direction = th.getAttribute("aria-sort") === "descending" ? "ascending" : "descending";
        headers.forEach(function (other) {
          if (other.hasAttribute("data-sort")) other.setAttribute("aria-sort", "none");
        });
        th.setAttribute("aria-sort", direction);
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = parseFloat(a.cells[index].getAttribute("data-value"));
          var y = parseFloat(b.cells[index].getAttribute("data-value"));
          if (isNaN(x) && isNaN(y)) return 0;
          if (isNaN(x)) return 1;
          if (isNaN(y)) return -1;
          return direction === "descending" ? y - x : x - y;
        });
        rows.forEach(function (row) {
          body.appendChild(row);
        });
      });
    });
  };

  /* Shared card scaffolding for the panes. ---------------------------------- */

  IV.card = function (title, sub) {
    var node = el(
      "section",
      { class: "iv-card" },
      title ? el("h3", { class: "tv-figure__title" }, title) : null,
      sub ? el("p", { class: "tv-figure__subtitle" }, sub) : null
    );
    return node;
  };

  IV.tile = function (label, value, sub, opts) {
    opts = opts || {};
    var valueNode = el("span", { class: IV.signClass(opts.signed, "iv-tile__value") || "iv-tile__value" }, value);
    return el(
      "div",
      { class: "iv-tile" + (opts.feature ? " iv-tile--feature" : "") },
      el("span", { class: "iv-tile__label" }, label),
      valueNode,
      sub ? el("span", { class: "iv-tile__sub" }, sub) : null
    );
  };

  IV.note = function (text) {
    return el("p", { class: "iv-note" }, text);
  };

  /* The destinations, in rail order. ---------------------------------------- */

  var DESTINATIONS = [
    { id: "overview", title: "Overview" },
    { id: "holdings", title: "Holdings" },
    { id: "performance", title: "Performance" },
    { id: "engine", title: "Engine read" },
    { id: "hygiene", title: "Hygiene" },
    { id: "ideas", title: "Ideas" },
    { id: "activity", title: "Activity" },
  ];

  var ICONS = {
    overview: [["path", { d: "M2.5 2.5h4.6v4.6H2.5zM8.9 2.5h4.6v4.6H8.9zM2.5 8.9h4.6v4.6H2.5zM8.9 8.9h4.6v4.6H8.9z" }]],
    holdings: [["path", { d: "M2.5 4.4h11M2.5 8h11M2.5 11.6h11" }]],
    performance: [["path", { d: "M2.5 12.5l3.2-4.2 2.8 1.9 3.4-5.4 1.6 2.2" }]],
    engine: [["path", { d: "M2.7 11.8a5.8 5.8 0 1 1 10.6 0" }], ["path", { d: "M8 11.8l2.6-3.4" }]],
    hygiene: [["circle", { cx: 8, cy: 8, r: 5.6 }], ["path", { d: "M8 2.4V8l3.9 2.9" }]],
    ideas: [["circle", { cx: 8, cy: 8, r: 5.6 }], ["circle", { cx: 8, cy: 8, r: 2 }]],
    activity: [["circle", { cx: 8, cy: 8, r: 5.6 }], ["path", { d: "M8 4.8V8l2.4 1.7" }]],
  };

  function icon(id) {
    var shapes = ICONS[id] || [];
    var node = svg("svg", { viewBox: "0 0 16 16", "aria-hidden": "true", focusable: "false" });
    shapes.forEach(function (shape) {
      node.appendChild(svg(shape[0], shape[1]));
    });
    return node;
  }

  /* Boot ---------------------------------------------------------------------- */

  function readSnapshot() {
    var node = document.getElementById(SNAPSHOT_ID);
    if (!node) return { error: "The page carries no portfolio snapshot." };
    try {
      var data = JSON.parse(node.textContent);
      if (!data || typeof data !== "object") throw new Error("not an object");
      if (!data.meta || data.meta.schema !== SCHEMA) {
        return { error: "This page reads snapshot schema " + SCHEMA + "." };
      }
      return { snapshot: data };
    } catch (err) {
      return { error: "The snapshot could not be read: " + err.message + "." };
    }
  }

  function currentPane() {
    var hash = (global.location.hash || "").replace("#", "");
    for (var i = 0; i < DESTINATIONS.length; i++) {
      if (DESTINATIONS[i].id === hash) return hash;
    }
    return DESTINATIONS[0].id;
  }

  function boot() {
    var read = readSnapshot();
    if (read.error) {
      document.body.appendChild(el("main", { class: "iv-shell" }, el("p", { class: "iv-note" }, read.error)));
      return;
    }
    var snapshot = read.snapshot;
    document.body.appendChild(shell(snapshot));
    global.addEventListener("hashchange", function () {
      draw(snapshot);
    });
    draw(snapshot);
  }

  var refs = {};

  function shell(snapshot) {
    var meta = snapshot.meta;
    refs.links = {};
    var nav = el(
      "ul",
      { class: "iv-nav" },
      DESTINATIONS.map(function (d) {
        var link = el("a", { class: "iv-nav__link", href: "#" + d.id }, icon(d.id), d.title);
        refs.links[d.id] = link;
        return el("li", null, link);
      })
    );
    var toggle = el("button", { class: "tv-btn", type: "button" }, "Theme");
    toggle.addEventListener("click", function () {
      var root = document.documentElement;
      var dark = root.getAttribute("data-theme") !== "light";
      root.setAttribute("data-theme", dark ? "light" : "dark");
    });
    refs.title = el("h1", { class: "iv-topbar__title" }, "");
    refs.pane = el("div", { class: "iv-pane" });
    return el(
      "div",
      { class: "iv-shell" },
      el(
        "nav",
        { class: "iv-rail", "aria-label": "Portfolio" },
        el(
          "div",
          { class: "iv-brand" },
          el("span", { class: "iv-brand__name" }, meta.name),
          el("span", { class: "iv-brand__sub" }, "techval invest")
        ),
        nav,
        el(
          "div",
          { class: "iv-rail__meta" },
          el("span", null, "As of " + meta.generated),
          el("span", null, "Prices: " + meta.price_source),
          el("span", null, "Benchmark: " + meta.benchmark)
        )
      ),
      el(
        "main",
        { class: "iv-main" },
        el(
          "header",
          { class: "iv-topbar" },
          refs.title,
          el("span", { class: "iv-topbar__meta" }, "As of " + meta.generated),
          toggle
        ),
        refs.pane
      )
    );
  }

  function draw(snapshot) {
    var id = currentPane();
    DESTINATIONS.forEach(function (d) {
      if (id === d.id) refs.links[d.id].setAttribute("aria-current", "page");
      else refs.links[d.id].removeAttribute("aria-current");
      if (id === d.id) refs.title.textContent = d.title;
    });
    TV.clear(refs.pane);
    refs.pane.classList.remove("is-entering");
    var pane = IV.panes[id];
    if (!pane) {
      refs.pane.appendChild(
        el("p", { class: "iv-note" }, "The " + id + " pane is not on this build of the page.")
      );
      return;
    }
    pane.render(refs.pane, snapshot);
    if (motionAllowed()) {
      /* Force a style flush so re-adding the class replays the entrance. */
      void refs.pane.offsetWidth;
      refs.pane.classList.add("is-entering");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})(window);
