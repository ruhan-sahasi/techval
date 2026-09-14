/*
 * techval dashboard chart kit: window.TV.
 *
 * Every element is built with TV.el or TV.svg and every piece of data enters
 * the DOM as a text node. Series names, tickers and figure titles are data, so
 * nothing in this file writes markup from a string.
 *
 * Colours are never hex values here. Marks are painted with var(--token)
 * references, so a theme change repaints a chart without redrawing it, and
 * colour follows the entity: a series with role "model" is --c-model wherever
 * it appears.
 *
 * Mark rules the kit enforces rather than suggests: bars at most 24px thick
 * with a 4px rounded data end and a square baseline end, 2px lines with round
 * joins, markers of radius 4 with a 2px surface ring, solid 1px hairlines for
 * grids and references, one y axis, a legend only for two or more series, and
 * a table view for every chart so no value is reachable only by hover.
 */
(function (global) {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var MINUS = "−";
  var BAR_MAX = 24;
  var RADIUS = 4;
  var GAP = 2;
  var HIT_MIN = 24;

  var TV = global.TV || {};
  global.TV = TV;

  /* Small utilities ------------------------------------------------------- */

  function isNil(v) {
    return v === null || v === undefined;
  }

  function isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function finite(list) {
    return list.filter(isNum);
  }

  function extent(list) {
    var vals = finite(list);
    if (!vals.length) return null;
    return [Math.min.apply(null, vals), Math.max.apply(null, vals)];
  }

  function distinct(list) {
    var seen = [];
    list.forEach(function (v) {
      if (seen.indexOf(v) < 0) seen.push(v);
    });
    return seen;
  }

  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  /* Element builders ------------------------------------------------------ */

  function setAttrs(node, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (key) {
      var value = attrs[key];
      if (isNil(value) || value === false) return;
      if (key === "style" && typeof value === "object") {
        Object.keys(value).forEach(function (prop) {
          if (isNil(value[prop])) return;
          if (prop.indexOf("--") === 0) node.style.setProperty(prop, value[prop]);
          else node.style[prop] = value[prop];
        });
        return;
      }
      if (key.indexOf("on") === 0 && typeof value === "function") {
        node.addEventListener(key.slice(2).toLowerCase(), value);
        return;
      }
      node.setAttribute(key, value === true ? "" : String(value));
    });
  }

  function appendChildren(node, children) {
    children.forEach(function (child) {
      if (isNil(child) || child === false) return;
      if (Array.isArray(child)) {
        appendChildren(node, child);
      } else if (typeof child === "string" || typeof child === "number") {
        node.appendChild(document.createTextNode(String(child)));
      } else {
        node.appendChild(child);
      }
    });
    return node;
  }

  function el(tag, attrs) {
    var node = document.createElement(tag);
    setAttrs(node, attrs);
    return appendChildren(node, Array.prototype.slice.call(arguments, 2));
  }

  function svg(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    setAttrs(node, attrs);
    return appendChildren(node, Array.prototype.slice.call(arguments, 2));
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  TV.el = el;
  TV.svg = svg;
  TV.clear = clear;

  /* Formatting ------------------------------------------------------------ */

  var numberFormats = {};

  function fixed(dp) {
    if (!numberFormats[dp]) {
      numberFormats[dp] = new Intl.NumberFormat("en-US", {
        minimumFractionDigits: dp,
        maximumFractionDigits: dp,
      });
    }
    return numberFormats[dp];
  }

  function dpOr(dp, fallback) {
    return isNum(dp) ? Math.max(0, Math.min(10, Math.round(dp))) : fallback;
  }

  function num(v, dp) {
    if (!isNum(v)) return "n/a";
    var body = fixed(dpOr(dp, 2)).format(Math.abs(v));
    return (v < 0 && /[1-9]/.test(body) ? MINUS : "") + body;
  }

  function pct(v, dp) {
    if (!isNum(v)) return "n/a";
    return num(v * 100, dpOr(dp, 1)) + "%";
  }

  function signed(v, dp) {
    if (!isNum(v)) return "n/a";
    var body = fixed(dpOr(dp, 2)).format(Math.abs(v));
    if (!/[1-9]/.test(body)) return body;
    return (v > 0 ? "+" : MINUS) + body;
  }

  function compact(v) {
    if (!isNum(v)) return "n/a";
    var abs = Math.abs(v);
    if (abs < 1000) {
      return num(v, Number.isInteger(v) ? 0 : abs < 10 ? 2 : 1);
    }
    var units = [
      [1e12, "T"],
      [1e9, "B"],
      [1e6, "M"],
      [1e3, "K"],
    ];
    for (var i = 0; i < units.length; i++) {
      if (abs >= units[i][0]) {
        var scaled = abs / units[i][0];
        var dp = scaled < 100 ? 1 : 0;
        var text = fixed(dp).format(scaled);
        if (parseFloat(text.replace(/,/g, "")) >= 1000 && i > 0) {
          scaled = abs / units[i - 1][0];
          text = fixed(1).format(scaled);
          return (v < 0 ? MINUS : "") + text + units[i - 1][1];
        }
        return (v < 0 ? MINUS : "") + text + units[i][1];
      }
    }
    return num(v, 0);
  }

  /* Millions, which is how the engine states money: 12,345mm. */
  function mm(v, dp) {
    if (!isNum(v)) return "n/a";
    return num(v, dpOr(dp, 0)) + "mm";
  }

  function mult(v, dp) {
    if (!isNum(v)) return "n/a";
    return num(v, dpOr(dp, 1)) + "x";
  }

  function auto(v) {
    if (!isNum(v)) return "n/a";
    var abs = Math.abs(v);
    if (Number.isInteger(v)) return num(v, 0);
    if (abs >= 100) return num(v, 0);
    if (abs >= 10) return num(v, 1);
    if (abs >= 1) return num(v, 2);
    return num(v, 4);
  }

  TV.fmt = {
    num: num,
    pct: pct,
    signed: signed,
    compact: compact,
    mm: mm,
    mult: mult,
    int: function (v) {
      return num(v, 0);
    },
    auto: auto,
  };

  /*
   * Figure data arrives as JSON, so a format is usually a string: "num:4",
   * "pct:1", "signed:4", "compact", "mm", "mult:1". A function also works.
   */
  function format(spec, v) {
    if (typeof spec === "function") return spec(v);
    if (typeof v === "string") return v;
    if (!spec) return auto(v);
    var parts = String(spec).split(":");
    var fn = TV.fmt[parts[0]];
    if (!fn) return auto(v);
    return fn(v, parts.length > 1 ? parseInt(parts[1], 10) : undefined);
  }

  TV.format = format;

  /*
   * Axis ticks sit on round numbers, so they carry only the decimals the step
   * needs: a "num:1" chart ticks 0, 5, 10 rather than 0.0, 5.0, 10.0. The unit
   * and sign conventions of the format are kept.
   */
  var TICK_TRIM = { num: true, pct: true, signed: true, mult: true, mm: true };

  function tickFormatter(spec, ticks) {
    if (typeof spec !== "string" && !isNil(spec)) return function (v) { return format(spec, v); };
    var parts = String(spec || "num").split(":");
    var name = parts[0];
    if (!Object.prototype.hasOwnProperty.call(TICK_TRIM, name)) {
      return function (v) { return format(spec, v); };
    }
    var scaleBy = name === "pct" ? 100 : 1;
    var need = 0;
    (ticks || []).forEach(function (t) {
      var text = String(tidy(t * scaleBy));
      var dot = text.indexOf(".");
      if (dot >= 0 && text.indexOf("e") < 0) need = Math.max(need, text.length - dot - 1);
    });
    /* The step decides, not the format's own decimals: fewer would print one label twice. */
    var dp = Math.min(need, 6);
    return function (v) {
      return TV.fmt[name](v, dp);
    };
  }

  /* Tokens and colour ----------------------------------------------------- */

  TV.token = function (name) {
    var prop = String(name).indexOf("--") === 0 ? name : "--" + name;
    return getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
  };

  var ROLE_VARS = {
    model: "var(--c-model)",
    baseline: "var(--c-baseline)",
    alt: "var(--c-alt)",
    third: "var(--c-third)",
    total: "var(--c-total)",
    pos: "var(--div-pos)",
    neg: "var(--div-neg)",
  };

  var ROLE_LABELS = {
    model: "Model",
    baseline: "Baseline",
    alt: "Comparison",
    third: "Third",
  };

  function color(ref) {
    if (isNil(ref)) return ROLE_VARS.model;
    if (ROLE_VARS[ref]) return ROLE_VARS[ref];
    if (String(ref).indexOf("--") === 0) return "var(" + ref + ")";
    return String(ref);
  }

  TV.color = color;

  /* Scales ---------------------------------------------------------------- */

  function tickStep(start, stop, count) {
    var raw = Math.abs(stop - start) / Math.max(1, count);
    if (!(raw > 0)) return 1;
    var step = Math.pow(10, Math.floor(Math.log10(raw)));
    var err = raw / step;
    if (err >= 7.07) step *= 10;
    else if (err >= 3.16) step *= 5;
    else if (err >= 1.41) step *= 2;
    return step;
  }

  function tidy(v) {
    return Math.abs(v) < 1e-12 ? 0 : parseFloat(v.toPrecision(12));
  }

  function linear(domain, range, opts) {
    opts = opts || {};
    var d0 = Number(domain[0]);
    var d1 = Number(domain[1]);
    if (!isFinite(d0) || !isFinite(d1)) {
      d0 = 0;
      d1 = 1;
    }
    if (d0 > d1) {
      var t = d0;
      d0 = d1;
      d1 = t;
    }
    if (d0 === d1) {
      var pad = d0 === 0 ? 1 : Math.abs(d0) * 0.1;
      d0 -= pad;
      d1 += pad;
    }
    var count = opts.ticks || 5;
    if (opts.nice) {
      for (var i = 0; i < 2; i++) {
        var step = tickStep(d0, d1, count);
        d0 = tidy(Math.floor(d0 / step) * step);
        d1 = tidy(Math.ceil(d1 / step) * step);
      }
    }
    var r0 = range[0];
    var r1 = range[1];
    var scale = function (v) {
      return r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
    };
    scale.domain = [d0, d1];
    scale.range = [r0, r1];
    scale.ticks = function (n) {
      var step = tickStep(d0, d1, n || count);
      var out = [];
      var first = Math.ceil(d0 / step - 1e-9);
      var last = Math.floor(d1 / step + 1e-9);
      for (var k = first; k <= last; k++) out.push(tidy(k * step));
      return out;
    };
    scale.invert = function (px) {
      return d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
    };
    return scale;
  }

  function band(keys, range, opts) {
    opts = opts || {};
    var padding = isNum(opts.padding) ? clamp(opts.padding, 0, 0.95) : 0.2;
    var n = Math.max(1, keys.length);
    var r0 = range[0];
    var step = (range[1] - r0) / n;
    var width = step * (1 - padding);
    var scale = function (key) {
      var i = keys.indexOf(key);
      return i < 0 ? NaN : scale.at(i);
    };
    scale.at = function (i) {
      return r0 + i * step + (step - width) / 2;
    };
    scale.bandwidth = function () {
      return width;
    };
    scale.step = function () {
      return step;
    };
    scale.domain = keys.slice();
    scale.range = [r0, range[1]];
    return scale;
  }

  TV.scale = { linear: linear, band: band };

  /* Text measurement and wrapping ----------------------------------------- */

  var measureCtx = null;
  var fontFamily = null;

  function measure(text, size, weight) {
    if (!measureCtx) {
      measureCtx = document.createElement("canvas").getContext("2d");
    }
    if (!fontFamily) {
      fontFamily = TV.token("--font-sans") || "system-ui, sans-serif";
    }
    measureCtx.font = (weight || 400) + " " + (size || 12) + "px " + fontFamily;
    return Math.ceil(measureCtx.measureText(String(text)).width * 1.04) + 1;
  }

  function wrapText(text, maxWidth, size, maxLines, weight) {
    var words = String(isNil(text) ? "" : text).split(/\s+/).filter(Boolean);
    var lines = [];
    var line = "";
    words.forEach(function (word) {
      var next = line ? line + " " + word : word;
      if (!line || measure(next, size, weight) <= maxWidth) {
        line = next;
      } else {
        lines.push(line);
        line = word;
      }
    });
    if (line) lines.push(line);
    if (!lines.length) lines.push("");
    if (maxLines && lines.length > maxLines) {
      var head = lines.slice(0, maxLines - 1);
      head.push(lines.slice(maxLines - 1).join(" "));
      lines = head;
    }
    var width = Math.max.apply(
      null,
      lines.map(function (l) {
        return measure(l, size, weight);
      })
    );
    return { lines: lines, width: width };
  }

  TV.measure = measure;

  /* Draw a multi-line label with its block centred on y. */
  function textBlock(parent, lines, x, y, attrs, lineHeight) {
    lineHeight = lineHeight || 14;
    var first = y - ((lines.length - 1) * lineHeight) / 2;
    var node = svg("text", Object.assign({ x: x, y: first, dy: "0.35em" }, attrs));
    lines.forEach(function (line, i) {
      node.appendChild(svg("tspan", { x: x, dy: i === 0 ? "0.35em" : lineHeight }, line));
    });
    node.removeAttribute("dy");
    parent.appendChild(node);
    return node;
  }

  /* Status chips ---------------------------------------------------------- */

  var STATUS = {
    beats: { label: "Beats baseline", short: "Beats", tone: "good", glyph: "check" },
    inside_noise: { label: "Inside noise", short: "In noise", tone: "warning", glyph: "tilde" },
    ties: { label: "Ties baseline", short: "Ties", tone: "neutral", glyph: "equals" },
    loses: { label: "Loses to baseline", short: "Loses", tone: "critical", glyph: "cross" },
    not_significant: { label: "Not significant", short: "Not sig.", tone: "warning", glyph: "ring" },
    refused: { label: "Refused", short: "Refused", tone: "neutral", glyph: "slash" },
  };

  var GLYPHS = {
    check: [["path", { d: "M4.6 8.3 L7 10.6 L11.4 5.6" }]],
    tilde: [["path", { d: "M4.2 9 C5.6 6.4 6.9 6.4 8 8 C9.1 9.6 10.4 9.6 11.8 7" }]],
    equals: [["path", { d: "M5 6.3 H11 M5 9.7 H11" }]],
    cross: [["path", { d: "M5.6 5.6 L10.4 10.4 M10.4 5.6 L5.6 10.4" }]],
    ring: [["circle", { cx: 8, cy: 8, r: 2.6 }]],
    slash: [["path", { d: "M5.2 10.8 L10.8 5.2" }]],
  };

  function statusInfo(status) {
    return STATUS[status] || { label: String(status || "Unknown"), tone: "neutral", glyph: "ring" };
  }

  function chipIcon(status) {
    var info = statusInfo(status);
    var icon = svg(
      "svg",
      {
        class: "tv-chip__icon tv-chip--" + info.tone,
        viewBox: "0 0 16 16",
        "aria-hidden": "true",
        focusable: "false",
      },
      svg("circle", { class: "tv-chip__disc", cx: 8, cy: 8, r: 8 })
    );
    GLYPHS[info.glyph].forEach(function (part) {
      icon.appendChild(svg(part[0], Object.assign({ class: "tv-chip__glyph" }, part[1])));
    });
    return icon;
  }

  TV.chip = function (status, text, opts) {
    var info = statusInfo(status);
    opts = opts || {};
    return el(
      "span",
      {
        class: "tv-chip tv-chip--" + info.tone + (opts.compact ? " tv-chip--compact" : ""),
        "data-status": status,
      },
      chipIcon(status),
      el("span", { class: "tv-chip__label" }, isNil(text) ? (opts.compact ? info.short || info.label : info.label) : text)
    );
  };

  TV.chip.icon = chipIcon;
  TV.chip.label = function (status) {
    return statusInfo(status).label;
  };
  TV.chip.known = function (status) {
    return Object.prototype.hasOwnProperty.call(STATUS, status);
  };
  TV.STATUS = STATUS;

  /* Figure cards ---------------------------------------------------------- */

  function refusalNote(refusal) {
    var what = refusal && refusal.what;
    var why = refusal && refusal.why;
    return el(
      "div",
      { class: "tv-note", role: "note" },
      chipIcon("refused"),
      el(
        "p",
        null,
        el("strong", null, "Refused" + (what ? ": " + what : "")),
        why ? ". " + why : ""
      )
    );
  }

  TV.refusalNote = refusalNote;

  TV.figure = function (parent, opts) {
    opts = opts || {};
    var title = el("h3", { class: "tv-figure__title" }, opts.title || "Untitled figure");
    var head = el(
      "figcaption",
      { class: "tv-figure__head" },
      title,
      opts.subtitle ? el("p", { class: "tv-figure__subtitle" }, opts.subtitle) : null
    );
    var legend = el("div", { class: "tv-figure__legend" });
    var body = el("div", { class: "tv-figure__body" });
    var table = el("div", { class: "tv-figure__table", hidden: true });
    var notes = el("div", { class: "tv-figure__notes" });
    var toggle = el("button", { class: "tv-btn", type: "button", hidden: true }, "View table");

    var provenance = opts.provenance;
    var entry = null;
    if (typeof provenance === "string") entry = provenance;
    else if (provenance && provenance.entry_point) entry = provenance.entry_point;
    var source = entry
      ? el(
          "span",
          {
            class: "tv-figure__source",
            title: provenance && provenance.inputs ? provenance.inputs.join("\n") : null,
          },
          entry
        )
      : el("span");
    var foot = el("footer", { class: "tv-figure__foot" }, toggle, source);

    var root = el(
      "figure",
      {
        class: "tv-figure" + (opts.wide ? " tv-figure--wide" : ""),
        "data-figure-id": opts.id || null,
        id: opts.anchor || (opts.id ? "fig-" + opts.id : null),
      },
      head,
      legend,
      body,
      table,
      notes,
      foot
    );

    var noted = [];
    var handle = {
      root: root,
      body: body,
      legend: legend,
      footer: foot,
      table: table,
      notes: notes,
      toggle: toggle,
      title: opts.title || "",
      addNote: function (refusal) {
        var key = (refusal && refusal.what) + " " + (refusal && refusal.why);
        if (noted.indexOf(key) >= 0) return;
        noted.push(key);
        notes.appendChild(refusalNote(refusal));
      },
      hasNote: function (refusal) {
        return noted.indexOf((refusal && refusal.what) + " " + (refusal && refusal.why)) >= 0;
      },
    };

    toggle.addEventListener("click", function () {
      var showTable = table.hidden;
      table.hidden = !showTable;
      body.hidden = showTable;
      legend.hidden = showTable;
      toggle.textContent = showTable ? "View chart" : "View table";
    });

    root.__tvHandle = handle;
    if (opts.note) {
      (Array.isArray(opts.note) ? opts.note : [opts.note]).forEach(function (n) {
        handle.addNote(typeof n === "string" ? { what: "", why: n } : n);
      });
    }
    if (parent) parent.appendChild(root);
    return handle;
  };

  function handleFor(body) {
    var fig = body && body.closest ? body.closest(".tv-figure") : null;
    return fig && fig.__tvHandle ? fig.__tvHandle : null;
  }

  /* Legend ---------------------------------------------------------------- */

  TV.legend = function (slot, items) {
    clear(slot);
    var list = el("ul", { class: "tv-legend" });
    (items || []).forEach(function (item) {
      list.appendChild(
        el(
          "li",
          { class: "tv-legend__item" },
          el("span", {
            class: "tv-key tv-key--" + (item.shape || "rect"),
            style: { "--key": color(item.color) },
            "aria-hidden": "true",
          }),
          el("span", null, item.label)
        )
      );
    });
    slot.appendChild(list);
    return list;
  };

  function legendFor(body, items) {
    var handle = handleFor(body);
    if (handle) {
      TV.legend(handle.legend, items);
    } else {
      var slot = el("div", { class: "tv-figure__legend" });
      body.appendChild(slot);
      TV.legend(slot, items);
    }
  }

  /* Tooltip --------------------------------------------------------------- */

  var tipNode = null;
  var tipOwner = null;

  function tipEnsure() {
    if (!tipNode) {
      tipNode = el("div", { class: "tv-tooltip", role: "tooltip", hidden: true });
      document.body.appendChild(tipNode);
    }
    return tipNode;
  }

  function tipFill(content) {
    var node = clear(tipEnsure());
    content = content || {};
    if (content.title) node.appendChild(el("p", { class: "tv-tooltip__title" }, content.title));
    var rows = el("div", { class: "tv-tooltip__rows" });
    (content.rows || []).forEach(function (row) {
      rows.appendChild(
        row.color
          ? el("span", {
              class: "tv-key tv-key--" + (row.shape || "line"),
              style: { "--key": color(row.color) },
              "aria-hidden": "true",
            })
          : el("span", { class: "tv-tooltip__nokey" })
      );
      rows.appendChild(el("span", { class: "tv-tooltip__label" }, row.label));
      rows.appendChild(el("span", { class: "tv-tooltip__value" }, row.value));
    });
    node.appendChild(rows);
  }

  function tipPlace(x, y) {
    var node = tipEnsure();
    node.hidden = false;
    var w = node.offsetWidth;
    var h = node.offsetHeight;
    var vw = document.documentElement.clientWidth;
    var vh = document.documentElement.clientHeight;
    var left = x + 14;
    if (left + w > vw - 8) left = x - 14 - w;
    left = clamp(left, 8, Math.max(8, vw - w - 8));
    var top = y - h - 12;
    if (top < 8) top = y + 18;
    top = clamp(top, 8, Math.max(8, vh - h - 8));
    node.style.transform = "translate(" + Math.round(left) + "px," + Math.round(top) + "px)";
  }

  var tooltip = {
    show: function (content, x, y, owner) {
      tipFill(content);
      tipOwner = owner || null;
      tipPlace(x, y);
    },
    move: function (x, y) {
      if (tipNode && !tipNode.hidden) tipPlace(x, y);
    },
    hide: function (owner) {
      if (owner && tipOwner && owner !== tipOwner) return;
      if (tipNode) tipNode.hidden = true;
      tipOwner = null;
    },
    attach: function (mark, getContent) {
      function hot(on) {
        mark.classList.toggle("is-hot", on);
      }
      mark.addEventListener("pointerenter", function (e) {
        hot(true);
        tooltip.show(getContent(), e.clientX, e.clientY, mark);
      });
      mark.addEventListener("pointermove", function (e) {
        tooltip.move(e.clientX, e.clientY);
      });
      mark.addEventListener("pointerleave", function () {
        hot(false);
        tooltip.hide(mark);
      });
      mark.addEventListener("focus", function () {
        hot(true);
        var r = mark.getBoundingClientRect();
        tooltip.show(getContent(), r.left + r.width / 2, r.top, mark);
      });
      mark.addEventListener("blur", function () {
        hot(false);
        tooltip.hide(mark);
      });
      mark.addEventListener("keydown", function (e) {
        if (e.key === "Escape") {
          hot(false);
          tooltip.hide(mark);
        }
      });
      return mark;
    },
  };

  TV.tooltip = tooltip;

  /* Table view ------------------------------------------------------------ */

  TV.tableView = function (handle, spec) {
    spec = spec || {};
    var columns = spec.columns || [];
    var rows = spec.rows || [];
    var target = handle && handle.table ? handle.table : handle;
    if (!target) return null;
    var head = el(
      "tr",
      null,
      columns.map(function (c) {
        return el("th", { scope: "col", class: c.align === "right" ? "tv-num" : null }, c.label);
      })
    );
    var body = el(
      "tbody",
      null,
      rows.map(function (row) {
        return el(
          "tr",
          null,
          columns.map(function (c, i) {
            var raw = row[c.key];
            var text = c.format || isNum(raw) ? format(c.format, raw) : isNil(raw) ? "n/a" : String(raw);
            var cls = [c.align === "right" ? "tv-num" : null, c.mono ? "tv-mono" : null]
              .filter(Boolean)
              .join(" ");
            return i === 0
              ? el("th", { scope: "row", class: cls || null }, text)
              : el("td", { class: cls || null }, text);
          })
        );
      })
    );
    var label = (spec.caption || (handle && handle.title) || "Figure") + ", table";
    var table = el(
      "table",
      { class: "tv-table" },
      el("caption", { class: "tv-visually-hidden" }, label),
      el("thead", null, head),
      body
    );
    var wrap = el("div", { class: "tv-table-wrap", tabindex: "0", role: "region", "aria-label": label }, table);
    target.appendChild(wrap);
    if (handle && handle.toggle) handle.toggle.hidden = false;
    return wrap;
  };

  function tableFor(body, spec) {
    var handle = handleFor(body);
    if (handle) TV.tableView(handle, spec);
  }

  /* Chart frame: one container, redrawn when its width changes ------------- */

  var liveCharts = [];

  function frame(body, spec, kind, draw) {
    var wrap = el("div", { class: "tv-chart tv-chart--" + kind });
    body.appendChild(wrap);
    var lastWidth = -1;
    function render(force) {
      var width = Math.floor(wrap.clientWidth);
      if (!width) return;
      if (!force && width === lastWidth) return;
      lastWidth = width;
      clear(wrap);
      draw(wrap, Math.max(width, 260));
    }
    wrap.__tvRender = render;
    render(true);
    if (typeof ResizeObserver !== "undefined") {
      /* Redraw on the next frame, so a redraw that changes the page never feeds back into this callback. */
      var pending = false;
      new ResizeObserver(function () {
        if (pending) return;
        pending = true;
        global.requestAnimationFrame(function () {
          pending = false;
          render(false);
        });
      }).observe(wrap);
    }
    liveCharts.push(wrap);
    return wrap;
  }

  function redrawAll() {
    fontFamily = null;
    liveCharts.forEach(function (wrap) {
      if (wrap.isConnected) wrap.__tvRender(true);
    });
  }

  if (document.fonts) {
    if (document.fonts.ready) document.fonts.ready.then(redrawAll);
    if (document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", redrawAll);
  }

  function chartSvg(width, height, spec, kind) {
    return svg("svg", {
      class: "tv-svg",
      width: width,
      height: height,
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": spec.ariaLabel || spec.title || kind + " chart",
    });
  }

  function empty(body, text) {
    body.appendChild(el("p", { class: "tv-empty" }, text || "No values to draw."));
  }

  /*
   * A bar with a rounded data end and a square baseline end. orient "h" grows
   * along x from base to end; "v" grows along y. cross is the left (h) or top
   * (v) edge of the bar across its thickness.
   */
  function barPath(orient, base, end, cross, thickness, roundBoth) {
    var len = Math.abs(end - base);
    var r = Math.min(RADIUS, thickness / 2, roundBoth ? len / 2 : len);
    var s = end >= base ? 1 : -1;
    var t = thickness;
    var c = cross;
    if (len < 0.5) return "";
    if (orient === "h") {
      if (roundBoth) {
        var x0 = Math.min(base, end);
        var x1 = Math.max(base, end);
        return [
          "M", x0 + r, c, "H", x1 - r,
          "A", r, r, 0, 0, 1, x1, c + r,
          "V", c + t - r,
          "A", r, r, 0, 0, 1, x1 - r, c + t,
          "H", x0 + r,
          "A", r, r, 0, 0, 1, x0, c + t - r,
          "V", c + r,
          "A", r, r, 0, 0, 1, x0 + r, c,
          "Z",
        ].join(" ");
      }
      return [
        "M", base, c, "H", end - s * r,
        "A", r, r, 0, 0, s > 0 ? 1 : 0, end, c + r,
        "V", c + t - r,
        "A", r, r, 0, 0, s > 0 ? 1 : 0, end - s * r, c + t,
        "H", base,
        "Z",
      ].join(" ");
    }
    /* vertical: y grows downward, so a positive bar has end < base */
    var dir = end <= base ? -1 : 1;
    return [
      "M", c, base, "V", end - dir * r,
      "A", r, r, 0, 0, dir < 0 ? 1 : 0, c + r, end,
      "H", c + t - r,
      "A", r, r, 0, 0, dir < 0 ? 1 : 0, c + t, end - dir * r,
      "V", base,
      "Z",
    ].join(" ");
  }

  TV.barPath = barPath;

  function markGroup(parent, hit, mark, label) {
    var g = svg("g", { class: "tv-markg", tabindex: "0", "aria-label": label });
    g.appendChild(svg("rect", Object.assign({ class: "tv-hit" }, hit)));
    if (mark) {
      (Array.isArray(mark) ? mark : [mark]).forEach(function (m) {
        g.appendChild(m);
      });
    }
    parent.appendChild(g);
    return g;
  }

  function xAxisTicks(g, scale, y0, y1, fmt, count, maxWidth) {
    var ticks = scale.ticks(count);
    ticks.forEach(function (t) {
      var x = crisp(scale(t));
      g.appendChild(svg("line", { class: "tv-gridline", x1: x, x2: x, y1: y0, y2: y1 }));
    });
    var lastRight = -Infinity;
    var tickText = tickFormatter(fmt, ticks);
    ticks.forEach(function (t, i) {
      var label = tickText(t);
      var w = measure(label, 11);
      var x = scale(t);
      var anchor = "middle";
      var left = x - w / 2;
      if (i === ticks.length - 1 && x + w / 2 > scale.range[1] + 12) {
        anchor = "end";
        left = x - w;
      }
      if (left < lastRight + 6) return;
      lastRight = left + w;
      g.appendChild(svg("text", { class: "tv-tick", x: x, y: y1 + 16, "text-anchor": anchor }, label));
    });
  }

  function yAxisTicks(g, scale, x0, x1, fmt, count) {
    var ticks = scale.ticks(count);
    var tickText = tickFormatter(fmt, ticks);
    ticks.forEach(function (t) {
      var y = crisp(scale(t));
      g.appendChild(svg("line", { class: "tv-gridline", x1: x0, x2: x1, y1: y, y2: y }));
      g.appendChild(
        svg("text", { class: "tv-tick", x: x0 - 8, y: y, dy: "0.35em", "text-anchor": "end" }, tickText(t))
      );
    });
  }

  function yTickWidth(scale, fmt, count) {
    var ticks = scale.ticks(count);
    var tickText = tickFormatter(fmt, ticks);
    return Math.max.apply(
      null,
      [0].concat(
        ticks.map(function (t) {
          return measure(tickText(t), 11);
        })
      )
    );
  }

  function referenceValues(spec) {
    return (spec.reference || [])
      .map(function (r) {
        return r && r.value;
      })
      .filter(isNum);
  }

  /* Vertical reference rules with labels stacked above the plot. */
  function verticalRefs(g, spec, x, top, bottom, strong) {
    var refs = (spec.reference || []).filter(function (r) {
      return r && isNum(r.value);
    });
    refs.forEach(function (ref, i) {
      var px = crisp(x(ref.value));
      g.appendChild(
        svg("line", { class: "tv-ref" + (strong ? " tv-ref--strong" : ""), x1: px, x2: px, y1: top - 4, y2: bottom })
      );
      var text = ref.label || format(spec.format, ref.value);
      var w = measure(text, 11, 500);
      var lx = clamp(px, x.range[0] + w / 2, x.range[1] - w / 2);
      g.appendChild(
        svg("text", { class: "tv-ref-label", x: lx, y: top - 10 - (refs.length - 1 - i) * 14, "text-anchor": "middle" }, text)
      );
    });
    return refs.length ? refs.length * 14 + 6 : 0;
  }

  function horizontalRefs(g, spec, y, left, right) {
    (spec.reference || []).forEach(function (ref) {
      if (!ref || !isNum(ref.value)) return;
      var py = crisp(y(ref.value));
      g.appendChild(svg("line", { class: "tv-ref", x1: left, x2: right, y1: py, y2: py }));
      if (ref.label) {
        g.appendChild(svg("text", { class: "tv-ref-label", x: left + 4, y: py - 6 }, ref.label));
      }
    });
  }

  function roleLegend(body, spec, roles, shape) {
    if (spec.legend) {
      legendFor(
        body,
        spec.legend.map(function (item) {
          return { label: item.label, color: item.role || item.color, shape: item.shape || shape };
        })
      );
      return;
    }
    if (roles.length < 2) return;
    var names = spec.roleLabels || {};
    legendFor(
      body,
      roles.map(function (role) {
        return { label: names[role] || ROLE_LABELS[role] || role, color: role, shape: shape };
      })
    );
  }

  /* Category rows with wrapped labels, shared by hbar, dot and range. */
  function rowLabels(rows, width) {
    var maxWidth = Math.min(Math.max(80, width * 0.34), 240);
    var wrapped = rows.map(function (r) {
      return wrapText(r.label, maxWidth, 12, 2);
    });
    var labelWidth = Math.max.apply(
      null,
      [40].concat(
        wrapped.map(function (w) {
          return w.width;
        })
      )
    );
    return { wrapped: wrapped, width: Math.min(labelWidth, width * 0.5) };
  }

  /* hbar ------------------------------------------------------------------ */

  function hbar(body, spec) {
    spec = spec || {};
    var rows = (spec.rows || []).filter(Boolean);
    var valueName = spec.valueLabel || "Value";
    tableFor(body, {
      columns: [
        { key: "label", label: spec.labelHeader || "Label" },
        { key: "value", label: valueName, align: "right", format: spec.format },
      ],
      rows: rows,
    });
    if (!rows.length) return empty(body);
    roleLegend(
      body,
      spec,
      distinct(
        rows.map(function (r) {
          return r.role || "model";
        })
      ),
      "rect"
    );

    frame(body, spec, "hbar", function (wrap, W) {
      var values = finite(
        rows.map(function (r) {
          return r.value;
        })
      );
      var refs = referenceValues(spec);
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : Math.min.apply(null, [0].concat(values, refs));
      var hi = isNum(dom[1]) ? dom[1] : Math.max.apply(null, [0].concat(values, refs));
      var labels = rowLabels(rows, W);
      var mode = spec.labels || (rows.length <= 12 ? "all" : "extreme");
      var valueWidth =
        mode === "none"
          ? 0
          : Math.max.apply(
              null,
              [0].concat(
                rows.map(function (r) {
                  return measure(format(spec.format, r.value), 11, 500);
                })
              )
            ) + 8;
      var refBand = (spec.reference || []).length * 14 + ((spec.reference || []).length ? 8 : 0);
      var m = {
        top: 6 + refBand,
        right: 8 + (hi > 0 ? valueWidth : 0),
        bottom: 26,
        left: labels.width + 12 + (lo < 0 ? valueWidth : 0),
      };
      var n = rows.length;
      var rowH = spec.height ? Math.max(HIT_MIN, (spec.height - m.top - m.bottom) / n) : 32;
      var plotH = rowH * n;
      var H = Math.round(m.top + plotH + m.bottom);
      var x = linear([lo, hi], [m.left, W - m.right], { nice: !spec.domain });
      var y = band(
        rows.map(function (_, i) {
          return i;
        }),
        [m.top, m.top + plotH],
        { padding: 0.3 }
      );
      var thick = Math.min(BAR_MAX, y.bandwidth());
      var root = chartSvg(W, H, spec, "Bar");
      var grid = svg("g");
      xAxisTicks(grid, x, m.top, m.top + plotH, spec.format, Math.max(2, Math.floor((W - m.left - m.right) / 90)));
      root.appendChild(grid);
      var zero = crisp(x(clamp(0, x.domain[0], x.domain[1])));
      root.appendChild(svg("line", { class: "tv-axisline", x1: zero, x2: zero, y1: m.top, y2: m.top + plotH }));

      var extremeIndex = -1;
      rows.forEach(function (r, i) {
        if (isNum(r.value) && (extremeIndex < 0 || Math.abs(r.value) > Math.abs(rows[extremeIndex].value))) {
          extremeIndex = i;
        }
      });

      var marks = svg("g");
      var valueLayer = svg("g", { class: "tv-values" });
      rows.forEach(function (r, i) {
        var cy = y.step() * i + m.top + y.step() / 2;
        textBlock(marks, labels.wrapped[i].lines, labels.width + 2, cy, { class: "tv-label", "text-anchor": "end" });
        var fill = color(r.role || "model");
        var parts = [];
        if (isNum(r.value)) {
          var d = barPath("h", x(clamp(0, x.domain[0], x.domain[1])), x(r.value), cy - thick / 2, thick);
          if (d) parts.push(svg("path", { class: "tv-mark", d: d, style: { fill: fill } }));
        }
        var g = markGroup(
          marks,
          { x: 0, y: m.top + y.step() * i, width: W, height: y.step() },
          parts,
          r.label + ": " + format(spec.format, r.value)
        );
        if (isNum(r.value) && (mode === "all" || (mode === "extreme" && i === extremeIndex))) {
          var tip = x(r.value);
          var right = r.value >= 0;
          valueLayer.appendChild(
            svg(
              "text",
              { class: "tv-value", x: tip + (right ? 6 : -6), y: cy, dy: "0.35em", "text-anchor": right ? "start" : "end" },
              format(spec.format, r.value)
            )
          );
        }
        tooltip.attach(g, function () {
          var out = [{ label: valueName, value: format(spec.format, r.value), color: r.role || "model", shape: "rect" }];
          (spec.reference || []).forEach(function (ref) {
            if (ref && isNum(ref.value)) out.push({ label: ref.label || "Reference", value: format(spec.format, ref.value) });
          });
          if (r.note) out.push({ label: r.note, value: "" });
          return { title: r.label, rows: out };
        });
      });
      root.appendChild(marks);
      var refs2 = svg("g");
      verticalRefs(refs2, spec, x, m.top, m.top + plotH, false);
      root.appendChild(refs2);
      root.appendChild(valueLayer);
      wrap.appendChild(root);
    });
  }

  /* column ---------------------------------------------------------------- */

  function column(body, spec) {
    spec = spec || {};
    var rows = (spec.rows || []).filter(Boolean);
    var valueName = spec.valueLabel || "Value";
    tableFor(body, {
      columns: [
        { key: "label", label: spec.labelHeader || "Label" },
        { key: "value", label: valueName, align: "right", format: spec.format },
      ],
      rows: rows,
    });
    if (!rows.length) return empty(body);
    if (spec.diverging) {
      if (spec.legend) roleLegend(body, spec, [], "rect");
    } else {
      roleLegend(
        body,
        spec,
        distinct(
          rows.map(function (r) {
            return r.role || "model";
          })
        ),
        "rect"
      );
    }

    frame(body, spec, "column", function (wrap, W) {
      var values = finite(
        rows.map(function (r) {
          return r.value;
        })
      );
      var refs = referenceValues(spec);
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : Math.min.apply(null, [0].concat(values, refs));
      var hi = isNum(dom[1]) ? dom[1] : Math.max.apply(null, [0].concat(values, refs));
      var H = spec.height || 260;
      var m = { top: 20, right: 8, bottom: 30, left: 0 };
      var y = linear([lo, hi], [H - m.bottom, m.top], { nice: !spec.domain });
      m.left = yTickWidth(y, spec.format, 5) + 14;
      var n = rows.length;
      var x = band(
        rows.map(function (_, i) {
          return i;
        }),
        [m.left, W - m.right],
        { padding: 0.25 }
      );
      var thick = Math.min(BAR_MAX, x.bandwidth(), Math.max(1, x.step() - GAP));
      var root = chartSvg(W, H, spec, "Column");
      var grid = svg("g");
      yAxisTicks(grid, y, m.left, W - m.right, spec.format, 5);
      root.appendChild(grid);
      var zeroV = clamp(0, y.domain[0], y.domain[1]);
      var base = y(zeroV);

      var labelWidths = rows.map(function (r) {
        return measure(r.label, 11);
      });
      var every = Math.max(1, Math.ceil((Math.max.apply(null, labelWidths) + 8) / x.step()));

      var maxI = -1;
      var minI = -1;
      rows.forEach(function (r, i) {
        if (!isNum(r.value)) return;
        if (maxI < 0 || r.value > rows[maxI].value) maxI = i;
        if (minI < 0 || r.value < rows[minI].value) minI = i;
      });
      var mode = spec.labels || "extreme";

      var marks = svg("g");
      var valueLayer = svg("g", { class: "tv-values" });
      rows.forEach(function (r, i) {
        var cx = x.step() * i + m.left + x.step() / 2;
        var fill = spec.diverging ? color(r.value >= 0 ? "pos" : "neg") : color(r.role || "model");
        var parts = [];
        if (isNum(r.value)) {
          var d = barPath("v", base, y(r.value), cx - thick / 2, thick);
          if (d) parts.push(svg("path", { class: "tv-mark", d: d, style: { fill: fill } }));
        }
        var g = markGroup(
          marks,
          { x: m.left + x.step() * i, y: m.top, width: x.step(), height: H - m.top - m.bottom },
          parts,
          r.label + ": " + format(spec.format, r.value)
        );
        var labelled = mode === "all" || (mode === "extreme" && (i === maxI || (i === minI && rows[minI].value < 0)));
        if (isNum(r.value) && labelled) {
          var up = r.value >= 0;
          valueLayer.appendChild(
            svg(
              "text",
              { class: "tv-value", x: cx, y: y(r.value) + (up ? -6 : 14), "text-anchor": "middle" },
              format(spec.format, r.value)
            )
          );
        }
        if (i % every === 0) {
          marks.appendChild(svg("text", { class: "tv-tick", x: cx, y: H - m.bottom + 16, "text-anchor": "middle" }, r.label));
        }
        tooltip.attach(g, function () {
          return {
            title: r.label,
            rows: [{ label: valueName, value: format(spec.format, r.value), color: fill, shape: "rect" }],
          };
        });
      });
      root.appendChild(svg("line", { class: "tv-axisline", x1: m.left, x2: W - m.right, y1: crisp(base), y2: crisp(base) }));
      root.appendChild(marks);
      var refs2 = svg("g");
      horizontalRefs(refs2, spec, y, m.left, W - m.right);
      root.appendChild(refs2);
      root.appendChild(valueLayer);
      wrap.appendChild(root);
    });
  }

  /* dot: a dot plot, or a dumbbell when there are two series --------------- */

  function dot(body, spec) {
    spec = spec || {};
    var rows = (spec.rows || []).filter(Boolean);
    var series = (spec.series || [{ key: "value", name: spec.valueLabel || "Value", role: "model" }]).slice(0, 3);
    var columns = [{ key: "label", label: spec.labelHeader || "Label" }];
    series.forEach(function (s) {
      columns.push({ key: "s:" + s.key, label: s.name, align: "right", format: spec.format });
    });
    if (series.length === 2) columns.push({ key: "gap", label: spec.gapLabel || "Difference", align: "right", format: spec.gapFormat || "signed:" + (spec.gapDp || 2) });
    tableFor(body, {
      columns: columns,
      rows: rows.map(function (r) {
        var out = { label: r.label };
        series.forEach(function (s) {
          out["s:" + s.key] = r.values ? r.values[s.key] : null;
        });
        if (series.length === 2) {
          var a = out["s:" + series[0].key];
          var b = out["s:" + series[1].key];
          out.gap = isNum(a) && isNum(b) ? a - b : null;
        }
        return out;
      }),
    });
    if (!rows.length) return empty(body);
    if (series.length > 1) {
      legendFor(
        body,
        series.map(function (s) {
          return { label: s.name, color: s.role, shape: "dot" };
        })
      );
    }

    frame(body, spec, "dot", function (wrap, W) {
      var all = [];
      rows.forEach(function (r) {
        series.forEach(function (s) {
          all.push(r.values ? r.values[s.key] : null);
        });
      });
      var ext = extent(all.concat(referenceValues(spec))) || [0, 1];
      if (spec.zero) ext = [Math.min(0, ext[0]), Math.max(0, ext[1])];
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : ext[0];
      var hi = isNum(dom[1]) ? dom[1] : ext[1];
      var labels = rowLabels(rows, W);
      var extremeText = 0;
      var extremeIndex = -1;
      var first = series[0];
      rows.forEach(function (r, i) {
        var a = r.values ? r.values[first.key] : null;
        if (!isNum(a)) return;
        var score = a;
        if (series.length === 2) {
          var b = r.values[series[1].key];
          score = isNum(b) ? Math.abs(a - b) : -Infinity;
        }
        if (extremeIndex < 0 || score > extremeText) {
          extremeText = score;
          extremeIndex = i;
        }
      });
      var labelW = extremeIndex >= 0 ? measure(format(spec.format, rows[extremeIndex].values[first.key]), 11, 500) + 12 : 0;
      var refBand = (spec.reference || []).length * 14 + ((spec.reference || []).length ? 8 : 0);
      var m = { top: 8 + refBand, right: 12 + labelW, bottom: 26, left: labels.width + 16 };
      var n = rows.length;
      var rowH = spec.height ? Math.max(HIT_MIN, (spec.height - m.top - m.bottom) / n) : 30;
      var plotH = rowH * n;
      var H = Math.round(m.top + plotH + m.bottom);
      var x = linear([lo, hi], [m.left + 6, W - m.right], { nice: !spec.domain });
      var root = chartSvg(W, H, spec, "Dot");
      var grid = svg("g");
      xAxisTicks(grid, x, m.top, m.top + plotH, spec.format, Math.max(2, Math.floor((W - m.left - m.right) / 90)));
      root.appendChild(grid);
      var marks = svg("g");
      var valueLayer = svg("g", { class: "tv-values" });
      rows.forEach(function (r, i) {
        var cy = m.top + rowH * i + rowH / 2;
        textBlock(marks, labels.wrapped[i].lines, labels.width + 2, cy, { class: "tv-label", "text-anchor": "end" });
        var parts = [];
        var vals = series.map(function (s) {
          return r.values ? r.values[s.key] : null;
        });
        var present = finite(vals);
        if (present.length > 1) {
          parts.push(
            svg("line", {
              class: "tv-hairline",
              x1: x(Math.min.apply(null, present)),
              x2: x(Math.max.apply(null, present)),
              y1: crisp(cy),
              y2: crisp(cy),
            })
          );
        }
        for (var k = series.length - 1; k >= 0; k--) {
          if (isNum(vals[k])) {
            parts.push(svg("circle", { class: "tv-mark tv-dot", cx: x(vals[k]), cy: cy, r: 4 + (k === 0 ? 0.5 : 0), style: { fill: color(series[k].role) } }));
          }
        }
        var g = markGroup(
          marks,
          { x: 0, y: m.top + rowH * i, width: W, height: rowH },
          parts,
          r.label +
            ": " +
            series
              .map(function (s, k) {
                return s.name + " " + format(spec.format, vals[k]);
              })
              .join(", ")
        );
        if (i === extremeIndex) {
          var right = series.length === 2 && isNum(vals[1]) && vals[1] > vals[0] ? vals[1] : vals[0];
          valueLayer.appendChild(
            svg("text", { class: "tv-value", x: x(Math.max(right, vals[0])) + 9, y: cy, dy: "0.35em" }, format(spec.format, vals[0]))
          );
        }
        tooltip.attach(g, function () {
          var out = series.map(function (s, k) {
            return { label: s.name, value: format(spec.format, vals[k]), color: s.role, shape: "dot" };
          });
          if (series.length === 2 && isNum(vals[0]) && isNum(vals[1])) {
            out.push({ label: spec.gapLabel || "Difference", value: format(spec.gapFormat || "signed:" + (spec.gapDp || 2), vals[0] - vals[1]) });
          }
          return { title: r.label, rows: out };
        });
      });
      root.appendChild(marks);
      var refs2 = svg("g");
      verticalRefs(refs2, spec, x, m.top, m.top + plotH, false);
      root.appendChild(refs2);
      root.appendChild(valueLayer);
      wrap.appendChild(root);
    });
  }

  /* line ------------------------------------------------------------------ */

  function line(body, spec) {
    spec = spec || {};
    var allSeries = (spec.series || []).filter(Boolean);
    var series = allSeries.slice(0, 4);
    var xSpec = spec.x || {};
    var xs = [];
    allSeries.forEach(function (s) {
      (s.values || []).forEach(function (p) {
        if (p && !isNil(p.x) && xs.indexOf(p.x) < 0) xs.push(p.x);
      });
    });
    var numericX = xs.length > 0 && xs.every(isNum);
    if (numericX) xs.sort(function (a, b) {
      return a - b;
    });

    function valueAt(s, xv) {
      var vals = s.values || [];
      for (var i = 0; i < vals.length; i++) if (vals[i] && vals[i].x === xv) return vals[i];
      return null;
    }

    var columns = [{ key: "x", label: xSpec.label || "x", format: xSpec.format }];
    allSeries.forEach(function (s, k) {
      columns.push({ key: "y" + k, label: s.name, align: "right", format: spec.format });
      if (s.values && s.values.some(function (p) { return p && isNum(p.lo); })) {
        columns.push({ key: "lo" + k, label: s.name + " low", align: "right", format: spec.format });
        columns.push({ key: "hi" + k, label: s.name + " high", align: "right", format: spec.format });
      }
    });
    tableFor(body, {
      columns: columns,
      rows: xs.map(function (xv) {
        var row = { x: xSpec.format ? xv : String(xv) };
        allSeries.forEach(function (s, k) {
          var p = valueAt(s, xv);
          row["y" + k] = p ? p.y : null;
          row["lo" + k] = p ? p.lo : null;
          row["hi" + k] = p ? p.hi : null;
        });
        return row;
      }),
    });
    if (!xs.length || !series.length) return empty(body);
    if (allSeries.length > 4) {
      var handle = handleFor(body);
      if (handle) handle.addNote({ what: "Series beyond four", why: "The chart draws the first four series; the table lists all " + allSeries.length + "." });
    }
    if (series.length > 1) {
      legendFor(
        body,
        series.map(function (s) {
          return { label: s.name, color: s.role, shape: "line" };
        })
      );
    }

    frame(body, spec, "line", function (wrap, W) {
      var ys = [];
      series.forEach(function (s) {
        (s.values || []).forEach(function (p) {
          if (!p) return;
          ys.push(p.y, p.lo, p.hi);
        });
      });
      var ext = extent(ys.concat(referenceValues(spec))) || [0, 1];
      if (spec.zero) ext = [Math.min(0, ext[0]), Math.max(0, ext[1])];
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : ext[0];
      var hi = isNum(dom[1]) ? dom[1] : ext[1];
      var H = spec.height || 280;
      var m = { top: 14, right: 16, bottom: 30, left: 0 };
      var y = linear([lo, hi], [H - m.bottom, m.top], { nice: !spec.domain });
      m.left = yTickWidth(y, spec.format, 5) + 14;

      /* End labels only when every series ends clear of its neighbours. */
      var ends = series
        .map(function (s, k) {
          var vals = (s.values || []).filter(function (p) {
            return p && isNum(p.y);
          });
          return vals.length ? { k: k, name: s.name, y: y(vals[vals.length - 1].y) } : null;
        })
        .filter(Boolean)
        .sort(function (a, b) {
          return a.y - b.y;
        });
      var endLabels = spec.endLabels !== false && ends.length > 0;
      for (var e = 1; e < ends.length; e++) if (ends[e].y - ends[e - 1].y < 14) endLabels = false;
      var endWidth = endLabels
        ? Math.max.apply(
            null,
            ends.map(function (d) {
              return measure(d.name, 12, 500);
            })
          ) + 14
        : 0;
      m.right = Math.max(m.right, endWidth);
      var plotL = m.left + 4;
      var plotR = W - m.right - (endLabels ? 0 : 4);
      var xAt;
      if (numericX) {
        var xScale = linear([xs[0], xs[xs.length - 1]], [plotL, plotR]);
        xAt = function (xv) {
          return xScale(xv);
        };
      } else {
        var stepX = xs.length > 1 ? (plotR - plotL) / (xs.length - 1) : 0;
        xAt = function (xv) {
          var i = xs.indexOf(xv);
          return xs.length > 1 ? plotL + i * stepX : (plotL + plotR) / 2;
        };
      }

      var root = chartSvg(W, H, spec, "Line");
      var grid = svg("g");
      yAxisTicks(grid, y, m.left, W - m.right, spec.format, 5);
      root.appendChild(grid);
      var baseV = clamp(0, y.domain[0], y.domain[1]);
      if (y.domain[0] <= 0 && y.domain[1] >= 0) {
        root.appendChild(svg("line", { class: "tv-axisline", x1: m.left, x2: W - m.right, y1: crisp(y(baseV)), y2: crisp(y(baseV)) }));
      } else {
        root.appendChild(svg("line", { class: "tv-axisline", x1: m.left, x2: W - m.right, y1: crisp(H - m.bottom), y2: crisp(H - m.bottom) }));
      }

      /* x tick labels, thinned to fit */
      var xLabels = xs.map(function (xv) {
        return xSpec.format ? format(xSpec.format, xv) : String(xv);
      });
      var widest = Math.max.apply(
        null,
        xLabels.map(function (t) {
          return measure(t, 11);
        })
      );
      var spacing = xs.length > 1 ? Math.abs(xAt(xs[1]) - xAt(xs[0])) : Infinity;
      var every = numericX ? 1 : Math.max(1, Math.ceil((widest + 10) / spacing));
      var tickXs = xs;
      if (numericX && xs.length > 8) {
        tickXs = linear([xs[0], xs[xs.length - 1]], [0, 1]).ticks(Math.max(2, Math.floor((plotR - plotL) / (widest + 24))));
      }
      tickXs.forEach(function (xv, i) {
        if (i % every !== 0) return;
        var px = xAt(xv);
        var label = xSpec.format ? format(xSpec.format, xv) : String(xv);
        var w = measure(label, 11);
        var anchor = px - w / 2 < 0 ? "start" : px + w / 2 > W ? "end" : "middle";
        grid.appendChild(svg("text", { class: "tv-tick", x: px, y: H - m.bottom + 16, "text-anchor": anchor }, label));
      });

      var layer = svg("g");
      series.forEach(function (s) {
        var pts = xs
          .map(function (xv) {
            var p = valueAt(s, xv);
            return p ? { x: xAt(xv), p: p } : null;
          })
          .filter(Boolean);
        var bandPts = pts.filter(function (d) {
          return isNum(d.p.lo) && isNum(d.p.hi);
        });
        if (bandPts.length > 1) {
          var dArea =
            "M" +
            bandPts
              .map(function (d) {
                return d.x + "," + y(d.p.hi);
              })
              .join("L") +
            "L" +
            bandPts
              .slice()
              .reverse()
              .map(function (d) {
                return d.x + "," + y(d.p.lo);
              })
              .join("L") +
            "Z";
          layer.appendChild(svg("path", { class: "tv-band", d: dArea, style: { fill: color(s.role) } }));
        }
      });
      series
        .slice()
        .reverse()
        .forEach(function (s) {
          var d = "";
          var pen = false;
          xs.forEach(function (xv) {
            var p = valueAt(s, xv);
            if (p && isNum(p.y)) {
              d += (pen ? "L" : "M") + xAt(xv) + "," + y(p.y);
              pen = true;
            } else {
              pen = false;
            }
          });
          layer.appendChild(svg("path", { class: "tv-line", d: d, style: { stroke: color(s.role) } }));
          var vals = (s.values || []).filter(function (p) {
            return p && isNum(p.y);
          });
          if (vals.length) {
            var last = vals[vals.length - 1];
            layer.appendChild(svg("circle", { class: "tv-dot", cx: xAt(last.x), cy: y(last.y), r: 4, style: { fill: color(s.role) } }));
          }
        });
      root.appendChild(layer);

      if (endLabels) {
        ends.forEach(function (d) {
          root.appendChild(svg("text", { class: "tv-endlabel", x: plotR + 10, y: d.y, dy: "0.35em" }, d.name));
        });
      }
      var refs2 = svg("g");
      horizontalRefs(refs2, spec, y, m.left, W - m.right);
      root.appendChild(refs2);

      /* Crosshair: the pointer finds the nearest x; arrows move it. */
      var hover = svg("g", { hidden: true });
      var hair = svg("line", { class: "tv-crosshair", x1: 0, x2: 0, y1: m.top, y2: H - m.bottom });
      hover.appendChild(hair);
      var hoverDots = series.map(function (s) {
        var c = svg("circle", { class: "tv-dot", r: 4, style: { fill: color(s.role) } });
        hover.appendChild(c);
        return c;
      });
      var plot = svg("g", {
        class: "tv-plot",
        tabindex: "0",
        "aria-label": (spec.title || "Line chart") + ". Use the arrow keys to read each point.",
      });
      plot.appendChild(svg("rect", { class: "tv-hit", x: m.left, y: m.top, width: Math.max(0, W - m.left - m.right + (endLabels ? endWidth : 0)), height: H - m.top - m.bottom }));
      root.appendChild(hover);
      root.appendChild(plot);
      var current = -1;

      function content(i) {
        var xv = xs[i];
        return {
          title: xSpec.format ? format(xSpec.format, xv) : String(xv),
          rows: series.map(function (s) {
            var p = valueAt(s, xv);
            var text = p ? format(spec.format, p.y) : "n/a";
            if (p && isNum(p.lo) && isNum(p.hi)) {
              text += " (" + format(spec.format, p.lo) + " to " + format(spec.format, p.hi) + ")";
            }
            return { label: s.name, value: text, color: s.role, shape: "line" };
          }),
        };
      }

      function showAt(i, clientX, clientY) {
        current = clamp(i, 0, xs.length - 1);
        var px = xAt(xs[current]);
        hair.setAttribute("x1", crisp(px));
        hair.setAttribute("x2", crisp(px));
        series.forEach(function (s, k) {
          var p = valueAt(s, xs[current]);
          if (p && isNum(p.y)) {
            hoverDots[k].removeAttribute("hidden");
            hoverDots[k].setAttribute("cx", px);
            hoverDots[k].setAttribute("cy", y(p.y));
          } else {
            hoverDots[k].setAttribute("hidden", "");
          }
        });
        hover.removeAttribute("hidden");
        if (isNil(clientX)) {
          var box = root.getBoundingClientRect();
          clientX = box.left + px;
          clientY = box.top + m.top;
        }
        tooltip.show(content(current), clientX, clientY, plot);
      }

      function nearest(clientX) {
        var box = root.getBoundingClientRect();
        var px = clientX - box.left;
        var best = 0;
        var bestD = Infinity;
        xs.forEach(function (xv, i) {
          var dd = Math.abs(xAt(xv) - px);
          if (dd < bestD) {
            bestD = dd;
            best = i;
          }
        });
        return best;
      }

      function hideHover() {
        hover.setAttribute("hidden", "");
        tooltip.hide(plot);
      }

      plot.addEventListener("pointermove", function (ev) {
        showAt(nearest(ev.clientX), ev.clientX, ev.clientY);
      });
      plot.addEventListener("pointerleave", hideHover);
      plot.addEventListener("focus", function () {
        showAt(current < 0 ? xs.length - 1 : current);
      });
      plot.addEventListener("blur", hideHover);
      plot.addEventListener("keydown", function (ev) {
        var next = current;
        if (ev.key === "ArrowRight") next = current + 1;
        else if (ev.key === "ArrowLeft") next = current - 1;
        else if (ev.key === "Home") next = 0;
        else if (ev.key === "End") next = xs.length - 1;
        else if (ev.key === "Escape") return hideHover();
        else return;
        ev.preventDefault();
        showAt(next);
      });
      wrap.appendChild(root);
    });
  }

  /* heat ------------------------------------------------------------------ */

  function heat(body, spec) {
    spec = spec || {};
    var rowsL = spec.rows || [];
    var colsL = spec.cols || spec.columns || [];
    var values = spec.values || [];
    var diverging = spec.scale === "diverging";
    var valueName = spec.valueLabel || "Value";

    var tableCols = [{ key: "row", label: spec.rowHeader || "Row" }].concat(
      colsL.map(function (c, j) {
        return { key: "c" + j, label: String(c), align: "right", format: spec.format };
      })
    );
    tableFor(body, {
      columns: tableCols,
      rows: rowsL.map(function (r, i) {
        var out = { row: r };
        colsL.forEach(function (_, j) {
          out["c" + j] = values[i] ? values[i][j] : null;
        });
        return out;
      }),
    });
    if (!rowsL.length || !colsL.length) return empty(body);

    var flat = [];
    values.forEach(function (row) {
      (row || []).forEach(function (v) {
        flat.push(v);
      });
    });
    var ext = extent(flat) || [0, 1];
    var dom = spec.domain || [];
    var maxAbs = Math.max(Math.abs(ext[0]), Math.abs(ext[1])) || 1;
    var lo = isNum(dom[0]) ? dom[0] : diverging ? -maxAbs : ext[0];
    var hi = isNum(dom[1]) ? dom[1] : diverging ? maxAbs : ext[1];
    var hasMissing = flat.some(function (v) {
      return !isNum(v);
    });

    function classOf(v) {
      if (!isNum(v)) return null;
      if (diverging) {
        var bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
        var k = clamp(Math.round((v / bound) * 3), -3, 3);
        return k;
      }
      if (hi === lo) return 4;
      return clamp(Math.floor(((v - lo) / (hi - lo)) * 7) + 1, 1, 7);
    }

    function fillOf(k) {
      if (k === null) return "var(--missing)";
      if (diverging) return k === 0 ? "var(--div-0)" : k > 0 ? "var(--div-pos-" + k + ")" : "var(--div-neg-" + -k + ")";
      return "var(--seq-" + k + ")";
    }

    function inkOf(k) {
      if (k === null) return "var(--ink-3)";
      return diverging ? "var(--on-div-" + Math.abs(k) + ")" : "var(--on-seq-" + k + ")";
    }

    frame(body, spec, "heat", function (wrap, W) {
      var labelMax = Math.min(Math.max(60, W * 0.28), 200);
      var rowWrapped = rowsL.map(function (r) {
        return wrapText(r, labelMax, 12, 1);
      });
      var rowW = Math.max.apply(
        null,
        [30].concat(
          rowWrapped.map(function (w) {
            return w.width;
          })
        )
      );
      var m = { top: 24, right: 4, bottom: 4, left: Math.min(rowW, labelMax) + 12 };
      var cellW = Math.max(HIT_MIN, (W - m.left - m.right) / colsL.length);
      var cellH = Math.max(HIT_MIN, spec.cellHeight || 30);
      var gridW = cellW * colsL.length;
      var H = Math.round(m.top + cellH * rowsL.length + m.bottom);
      var root = chartSvg(Math.max(W, m.left + gridW + m.right), H, spec, "Heat map");
      var colWidths = colsL.map(function (c) {
        return measure(c, 11);
      });
      var every = Math.max(1, Math.ceil((Math.max.apply(null, colWidths) + 6) / cellW));
      colsL.forEach(function (c, j) {
        if (j % every !== 0) return;
        root.appendChild(svg("text", { class: "tv-tick", x: m.left + cellW * j + cellW / 2, y: m.top - 9, "text-anchor": "middle" }, String(c)));
      });
      var showLabels =
        spec.cellLabels !== false &&
        flat.every(function (v) {
          return measure(format(spec.format, v), 11) + 8 <= cellW - GAP;
        });
      var marks = svg("g");
      rowsL.forEach(function (r, i) {
        var cy = m.top + cellH * i + cellH / 2;
        marks.appendChild(svg("text", { class: "tv-label", x: m.left - 10, y: cy, dy: "0.35em", "text-anchor": "end" }, rowWrapped[i].lines[0]));
        colsL.forEach(function (c, j) {
          var v = values[i] ? values[i][j] : null;
          var k = classOf(v);
          var cx = m.left + cellW * j;
          var cell = svg("rect", {
            class: "tv-mark",
            x: cx + GAP / 2,
            y: m.top + cellH * i + GAP / 2,
            width: Math.max(0, cellW - GAP),
            height: Math.max(0, cellH - GAP),
            rx: 2,
            style: { fill: fillOf(k) },
          });
          var g = markGroup(
            marks,
            { x: cx, y: m.top + cellH * i, width: cellW, height: cellH },
            cell,
            r + ", " + c + ": " + format(spec.format, v)
          );
          if (showLabels) {
            g.appendChild(
              svg(
                "text",
                { class: "tv-cell-label", x: cx + cellW / 2, y: cy, dy: "0.35em", "text-anchor": "middle", style: { fill: inkOf(k) } },
                format(spec.format, v)
              )
            );
          }
          tooltip.attach(g, function () {
            return {
              title: r + " · " + c,
              rows: [{ label: valueName, value: format(spec.format, v), color: fillOf(k), shape: "rect" }],
            };
          });
        });
      });
      root.appendChild(marks);
      wrap.appendChild(root);

      /* The scale legend: one swatch per class, with its bounds in text. */
      var classes = diverging ? [-3, -2, -1, 0, 1, 2, 3] : [1, 2, 3, 4, 5, 6, 7];
      var ticks = diverging
        ? [format(spec.format, -Math.max(Math.abs(lo), Math.abs(hi))), format(spec.format, 0), format(spec.format, Math.max(Math.abs(lo), Math.abs(hi)))]
        : [format(spec.format, lo), format(spec.format, hi)];
      var scaleNode = el(
        "div",
        { class: "tv-scale" },
        el(
          "div",
          { class: "tv-scale__ramp" },
          spec.scaleLabel ? el("span", { class: "tv-scale__title" }, spec.scaleLabel) : null,
          el(
            "div",
            { class: "tv-scale__swatches", "aria-hidden": "true" },
            classes.map(function (k) {
              return el("span", { class: "tv-scale__swatch", style: { background: fillOf(k) } });
            })
          ),
          el(
            "div",
            { class: "tv-scale__ticks" },
            ticks.map(function (t) {
              return el("span", null, t);
            })
          )
        ),
        hasMissing
          ? el(
              "span",
              { class: "tv-scale__missing" },
              el("span", { class: "tv-key tv-key--rect", style: { "--key": "var(--missing)" }, "aria-hidden": "true" }),
              "No value"
            )
          : null
      );
      wrap.appendChild(scaleNode);
    });
  }

  /* hist: pre-binned counts, up to two series ------------------------------ */

  function hist(body, spec) {
    spec = spec || {};
    var edges = spec.edges || [];
    var series = (spec.series || []).filter(Boolean).slice(0, 2);
    var nBins = Math.max(0, edges.length - 1);
    var columns = [
      { key: "from", label: spec.binLabel ? spec.binLabel + " from" : "From", align: "right", format: spec.format },
      { key: "to", label: "To", align: "right", format: spec.format },
    ];
    series.forEach(function (s, k) {
      columns.push({ key: "n" + k, label: s.name, align: "right", format: "int" });
    });
    var tableRows = [];
    for (var b = 0; b < nBins; b++) {
      var row = { from: edges[b], to: edges[b + 1] };
      series.forEach(function (s, k) {
        row["n" + k] = s.counts ? s.counts[b] : null;
      });
      tableRows.push(row);
    }
    tableFor(body, { columns: columns, rows: tableRows });
    if (!nBins || !series.length) return empty(body);
    if (series.length > 1) {
      legendFor(
        body,
        series.map(function (s) {
          return { label: s.name, color: s.role, shape: "rect" };
        })
      );
    }

    frame(body, spec, "hist", function (wrap, W) {
      var counts = [];
      series.forEach(function (s) {
        counts = counts.concat(s.counts || []);
      });
      var maxCount = Math.max.apply(null, [1].concat(finite(counts)));
      var H = spec.height || 240;
      var refBand = (spec.reference || []).length * 14 + ((spec.reference || []).length ? 6 : 0);
      var m = { top: 12 + refBand, right: 12, bottom: 30, left: 0 };
      var y = linear([0, maxCount], [H - m.bottom, m.top], { nice: true });
      var intFmt = function (v) {
        return num(v, 0);
      };
      var yTicks = y.ticks(5).filter(function (t) {
        return Number.isInteger(t);
      });
      m.left =
        Math.max.apply(
          null,
          [0].concat(
            yTicks.map(function (t) {
              return measure(intFmt(t), 11);
            })
          )
        ) + 14;
      var x = linear([edges[0], edges[nBins]], [m.left + 2, W - m.right]);
      var root = chartSvg(W, H, spec, "Histogram");
      var grid = svg("g");
      yTicks.forEach(function (t) {
        var py = crisp(y(t));
        grid.appendChild(svg("line", { class: "tv-gridline", x1: m.left, x2: W - m.right, y1: py, y2: py }));
        grid.appendChild(svg("text", { class: "tv-tick", x: m.left - 8, y: py, dy: "0.35em", "text-anchor": "end" }, intFmt(t)));
      });
      var xt = x.ticks(Math.max(2, Math.floor((W - m.left - m.right) / 80)));
      var lastRight = -Infinity;
      var xTickText = tickFormatter(spec.format, xt);
      xt.forEach(function (t) {
        var label = xTickText(t);
        var w = measure(label, 11);
        var px = x(t);
        if (px - w / 2 < lastRight + 6) return;
        lastRight = px + w / 2;
        grid.appendChild(svg("text", { class: "tv-tick", x: px, y: H - m.bottom + 16, "text-anchor": "middle" }, label));
      });
      root.appendChild(grid);
      var base = y(0);
      var marks = svg("g");
      for (var i = 0; i < nBins; i++) {
        (function (i) {
          var x0 = x(edges[i]);
          var x1 = x(edges[i + 1]);
          var avail = Math.max(1, x1 - x0 - GAP);
          var each = series.length === 2 ? Math.min(BAR_MAX, (avail - GAP) / 2) : Math.min(BAR_MAX, avail);
          var total = series.length === 2 ? each * 2 + GAP : each;
          var start = x0 + (x1 - x0 - total) / 2;
          var parts = [];
          series.forEach(function (s, k) {
            var c = s.counts ? s.counts[i] : null;
            if (!isNum(c) || c <= 0) return;
            var d = barPath("v", base, y(c), start + k * (each + GAP), each);
            if (d) parts.push(svg("path", { class: "tv-mark", d: d, style: { fill: color(s.role) } }));
          });
          var title = format(spec.format, edges[i]) + " to " + format(spec.format, edges[i + 1]);
          var g = markGroup(
            marks,
            { x: x0, y: m.top, width: Math.max(HIT_MIN, x1 - x0), height: H - m.top - m.bottom },
            parts,
            title
          );
          tooltip.attach(g, function () {
            return {
              title: title,
              rows: series.map(function (s) {
                return { label: s.name, value: intFmt(s.counts ? s.counts[i] : null), color: s.role, shape: "rect" };
              }),
            };
          });
        })(i);
      }
      root.appendChild(svg("line", { class: "tv-axisline", x1: m.left, x2: W - m.right, y1: crisp(base), y2: crisp(base) }));
      root.appendChild(marks);
      var refs2 = svg("g");
      verticalRefs(refs2, spec, x, m.top, H - m.bottom, false);
      root.appendChild(refs2);
      wrap.appendChild(root);
    });
  }

  /* range: the football field --------------------------------------------- */

  function range(body, spec) {
    spec = spec || {};
    var rows = (spec.rows || []).filter(Boolean);
    var hasMid = rows.some(function (r) {
      return isNum(r.mid);
    });
    var columns = [
      { key: "label", label: spec.labelHeader || "Method" },
      { key: "lo", label: "Low", align: "right", format: spec.format },
    ];
    if (hasMid) columns.push({ key: "mid", label: "Mid", align: "right", format: spec.format });
    columns.push({ key: "hi", label: "High", align: "right", format: spec.format });
    tableFor(body, { columns: columns, rows: rows });
    if (!rows.length) return empty(body);
    roleLegend(
      body,
      spec,
      distinct(
        rows.map(function (r) {
          return r.role || "model";
        })
      ),
      "rect"
    );

    frame(body, spec, "range", function (wrap, W) {
      var all = [];
      rows.forEach(function (r) {
        all.push(r.lo, r.hi, r.mid);
      });
      var ext = extent(all.concat(referenceValues(spec))) || [0, 1];
      if (spec.zero) ext = [Math.min(0, ext[0]), Math.max(0, ext[1])];
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : ext[0];
      var hi = isNum(dom[1]) ? dom[1] : ext[1];
      var labels = rowLabels(rows, W);
      var endW = Math.max.apply(
        null,
        [0].concat(
          rows.map(function (r) {
            return Math.max(measure(format(spec.format, r.lo), 11, 500), measure(format(spec.format, r.hi), 11, 500));
          })
        )
      ) + 8;
      var refCount = (spec.reference || []).length;
      var m = { top: 8 + (refCount ? refCount * 14 + 8 : 0), right: 8 + endW, bottom: 26, left: labels.width + 12 + endW };
      var n = rows.length;
      var rowH = spec.height ? Math.max(HIT_MIN, (spec.height - m.top - m.bottom) / n) : 34;
      var plotH = rowH * n;
      var H = Math.round(m.top + plotH + m.bottom);
      var x = linear([lo, hi], [m.left, W - m.right], { nice: !spec.domain });
      var thick = Math.min(BAR_MAX, 18, rowH * 0.6);
      var root = chartSvg(W, H, spec, "Range");
      var grid = svg("g");
      xAxisTicks(grid, x, m.top, m.top + plotH, spec.format, Math.max(2, Math.floor((W - m.left - m.right) / 90)));
      root.appendChild(grid);
      var marks = svg("g");
      var valueLayer = svg("g", { class: "tv-values" });
      rows.forEach(function (r, i) {
        var cy = m.top + rowH * i + rowH / 2;
        textBlock(marks, labels.wrapped[i].lines, labels.width + 2, cy, { class: "tv-label", "text-anchor": "end" });
        var parts = [];
        if (isNum(r.lo) && isNum(r.hi)) {
          var d = barPath("h", x(Math.min(r.lo, r.hi)), x(Math.max(r.lo, r.hi)), cy - thick / 2, thick, true);
          if (d) parts.push(svg("path", { class: "tv-mark", d: d, style: { fill: color(r.role || "model") } }));
          if (isNum(r.mid)) {
            parts.push(svg("rect", { x: x(r.mid) - GAP / 2, y: cy - thick / 2, width: GAP, height: thick, style: { fill: "var(--surface)" } }));
          }
        }
        var g = markGroup(
          marks,
          { x: 0, y: m.top + rowH * i, width: W, height: rowH },
          parts,
          r.label + ": " + format(spec.format, r.lo) + " to " + format(spec.format, r.hi)
        );
        if (isNum(r.lo) && isNum(r.hi)) {
          valueLayer.appendChild(svg("text", { class: "tv-value", x: x(Math.min(r.lo, r.hi)) - 6, y: cy, dy: "0.35em", "text-anchor": "end" }, format(spec.format, Math.min(r.lo, r.hi))));
          valueLayer.appendChild(svg("text", { class: "tv-value", x: x(Math.max(r.lo, r.hi)) + 6, y: cy, dy: "0.35em" }, format(spec.format, Math.max(r.lo, r.hi))));
        }
        tooltip.attach(g, function () {
          var out = [
            { label: "Low", value: format(spec.format, r.lo), color: r.role || "model", shape: "rect" },
          ];
          if (isNum(r.mid)) out.push({ label: "Mid", value: format(spec.format, r.mid) });
          out.push({ label: "High", value: format(spec.format, r.hi) });
          (spec.reference || []).forEach(function (ref) {
            if (ref && isNum(ref.value)) out.push({ label: ref.label || "Reference", value: format(spec.format, ref.value) });
          });
          return { title: r.label, rows: out };
        });
      });
      root.appendChild(marks);
      var refs2 = svg("g");
      verticalRefs(refs2, spec, x, m.top, m.top + plotH, true);
      root.appendChild(refs2);
      root.appendChild(valueLayer);
      wrap.appendChild(root);
    });
  }

  /* waterfall: a sum-of-the-parts bridge ---------------------------------- */

  function waterfall(body, spec) {
    spec = spec || {};
    var steps = (spec.steps || []).filter(Boolean);
    var items = [];
    var running = 0;
    if (spec.start && isNum(spec.start.value)) {
      running = spec.start.value;
      items.push({ label: spec.start.label || "Start", kind: "total", from: 0, to: running, value: running });
    }
    steps.forEach(function (s) {
      var v = isNum(s.value) ? s.value : 0;
      items.push({ label: s.label, kind: v >= 0 ? "up" : "down", from: running, to: running + v, value: s.value });
      running += v;
    });
    var totalLabel = (spec.total && spec.total.label) || "Total";
    var reported = spec.total && isNum(spec.total.value) ? spec.total.value : null;
    items.push({ label: totalLabel, kind: "total", from: 0, to: running, value: running });

    tableFor(body, {
      columns: [
        { key: "label", label: spec.labelHeader || "Component" },
        { key: "value", label: spec.valueLabel || "Value", align: "right", format: spec.format },
        { key: "to", label: "Running total", align: "right", format: spec.format },
      ],
      rows: items,
    });
    if (!steps.length) return empty(body);
    var handle = handleFor(body);
    if (handle && reported !== null && Math.abs(reported - running) > Math.max(1e-6, Math.abs(running) * 1e-6)) {
      handle.addNote({
        what: totalLabel,
        why: "The components sum to " + format(spec.format, running) + " but the total supplied is " + format(spec.format, reported) + "; the bridge draws the sum.",
      });
    }
    legendFor(body, [
      { label: spec.upLabel || "Adds", color: "pos", shape: "rect" },
      { label: spec.downLabel || "Subtracts", color: "neg", shape: "rect" },
      { label: spec.totalLegend || "Total", color: "total", shape: "rect" },
    ]);

    frame(body, spec, "waterfall", function (wrap, W) {
      var levels = [0];
      items.forEach(function (it) {
        levels.push(it.from, it.to);
      });
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : Math.min.apply(null, levels);
      var hi = isNum(dom[1]) ? dom[1] : Math.max.apply(null, levels);
      var H0 = spec.height || 300;
      var m = { top: 20, right: 8, bottom: 0, left: 0 };
      var n = items.length;
      var yProbe = linear([lo, hi], [H0, m.top], { nice: !spec.domain });
      m.left = yTickWidth(yProbe, spec.format, 5) + 14;
      var stepW = (W - m.left - m.right) / n;
      var wrapped = items.map(function (it) {
        return wrapText(it.label, Math.max(40, stepW - 6), 11, 3);
      });
      var lines = Math.max.apply(
        null,
        wrapped.map(function (w) {
          return w.lines.length;
        })
      );
      m.bottom = 14 + lines * 13;
      var H = Math.max(H0, 160);
      var y = linear([lo, hi], [H - m.bottom, m.top], { nice: !spec.domain });
      var thick = Math.min(BAR_MAX, stepW * 0.6);
      var root = chartSvg(W, H, spec, "Waterfall");
      var grid = svg("g");
      yAxisTicks(grid, y, m.left, W - m.right, spec.format, 5);
      root.appendChild(grid);
      var marks = svg("g");
      var connectors = svg("g");
      var labelAll = n <= 9;
      items.forEach(function (it, i) {
        var cx = m.left + stepW * i + stepW / 2;
        var fill = color(it.kind === "total" ? "total" : it.kind === "up" ? "pos" : "neg");
        var parts = [];
        var d = barPath("v", y(it.from), y(it.to), cx - thick / 2, thick);
        if (d) parts.push(svg("path", { class: "tv-mark", d: d, style: { fill: fill } }));
        var g = markGroup(
          marks,
          { x: m.left + stepW * i, y: m.top, width: stepW, height: H - m.top - m.bottom },
          parts,
          it.label + ": " + format(spec.format, it.value)
        );
        if (labelAll || it.kind === "total") {
          var up = it.to >= it.from;
          var text = it.kind === "total" ? format(spec.format, it.value) : stepText(spec.format, it.value);
          g.appendChild(
            svg("text", { class: "tv-value", x: cx, y: y(it.to) + (up ? -6 : 14), "text-anchor": "middle" }, text)
          );
        }
        textBlock(marks, wrapped[i].lines, cx, H - m.bottom + 10 + ((wrapped[i].lines.length - 1) * 13) / 2, { class: "tv-tick", "text-anchor": "middle" }, 13);
        if (i < n - 1) {
          var level = crisp(y(it.to));
          connectors.appendChild(
            svg("line", { class: "tv-connector", x1: cx + thick / 2 + GAP, x2: cx + stepW - thick / 2 - GAP, y1: level, y2: level })
          );
        }
        tooltip.attach(g, function () {
          var rows = [];
          if (it.kind === "total") {
            rows.push({ label: it.label, value: format(spec.format, it.value), color: fill, shape: "rect" });
          } else {
            rows.push({ label: "Change", value: stepText(spec.format, it.value), color: fill, shape: "rect" });
            rows.push({ label: "Running total", value: format(spec.format, it.to) });
          }
          return { title: it.label, rows: rows };
        });
      });
      var zero = clamp(0, y.domain[0], y.domain[1]);
      root.appendChild(svg("line", { class: "tv-axisline", x1: m.left, x2: W - m.right, y1: crisp(y(zero)), y2: crisp(y(zero)) }));
      root.appendChild(connectors);
      root.appendChild(marks);
      wrap.appendChild(root);
    });
  }

  /* A step in the chart's own format with its sign in front: +1,452 or −97.8mm. */
  function stepText(spec, v) {
    if (!isNum(v)) return "n/a";
    var body = format(spec, Math.abs(v));
    if (!/[1-9]/.test(body)) return body;
    return (v > 0 ? "+" : MINUS) + body;
  }

  /* tiles: stat tiles ----------------------------------------------------- */

  function deltaNode(delta) {
    if (!delta || !isNum(delta.value)) return null;
    var v = delta.value;
    var higherIsBetter = delta.higher_is_better !== false;
    var tone = v === 0 ? "flat" : (v > 0) === higherIsBetter ? "good" : "bad";
    var icon = null;
    if (v !== 0) {
      icon = svg(
        "svg",
        { class: "tv-delta__icon", viewBox: "0 0 10 10", "aria-hidden": "true", focusable: "false" },
        svg("path", { d: v > 0 ? "M5 1.5 L9 8 H1 Z" : "M5 8.5 L1 2 H9 Z" })
      );
    }
    return el(
      "p",
      { class: "tv-delta tv-delta--" + tone },
      icon,
      el("span", null, format(delta.format || "signed:2", v)),
      delta.label ? el("span", { class: "tv-delta__label" }, delta.label) : null
    );
  }

  function tiles(body, input) {
    var list = Array.isArray(input) ? input : (input && input.tiles) || [];
    var grid = el("div", { class: "tv-tiles" });
    list.forEach(function (t) {
      if (!t) return;
      var value = isNum(t.value) ? format(t.format, t.value) : isNil(t.value) ? "n/a" : String(t.value);
      grid.appendChild(
        el(
          t.href ? "a" : "div",
          { class: "tv-tile", href: t.href || null },
          el("p", { class: "tv-tile__label" }, t.label),
          el("p", { class: "tv-tile__value" }, value),
          deltaNode(t.delta),
          t.sub ? el("p", { class: "tv-tile__sub" }, t.sub) : null,
          t.status ? TV.chip(t.status, t.statusText) : null
        )
      );
    });
    body.appendChild(grid);
    return grid;
  }

  TV.charts = {
    hbar: hbar,
    column: column,
    dot: dot,
    line: line,
    heat: heat,
    hist: hist,
    range: range,
    waterfall: waterfall,
    tiles: tiles,
  };

  /* Section registry ------------------------------------------------------ */

  var renderers = {};
  var registered = [];

  TV.sections = {
    /* Section scripts are inlined in page order, so registration order is page order. */
    register: function (id, fn) {
      if (typeof fn !== "function") throw new Error("Section " + id + " registered without a renderer.");
      if (registered.indexOf(id) < 0) registered.push(id);
      renderers[id] = fn;
    },
    has: function (id) {
      return Object.prototype.hasOwnProperty.call(renderers, id);
    },
    order: function () {
      return registered.slice();
    },
    render: function (id, root, data, snapshot) {
      if (!TV.sections.has(id)) throw new Error("No renderer is registered for section " + id + ".");
      return renderers[id](root, data, snapshot);
    },

    /*
     * The default drawing of a section's figures, in the snapshot's key order.
     * Stat tiles sit on the page plane; every other kind gets a card drawn by
     * the kit chart it names. A kind the kit does not have becomes a note on
     * its card, never a guess. Returns {figureId: handle} so the caller can
     * hang refusal notes under the figures they affect.
     */
    figures: function (root, data) {
      var figures = (data && data.figures) || {};
      var ids = Object.keys(figures);
      var handles = {};
      if (!ids.length) return handles;
      var provenance = (data && data.provenance) || [];
      var tileIds = ids.filter(function (id) {
        return figures[id] && figures[id].kind === "tiles";
      });
      tileIds.forEach(function (id) {
        var f = figures[id];
        var set = el(
          "div",
          { class: "tv-tileset", "data-figure-id": id },
          f.title ? el("h3", { class: "tv-tileset__title" }, f.title) : null
        );
        root.appendChild(set);
        tiles(set, f.data || {});
        handles[id] = { root: set, body: set, title: f.title || "", addNote: function (r) { set.appendChild(refusalNote(r)); } };
      });
      var cardIds = ids.filter(function (id) {
        return tileIds.indexOf(id) < 0;
      });
      if (!cardIds.length) return handles;
      var grid = el("div", { class: "tv-grid" });
      root.appendChild(grid);
      cardIds.forEach(function (id) {
        var f = figures[id] || {};
        var prov = provenance.filter(function (p) {
          return p && p.figure === id;
        })[0];
        var handle = TV.figure(grid, {
          id: id,
          anchor: data && data.id ? "fig-" + data.id + "-" + id : null,
          title: f.title,
          subtitle: f.subtitle,
          provenance: prov,
          wide: !!(f.wide || (f.data && f.data.wide)),
        });
        handles[id] = handle;
        var chart = TV.charts[f.kind];
        if (!chart) {
          handle.addNote({ what: id, why: "The chart kit has no chart named " + JSON.stringify(String(f.kind)) + "." });
          return;
        }
        chart(handle.body, Object.assign({ title: f.title }, f.data || {}));
      });
      return handles;
    },
  };
})(window);
