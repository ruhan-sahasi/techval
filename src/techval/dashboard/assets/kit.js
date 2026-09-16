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
 * it appears. The roles are model, model-muted (a tint of the model, for a
 * second reading of the same model), baseline, alt (the one named
 * comparison), third, total, and pos and neg for diverging bars.
 *
 * Mark rules the kit enforces rather than suggests: bars at most 24px thick
 * with a 4px rounded data end and a square baseline end, 2px lines with round
 * joins, markers of radius 4 with a 2px surface ring, solid 1px hairlines for
 * grids and references, one y axis, a legend only for two or more series, and
 * a table view for every chart so no value is reachable only by hover.
 *
 * Options every chart reads, beyond its own data:
 *
 *   legend        false (or []) draws no legend; a list of {label, role or
 *                 color, shape} draws exactly that list.
 *   table         a fuller table view in place of the drawn series: one
 *                 {columns, rows, caption} or a list of them. A cell may be a
 *                 string, a number, a node, or {text, sub, chip, mono, muted}.
 *   tableHeaders  {key: header} renames a column of the chart's own table.
 *   reference     rules at a value, labelled. A horizontal rule's label is
 *                 measured and set where it touches no mark: the right end
 *                 above or below the rule, then the left end, then outside
 *                 the plot beyond its right edge.
 *   labelWidth    where hbar, dot and range wrap their row labels: pixels, or
 *                 a share of the chart's width when at most 1 (default 0.34).
 *
 * Chart by chart, what is not obvious from the data:
 *
 *   hbar    rows [{label, value, role, note, lo, hi}]; lo and hi draw a whisker.
 *   column  rows [{label, value, role, labelled}]; diverging colours by sign.
 *           Column labels are thinned to fit and the first and last always
 *           stay. One value label names the highest bar, and the lowest when it
 *           is negative; where any row carries labelled, the flagged rows carry
 *           the labels instead. labels "all" labels every bar.
 *   dot     series [{key, name, role}] and rows [{label, values, lo, hi,
 *           intervals: {key: {lo, hi}}, group, text, tip, role, hollow,
 *           aside, asideStrong, labelled, gap}]. A value label names only the
 *           value of the dot it sits beside. labels: "none", "all", or
 *           {series, rows: "auto" | "all" | "flagged" | [index], text: "value"
 *           | "gap"}; by default the row that stands out (largest gap for two
 *           series, largest value otherwise) is labelled, and a row that
 *           carries text is labelled with that text instead. Points closer
 *           than a marker are drawn apart vertically, the row growing to hold
 *           them, unless dodge is false.
 *           aside is a column of text at the right edge, under asideHeader.
 *   line    series [{name, role, values: [{x, y, lo, hi}]}]; x {label,
 *           format, type: "date"}; yScale "log"; yTitle; shade [{from, to,
 *           label, tip}] with shadeLegend; points [{x, y, label, tip}].
 *   heat    rows, cols, values; scale "diverging" or sequential; breaks [a, b]
 *           for fixed diverging classes; mono and labelAlign for row labels,
 *           which are never clipped (labelWidth wraps them); groups [{label,
 *           count}]; colTitle; cellMax; rowNotes; rowTips.
 *   hist    edges and series [{name, role, counts}], at most two series.
 *   range   rows [{label, lo, mid, hi, role}], the football field.
 *   waterfall start, steps [{label, value}] and total; its legend lists only
 *           the kinds of step it draws.
 *   table   columns and rows, drawn as the figure itself.
 *   tiles   tiles [{label, value, format, sub, delta, status}]. Inside a
 *           figure card they carry their own table view.
 *
 * Figures in a section: TV.sections.figures(root, data, opts) draws each
 * figure by its kind, in reading order. A figure may carry order (a number),
 * wide, card (tiles drawn in a card), part (the id of the card it is drawn
 * inside), note or notes (a caution: the figure stands but is fragile), and
 * refusals (the names of section refusals that belong under it). opts:
 *
 *   order     a list of ids, lists of ids, or {title, ids}; each list starts
 *             a new group, a titled group gets a heading. Unnamed figures
 *             follow in their own order.
 *   figures   {id: override} merged over the snapshot's figure; its data may
 *             be a function of the figure's data, and draw or after a
 *             function of (handle, figure).
 *   parts     {childId: parentId}, the same as part on the figure.
 *   refusals  function (id, figure) returning more refusal names.
 *
 * Helpers the sections share: TV.frame (a chart container redrawn when its
 * width changes), TV.axis.x, TV.axis.y and TV.axis.yWidth, TV.tickFormat,
 * TV.measure (sans or mono), TV.wrapText, TV.thinLabels, TV.markGroup,
 * TV.chartSvg, TV.crisp, TV.barPath, TV.refusalNote and TV.cautionNote.
 */
(function (global) {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var MINUS = "−";
  var BAR_MAX = 24;
  var RADIUS = 4;
  var GAP = 2;
  var HIT_MIN = 24;
  var GROUP_ROW = 26;

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

  function maxOf(list, floor) {
    return Math.max.apply(null, [isNum(floor) ? floor : 0].concat(finite(list)));
  }

  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  TV.crisp = crisp;

  /* An ISO date, 2016-09-16, as UTC milliseconds; NaN for anything else. */
  function parseDay(text) {
    var s = String(isNil(text) ? "" : text);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return NaN;
    var t = Date.parse(s + "T00:00:00Z");
    return isFinite(t) ? t : NaN;
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

  function isNode(v) {
    return !!v && typeof v === "object" && typeof v.nodeType === "number";
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

  /* A difference of two probabilities in percentage points: +30.5 pts. */
  function points(v, dp) {
    if (!isNum(v)) return "n/a";
    return signed(v * 100, dpOr(dp, 1)) + " pts";
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
    points: points,
    int: function (v) {
      return num(v, 0);
    },
    auto: auto,
  };

  /*
   * Figure data arrives as JSON, so a format is usually a string: "num:4",
   * "pct:1", "signed:4", "compact", "mm", "mult:1", "points". A function also works.
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

  TV.tickFormat = tickFormatter;

  /* Tokens and colour ----------------------------------------------------- */

  TV.token = function (name) {
    var prop = String(name).indexOf("--") === 0 ? name : "--" + name;
    return getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
  };

  var ROLE_VARS = {
    model: "var(--c-model)",
    "model-muted": "var(--c-model-muted)",
    baseline: "var(--c-baseline)",
    alt: "var(--c-alt)",
    third: "var(--c-third)",
    total: "var(--c-total)",
    pos: "var(--div-pos)",
    neg: "var(--div-neg)",
  };

  var ROLE_LABELS = {
    model: "Model",
    "model-muted": "Model",
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

  /* A reference line's tooltip row; a label that already ends in the value ("Price 225.27") is not made to say it twice. */
  function refTipRow(ref, fmt) {
    var value = format(fmt, ref.value);
    var label = String(ref.label || "Reference");
    if (label.length > value.length && label.slice(-value.length) === value) label = label.slice(0, -value.length).trim();
    return { label: label, value: value };
  }

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

  /* A log scale on 1, 2 and 5 times a power of ten, between two positive bounds. */
  function niceLogBound(v, up) {
    var k = Math.floor(Math.log10(v));
    var steps = [1, 2, 5, 10];
    var base = Math.pow(10, k);
    if (up) {
      for (var i = 0; i < steps.length; i++) if (steps[i] * base >= v * 0.999999) return steps[i] * base;
    } else {
      for (var j = steps.length - 1; j >= 0; j--) if (steps[j] * base <= v * 1.000001) return steps[j] * base;
    }
    return v;
  }

  function logTicks(lo, hi, maxCount) {
    var out = [];
    var k0 = Math.floor(Math.log10(lo));
    var k1 = Math.ceil(Math.log10(hi));
    [[1, 2, 5], [1]].some(function (mults) {
      out = [];
      for (var k = k0; k <= k1; k++) {
        mults.forEach(function (m) {
          var v = m * Math.pow(10, k);
          if (v >= lo * 0.999 && v <= hi * 1.001) out.push(parseFloat(v.toPrecision(6)));
        });
      }
      return out.length <= maxCount;
    });
    return out;
  }

  function logScale(lo, hi, range) {
    var d0 = niceLogBound(lo, false);
    var d1 = niceLogBound(hi, true);
    if (d0 === d1) d1 = d0 * 10;
    var inner = linear([Math.log10(d0), Math.log10(d1)], range);
    var scale = function (v) {
      return inner(Math.log10(v));
    };
    scale.domain = [d0, d1];
    scale.range = range.slice();
    scale.ticks = function (n) {
      return logTicks(d0, d1, n || 8);
    };
    scale.log = true;
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

  TV.scale = { linear: linear, band: band, log: logScale };

  /* Text measurement and wrapping ----------------------------------------- */

  var measureCtx = null;
  var fontFamily = null;
  var monoFamily = null;

  /* family "mono" measures in the mono face; anything else in the sans face. */
  function measure(text, size, weight, family) {
    if (!measureCtx) {
      measureCtx = document.createElement("canvas").getContext("2d");
    }
    var face;
    if (family === "mono") {
      if (!monoFamily) monoFamily = TV.token("--font-mono") || "monospace";
      face = monoFamily;
    } else {
      if (!fontFamily) fontFamily = TV.token("--font-sans") || "system-ui, sans-serif";
      face = fontFamily;
    }
    measureCtx.font = (weight || 400) + " " + (size || 12) + "px " + face;
    return Math.ceil(measureCtx.measureText(String(text)).width * 1.04) + 1;
  }

  function wrapText(text, maxWidth, size, maxLines, weight, family) {
    var words = String(isNil(text) ? "" : text).split(/\s+/).filter(Boolean);
    var lines = [];
    var line = "";
    words.forEach(function (word) {
      var next = line ? line + " " + word : word;
      if (!line || measure(next, size, weight, family) <= maxWidth) {
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
        return measure(l, size, weight, family);
      })
    );
    return { lines: lines, width: width };
  }

  TV.measure = measure;
  TV.wrapText = wrapText;

  /*
   * Which labels along an axis to show: items are {center, width} in order.
   * The first label and the last always stay; a middle label is dropped only
   * when it would touch a label already kept or the last one.
   */
  function thinLabels(items, gap) {
    var n = items.length;
    if (!n) return [];
    gap = isNum(gap) ? gap : 6;
    function left(i) {
      return items[i].center - items[i].width / 2;
    }
    function right(i) {
      return items[i].center + items[i].width / 2;
    }
    var kept = [0];
    if (n === 1) return kept;
    for (var i = 1; i < n - 1; i++) {
      if (left(i) >= right(kept[kept.length - 1]) + gap && right(i) + gap <= left(n - 1)) kept.push(i);
    }
    if (left(n - 1) >= right(kept[kept.length - 1]) + gap) {
      kept.push(n - 1);
    } else if (kept.length > 1) {
      kept.pop();
      kept.push(n - 1);
    } else if (left(n - 1) >= right(0)) {
      kept.push(n - 1);
    }
    return kept;
  }

  TV.thinLabels = thinLabels;

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

  TV.textBlock = textBlock;

  /* Geometry for placing labels clear of marks ---------------------------- */

  function rectsOverlap(a, b, pad) {
    pad = pad || 0;
    return a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad;
  }

  function segmentHitsRect(s, r, pad) {
    var x0 = r.x - pad;
    var x1 = r.x + r.w + pad;
    var y0 = r.y - pad;
    var y1 = r.y + r.h + pad;
    if (Math.max(s[0], s[2]) < x0 || Math.min(s[0], s[2]) > x1 || Math.max(s[1], s[3]) < y0 || Math.min(s[1], s[3]) > y1) {
      return false;
    }
    var len = Math.max(Math.abs(s[2] - s[0]), Math.abs(s[3] - s[1]));
    var n = Math.max(1, Math.ceil(len / 2));
    for (var i = 0; i <= n; i++) {
      var t = i / n;
      var px = s[0] + (s[2] - s[0]) * t;
      var py = s[1] + (s[3] - s[1]) * t;
      if (px >= x0 && px <= x1 && py >= y0 && py <= y1) return true;
    }
    return false;
  }

  /* obstacles are {rect: {x, y, w, h}} or {seg: [x1, y1, x2, y2]}. */
  function hits(box, obstacles) {
    return obstacles.some(function (o) {
      if (o.rect) return rectsOverlap(box, o.rect, 3);
      if (o.seg) return segmentHitsRect(o.seg, box, 2);
      return false;
    });
  }

  /* An 11px label's box around its baseline. */
  function labelBox(x, base, w, anchor) {
    var left = anchor === "end" ? x - w : anchor === "middle" ? x - w / 2 : x;
    return { x: left, y: base - 9, w: w, h: 12 };
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

  /* Notes ------------------------------------------------------------------ */

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

  /*
   * A caution: the figure above it stands, but a limit of its input makes it
   * fragile. It carries the warning glyph, never the refusal chip. A string, or
   * {what, why} for a caution with a name.
   */
  function cautionNote(note) {
    var what = typeof note === "string" ? "" : note && note.what;
    var why = typeof note === "string" ? note : note && note.why;
    return el(
      "div",
      { class: "tv-note tv-note--caution", role: "note" },
      chipIcon("inside_noise"),
      el("p", null, what ? el("strong", null, what) : null, what && why ? ". " : "", why || "")
    );
  }

  TV.refusalNote = refusalNote;
  TV.cautionNote = cautionNote;

  /* Figure cards ---------------------------------------------------------- */

  function provenanceRows(provenance) {
    var list = Array.isArray(provenance) ? provenance : isNil(provenance) ? [] : [provenance];
    return list
      .map(function (p) {
        return typeof p === "string" ? { entry_point: p, inputs: [] } : p;
      })
      .filter(function (p) {
        return p && p.entry_point;
      });
  }

  /*
   * Every entry point behind a figure, compactly: entry points in one module
   * share its path, so techval.commands_peers._load_groups, _load_panel.
   */
  function compactEntries(rows) {
    var groups = [];
    distinct(
      rows.map(function (p) {
        return String(p.entry_point);
      })
    ).forEach(function (entry) {
      var cut = entry.lastIndexOf(".");
      var mod = cut > 0 ? entry.slice(0, cut) : "";
      var name = cut > 0 ? entry.slice(cut + 1) : entry;
      var group = groups.filter(function (g) {
        return g.mod === mod;
      })[0];
      if (group) group.names.push(name);
      else groups.push({ mod: mod, names: [name] });
    });
    return groups
      .map(function (g) {
        return (g.mod ? g.mod + "." : "") + g.names.join(", ");
      })
      .join(" · ");
  }

  function sourceTitle(rows) {
    return rows
      .map(function (p) {
        var inputs = p.inputs && p.inputs.length ? p.inputs.join(", ") : "no inputs";
        return p.entry_point + ": " + inputs;
      })
      .join("\n");
  }

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
    var toggle = el("button", { class: "tv-btn", type: "button", hidden: true }, "Show data");
    var source = el("span", { class: "tv-figure__source" });
    var foot = el("footer", { class: "tv-figure__foot" }, source, toggle);
    var entries = [];

    function addProvenance(provenance) {
      entries = entries.concat(provenanceRows(provenance));
      clear(source);
      if (entries.length) {
        source.appendChild(document.createTextNode(compactEntries(entries)));
        source.setAttribute("title", sourceTitle(entries));
      }
    }
    addProvenance(opts.provenance);

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
        var key = (refusal && refusal.what) + " " + (refusal && refusal.why);
        if (noted.indexOf(key) >= 0) return;
        noted.push(key);
        notes.appendChild(refusalNote(refusal));
      },
      hasNote: function (refusal) {
        return noted.indexOf((refusal && refusal.what) + " " + (refusal && refusal.why)) >= 0;
      },
      addCaution: function (note) {
        notes.appendChild(cautionNote(note));
      },
      addProvenance: addProvenance,
      /*
       * A second figure drawn inside this card, under its own heading. Its table
       * view joins the card's, its entry points join the card's source line.
       */
      part: function (o) {
        o = o || {};
        var partLegend = el("div", { class: "tv-figure__legend" });
        var partBody = el("div", { class: "tv-figure__body" });
        var block = el(
          "div",
          { class: "tv-figure__part", "data-figure-id": o.id || null },
          el(
            "div",
            { class: "tv-figure__head" },
            el("h4", { class: "tv-figure__title" }, o.title || ""),
            o.subtitle ? el("p", { class: "tv-figure__subtitle" }, o.subtitle) : null
          ),
          partLegend,
          partBody
        );
        body.appendChild(block);
        addProvenance(o.provenance);
        var sub = {
          root: block,
          body: partBody,
          legend: partLegend,
          footer: foot,
          table: table,
          notes: notes,
          toggle: toggle,
          title: o.title || "",
          addNote: handle.addNote,
          hasNote: handle.hasNote,
          addCaution: handle.addCaution,
          addProvenance: addProvenance,
        };
        block.__tvHandle = sub;
        return sub;
      },
    };

    toggle.addEventListener("click", function () {
      var showTable = table.hidden;
      table.hidden = !showTable;
      body.hidden = showTable;
      legend.hidden = showTable;
      toggle.textContent = showTable ? "Show chart" : "Show data";
    });

    root.__tvHandle = handle;
    if (opts.note) {
      (Array.isArray(opts.note) ? opts.note : [opts.note]).forEach(function (n) {
        handle.addNote(typeof n === "string" ? { what: "", why: n } : n);
      });
    }
    if (opts.caution) {
      (Array.isArray(opts.caution) ? opts.caution : [opts.caution]).forEach(handle.addCaution);
    }
    if (parent) parent.appendChild(root);
    return handle;
  };

  function handleFor(body) {
    var fig = body && body.closest ? body.closest(".tv-figure__part, .tv-figure, .tv-tileset") : null;
    return fig && fig.__tvHandle ? fig.__tvHandle : null;
  }

  /* Legend ---------------------------------------------------------------- */

  TV.legend = function (slot, items) {
    clear(slot);
    if (!items || !items.length) return null;
    var list = el("ul", { class: "tv-legend" });
    items.forEach(function (item) {
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

  function legendOff(spec) {
    return spec.legend === false || (Array.isArray(spec.legend) && !spec.legend.length);
  }

  /* A chart's legend: none when the spec turns it off, the spec's own list when it gives one. */
  function chartLegend(body, spec, items, shape) {
    if (legendOff(spec)) return;
    if (Array.isArray(spec.legend)) {
      legendFor(
        body,
        spec.legend.map(function (item) {
          return { label: item.label, color: item.role || item.color, shape: item.shape || shape };
        })
      );
      return;
    }
    if (items && items.length > 1) legendFor(body, items);
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

  /* Extra tooltip rows a figure's data carries: [{label, value, format}]. */
  function tipRows(list) {
    return (Array.isArray(list) ? list : list ? [list] : [])
      .filter(function (t) {
        return t && !isNil(t.label);
      })
      .map(function (t) {
        return { label: t.label, value: typeof t.value === "string" ? t.value : format(t.format, t.value) };
      });
  }

  /* Table view ------------------------------------------------------------ */

  /*
   * A cell is a string, a number (formatted by its column), a node, or
   * {text, sub, chip, chipText, mono, muted}: a first line, a muted second line
   * and a status chip.
   */
  function cellContent(raw, column) {
    if (isNode(raw)) return raw;
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      var first = null;
      if (raw.chip) {
        first = TV.chip(raw.chip, isNil(raw.chipText) ? null : raw.chipText, { compact: true });
      } else if (!isNil(raw.text)) {
        var text = isNum(raw.text) ? format(column.format, raw.text) : String(raw.text);
        first = raw.mono || raw.muted ? el("span", { class: (raw.mono ? "tv-mono" : "") + (raw.muted ? " tv-muted" : "") }, text) : text;
      }
      if (isNil(raw.sub) || raw.sub === "") return first;
      return [el("div", null, first), el("div", { class: "tv-muted" }, String(raw.sub))];
    }
    if (column.format || isNum(raw)) return format(column.format, raw);
    return isNil(raw) ? "n/a" : String(raw);
  }

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
            var cls = [c.align === "right" ? "tv-num" : null, c.mono ? "tv-mono" : null, c.nowrap ? "tv-nowrap" : null]
              .filter(Boolean)
              .join(" ");
            var content = cellContent(row ? row[c.key] : null, c);
            return i === 0
              ? el("th", { scope: "row", class: cls || null }, content)
              : el("td", { class: cls || null }, content);
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
    if (handle && handle.toggle && handle.table) handle.toggle.hidden = false;
    return wrap;
  };

  /*
   * A chart's table view: the spec's fuller table when it gives one, otherwise
   * the chart's own columns with any headers the spec renames.
   */
  function tableFor(body, def, spec) {
    var handle = handleFor(body);
    if (!handle) return;
    var custom = spec && spec.table;
    var list = (Array.isArray(custom) ? custom : custom ? [custom] : []).filter(function (t) {
      return t && Array.isArray(t.columns);
    });
    if (list.length) {
      list.forEach(function (t) {
        TV.tableView(handle, t);
      });
      return;
    }
    var headers = (spec && spec.tableHeaders) || {};
    TV.tableView(
      handle,
      Object.assign({}, def, {
        columns: def.columns.map(function (c) {
          return Object.prototype.hasOwnProperty.call(headers, c.key) ? Object.assign({}, c, { label: headers[c.key] }) : c;
        }),
      })
    );
  }

  /* Chart frame: one container, redrawn when its width changes ------------- */

  var liveCharts = [];

  function frame(body, kind, draw) {
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

  TV.frame = frame;

  function redrawAll() {
    fontFamily = null;
    monoFamily = null;
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
      "aria-label": (spec && (spec.ariaLabel || spec.title)) || kind + " chart",
    });
  }

  TV.chartSvg = chartSvg;

  function empty(body, text) {
    body.appendChild(el("p", { class: "tv-empty" }, text || "No values to draw."));
  }

  /*
   * A bar with a rounded data end and a square baseline end. orient "h" grows
   * along x from base to end; "v" grows along y. cross is the left (h) or top
   * (v) edge of the bar across its thickness.
   */
  function barPath(orient, base, end, cross, thickness, roundBoth) {
    if (end === base) return "";
    var s = end > base ? 1 : -1;
    /* A value too small for a pixel still draws one, so a -0.003 is never an empty slot beside its neighbours. */
    if (Math.abs(end - base) < 1) end = base + s;
    var len = Math.abs(end - base);
    var r = Math.min(RADIUS, thickness / 2, roundBoth ? len / 2 : len);
    var t = thickness;
    var c = cross;
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

  TV.markGroup = markGroup;

  /* limit is the right edge a label may reach, the svg's own width where the caller knows it. */
  function xAxisTicks(g, scale, y0, y1, fmt, count, limit) {
    var edge = isNum(limit) ? limit : scale.range[1] + 12;
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
      if (i === ticks.length - 1 && x + w / 2 > edge) {
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

  /*
   * How many ticks to ask a horizontal axis for: one per 90px and at least
   * three, so a chart in a half-width card still reads its scale, and never so
   * few that a finely niced domain is left with a single label. Labels that
   * would touch are dropped as they are drawn.
   */
  function labelTickCount(scale, plotWidth) {
    var count = Math.max(3, Math.floor(plotWidth / 90));
    while (scale.ticks(count).length < 2 && count < 10) count++;
    return count;
  }

  TV.axis = { x: xAxisTicks, y: yAxisTicks, yWidth: yTickWidth, count: labelTickCount };

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

  /*
   * Where each horizontal reference label goes. Tried in turn: the right end
   * above the rule, the right end below it, the left end above and the left
   * end below; the first spot that touches no obstacle (a bar, a value label,
   * a line, another reference label) wins. With none free the label goes
   * outside the plot beyond its right edge, and outside is how much room that
   * needs. forced keeps a label outside once a caller has made room for it.
   */
  function placeHorizontalRefs(spec, y, left, right, top, bottom, obstacles, forced) {
    var list = [];
    var outside = 0;
    var taken = [];
    (spec.reference || []).forEach(function (ref) {
      if (!ref || !isNum(ref.value)) return;
      var index = list.length;
      var py = crisp(y(ref.value));
      var item = { ref: ref, py: py, text: ref.label ? String(ref.label) : "", outside: false };
      list.push(item);
      if (!item.text) return;
      var w = measure(item.text, 11, 500);
      item.width = w;
      var spots = [
        { x: right - 4, anchor: "end", base: py - 5 },
        { x: right - 4, anchor: "end", base: py + 13 },
        { x: left + 4, anchor: "start", base: py - 5 },
        { x: left + 4, anchor: "start", base: py + 13 },
      ];
      var chosen = null;
      if (!(forced && forced[index])) {
        for (var k = 0; k < spots.length && !chosen; k++) {
          var box = labelBox(spots[k].x, spots[k].base, w, spots[k].anchor);
          if (box.y < 0 || box.y + box.h > bottom + 2) continue;
          if (hits(box, obstacles) || hits(box, taken)) continue;
          chosen = spots[k];
          taken.push({ rect: box });
        }
      }
      if (chosen) {
        item.x = chosen.x;
        item.y = chosen.base;
        item.anchor = chosen.anchor;
      } else {
        item.outside = true;
        item.x = right + 6;
        item.y = py + 4;
        item.anchor = "start";
        outside = Math.max(outside, w + 12);
      }
    });
    /* Outside labels never sit on each other. */
    var out = list
      .filter(function (d) {
        return d.outside && d.text;
      })
      .sort(function (a, b) {
        return a.y - b.y;
      });
    for (var i = 1; i < out.length; i++) {
      if (out[i].y - out[i - 1].y < 13) out[i].y = out[i - 1].y + 13;
    }
    return { list: list, outside: outside };
  }

  function drawHorizontalRefs(g, placed, left, right) {
    placed.list.forEach(function (item) {
      g.appendChild(svg("line", { class: "tv-ref", x1: left, x2: right, y1: item.py, y2: item.py }));
    });
    placed.list.forEach(function (item) {
      if (!item.text) return;
      g.appendChild(
        svg("text", { class: "tv-ref-label", x: item.x, y: item.y, "text-anchor": item.anchor }, item.text)
      );
    });
  }

  function roleLegend(body, spec, roles, shape) {
    var names = spec.roleLabels || {};
    chartLegend(
      body,
      spec,
      roles.map(function (role) {
        return { label: names[role] || ROLE_LABELS[role] || role, color: role, shape: shape };
      }),
      shape
    );
  }

  /*
   * Category rows with wrapped labels, shared by hbar, dot and range. cap is the
   * spec's labelWidth: pixels, or a share of the chart's width when at most 1,
   * so a chart that redraws on resize keeps its proportions.
   */
  function rowLabels(rows, width, cap) {
    var maxWidth =
      isNum(cap) && cap > 0 && cap <= 1
        ? Math.min(Math.max(80, width * cap), 240)
        : isNum(cap)
          ? cap
          : Math.min(Math.max(80, width * 0.34), 240);
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

  /* A whisker from lo to hi with end caps, centred on cy. */
  function whisker(x0, x1, cy, cap, stroke) {
    return [
      svg("line", { class: "tv-whisker tv-mark", x1: x0, x2: x1, y1: cy, y2: cy, style: { stroke: stroke } }),
      svg("line", { class: "tv-whisker tv-mark", x1: x0, x2: x0, y1: cy - cap, y2: cy + cap, style: { stroke: stroke } }),
      svg("line", { class: "tv-whisker tv-mark", x1: x1, x2: x1, y1: cy - cap, y2: cy + cap, style: { stroke: stroke } }),
    ];
  }

  /* hbar ------------------------------------------------------------------ */

  function hbar(body, spec) {
    spec = spec || {};
    var rows = (spec.rows || []).filter(Boolean);
    var valueName = spec.valueLabel || "Value";
    var hasInterval = rows.some(function (r) {
      return isNum(r.lo) && isNum(r.hi);
    });
    var columns = [
      { key: "label", label: spec.labelHeader || "Label" },
      { key: "value", label: valueName, align: "right", format: spec.format },
    ];
    if (hasInterval) {
      columns.push({ key: "lo", label: "Low", align: "right", format: spec.format });
      columns.push({ key: "hi", label: "High", align: "right", format: spec.format });
    }
    tableFor(body, { columns: columns, rows: rows }, spec);
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

    frame(body, "hbar", function (wrap, W) {
      var values = [];
      rows.forEach(function (r) {
        values.push(r.value, r.lo, r.hi);
      });
      values = finite(values);
      var refs = referenceValues(spec);
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : Math.min.apply(null, [0].concat(values, refs));
      var hi = isNum(dom[1]) ? dom[1] : Math.max.apply(null, [0].concat(values, refs));
      var labels = rowLabels(rows, W, spec.labelWidth);
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
      xAxisTicks(grid, x, m.top, m.top + plotH, spec.format, labelTickCount(x, W - m.left - m.right), W - 1);
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
        var interval = isNum(r.lo) && isNum(r.hi);
        if (isNum(r.value)) {
          var d = barPath("h", x(clamp(0, x.domain[0], x.domain[1])), x(r.value), cy - thick / 2, thick);
          if (d) parts.push(svg("path", { class: "tv-mark", d: d, style: { fill: fill } }));
        }
        if (interval) {
          parts = parts.concat(whisker(x(r.lo), x(r.hi), cy, Math.max(3, thick / 2 - 4), "var(--ink-1)"));
        }
        var g = markGroup(
          marks,
          { x: 0, y: m.top + y.step() * i, width: W, height: y.step() },
          parts,
          r.label + ": " + format(spec.format, r.value) + (interval ? ", " + format(spec.format, r.lo) + " to " + format(spec.format, r.hi) : "")
        );
        if (isNum(r.value) && (mode === "all" || (mode === "extreme" && i === extremeIndex))) {
          var right = r.value >= 0;
          var tip = right ? Math.max(x(r.value), interval ? x(r.hi) : -Infinity) : Math.min(x(r.value), interval ? x(r.lo) : Infinity);
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
          if (interval) {
            out.push({ label: spec.intervalLabel || "Interval", value: format(spec.format, r.lo) + " to " + format(spec.format, r.hi) });
          }
          (spec.reference || []).forEach(function (ref) {
            if (ref && isNum(ref.value)) out.push(refTipRow(ref, spec.format));
          });
          if (r.note) out.push({ label: r.note, value: "" });
          return { title: r.label, rows: out.concat(tipRows(r.tip)) };
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
    tableFor(
      body,
      {
        columns: [
          { key: "label", label: spec.labelHeader || "Label" },
          { key: "value", label: valueName, align: "right", format: spec.format },
        ],
        rows: rows,
      },
      spec
    );
    if (!rows.length) return empty(body);
    if (spec.diverging) {
      if (Array.isArray(spec.legend)) roleLegend(body, spec, [], "rect");
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

    frame(body, "column", function (wrap, W) {
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
      var top = 20;
      var bottom = 30;
      var y = linear([lo, hi], [H - bottom, top], { nice: !spec.domain });
      var left = yTickWidth(y, spec.format, 5) + 14;
      var zeroV = clamp(0, y.domain[0], y.domain[1]);
      var base = y(zeroV);

      var maxI = -1;
      var minI = -1;
      rows.forEach(function (r, i) {
        if (!isNum(r.value)) return;
        if (maxI < 0 || r.value > rows[maxI].value) maxI = i;
        if (minI < 0 || r.value < rows[minI].value) minI = i;
      });
      var mode = spec.labels || "extreme";
      /* A row may name itself, as a dot chart's rows do, so a title that quotes two bars can label those two. */
      var flagged = rows.some(function (r) {
        return typeof r.labelled === "boolean";
      });
      function labelled(i) {
        if (mode !== "all" && flagged) return rows[i].labelled === true;
        return mode === "all" || (mode === "extreme" && (i === maxI || (i === minI && rows[minI].value < 0)));
      }

      function geometry(extraRight) {
        var right = 8 + extraRight;
        var x = band(
          rows.map(function (_, i) {
            return i;
          }),
          [left, W - right],
          { padding: 0.25 }
        );
        var thick = Math.min(BAR_MAX, x.bandwidth(), Math.max(1, x.step() - GAP));
        var items = rows.map(function (r, i) {
          var cx = x.step() * i + left + x.step() / 2;
          var item = { cx: cx, bar: null, value: null };
          if (isNum(r.value)) {
            var yv = y(r.value);
            item.bar = { x: cx - thick / 2, y: Math.min(yv, base), w: thick, h: Math.abs(base - yv) };
            if (labelled(i)) {
              var up = r.value >= 0;
              var text = format(spec.format, r.value);
              var ty = yv + (up ? -6 : 14);
              item.value = { text: text, y: ty, box: labelBox(cx, ty, measure(text, 11, 500), "middle") };
            }
          }
          return item;
        });
        return { right: right, x: x, thick: thick, items: items };
      }

      var geo = null;
      var placed = null;
      var extra = 0;
      var forced = {};
      for (var pass = 0; pass < 3; pass++) {
        geo = geometry(extra);
        var obstacles = [];
        geo.items.forEach(function (it) {
          if (it.bar) obstacles.push({ rect: it.bar });
          if (it.value) obstacles.push({ rect: it.value.box });
        });
        placed = placeHorizontalRefs(spec, y, left, W - geo.right, top, H - bottom, obstacles, forced);
        if (placed.outside <= extra) break;
        extra = placed.outside;
        placed.list.forEach(function (p, i) {
          if (p.outside) forced[i] = true;
        });
      }

      var x = geo.x;
      var thick = geo.thick;
      var plotRight = W - geo.right;
      var root = chartSvg(W, H, spec, "Column");
      var grid = svg("g");
      yAxisTicks(grid, y, left, plotRight, spec.format, 5);
      root.appendChild(grid);

      var shown = thinLabels(
        rows.map(function (r, i) {
          return { center: geo.items[i].cx, width: measure(r.label, 11) };
        }),
        8
      );

      var marks = svg("g");
      var valueLayer = svg("g", { class: "tv-values" });
      rows.forEach(function (r, i) {
        var cx = geo.items[i].cx;
        var fill = spec.diverging ? color(r.value >= 0 ? "pos" : "neg") : color(r.role || "model");
        var parts = [];
        if (isNum(r.value)) {
          var d = barPath("v", base, y(r.value), cx - thick / 2, thick);
          if (d) parts.push(svg("path", { class: "tv-mark", d: d, style: { fill: fill } }));
        }
        var g = markGroup(
          marks,
          { x: left + x.step() * i, y: top, width: x.step(), height: H - top - bottom },
          parts,
          r.label + ": " + format(spec.format, r.value)
        );
        var v = geo.items[i].value;
        if (v) {
          valueLayer.appendChild(svg("text", { class: "tv-value", x: cx, y: v.y, "text-anchor": "middle" }, v.text));
        }
        if (shown.indexOf(i) >= 0) {
          var tickW = measure(r.label, 11);
        var tickAnchor = cx - tickW / 2 < 0 ? "start" : cx + tickW / 2 > W ? "end" : "middle";
        marks.appendChild(svg("text", { class: "tv-tick", x: cx, y: H - bottom + 16, "text-anchor": tickAnchor }, r.label));
        }
        tooltip.attach(g, function () {
          return {
            title: r.label,
            rows: [{ label: valueName, value: format(spec.format, r.value), color: fill, shape: "rect" }].concat(tipRows(r.tip)),
          };
        });
      });
      root.appendChild(svg("line", { class: "tv-axisline", x1: left, x2: plotRight, y1: crisp(base), y2: crisp(base) }));
      root.appendChild(marks);
      var refLayer = svg("g");
      drawHorizontalRefs(refLayer, placed, left, plotRight);
      root.appendChild(refLayer);
      root.appendChild(valueLayer);
      wrap.appendChild(root);
    });
  }

  /* dot: a dot plot, or a dumbbell when there are two series --------------- */

  /* Points closer than a marker along x, in runs: items carry px. */
  function clusters(points) {
    var sorted = points.slice().sort(function (a, b) {
      return a.px - b.px;
    });
    var out = [];
    var current = [];
    sorted.forEach(function (p) {
      if (current.length && p.px - current[current.length - 1].px >= 9) {
        out.push(current);
        current = [];
      }
      current.push(p);
    });
    if (current.length) out.push(current);
    return out;
  }

  /* The most points one row has to draw apart, from their x positions. */
  function clusterSize(pxs) {
    return maxOf(
      clusters(
        pxs.map(function (px) {
          return { px: px };
        })
      ).map(function (c) {
        return c.length;
      }),
      1
    );
  }

  /* Points in one row closer than a marker are moved apart across the row. */
  function dodge(points, rowH) {
    clusters(points).forEach(function (cluster) {
      if (cluster.length < 2) return;
      /* Centres at least a marker apart, and a gap left to the next row's markers. */
      var step = cluster.length === 2 ? 10 : clamp((rowH - 14) / (cluster.length - 1), 8, 10);
      cluster
        .sort(function (a, b) {
          return a.k - b.k;
        })
        .forEach(function (p, j) {
          p.dy = (j - (cluster.length - 1) / 2) * step;
        });
    });
  }

  function dot(body, spec) {
    spec = spec || {};
    var rows = (spec.rows || []).filter(Boolean);
    var series = (spec.series || [{ key: "value", name: spec.valueLabel || "Value", role: "model" }]).slice(0, 3);
    var two = series.length === 2;
    var gapFormat = spec.gapFormat || "signed:" + (spec.gapDp || 2);
    var valueFormat = spec.tableFormat || spec.format;

    function valueOf(r, s) {
      return r.values ? r.values[s.key] : null;
    }
    function intervalOf(r, s, k) {
      var iv = r.intervals && r.intervals[s.key];
      if (iv && isNum(iv.lo) && isNum(iv.hi)) return iv;
      if (k === 0 && isNum(r.lo) && isNum(r.hi)) return { lo: r.lo, hi: r.hi };
      return null;
    }
    /* A row's gap is the collector's when it gives one, otherwise first less second. */
    function gapOf(r) {
      if (!two) return null;
      if (isNum(r.gap)) return r.gap;
      var a = valueOf(r, series[0]);
      var b = valueOf(r, series[1]);
      return isNum(a) && isNum(b) ? a - b : null;
    }
    var hasGroups = rows.some(function (r) {
      return !isNil(r.group) && r.group !== "";
    });
    var hasAside = rows.some(function (r) {
      return !isNil(r.aside) && r.aside !== "";
    });

    var columns = [{ key: "label", label: spec.labelHeader || "Label" }];
    if (hasGroups) columns.push({ key: "group", label: spec.groupHeader || "Group" });
    series.forEach(function (s, k) {
      columns.push({ key: "s:" + s.key, label: s.name, align: "right", format: valueFormat });
      if (
        rows.some(function (r) {
          return intervalOf(r, s, k);
        })
      ) {
        columns.push({ key: "lo:" + s.key, label: s.name + ", low", align: "right", format: valueFormat });
        columns.push({ key: "hi:" + s.key, label: s.name + ", high", align: "right", format: valueFormat });
      }
    });
    if (two) columns.push({ key: "gap", label: spec.gapLabel || "Difference", align: "right", format: gapFormat });
    if (hasAside) columns.push({ key: "aside", label: spec.asideHeader || "Note" });
    tableFor(
      body,
      {
        columns: columns,
        rows: rows.map(function (r) {
          var out = { label: r.label, group: r.group, gap: gapOf(r), aside: r.aside };
          series.forEach(function (s, k) {
            out["s:" + s.key] = valueOf(r, s);
            var iv = intervalOf(r, s, k);
            out["lo:" + s.key] = iv ? iv.lo : null;
            out["hi:" + s.key] = iv ? iv.hi : null;
          });
          return out;
        }),
      },
      spec
    );
    if (!rows.length) return empty(body);
    var legendItems = series.length > 1
      ? series.map(function (s) {
          return { label: s.name, color: s.role, shape: "dot" };
        })
      : [];
    if (
      spec.hollowLegend &&
      rows.some(function (r) {
        return r.hollow;
      })
    ) {
      if (!legendItems.length) legendItems = [{ label: series[0].name, color: series[0].role, shape: "dot" }];
      legendItems.push({ label: spec.hollowLegend, color: "--ink-3", shape: "ring" });
    }
    chartLegend(body, spec, legendItems, "dot");

    /* Which rows carry a label, and what the label says. */
    function labelPlan() {
      var cfg = spec.labels;
      if (cfg === "none" || cfg === false) return null;
      var anyText = rows.some(function (r) {
        return typeof r.text === "string" && r.text;
      });
      if (isNil(cfg) && anyText) return null;
      if (typeof cfg === "string" || isNil(cfg)) cfg = { rows: cfg === "all" ? "all" : "auto" };
      var sIndex = 0;
      series.forEach(function (s, k) {
        if (s.key === cfg.series) sIndex = k;
      });
      var text = cfg.text === "gap" && two ? "gap" : "value";
      function score(r) {
        return text === "gap" || (two && cfg.series === undefined) ? gapOf(r) : valueOf(r, series[sIndex]);
      }
      var picked = {};
      var flagged = rows.some(function (r) {
        return typeof r.labelled === "boolean";
      });
      if (Array.isArray(cfg.rows)) {
        cfg.rows.forEach(function (i) {
          if (isNum(i)) picked[i] = true;
        });
      } else if (cfg.rows === "all") {
        rows.forEach(function (_, i) {
          picked[i] = true;
        });
      } else if (cfg.rows === "flagged" || (cfg.rows !== "all" && flagged)) {
        rows.forEach(function (r, i) {
          if (r.labelled === true) picked[i] = true;
        });
      } else {
        /* The row that stands out; none when nothing does, so an all-zero chart carries no stray label. */
        var best = -1;
        var bestScore = 0;
        rows.forEach(function (r, i) {
          var s = score(r);
          if (isNum(s) && Math.abs(s) > bestScore) {
            best = i;
            bestScore = Math.abs(s);
          }
        });
        if (best >= 0) picked[best] = true;
      }
      return { text: text, sIndex: sIndex, rows: picked };
    }

    frame(body, "dot", function (wrap, W) {
      var all = [];
      rows.forEach(function (r) {
        series.forEach(function (s, k) {
          all.push(valueOf(r, s));
          var iv = intervalOf(r, s, k);
          if (iv) all.push(iv.lo, iv.hi);
        });
      });
      var ext = extent(all.concat(referenceValues(spec))) || [0, 1];
      if (spec.zero) ext = [Math.min(0, ext[0]), Math.max(0, ext[1])];
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : ext[0];
      var hi = isNum(dom[1]) ? dom[1] : ext[1];
      var labels = rowLabels(rows, W, spec.labelWidth);
      var plan = labelPlan();

      var texts = rows.map(function (r, i) {
        if (typeof r.text === "string" && r.text) return { text: r.text, kind: "text" };
        if (!plan || !plan.rows[i]) return null;
        if (plan.text === "gap") {
          var g = gapOf(r);
          return isNum(g) ? { text: format(gapFormat, g), kind: "gap" } : null;
        }
        var v = valueOf(r, series[plan.sIndex]);
        return isNum(v) ? { text: format(spec.format, v), kind: "value", k: plan.sIndex } : null;
      });
      var labelW = maxOf(
        texts.map(function (t) {
          return t ? measure(t.text, 11, 500) : 0;
        })
      );
      labelW = labelW ? labelW + 12 : 0;
      var asideTexts = rows.map(function (r) {
        return isNil(r.aside) ? "" : String(r.aside);
      });
      var asideW = hasAside
        ? maxOf(
            asideTexts.map(function (t, i) {
              return measure(t, 11, rows[i].asideStrong ? 600 : 400);
            }).concat(spec.asideHeader ? [measure(spec.asideHeader, 11)] : [])
          ) + 20
        : 0;
      var refCount = (spec.reference || []).length;
      var refBand = refCount * 14 + (refCount ? 8 : 0);
      var m = { top: 8 + refBand, right: 12 + labelW + asideW, bottom: 26, left: labels.width + 16 };
      if (hasAside && spec.asideHeader) m.top = Math.max(m.top, 24);

      var lines = [];
      var lastGroup = null;
      rows.forEach(function (r, i) {
        if (hasGroups && !isNil(r.group) && r.group !== lastGroup) {
          lines.push({ group: r.group });
          lastGroup = r.group;
        }
        lines.push({ i: i });
      });
      var groupCount = lines.length - rows.length;
      var n = rows.length;
      var baseH = spec.height
        ? Math.max(HIT_MIN, (spec.height - m.top - m.bottom - groupCount * GROUP_ROW) / n)
        : isNum(spec.rowHeight)
          ? Math.max(HIT_MIN, spec.rowHeight)
          : 30;
      var x = linear([lo, hi], [m.left + 6, W - m.right], { nice: !spec.domain });
      /*
       * A row whose label wraps gets the height its lines need, and a row whose
       * points are drawn apart gets the height they take, so three coincident
       * points never run into the next row's.
       */
      var heights = rows.map(function (r, i) {
        var h = Math.max(baseH, labels.wrapped[i].lines.length * 14 + 12);
        if (spec.dodge !== false) {
          var pxs = [];
          series.forEach(function (s) {
            var v = valueOf(r, s);
            if (isNum(v)) pxs.push(x(v));
          });
          h = Math.max(h, (clusterSize(pxs) - 1) * 10 + 22);
        }
        return h;
      });
      var plotH =
        heights.reduce(function (s, h) {
          return s + h;
        }, 0) +
        groupCount * GROUP_ROW;
      var H = Math.round(m.top + plotH + m.bottom);
      var tickCount = labelTickCount(x, W - m.left - m.right);
      var root = chartSvg(W, H, spec, "Dot");
      var grid = svg("g");
      xAxisTicks(grid, x, m.top, m.top + plotH, spec.format, tickCount, hasAside ? W - asideW : W - 1);
      if (spec.zeroLine && x.domain[0] <= 0 && x.domain[1] >= 0) {
        var zx = crisp(x(0));
        grid.appendChild(svg("line", { class: "tv-axisline", x1: zx, x2: zx, y1: m.top, y2: m.top + plotH }));
      }
      if (hasAside && spec.asideHeader) {
        grid.appendChild(
          svg(
            "text",
            spec.asideAlign === "end"
              ? { class: "tv-tick", x: W, y: m.top - 8, "text-anchor": "end" }
              : { class: "tv-tick", x: W - asideW + 14, y: m.top - 8, "text-anchor": "start" },
            spec.asideHeader
          )
        );
      }
      root.appendChild(grid);
      var marks = svg("g");
      var valueLayer = svg("g", { class: "tv-values" });
      var top = m.top;
      lines.forEach(function (line) {
        if (line.group) {
          marks.appendChild(svg("text", { class: "tv-group-label", x: 0, y: top + GROUP_ROW / 2 + 2, dy: "0.35em" }, String(line.group)));
          top += GROUP_ROW;
          return;
        }
        var i = line.i;
        var r = rows[i];
        var rowH = heights[i];
        var cy = top + rowH / 2;
        var rowTop = top;
        top += rowH;
        textBlock(marks, labels.wrapped[i].lines, labels.width + 2, cy, {
          class: "tv-label",
          "text-anchor": "end",
          style: r.hollow ? { fill: "var(--ink-3)" } : null,
        });
        var pts = [];
        series.forEach(function (s, k) {
          var v = valueOf(r, s);
          if (!isNum(v)) return;
          var iv = intervalOf(r, s, k);
          pts.push({ k: k, s: s, v: v, px: x(v), dy: 0, iv: iv, paint: color(r.role || s.role) });
        });
        if (spec.dodge !== false) dodge(pts, rowH);
        var parts = [];
        if (pts.length > 1) {
          var pxs = pts.map(function (p) {
            return p.px;
          });
          parts.push(
            svg("line", {
              class: "tv-hairline",
              x1: Math.min.apply(null, pxs),
              x2: Math.max.apply(null, pxs),
              y1: crisp(cy),
              y2: crisp(cy),
            })
          );
        }
        pts.forEach(function (p) {
          if (p.iv) parts = parts.concat(whisker(x(p.iv.lo), x(p.iv.hi), cy + p.dy, 5, p.paint));
        });
        /* The first series is drawn last and a half pixel larger, so it wins a tie. */
        pts
          .slice()
          .sort(function (a, b) {
            return b.k - a.k;
          })
          .forEach(function (p) {
            var radius = 4 + (p.k === 0 ? 0.5 : 0);
            parts.push(
              svg("circle", {
                class: r.hollow ? "tv-mark" : "tv-mark tv-dot",
                cx: p.px,
                cy: cy + p.dy,
                r: r.hollow ? radius - 0.75 : radius,
                "data-series": p.s.key,
                "data-value": format(spec.format, p.v),
                style: r.hollow ? { fill: "var(--surface)", stroke: p.paint, strokeWidth: "1.5" } : { fill: p.paint },
              })
            );
          });
        var t = texts[i];
        var aria =
          r.label +
          (hasGroups && r.group ? ", " + r.group : "") +
          ": " +
          series
            .map(function (s) {
              return s.name + " " + format(spec.format, valueOf(r, s));
            })
            .join(", ") +
          (t && t.kind !== "value" ? ", " + t.text : "");
        var g = markGroup(marks, { x: 0, y: rowTop, width: W, height: rowH }, parts, aria);

        if (t) {
          var w = measure(t.text, 11, 500);
          var extentOf = function (p) {
            return [
              Math.min(p.px - 5, p.iv ? x(p.iv.lo) : Infinity),
              Math.max(p.px + 5, p.iv ? x(p.iv.hi) : -Infinity),
            ];
          };
          var rowRight = maxOf(
            pts.map(function (p) {
              return extentOf(p)[1];
            }),
            -Infinity
          );
          var attrs = null;
          if (t.kind === "value") {
            var own = pts.filter(function (p) {
              return p.k === t.k;
            })[0];
            if (own) {
              var ex = extentOf(own);
              var others = pts
                .filter(function (p) {
                  return p !== own;
                })
                .map(extentOf);
              var free = function (a, b) {
                return others.every(function (o) {
                  return b + 2 <= o[0] || a - 2 >= o[1];
                });
              };
              var baseY = cy + own.dy;
              var rightSpot = free(ex[1] + 4, ex[1] + 4 + w) && ex[1] + 4 + w <= W
                ? { x: ex[1] + 4, y: baseY, dy: "0.35em", "text-anchor": "start" }
                : null;
              var leftSpot = free(ex[0] - 4 - w, ex[0] - 4) && ex[0] - 4 - w >= m.left
                ? { x: ex[0] - 4, y: baseY, dy: "0.35em", "text-anchor": "end" }
                : null;
              /* A dot left of every other point in its row takes its label on the outside, off the connecting rule. */
              var outsideLeft =
                others.length > 0 &&
                others.every(function (o) {
                  return o[0] >= ex[1];
                });
              attrs = (outsideLeft ? leftSpot || rightSpot : rightSpot || leftSpot) || {
                x: own.px,
                y: baseY - 9,
                "text-anchor": "middle",
              };
              attrs["data-series"] = own.s.key;
            }
          } else if (isFinite(rowRight)) {
            attrs = { x: rowRight + 4, y: cy, dy: "0.35em", "text-anchor": "start" };
          }
          if (attrs) valueLayer.appendChild(svg("text", Object.assign({ class: "tv-value" }, attrs), t.text));
        }
        if (hasAside && asideTexts[i]) {
          marks.appendChild(
            svg(
              "text",
              Object.assign(
                {
                  class: r.asideStrong ? "tv-value" : "tv-tick",
                  y: cy,
                  dy: "0.35em",
                  style: Object.assign(r.asideStrong ? { fontWeight: "600" } : {}, r.hollow ? { fill: "var(--ink-3)" } : {}),
                },
                spec.asideAlign === "end" ? { x: W, "text-anchor": "end" } : { x: W - asideW + 14, "text-anchor": "start" }
              ),
              asideTexts[i]
            )
          );
        }
        tooltip.attach(g, function () {
          var out = [];
          series.forEach(function (s, k) {
            var v = valueOf(r, s);
            out.push({ label: s.name, value: format(valueFormat, v), color: r.role || s.role, shape: "dot" });
            var iv = intervalOf(r, s, k);
            if (iv) {
              out.push({
                label: spec.intervalLabel || "Interval",
                value: format(valueFormat, iv.lo) + " to " + format(valueFormat, iv.hi),
              });
            }
          });
          var gap = gapOf(r);
          if (two && isNum(gap)) out.push({ label: spec.gapLabel || "Difference", value: format(gapFormat, gap) });
          if (hasAside && asideTexts[i]) out.push({ label: spec.asideHeader || "Note", value: asideTexts[i] });
          return { title: hasGroups && r.group ? r.label + ", " + r.group : r.label, rows: out.concat(tipRows(r.tip)) };
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
    var dates = xSpec.type === "date";
    var xs = [];
    allSeries.forEach(function (s) {
      (s.values || []).forEach(function (p) {
        if (p && !isNil(p.x) && xs.indexOf(p.x) < 0) xs.push(p.x);
      });
    });
    var numericX = xs.length > 0 && (dates ? xs.every(function (v) { return isFinite(parseDay(v)); }) : xs.every(isNum));
    function xNum(v) {
      return dates ? parseDay(v) : v;
    }
    if (numericX) {
      xs.sort(function (a, b) {
        return xNum(a) - xNum(b);
      });
    }
    var lookups = allSeries.map(function (s) {
      var map = {};
      (s.values || []).forEach(function (p) {
        if (p && !isNil(p.x)) map[String(p.x)] = p;
      });
      return map;
    });
    function valueAt(k, xv) {
      return lookups[k][String(xv)] || null;
    }
    function xText(xv) {
      return xSpec.format ? format(xSpec.format, xv) : String(xv);
    }

    var columns = [{ key: "x", label: xSpec.label || "x", format: xSpec.format, mono: dates }];
    allSeries.forEach(function (s, k) {
      columns.push({ key: "y" + k, label: s.name, align: "right", format: spec.format });
      if (s.values && s.values.some(function (p) { return p && isNum(p.lo); })) {
        columns.push({ key: "lo" + k, label: s.name + " low", align: "right", format: spec.format });
        columns.push({ key: "hi" + k, label: s.name + " high", align: "right", format: spec.format });
      }
    });
    tableFor(
      body,
      {
        columns: columns,
        rows: xs.map(function (xv) {
          var row = { x: xSpec.format ? xv : String(xv) };
          allSeries.forEach(function (s, k) {
            var p = valueAt(k, xv);
            row["y" + k] = p ? p.y : null;
            row["lo" + k] = p ? p.lo : null;
            row["hi" + k] = p ? p.hi : null;
          });
          return row;
        }),
      },
      spec
    );
    if (!xs.length || !series.length) return empty(body);
    var handle = handleFor(body);
    if (allSeries.length > 4 && handle) {
      handle.addNote({ what: "Series beyond four", why: "The chart draws the first four series; the table lists all " + allSeries.length + "." });
    }
    var logY = spec.yScale === "log";
    if (logY) {
      var positive = true;
      series.forEach(function (s) {
        (s.values || []).forEach(function (p) {
          if (p && isNum(p.y) && p.y <= 0) positive = false;
        });
      });
      if (!positive) {
        logY = false;
        if (handle) handle.addNote({ what: "Log scale", why: "A log scale needs every value above zero, so the chart is drawn on a linear scale." });
      }
    }
    var shades = (spec.shade || []).filter(function (s) {
      return s && !isNil(s.from) && !isNil(s.to);
    });
    var legendItems = [];
    /* One series needs no key: the title names it. A shaded window still does. */
    if (series.length > 1) {
      legendItems = series.map(function (s) {
        return { label: s.name, color: s.role, shape: "line" };
      });
    }
    if (spec.shadeLegend && shades.length) legendItems.push({ label: spec.shadeLegend, color: "--wash-strong", shape: "rect" });
    /*
     * chartLegend draws nothing for a single item, which is right for one series
     * the title already names and wrong for a shaded window, which nothing else
     * on the chart explains. So a lone shade key is drawn here.
     */
    if (legendItems.length === 1 && spec.shadeLegend && !legendOff(spec)) legendFor(body, legendItems);
    else chartLegend(body, spec, legendItems, "line");

    frame(body, "line", function (wrap, W) {
      var ys = [];
      series.forEach(function (s) {
        (s.values || []).forEach(function (p) {
          if (!p) return;
          ys.push(p.y, p.lo, p.hi);
        });
      });
      var ext = extent(ys.concat(logY ? [] : referenceValues(spec))) || [0, 1];
      if (spec.zero && !logY) ext = [Math.min(0, ext[0]), Math.max(0, ext[1])];
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : ext[0];
      var hi = isNum(dom[1]) ? dom[1] : ext[1];
      var H = spec.height || 280;
      var bottom = H - 30;
      function makeY(top) {
        return logY ? logScale(lo, hi, [bottom, top]) : linear([lo, hi], [bottom, top], { nice: !spec.domain });
      }
      var probe = makeY(14);
      var tickCount = logY ? Math.max(3, Math.floor((bottom - 14) / 22)) : 5;
      var left = yTickWidth(probe, spec.format, tickCount) + 14;
      var plotL = left + 4;
      var yTitleH = spec.yTitle ? 18 : 0;

      /* Where a date or number sits along x. */
      function xAtFor(plotR) {
        if (numericX) {
          var scale = linear([xNum(xs[0]), xNum(xs[xs.length - 1])], [plotL, plotR]);
          return function (xv) {
            var v = xNum(xv);
            return isFinite(v) ? scale(v) : NaN;
          };
        }
        var stepX = xs.length > 1 ? (plotR - plotL) / (xs.length - 1) : 0;
        return function (xv) {
          var i = xs.indexOf(xv);
          if (i < 0) return NaN;
          return xs.length > 1 ? plotL + i * stepX : (plotL + plotR) / 2;
        };
      }

      function layout(extraRight) {
        var provisional = makeY(14 + yTitleH);
        var ends = series
          .map(function (s, k) {
            var vals = (s.values || []).filter(function (p) {
              return p && isNum(p.y);
            });
            return vals.length ? { k: k, name: s.name, last: vals[vals.length - 1] } : null;
          })
          .filter(Boolean);
        var endLabels = spec.endLabels !== false && ends.length > 0;
        function collide(scale) {
          var sorted = ends
            .map(function (d) {
              return scale(d.last.y);
            })
            .sort(function (a, b) {
              return a - b;
            });
          for (var e = 1; e < sorted.length; e++) if (sorted[e] - sorted[e - 1] < 14) return true;
          return false;
        }
        if (endLabels && collide(provisional)) endLabels = false;
        var endWidth = endLabels
          ? maxOf(
              ends.map(function (d) {
                return measure(d.name, 12, 500);
              })
            ) + 14
          : 0;
        var right = Math.max(16, endWidth) + extraRight;
        var plotR = W - right - (endLabels ? 0 : 4);
        var xAt = xAtFor(plotR);

        /* Shaded ranges, their names stacked above the plot. */
        var shadeRows = 0;
        var lastRight = [];
        var shadeBoxes = shades.map(function (sh) {
          var x0 = clamp(xAt(sh.from), plotL, plotR);
          var x1 = clamp(xAt(sh.to), plotL, plotR);
          if (!isFinite(x0) || !isFinite(x1)) return null;
          if (x1 - x0 < 1) x1 = x0 + 1;
          var box = { x0: x0, x1: x1, text: sh.label ? String(sh.label) : "", row: 0 };
          if (box.text) {
            var tw = measure(box.text, 11, 500);
            box.cx = clamp((x0 + x1) / 2, plotL + tw / 2, plotR - tw / 2);
            var row = 0;
            while (!isNil(lastRight[row]) && box.cx - tw / 2 < lastRight[row] + 6) row++;
            lastRight[row] = box.cx + tw / 2;
            box.row = row;
            shadeRows = Math.max(shadeRows, row + 1);
          }
          return box;
        });
        var top = 14 + yTitleH + (shadeRows ? shadeRows * 13 + 8 : 0);
        var y = makeY(top);
        if (endLabels && collide(y)) endLabels = false;

        var obstacles = [];
        var segments = [];
        series.forEach(function (s, k) {
          var prev = null;
          xs.forEach(function (xv) {
            var p = valueAt(k, xv);
            if (p && isNum(p.y)) {
              var pt = [xAt(xv), y(p.y)];
              if (prev) segments.push([prev[0], prev[1], pt[0], pt[1]]);
              prev = pt;
            } else {
              prev = null;
            }
          });
          var vals = (s.values || []).filter(function (p) {
            return p && isNum(p.y);
          });
          if (vals.length) {
            var last = vals[vals.length - 1];
            obstacles.push({ rect: { x: xAt(last.x) - 5, y: y(last.y) - 5, w: 10, h: 10 } });
          }
        });
        segments.forEach(function (sg) {
          obstacles.push({ seg: sg });
        });

        /* Labelled points: beside the dot, below the line where there is room. */
        var pointItems = (spec.points || [])
          .filter(function (pt) {
            return pt && !isNil(pt.x);
          })
          .map(function (pt) {
            var px = xAt(pt.x);
            var first = valueAt(0, pt.x);
            var yv = isNum(pt.y) ? pt.y : first && isNum(first.y) ? first.y : null;
            if (!isFinite(px) || !isNum(yv) || (logY && yv <= 0)) return null;
            var py = y(yv);
            var item = { pt: pt, px: px, py: py };
            obstacles.push({ rect: { x: px - 5, y: py - 5, w: 10, h: 10 } });
            if (pt.label) {
              var tw = measure(pt.label, 11, 500);
              var spots = [
                { x: px + 8, base: py + 16, anchor: "start" },
                { x: px + 8, base: py - 10, anchor: "start" },
                { x: px - 8, base: py + 16, anchor: "end" },
                { x: px - 8, base: py - 10, anchor: "end" },
              ];
              var lines = segments.map(function (sg) {
                return { seg: sg };
              });
              var chosen = null;
              /* Clear of the lines and inside the plot; failing that, inside the plot. */
              [true, false].forEach(function (clearOfLines) {
                spots.forEach(function (sp) {
                  if (chosen) return;
                  var box = labelBox(sp.x, sp.base, tw, sp.anchor);
                  if (box.x < plotL || box.x + box.w > plotR || box.y < top || box.y + box.h > bottom) return;
                  if (clearOfLines && hits(box, lines)) return;
                  chosen = sp;
                  item.box = box;
                });
              });
              chosen = chosen || spots[0];
              item.label = { x: chosen.x, y: chosen.base, anchor: chosen.anchor };
              item.box = item.box || labelBox(chosen.x, chosen.base, tw, chosen.anchor);
            }
            return item;
          })
          .filter(Boolean);
        pointItems.forEach(function (it) {
          if (it.box) obstacles.push({ rect: it.box });
        });

        return {
          right: right,
          plotR: plotR,
          xAt: xAt,
          y: y,
          top: top,
          endLabels: endLabels,
          endWidth: endWidth,
          ends: ends,
          shadeBoxes: shadeBoxes,
          pointItems: pointItems,
          obstacles: obstacles,
        };
      }

      var geo = null;
      var placed = null;
      var extra = 0;
      var forced = {};
      for (var pass = 0; pass < 3; pass++) {
        geo = layout(extra);
        var refRight = W - geo.right + (geo.endLabels ? 0 : 0);
        placed = placeHorizontalRefs(spec, geo.y, left, refRight, geo.top, bottom, geo.obstacles, forced);
        if (placed.outside <= extra) break;
        extra = placed.outside;
        placed.list.forEach(function (p, i) {
          if (p.outside) forced[i] = true;
        });
      }
      /* An outside label sits past the end labels, not on them. */
      placed.list.forEach(function (p) {
        if (p.outside) p.x = W - extra + 6;
      });

      var y = geo.y;
      var xAt = geo.xAt;
      var plotR = geo.plotR;
      var top = geo.top;
      var gridRight = W - geo.right;
      var root = chartSvg(W, H, spec, "Line");
      var grid = svg("g");
      yAxisTicks(grid, y, left, gridRight, spec.format, tickCount);
      if (spec.yTitle) {
        grid.appendChild(svg("text", { class: "tv-tick", x: 0, y: 11, "text-anchor": "start" }, spec.yTitle));
      }
      root.appendChild(grid);
      if (!logY && y.domain[0] <= 0 && y.domain[1] >= 0) {
        root.appendChild(svg("line", { class: "tv-axisline", x1: left, x2: gridRight, y1: crisp(y(0)), y2: crisp(y(0)) }));
      } else {
        root.appendChild(svg("line", { class: "tv-axisline", x1: left, x2: gridRight, y1: crisp(bottom), y2: crisp(bottom) }));
      }

      /* x tick labels, thinned to fit, the first and last kept */
      var tickItems = [];
      if (dates) {
        var t0 = parseDay(xs[0]);
        var t1 = parseDay(xs[xs.length - 1]);
        for (var yr = new Date(t0).getUTCFullYear(); yr <= new Date(t1).getUTCFullYear() + 1; yr++) {
          var jan = Date.UTC(yr, 0, 1);
          if (jan >= t0 && jan <= t1) {
            var label = String(yr);
            var pos = linear([t0, t1], [plotL, plotR])(jan);
            tickItems.push({ center: pos, width: measure(label, 11), text: label });
          }
        }
        if (tickItems.length < 2) {
          tickItems = [xs[0], xs[xs.length - 1]].map(function (xv) {
            return { center: xAt(xv), width: measure(String(xv), 11), text: String(xv) };
          });
        }
      } else {
        var tickXs = xs;
        if (numericX && xs.length > 8) {
          var widest = maxOf(
            xs.map(function (v) {
              return measure(xText(v), 11);
            })
          );
          tickXs = linear([xs[0], xs[xs.length - 1]], [0, 1]).ticks(Math.max(2, Math.floor((plotR - plotL) / (widest + 24))));
        }
        tickItems = tickXs.map(function (xv) {
          var text = xText(xv);
          return { center: xAt(xv), width: measure(text, 11), text: text };
        });
      }
      thinLabels(tickItems, 10).forEach(function (i) {
        var it = tickItems[i];
        var anchor = it.center - it.width / 2 < 0 ? "start" : it.center + it.width / 2 > W ? "end" : "middle";
        /* Set a little lower than other charts' ticks, so the first clears the lowest y label in the corner. */
        grid.appendChild(svg("text", { class: "tv-tick", x: it.center, y: bottom + 19, "text-anchor": anchor }, it.text));
      });

      var shadeLayer = svg("g");
      geo.shadeBoxes.forEach(function (box) {
        if (!box) return;
        shadeLayer.appendChild(svg("rect", { class: "tv-shade", x: box.x0, y: top, width: box.x1 - box.x0, height: bottom - top }));
        if (box.text) {
          shadeLayer.appendChild(
            svg("text", { class: "tv-ref-label", x: box.cx, y: top - 8 - box.row * 13, "text-anchor": "middle" }, box.text)
          );
        }
      });
      root.appendChild(shadeLayer);

      var layer = svg("g");
      series.forEach(function (s, k) {
        var bandPts = xs
          .map(function (xv) {
            var p = valueAt(k, xv);
            return p && isNum(p.lo) && isNum(p.hi) ? { x: xAt(xv), p: p } : null;
          })
          .filter(Boolean);
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
        .map(function (s, k) {
          return { s: s, k: k };
        })
        .reverse()
        .forEach(function (d) {
          var path = "";
          var pen = false;
          xs.forEach(function (xv) {
            var p = valueAt(d.k, xv);
            if (p && isNum(p.y)) {
              path += (pen ? "L" : "M") + xAt(xv).toFixed(1) + "," + y(p.y).toFixed(1);
              pen = true;
            } else {
              pen = false;
            }
          });
          layer.appendChild(svg("path", { class: "tv-line", d: path, style: { stroke: color(d.s.role) } }));
          var vals = (d.s.values || []).filter(function (p) {
            return p && isNum(p.y);
          });
          if (vals.length) {
            var last = vals[vals.length - 1];
            layer.appendChild(svg("circle", { class: "tv-dot", cx: xAt(last.x), cy: y(last.y), r: 4, style: { fill: color(d.s.role) } }));
          }
        });
      root.appendChild(layer);

      if (geo.endLabels) {
        geo.ends.forEach(function (d) {
          root.appendChild(svg("text", { class: "tv-endlabel", x: plotR + 10, y: y(d.last.y), dy: "0.35em" }, d.name));
        });
      }
      var refLayer = svg("g");
      drawHorizontalRefs(refLayer, placed, left, gridRight);
      root.appendChild(refLayer);

      var pointLayer = svg("g", { class: "tv-values" });
      geo.pointItems.forEach(function (it) {
        root.appendChild(
          svg("circle", { class: "tv-dot", cx: it.px, cy: it.py, r: 4, style: { fill: color(it.pt.role || series[0].role) } })
        );
        if (it.label) {
          pointLayer.appendChild(
            svg("text", { class: "tv-value", x: it.label.x, y: it.label.y, "text-anchor": it.label.anchor }, it.pt.label)
          );
        }
      });
      root.appendChild(pointLayer);

      /* Crosshair: the pointer finds the nearest x; arrows move it. */
      var hover = svg("g", { hidden: true });
      var hair = svg("line", { class: "tv-crosshair", x1: 0, x2: 0, y1: top, y2: bottom });
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
      plot.appendChild(
        svg("rect", {
          class: "tv-hit",
          x: left,
          y: top,
          width: Math.max(0, gridRight - left + (geo.endLabels ? geo.endWidth : 0)),
          height: Math.max(HIT_MIN, bottom - top),
        })
      );
      root.appendChild(hover);
      root.appendChild(plot);
      var current = -1;

      function content(i) {
        var xv = xs[i];
        var out = series.map(function (s, k) {
          var p = valueAt(k, xv);
          var text = p ? format(spec.format, p.y) : "n/a";
          if (p && isNum(p.lo) && isNum(p.hi)) {
            text += " (" + format(spec.format, p.lo) + " to " + format(spec.format, p.hi) + ")";
          }
          return { label: s.name, value: text, color: s.role, shape: "line" };
        });
        var at = xNum(xv);
        shades.forEach(function (sh) {
          var from = xNum(sh.from);
          var to = xNum(sh.to);
          var inside = numericX ? at > from && at <= to : xs.indexOf(xv) > xs.indexOf(sh.from) && xs.indexOf(xv) <= xs.indexOf(sh.to);
          if (inside) out = out.concat(tipRows(sh.tip));
        });
        (spec.points || []).forEach(function (pt) {
          if (pt && String(pt.x) === String(xv)) out = out.concat(tipRows(pt.tip));
        });
        return { title: xText(xv), rows: out };
      }

      function showAt(i, clientX, clientY) {
        current = clamp(i, 0, xs.length - 1);
        var px = xAt(xs[current]);
        hair.setAttribute("x1", crisp(px));
        hair.setAttribute("x2", crisp(px));
        series.forEach(function (s, k) {
          var p = valueAt(k, xs[current]);
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
          clientY = box.top + top;
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
    /*
     * scale "plain" sets the grid the way a banker's sensitivity table is set:
     * numbers on the page, a rule under each row, no colour, and the base case
     * boxed. spec.base is that cell as [row, column].
     */
    var plain = spec.scale === "plain";
    var base = Array.isArray(spec.base) && isNum(spec.base[0]) && isNum(spec.base[1]) ? spec.base : null;
    var valueName = spec.valueLabel || "Value";
    var breaks =
      diverging && Array.isArray(spec.breaks) && spec.breaks.length === 2 && isNum(spec.breaks[0]) && isNum(spec.breaks[1])
        ? [Math.abs(spec.breaks[0]), Math.abs(spec.breaks[1])].sort(function (a, b) {
            return a - b;
          })
        : null;
    var family = spec.mono ? "mono" : null;

    var tableCols = [{ key: "row", label: spec.rowHeader || "Row", mono: !!spec.mono }].concat(
      colsL.map(function (c, j) {
        return { key: "c" + j, label: String(c), align: "right", format: spec.format };
      })
    );
    tableFor(
      body,
      {
        columns: tableCols,
        rows: rowsL.map(function (r, i) {
          var out = { row: r };
          colsL.forEach(function (_, j) {
            out["c" + j] = values[i] ? values[i][j] : null;
          });
          return out;
        }),
      },
      spec
    );
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
    var hasZero = flat.some(function (v) {
      return v === 0;
    });

    function classOf(v) {
      if (!isNum(v)) return null;
      if (diverging && breaks) {
        if (v === 0) return 0;
        var a = Math.abs(v);
        var c = a < breaks[0] ? 1 : a < breaks[1] ? 2 : 3;
        return v > 0 ? c : -c;
      }
      if (diverging) {
        var bound = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
        return clamp(Math.round((v / bound) * 3), -3, 3);
      }
      if (hi === lo) return 4;
      return clamp(Math.floor(((v - lo) / (hi - lo)) * 7) + 1, 1, 7);
    }

    function fillOf(k) {
      if (plain) return "transparent";
      if (k === null) return "var(--missing)";
      if (diverging) return k === 0 ? "var(--div-0)" : k > 0 ? "var(--div-pos-" + k + ")" : "var(--div-neg-" + -k + ")";
      return "var(--seq-" + k + ")";
    }

    function inkOf(k) {
      if (k === null) return "var(--ink-3)";
      if (plain) return "var(--ink-1)";
      return diverging ? "var(--on-div-" + Math.abs(k) + ")" : "var(--on-seq-" + k + ")";
    }

    /* Rows under group headings, in order; rows no group counts come last, unheaded. */
    var blocks = [];
    var cursor = 0;
    (Array.isArray(spec.groups) ? spec.groups : []).forEach(function (g) {
      if (!g || !isNum(g.count) || g.count <= 0 || cursor >= rowsL.length) return;
      var count = Math.min(Math.round(g.count), rowsL.length - cursor);
      blocks.push({ label: g.label ? String(g.label) : "", from: cursor, count: count });
      cursor += count;
    });
    if (cursor < rowsL.length) blocks.push({ label: "", from: cursor, count: rowsL.length - cursor, bare: true });

    frame(body, "heat", function (wrap, W) {
      /* Row labels are never clipped: wrapped at labelWidth when given, otherwise at full length. */
      var cap = isNum(spec.labelWidth) ? spec.labelWidth : null;
      var rowWrapped = rowsL.map(function (r) {
        return cap ? wrapText(r, cap, 12, 2, 400, family) : { lines: [String(r)], width: measure(r, 12, 400, family) };
      });
      var rowW = maxOf(
        rowWrapped.map(function (w) {
          return w.width;
        }),
        30
      );
      var colTitleH = spec.colTitle ? 16 : 0;
      var m = { top: 24 + colTitleH, right: 4, bottom: 4, left: rowW + 12 };
      var notes = Array.isArray(spec.rowNotes) ? spec.rowNotes : [];
      var notesW = maxOf(
        notes.map(function (t) {
          return t ? measure(t, 11, 500) : 0;
        })
      );
      notesW = notesW ? notesW + 12 : 0;
      function cellWidth(reserve) {
        var w = Math.max(HIT_MIN, (W - m.left - m.right - reserve) / colsL.length);
        return isNum(spec.cellMax) ? Math.min(Math.max(HIT_MIN, spec.cellMax), w) : w;
      }
      var cellW = cellWidth(notesW);
      if (notesW && m.left + cellW * colsL.length + notesW > W) {
        notesW = 0;
        cellW = cellWidth(0);
      }
      var cellH = Math.max(HIT_MIN, spec.cellHeight || 30);
      var groupH = 24;
      var gridW = cellW * colsL.length;
      var headed = blocks.filter(function (b) {
        return b.label;
      }).length;
      var H = Math.round(m.top + headed * groupH + cellH * rowsL.length + m.bottom);
      var root = chartSvg(Math.max(W, m.left + gridW + notesW + m.right), H, spec, "Heat map");
      if (spec.colTitle) {
        root.appendChild(
          svg("text", { class: "tv-group-label", x: m.left + gridW / 2, y: 11, "text-anchor": "middle" }, String(spec.colTitle))
        );
      }
      var colItems = colsL.map(function (c, j) {
        return { center: m.left + cellW * j + cellW / 2, width: measure(c, 11) };
      });
      thinLabels(colItems, 6).forEach(function (j) {
        root.appendChild(svg("text", { class: "tv-tick", x: colItems[j].center, y: m.top - 9, "text-anchor": "middle" }, String(colsL[j])));
      });
      if (plain) {
        if (spec.rowHeader) {
          root.appendChild(svg("text", { class: "tv-tick", x: m.left - 10, y: m.top - 9, "text-anchor": "end" }, String(spec.rowHeader)));
        }
        root.appendChild(svg("line", { class: "tv-axisline", x1: 0, x2: m.left + gridW, y1: crisp(m.top - 2), y2: crisp(m.top - 2) }));
      }
      var showLabels =
        spec.cellLabels !== false &&
        flat.every(function (v) {
          return measure(format(spec.format, v), 11) + 8 <= cellW - GAP;
        });
      var marks = svg("g");
      var labelStyle = spec.mono ? { fontFamily: "var(--font-mono)" } : null;
      var startAligned = spec.labelAlign === "start";
      var top = m.top;
      blocks.forEach(function (b) {
        if (b.label) {
          marks.appendChild(svg("text", { class: "tv-group-label", x: 0, y: top + groupH - 8 }, b.label));
          marks.appendChild(
            svg("line", { class: "tv-axisline", x1: 0, x2: m.left + gridW, y1: crisp(top + groupH - 2), y2: crisp(top + groupH - 2) })
          );
          top += groupH;
        }
        for (var i = b.from; i < b.from + b.count; i++) {
          drawRow(i, top);
          top += cellH;
        }
      });

      function drawRow(i, rowTop) {
        var r = rowsL[i];
        var cy = rowTop + cellH / 2;
        textBlock(marks, rowWrapped[i].lines, startAligned ? 0 : m.left - 10, cy, {
          class: "tv-label",
          "text-anchor": startAligned ? "start" : "end",
          style: labelStyle,
        }, 13);
        colsL.forEach(function (c, j) {
          var v = values[i] ? values[i][j] : null;
          var k = classOf(v);
          var cx = m.left + cellW * j;
          var cell = svg("rect", {
            class: "tv-mark",
            x: cx + GAP / 2,
            y: rowTop + GAP / 2,
            width: Math.max(0, cellW - GAP),
            height: Math.max(0, cellH - GAP),
            rx: 2,
            style: { fill: fillOf(k) },
          });
          var g = markGroup(marks, { x: cx, y: rowTop, width: cellW, height: cellH }, cell, r + ", " + c + ": " + format(spec.format, v));
          if (showLabels) {
            var isBase = base && base[0] === i && base[1] === j;
            g.appendChild(
              svg(
                "text",
                { class: "tv-cell-label" + (isBase ? " tv-cell-label--base" : ""), x: cx + cellW / 2, y: cy, dy: "0.35em", "text-anchor": "middle", style: { fill: inkOf(k) } },
                format(spec.format, v)
              )
            );
          }
          tooltip.attach(g, function () {
            return {
              title: r + " · " + c,
              rows: [{ label: valueName, value: format(spec.format, v), color: fillOf(k), shape: "rect" }].concat(
                tipRows(Array.isArray(spec.rowTips) ? spec.rowTips[i] : null)
              ),
            };
          });
        });
        if (notesW && notes[i]) {
          marks.appendChild(svg("text", { class: "tv-value", x: m.left + gridW + 10, y: cy, dy: "0.35em" }, notes[i]));
        }
        if (plain) {
          marks.appendChild(svg("line", { class: "tv-gridline", x1: 0, x2: m.left + gridW, y1: crisp(rowTop + cellH), y2: crisp(rowTop + cellH) }));
        }
        if (base && base[0] === i) {
          var bx = m.left + cellW * base[1];
          marks.appendChild(
            svg("rect", { class: "tv-heat-base", x: bx + 1.5, y: rowTop + 1.5, width: Math.max(0, cellW - 3), height: Math.max(0, cellH - 3) })
          );
        }
      }
      root.appendChild(marks);
      wrap.appendChild(root);

      /* The scale legend: one swatch per class, with its bounds in text. A plain grid has no scale. */
      if (plain) return;
      var classes;
      var ticks;
      var title = spec.scaleLabel || null;
      if (breaks) {
        classes = hasZero ? [-3, -2, -1, 0, 1, 2, 3] : [-3, -2, -1, 1, 2, 3];
        ticks = [format("signed:1", -breaks[1]) + " or less", "0", format("signed:1", breaks[1]) + " or more"];
        title = (spec.scaleLabel || valueName) + ", classes break at " + format("num:1", breaks[0]) + " and " + format("num:1", breaks[1]);
      } else {
        classes = diverging ? [-3, -2, -1, 0, 1, 2, 3] : [1, 2, 3, 4, 5, 6, 7];
        ticks = diverging
          ? [format(spec.format, -Math.max(Math.abs(lo), Math.abs(hi))), format(spec.format, 0), format(spec.format, Math.max(Math.abs(lo), Math.abs(hi)))]
          : [format(spec.format, lo), format(spec.format, hi)];
      }
      var scaleNode = el(
        "div",
        { class: "tv-scale" },
        el(
          "div",
          { class: "tv-scale__ramp" },
          title ? el("span", { class: "tv-scale__title" }, title) : null,
          el(
            "div",
            { class: "tv-scale__bar" },
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
    tableFor(body, { columns: columns, rows: tableRows }, spec);
    if (!nBins || !series.length) return empty(body);
    chartLegend(
      body,
      spec,
      series.map(function (s) {
        return { label: s.name, color: s.role, shape: "rect" };
      }),
      "rect"
    );

    frame(body, "hist", function (wrap, W) {
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
    tableFor(body, { columns: columns, rows: rows }, spec);
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

    frame(body, "range", function (wrap, W) {
      var all = [];
      rows.forEach(function (r) {
        all.push(r.lo, r.hi, r.mid);
      });
      var ext = extent(all.concat(referenceValues(spec))) || [0, 1];
      if (spec.zero) ext = [Math.min(0, ext[0]), Math.max(0, ext[1])];
      var dom = spec.domain || [];
      var lo = isNum(dom[0]) ? dom[0] : ext[0];
      var hi = isNum(dom[1]) ? dom[1] : ext[1];
      var labels = rowLabels(rows, W, spec.labelWidth);
      var endW = Math.max.apply(
        null,
        [0].concat(
          rows.map(function (r) {
            var loT = format(spec.format, Math.min(r.lo, r.hi));
            var hiT = format(spec.format, Math.max(r.lo, r.hi));
            /* A bar starting right of a reference line may carry both ends on its right. */
            var joins = referenceValues(spec).some(function (v) {
              return Math.min(r.lo, r.hi) >= v;
            });
            return Math.max(measure(loT, 11, 500), measure(joins ? loT + " to " + hiT : hiT, 11, 500));
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
      var tickCount = labelTickCount(x, W - m.left - m.right);
      var thick = Math.min(BAR_MAX, 18, rowH * 0.6);
      var root = chartSvg(W, H, spec, "Range");
      var grid = svg("g");
      xAxisTicks(grid, x, m.top, m.top + plotH, spec.format, tickCount, W - 1);
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
          var loText = format(spec.format, Math.min(r.lo, r.hi));
          var hiText = format(spec.format, Math.max(r.lo, r.hi));
          var loRight = x(Math.min(r.lo, r.hi)) - 6;
          var loLeft = loRight - measure(loText, 11, 500) - 3;
          /* A low label that would sit across a reference line (the zero of a lift) joins the high one instead. */
          var crossesRef = referenceValues(spec).some(function (v) {
            return x(v) > loLeft && x(v) < loRight + 6;
          });
          if (!crossesRef) {
            valueLayer.appendChild(svg("text", { class: "tv-value", x: loRight, y: cy, dy: "0.35em", "text-anchor": "end" }, loText));
          }
          valueLayer.appendChild(
            svg("text", { class: "tv-value", x: x(Math.max(r.lo, r.hi)) + 6, y: cy, dy: "0.35em" }, crossesRef ? loText + " to " + hiText : hiText)
          );
        }
        tooltip.attach(g, function () {
          var out = [
            { label: "Low", value: format(spec.format, r.lo), color: r.role || "model", shape: "rect" },
          ];
          if (isNum(r.mid)) out.push({ label: "Mid", value: format(spec.format, r.mid) });
          out.push({ label: "High", value: format(spec.format, r.hi) });
          (spec.reference || []).forEach(function (ref) {
            if (ref && isNum(ref.value)) out.push(refTipRow(ref, spec.format));
          });
          return { title: r.label, rows: out.concat(tipRows(r.tip)) };
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

    tableFor(
      body,
      {
        columns: [
          { key: "label", label: spec.labelHeader || "Component" },
          { key: "value", label: spec.valueLabel || "Value", align: "right", format: spec.format },
          { key: "to", label: "Running total", align: "right", format: spec.format },
        ],
        rows: items,
      },
      spec
    );
    if (!steps.length) return empty(body);
    var handle = handleFor(body);
    if (handle && reported !== null && Math.abs(reported - running) > Math.max(1e-6, Math.abs(running) * 1e-6)) {
      handle.addNote({
        what: totalLabel,
        why: "The components sum to " + format(spec.format, running) + " but the total supplied is " + format(spec.format, reported) + "; the bridge draws the sum.",
      });
    }
    /* A legend entry for each kind of bar the bridge draws, and no other. */
    var legendItems = [];
    if (
      steps.some(function (s) {
        return isNum(s.value) && s.value > 0;
      })
    ) {
      legendItems.push({ label: spec.upLabel || "Adds", color: "pos", shape: "rect" });
    }
    if (
      steps.some(function (s) {
        return isNum(s.value) && s.value < 0;
      })
    ) {
      legendItems.push({ label: spec.downLabel || "Subtracts", color: "neg", shape: "rect" });
    }
    legendItems.push({ label: spec.totalLegend || "Total", color: "total", shape: "rect" });
    chartLegend(body, spec, legendItems, "rect");

    frame(body, "waterfall", function (wrap, W) {
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
        /* A step of exactly zero is the finding, so it draws a rule rather than an empty column under its label. */
        else if (isNum(it.value) && it.kind !== "total") {
          parts.push(
            svg("rect", { class: "tv-mark", x: cx - thick / 2, y: crisp(y(it.to)) - 1, width: thick, height: 2, style: { fill: fill } })
          );
        }
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

  /* table: the figure is the table ---------------------------------------- */

  function table(body, spec) {
    spec = spec || {};
    var rows = (spec.rows || []).filter(Boolean);
    if (!rows.length || !(spec.columns || []).length) return empty(body, "No rows to show.");
    var handle = handleFor(body);
    return TV.tableView(
      { table: body, title: handle ? handle.title : spec.title },
      { columns: spec.columns, rows: rows, caption: spec.caption || spec.title }
    );
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

  function tileValue(t) {
    return isNum(t.value) ? format(t.format, t.value) : isNil(t.value) ? "n/a" : String(t.value);
  }

  function tiles(body, input) {
    var list = (Array.isArray(input) ? input : (input && input.tiles) || []).filter(Boolean);
    var grid = el("div", { class: "tv-tiles" });
    list.forEach(function (t) {
      grid.appendChild(
        el(
          t.href ? "a" : "div",
          { class: "tv-tile", href: t.href || null },
          el("p", { class: "tv-tile__label" }, t.label),
          el("p", { class: "tv-tile__value" }, tileValue(t)),
          deltaNode(t.delta),
          t.sub ? el("p", { class: "tv-tile__sub" }, t.sub) : null,
          t.status ? TV.chip(t.status, t.statusText) : null
        )
      );
    });
    body.appendChild(grid);
    /* Inside a figure card the tiles are a figure, so they get a table view like any other. */
    var handle = handleFor(body);
    if (handle && list.length) {
      TV.tableView(handle, {
        columns: [
          { key: "label", label: "Measure" },
          { key: "value", label: "Value", align: "right" },
          { key: "sub", label: "Detail" },
        ],
        rows: list.map(function (t) {
          return { label: t.label, value: tileValue(t), sub: t.sub || "" };
        }),
      });
    }
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
    table: table,
    tiles: tiles,
  };

  /* Section registry ------------------------------------------------------ */

  var renderers = {};
  var registered = [];

  /* A snapshot figure with a section's override merged over it. */
  function resolveFigure(figure, override) {
    var f = Object.assign({}, figure || {});
    if (!override) return f;
    Object.keys(override).forEach(function (key) {
      if (key === "data") return;
      f[key] = override[key];
    });
    if (typeof override.data === "function") {
      f.data = override.data(Object.assign({}, f.data || {}), f) || f.data;
    } else if (override.data && typeof override.data === "object") {
      f.data = Object.assign({}, f.data || {}, override.data);
    }
    return f;
  }

  function cautionsOf(f) {
    var out = [];
    [f.note, f.notes, f.data && f.data.note, f.data && f.data.notes].forEach(function (n) {
      (Array.isArray(n) ? n : [n]).forEach(function (x) {
        if (typeof x === "string" ? x : x && (x.what || x.why)) out.push(x);
      });
    });
    return out;
  }

  /* Figures in reading order, as groups: each group is one run of cards and tiles. */
  function figureGroups(ids, figs, order) {
    function byOrder(list) {
      return list
        .map(function (id, i) {
          return { id: id, i: i, o: isNum(figs[id].order) ? figs[id].order : Infinity };
        })
        .sort(function (a, b) {
          return a.o === b.o ? a.i - b.i : a.o < b.o ? -1 : 1;
        })
        .map(function (d) {
          return d.id;
        });
    }
    if (Array.isArray(order) && order.length) {
      var groups = [];
      var loose = null;
      order.forEach(function (item) {
        if (typeof item === "string") {
          if (!loose) {
            loose = { title: null, ids: [] };
            groups.push(loose);
          }
          loose.ids.push(item);
          return;
        }
        loose = null;
        if (Array.isArray(item)) groups.push({ title: null, ids: item.slice() });
        else if (item && Array.isArray(item.ids)) groups.push({ title: item.title || null, ids: item.ids.slice() });
      });
      var named = {};
      groups.forEach(function (g) {
        g.ids = g.ids.filter(function (id) {
          if (named[id] || ids.indexOf(id) < 0) return false;
          named[id] = true;
          return true;
        });
      });
      groups = groups.filter(function (g) {
        return g.ids.length;
      });
      var rest = byOrder(
        ids.filter(function (id) {
          return !named[id];
        })
      );
      if (rest.length) groups.push({ title: null, ids: rest });
      return groups;
    }
    if (
      ids.some(function (id) {
        return isNum(figs[id].order);
      })
    ) {
      return [{ title: null, ids: byOrder(ids) }];
    }
    var tileIds = ids.filter(function (id) {
      return figs[id].kind === "tiles" && !figs[id].card;
    });
    return [
      {
        title: null,
        ids: tileIds.concat(
          ids.filter(function (id) {
            return tileIds.indexOf(id) < 0;
          })
        ),
      },
    ];
  }

  function drawFigure(handle, id, f) {
    var spec = Object.assign({ title: f.title }, f.data || {});
    if (typeof f.draw === "function") {
      f.draw(handle, f);
    } else if (f.kind === "tiles") {
      tiles(handle.body, spec);
    } else if (Object.prototype.hasOwnProperty.call(TV.charts, f.kind)) {
      TV.charts[f.kind](handle.body, spec);
    } else {
      handle.addNote({ what: id, why: "The chart kit has no chart named " + JSON.stringify(String(f.kind)) + "." });
    }
    cautionsOf(f).forEach(function (n) {
      handle.addCaution(n);
    });
    if (typeof f.after === "function") f.after(handle, f);
  }

  /*
   * A set of stat tiles on the page plane. It is a figure like any other, so it
   * carries the same table view, the same data toggle and the same source line;
   * only the card around it is missing.
   */
  function tileset(parent, id, f, provenance) {
    var set = el(
      "div",
      { class: "tv-tileset", "data-figure-id": id },
      f.title ? el("h3", { class: "tv-tileset__title" }, f.title) : null,
      f.subtitle ? el("p", { class: "tv-figure__subtitle" }, f.subtitle) : null
    );
    var body = el("div", { class: "tv-tileset__body" });
    var table = el("div", { class: "tv-figure__table", hidden: true });
    var notes = el("div", { class: "tv-figure__notes" });
    var toggle = el("button", { class: "tv-btn", type: "button", hidden: true }, "Show data");
    var source = el("span", { class: "tv-figure__source" });
    var foot = el("footer", { class: "tv-figure__foot" }, source, toggle);
    set.appendChild(body);
    set.appendChild(table);
    set.appendChild(notes);
    set.appendChild(foot);
    parent.appendChild(set);

    var entries = provenanceRows(provenance);
    if (entries.length) {
      source.appendChild(document.createTextNode(compactEntries(entries)));
      source.setAttribute("title", sourceTitle(entries));
    }

    toggle.addEventListener("click", function () {
      var showTable = table.hidden;
      table.hidden = !showTable;
      body.hidden = showTable;
      toggle.textContent = showTable ? "Hide data" : "Show data";
    });

    var handle = {
      root: set,
      body: body,
      table: table,
      toggle: toggle,
      title: f.title || "",
      addNote: function (r) {
        notes.appendChild(refusalNote(r));
      },
      addCaution: function (n) {
        notes.appendChild(cautionNote(n));
      },
    };
    set.__tvHandle = handle;
    return handle;
  }

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
     * The drawing of a section's figures, in reading order (see the header for
     * opts). Stat tiles sit on the page plane unless a figure asks for a card;
     * every other kind gets a card drawn by the kit chart it names. A kind the
     * kit does not have becomes a note on its card, never a guess. Returns
     * {figureId or refusal name: handle} so app.js can hang refusal notes under
     * the figures they affect.
     */
    figures: function (root, data, opts) {
      opts = opts || {};
      var source = (data && data.figures) || {};
      var ids = Object.keys(source);
      var handles = {};
      if (!ids.length) return handles;
      var provenance = (data && data.provenance) || [];
      var overrides = opts.figures || {};
      var figs = {};
      ids.forEach(function (id) {
        figs[id] = resolveFigure(source[id], overrides[id]);
      });
      function provenanceOf(id) {
        return provenance.filter(function (p) {
          return p && p.figure === id;
        });
      }
      function inCard(f) {
        return f.kind !== "tiles" || !!f.card;
      }
      var parentOf = {};
      ids.forEach(function (id) {
        var parent = (opts.parts && opts.parts[id]) || figs[id].part;
        if (typeof parent === "string" && parent !== id && figs[parent] && inCard(figs[parent]) && !figs[parent].part) {
          parentOf[id] = parent;
        }
      });
      var topIds = ids.filter(function (id) {
        return !parentOf[id];
      });

      figureGroups(topIds, figs, opts.order).forEach(function (group) {
        var block = root;
        if (group.title) {
          block = el("div", { class: "tv-section__body tv-figure-group" }, el("h3", { class: "tv-eyebrow" }, group.title));
          root.appendChild(block);
        }
        var grid = null;
        group.ids.forEach(function (id) {
          var f = figs[id];
          var handle;
          if (!inCard(f)) {
            handle = tileset(block, id, f, provenanceOf(id));
            grid = null;
            drawFigure(handle, id, f);
          } else {
            if (!grid) {
              grid = el("div", { class: "tv-grid" });
              block.appendChild(grid);
            }
            handle = TV.figure(grid, {
              id: id,
              anchor: data && data.id ? "fig-" + data.id + "-" + id : null,
              title: f.title,
              subtitle: f.subtitle,
              provenance: provenanceOf(id),
              wide: !!(f.wide || (f.data && f.data.wide)),
            });
            drawFigure(handle, id, f);
          }
          handles[id] = handle;
          ids.forEach(function (child) {
            if (parentOf[child] !== id) return;
            var cf = figs[child];
            var part = handle.part({ id: child, title: cf.title, subtitle: cf.subtitle, provenance: provenanceOf(child) });
            drawFigure(part, child, cf);
            handles[child] = part;
          });
        });
      });

      ids.forEach(function (id) {
        var f = figs[id];
        var names = [].concat(
          (opts.refusals && opts.refusals(id, f)) || [],
          f.refusals || [],
          (f.data && f.data.refusals) || []
        );
        names.forEach(function (name) {
          if (typeof name === "string" && name && handles[id] && !handles[name]) handles[name] = handles[id];
        });
      });
      return handles;
    },
  };
})(window);
