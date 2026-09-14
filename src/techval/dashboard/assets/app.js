/*
 * techval dashboard boot: read the snapshot, draw the header, the section rail
 * and every section in page order.
 *
 * This file owns the parts of the page the snapshot schema defines for every
 * section alike: the eyebrow, title and takeaway, the headline strip on model
 * sections, the not_built and refused states, refusal notes and the provenance
 * list. What goes between the header and the provenance is the section
 * renderer's business, registered by assets/sections/<id>.js.
 *
 * The page never states a number the snapshot does not hold. A missing or
 * unreadable snapshot is shown as a refusal, not an empty page with a title.
 */
(function (global) {
  "use strict";

  var TV = global.TV;
  var el = TV.el;

  var SNAPSHOT_ID = "tv-snapshot";
  var SCHEMA = 1;

  /*
   * ?theme=dark or ?theme=light stamps data-theme on the root, so a headless
   * screenshot can force either theme. Done at once rather than at boot, so
   * the page never paints in the wrong one first.
   */
  function stampTheme() {
    var match = /[?&]theme=(dark|light)(?:&|#|$)/.exec(global.location ? global.location.search : "");
    if (match) document.documentElement.setAttribute("data-theme", match[1]);
  }

  stampTheme();

  function readSnapshot() {
    var node = document.getElementById(SNAPSHOT_ID);
    if (!node) {
      return { error: "The page carries no results snapshot, so there is nothing it can show." };
    }
    try {
      var data = JSON.parse(node.textContent);
      if (!data || typeof data !== "object") throw new Error("the snapshot is not an object");
      if (data.schema !== SCHEMA) {
        return {
          error:
            "The results snapshot is schema " + JSON.stringify(data.schema) + " and this page reads schema " + SCHEMA + ".",
        };
      }
      return { snapshot: data };
    } catch (err) {
      return { error: "The results snapshot could not be read: " + err.message + "." };
    }
  }

  /* Formatting that follows the magnitude of the number, for headline figures. */
  function autoDp(v) {
    var abs = Math.abs(v);
    if (abs >= 100) return 0;
    if (abs >= 10) return 1;
    if (abs >= 1) return 2;
    return 4;
  }

  function score(v) {
    return typeof v === "number" && isFinite(v) ? TV.fmt.num(v, autoDp(v)) : "n/a";
  }

  function signedScore(v) {
    return typeof v === "number" && isFinite(v) ? TV.fmt.signed(v, autoDp(v)) : "n/a";
  }

  function sentenceCase(text) {
    var s = String(text || "");
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
  }

  function isModelSection(section) {
    return !!(section && section.headline);
  }

  /* Header ---------------------------------------------------------------- */

  function header(snapshot) {
    var fixtures = snapshot.fixtures || {};
    var paths = Object.keys(fixtures).sort();
    var meta = el(
      "p",
      { class: "tv-meta" },
      el("span", null, el("span", { class: "tv-meta__label" }, "Commit"), el("code", null, snapshot.techval_commit || "n/a")),
      el("span", null, el("span", { class: "tv-meta__label" }, "Collected"), snapshot.collected_at || "n/a"),
      el("span", null, paths.length === 1 ? "1 fixture" : paths.length + " fixtures"),
      el(
        "span",
        null,
        paths.length
          ? "All figures computed offline from committed fixtures"
          : "No fixture digests recorded, so no figure here is tied to a committed input"
      )
    );
    var list = null;
    if (paths.length) {
      list = el(
        "details",
        { class: "tv-fixtures" },
        el("summary", null, "Fixture digests"),
        el(
          "ul",
          null,
          paths.map(function (path) {
            var digest = String(fixtures[path] || "");
            return el(
              "li",
              null,
              el("code", null, path),
              el("code", { class: "tv-muted", title: digest }, "sha256 " + digest.slice(0, 12))
            );
          })
        )
      );
    }
    var node = el("header", { class: "tv-header" }, el("h1", { class: "tv-title" }, snapshot.title || "techval Results"), meta, list);
    /* A page may carry its own notice (the gallery says its data is synthetic). */
    var notice = document.getElementById("tv-notice");
    if (notice) {
      notice.hidden = false;
      node.appendChild(notice);
    }
    return node;
  }

  /* Rail ------------------------------------------------------------------ */

  function railItem(id, section) {
    var state = null;
    if (!section) {
      state = el("span", { class: "tv-rail__state" }, "Not collected");
    } else if (section.status === "refused") {
      state = TV.chip("refused", null, { compact: true });
    } else if (section.status === "not_built") {
      state = el("span", { class: "tv-rail__state" }, "Not collected");
    } else if (isModelSection(section) && section.headline.verdict_status) {
      state = TV.chip(section.headline.verdict_status, null, { compact: true });
    }
    var link = el(
      "a",
      { class: "tv-rail__link", href: "#" + id, "data-section": id },
      el("span", null, (section && section.title) || sentenceCase(id)),
      state
    );
    return el("li", null, link);
  }

  function rail(order, sections) {
    var list = el("ul", { class: "tv-rail__list" });
    order.forEach(function (id) {
      list.appendChild(railItem(id, sections[id]));
    });
    return el(
      "nav",
      { class: "tv-rail", "aria-label": "Sections" },
      el("p", { class: "tv-rail__title", "aria-hidden": "true" }, "Sections"),
      list
    );
  }

  function spy(nav) {
    if (typeof IntersectionObserver === "undefined") return;
    var links = {};
    Array.prototype.forEach.call(nav.querySelectorAll("[data-section]"), function (a) {
      links[a.getAttribute("data-section")] = a;
    });
    var visible = {};
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          visible[entry.target.id] = entry.isIntersecting;
        });
        var current = null;
        Object.keys(links).some(function (id) {
          if (visible[id]) {
            current = id;
            return true;
          }
          return false;
        });
        if (!current) return;
        Object.keys(links).forEach(function (id) {
          if (id === current) links[id].setAttribute("aria-current", "true");
          else links[id].removeAttribute("aria-current");
        });
        var active = links[current];
        var list = active.parentNode && active.parentNode.parentNode;
        if (list && list.scrollWidth > list.clientWidth) {
          var left = active.offsetLeft - list.clientWidth / 2 + active.offsetWidth / 2;
          list.scrollLeft = Math.max(0, left);
        }
      },
      { rootMargin: "-15% 0px -70% 0px" }
    );
    Object.keys(links).forEach(function (id) {
      var target = document.getElementById(id);
      if (target) observer.observe(target);
    });
  }

  /* Section states -------------------------------------------------------- */

  function headline(h) {
    var metric = h.metric || "score";
    var n = typeof h.n === "number" && isFinite(h.n) ? ", n = " + TV.fmt.int(h.n) : "";
    var chip = TV.chip.known(h.verdict_status) ? TV.chip(h.verdict_status) : TV.chip("refused", "No verdict");
    var strip = el(
      "div",
      { class: "tv-headline" },
      el(
        "div",
        { class: "tv-headline__verdict" },
        chip,
        h.verdict_text ? el("p", { class: "tv-headline__text" }, h.verdict_text) : null
      )
    );
    TV.charts.tiles(strip, [
      { label: "Model", value: score(h.score), sub: metric + n },
      { label: "Baseline", value: score(h.baseline_score), sub: h.baseline_name || "unnamed baseline" },
      {
        label: "Lift over baseline",
        value: signedScore(h.lift),
        sub: h.higher_is_better === false ? "Lower " + metric + " is better" : "Higher " + metric + " is better",
      },
    ]);
    return strip;
  }

  function placeholder(title, text) {
    return el(
      "div",
      { class: "tv-placeholder" },
      el("p", { class: "tv-placeholder__title" }, title),
      el("p", null, text)
    );
  }

  function refusalCard(section) {
    var refusals = section.refusals || [];
    return el(
      "div",
      { class: "tv-refusal", role: "note" },
      el(
        "div",
        { class: "tv-refusal__head" },
        TV.chip("refused"),
        el("span", null, "This section declined to produce figures rather than show numbers it could not reproduce.")
      ),
      refusals.length
        ? el(
            "ul",
            { class: "tv-refusal__list" },
            refusals.map(function (r) {
              return el(
                "li",
                null,
                el("p", { class: "tv-refusal__what" }, (r && r.what) || "Unnamed figure"),
                el("p", { class: "tv-refusal__why" }, (r && r.why) || "No reason was recorded.")
              );
            })
          )
        : el("p", { class: "tv-refusal__why" }, "No reason was recorded, which is itself a defect in the collector.")
    );
  }

  function provenanceList(section) {
    var rows = (section.provenance || []).filter(Boolean);
    if (!rows.length) return null;
    var details = el(
      "details",
      { class: "tv-provenance" },
      el("summary", null, rows.length === 1 ? "Provenance, 1 entry" : "Provenance, " + rows.length + " entries")
    );
    var holder = { table: details };
    TV.tableView(holder, {
      caption: (section.title || section.id) + " provenance",
      columns: [
        { key: "figure", label: "Figure", mono: true },
        { key: "entry_point", label: "Entry point", mono: true },
        { key: "inputs", label: "Inputs", mono: true },
        { key: "seconds", label: "Seconds", align: "right", format: "num:1" },
      ],
      rows: rows.map(function (p) {
        return {
          figure: p.figure,
          entry_point: p.entry_point,
          inputs: (p.inputs || []).length ? p.inputs.join(", ") : "none",
          seconds: p.seconds,
        };
      }),
    });
    return details;
  }

  /* A refusal names a figure by id or by title; anything else is section-wide. */
  function placeRefusals(section, handles, notes) {
    var byKey = {};
    Object.keys(handles || {}).forEach(function (id) {
      byKey[id.toLowerCase()] = handles[id];
      var t = handles[id].title;
      if (t) byKey[String(t).toLowerCase()] = handles[id];
    });
    (section.refusals || []).forEach(function (r) {
      if (!r) return;
      var target = byKey[String(r.what || "").toLowerCase()];
      if (target) target.addNote(r);
      else notes.appendChild(TV.refusalNote(r));
    });
  }

  function sectionNode(id, section, snapshot) {
    var head = el(
      "header",
      { class: "tv-section__head" },
      el("p", { class: "tv-eyebrow" }, id),
      el("h2", { class: "tv-section__title", id: id + "-title" }, (section && section.title) || sentenceCase(id)),
      section && section.takeaway ? el("p", { class: "tv-takeaway" }, section.takeaway) : null
    );
    var node = el("section", { class: "tv-section", id: id, "aria-labelledby": id + "-title" }, head);

    if (!section) {
      node.appendChild(placeholder("Not collected yet", "The snapshot holds no entry for this section, so the page shows nothing in its place."));
      return node;
    }
    if (section.status === "not_built") {
      node.appendChild(placeholder("Not collected yet", "This section has not been collected, so it has no figures to show."));
      return node;
    }
    if (section.status === "refused") {
      node.appendChild(refusalCard(section));
      var refusedProv = provenanceList(section);
      if (refusedProv) node.appendChild(refusedProv);
      return node;
    }
    if (section.status !== "ok") {
      node.appendChild(
        refusalCard({ refusals: [{ what: "Section status", why: "The snapshot gives this section the status " + JSON.stringify(section.status) + ", which this page does not know how to show." }] })
      );
      return node;
    }

    if (isModelSection(section)) node.appendChild(headline(section.headline));

    var body = el("div", { class: "tv-section__body" });
    node.appendChild(body);
    var notes = el("div", { class: "tv-section__notes" });
    var handles = {};
    try {
      if (TV.sections.has(id)) {
        handles = TV.sections.render(id, body, section, snapshot) || {};
      } else {
        handles = TV.sections.figures(body, section);
      }
    } catch (err) {
      notes.appendChild(
        TV.refusalNote({ what: "Section renderer", why: "The renderer for this section failed (" + err.message + "), so its figures are not shown." })
      );
    }
    if (!body.firstChild && !(section.refusals || []).length) {
      body.appendChild(placeholder("No figures", "The snapshot records this section as collected but holds no figures for it."));
    }
    placeRefusals(section, handles, notes);
    if (notes.firstChild) body.appendChild(notes);
    var prov = provenanceList(section);
    if (prov) node.appendChild(prov);
    return node;
  }

  /* Boot ------------------------------------------------------------------ */

  function pageOrder(sections) {
    var order = TV.sections.order();
    Object.keys(sections)
      .sort()
      .forEach(function (id) {
        if (order.indexOf(id) < 0) order.push(id);
      });
    return order;
  }

  function boot() {
    var read = readSnapshot();
    var main = el("main", { class: "tv-main", id: "tv-main" });
    if (read.error) {
      document.title = "techval Results";
      main.appendChild(el("header", { class: "tv-header" }, el("h1", { class: "tv-title" }, "techval Results")));
      main.appendChild(
        el(
          "div",
          { class: "tv-refusal", role: "note" },
          el("div", { class: "tv-refusal__head" }, TV.chip("refused", "No results")),
          el("p", { class: "tv-refusal__why" }, read.error)
        )
      );
      document.body.appendChild(el("div", { class: "tv-shell" }, main));
      return;
    }
    var snapshot = read.snapshot;
    var sections = snapshot.sections || {};
    var order = pageOrder(sections);
    var nav = rail(order, sections);
    main.appendChild(header(snapshot));
    var shell = el("div", { class: "tv-shell" }, nav, main);
    document.body.appendChild(shell);
    order.forEach(function (id) {
      main.appendChild(sectionNode(id, sections[id], snapshot));
    });
    spy(nav);
    if (global.location && global.location.hash) {
      var target = document.getElementById(decodeURIComponent(global.location.hash.slice(1)));
      if (target) target.scrollIntoView();
    }
    document.documentElement.setAttribute("data-tv-ready", "true");
  }

  TV.boot = boot;

  /* Section scripts follow this one, so wait for the whole document before drawing. */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    global.setTimeout(boot, 0);
  }
})(window);
