(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const icon = (name, size = 15) =>
    `<svg width="${size}" height="${size}" aria-hidden="true"><use href="#i-${name}"/></svg>`;

  const state = {
    status: null,
    bullets: null,
    academicProjects: [],
    selectedCourses: [],
    selectedTopics: [],
    jdSignals: {},
    companyName: "",
    positionName: "",
    jdText: "",
    stage: 0, // 0 = JD, 1 = generate, 2 = review, 3 = exported
    compiled: null,
  };

  const titleCase = (s) =>
    s.replace(/_/g, " ").replace(/\w\S*/g, (t) => t[0].toUpperCase() + t.slice(1));

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  function setStatus(el, text, stateName) {
    if (!text) {
      el.removeAttribute("data-state");
      el.textContent = "";
      return;
    }
    el.setAttribute("data-state", stateName);
    el.textContent = text;
  }

  // ── Card collapse ────────────────────────────────────────────────────────
  // Steps collapse once they're behind you, so a finished JD doesn't push the
  // bullets you're actually editing off the screen.

  function setCard(id, { open, summary }) {
    const card = $(id);
    if (!card) return;
    if (open !== undefined) {
      card.classList.toggle("collapsed", !open);
      const head = card.querySelector(".cardHead");
      if (head && head.tagName === "BUTTON") head.setAttribute("aria-expanded", String(open));
    }
    if (summary !== undefined) {
      const subEl = card.querySelector('[data-role="sub"]');
      const sumEl = card.querySelector('[data-role="summary"]');
      if (subEl && sumEl) {
        const showSummary = Boolean(summary) && card.classList.contains("collapsed");
        sumEl.textContent = summary || "";
        sumEl.hidden = !showSummary;
        subEl.hidden = showSummary;
      }
    }
  }

  document.querySelectorAll(".cardHead[data-toggle]").forEach((head) => {
    head.addEventListener("click", () => {
      const card = $(head.dataset.toggle);
      const willOpen = card.classList.contains("collapsed");
      setCard(head.dataset.toggle, { open: willOpen });
      setCard(head.dataset.toggle, { summary: cardSummary(head.dataset.toggle) });
    });
  });

  function cardSummary(id) {
    if (id === "card-jd") {
      const c = $("companyName").value.trim();
      const p = $("positionName").value.trim();
      if (!c && !p) return "";
      return [c, p].filter(Boolean).join(" — ");
    }
    if (id === "card-review" && state.bullets) {
      const roles = Object.keys(state.bullets).length;
      const n = Object.values(state.bullets).reduce((a, b) => a + b.length, 0);
      return `${n} bullets across ${roles} roles`;
    }
    return "";
  }

  // ── Stepper ──────────────────────────────────────────────────────────────

  const STEPS = [
    { label: "Job description", card: "card-jd" },
    { label: "Generate", card: "card-generate" },
    { label: "Review", card: "card-review" },
    { label: "Export", card: "card-export" },
  ];

  function renderStepper() {
    const el = $("stepper");
    el.innerHTML = "";
    STEPS.forEach((step, i) => {
      const btn = document.createElement("button");
      const cls = i < state.stage ? "done" : i === state.stage ? "active" : "";
      btn.className = `stepBtn ${cls}`;
      btn.type = "button";
      btn.innerHTML =
        `<span class="dot">${i < state.stage ? icon("check", 13) : i + 1}</span>` +
        `<span class="label">${step.label}</span>`;
      const reachable = i <= state.stage && !$(step.card).hidden;
      btn.disabled = !reachable;
      if (i === state.stage) btn.setAttribute("aria-current", "step");
      btn.addEventListener("click", () => {
        const card = $(step.card);
        if (card.hidden) return;
        setCard(step.card, { open: true });
        setCard(step.card, { summary: "" });
        card.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      el.appendChild(btn);
      if (i < STEPS.length - 1) {
        const line = document.createElement("span");
        line.className = `stepLine ${i < state.stage ? "done" : ""}`;
        el.appendChild(line);
      }
    });
  }

  // ── Sidebar (mobile disclosure) ──────────────────────────────────────────

  $("sidebarToggle").addEventListener("click", () => {
    const bar = $("sidebar");
    const open = bar.dataset.open !== "true";
    bar.dataset.open = String(open);
    $("sidebarToggle").setAttribute("aria-expanded", String(open));
  });

  // ── Bootstrap ────────────────────────────────────────────────────────────

  async function loadStatus() {
    const s = await api("/api/status");
    state.status = s;

    const model = $("modelSelect");
    model.innerHTML = "";
    s.model_choices.forEach((m) => {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = m;
      model.appendChild(o);
    });
    model.value = s.default_model;
    $("modeSelect").value = s.default_generation_mode;

    const ws = $("workspaceStatus");
    ws.innerHTML = "";
    if (s.evidence_files.length === 0) {
      ws.appendChild(statusRow("No evidence files found", "", false));
    } else {
      s.evidence_files.forEach((f) =>
        ws.appendChild(statusRow(titleCase(f.company), `${f.min}–${f.max}`, true))
      );
    }
    ws.appendChild(
      statusRow(
        s.template_exists ? "Template ready" : "main.tex not found",
        "",
        s.template_exists
      )
    );

    updateReadiness();
  }

  function statusRow(name, count, ok) {
    const row = document.createElement("div");
    row.className = `statusRow${ok ? "" : " bad"}`;
    row.innerHTML =
      `<span class="dot"></span><span>${name}</span>` +
      (count ? `<span class="count">${count}</span>` : "");
    return row;
  }

  function updateReadiness() {
    const s = state.status;
    const hint = $("generateHint");
    const reasons = [];
    if (s) {
      if (!s.evidence_files.length) reasons.push("no work_*.json evidence files in data/");
      if (!s.template_exists) reasons.push("data/main.tex is missing");
    }
    const hasJd = $("jdText").value.trim() && $("companyName").value.trim();
    if (!reasons.length && !hasJd) reasons.push("add a company name and job description above");

    $("generateBtn").disabled = reasons.length > 0;

    // The cover letter needs only the JD, so it appears as soon as one exists
    // rather than waiting behind the resume steps.
    const coverReady = Boolean(hasJd && s && s.cover_letter_template_exists);
    $("card-cover").hidden = !coverReady;
    $("coverBtn").disabled = !coverReady;
    if (reasons.length) {
      hint.hidden = false;
      hint.innerHTML = `${icon("alert", 15)}<span>Can't generate yet — ${reasons.join("; ")}.</span>`;
    } else {
      hint.hidden = true;
    }
  }

  ["jdText", "companyName", "positionName"].forEach((id) =>
    $(id).addEventListener("input", updateReadiness)
  );

  async function loadJdDefault() {
    try {
      const d = await api("/api/jd-default");
      $("companyName").value = d.company_name || "";
      $("positionName").value = d.position_name || "";
      $("jdText").value = d.jd_text || "";
    } catch (_) {}
  }

  async function loadAcademicProjects() {
    try {
      state.academicProjects = (await api("/api/academic-projects")).projects || [];
    } catch (_) {
      state.academicProjects = [];
    }
  }

  // ── JD upload ────────────────────────────────────────────────────────────

  $("jdFileBtn").addEventListener("click", () => $("jdFile").click());

  $("jdFile").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    $("jdFileName").textContent = file.name;
    const raw = (await file.text()).trim();
    try {
      const data = JSON.parse(raw);
      $("companyName").value = data.company_name || $("companyName").value;
      $("positionName").value = data.position_name || $("positionName").value;
      $("jdText").value = data.job_description || raw;
    } catch (_) {
      if (raw.includes("---")) {
        const [header, ...rest] = raw.split("---");
        header.trim().split("\n").forEach((line) => {
          const i = line.indexOf(":");
          if (i === -1) return;
          const k = line.slice(0, i).trim();
          const v = line.slice(i + 1).trim();
          if (k === "company_name") $("companyName").value = v;
          else if (k === "position_name") $("positionName").value = v;
        });
        $("jdText").value = rest.join("---").trim();
      } else {
        $("jdText").value = raw;
      }
    }
    updateReadiness();
  });

  // ── Generation progress ──────────────────────────────────────────────────
  // The wait is ~30s. These labels track the pipeline's real stages, and the
  // bar is deliberately indeterminate — the API returns once, so a percentage
  // would be invented.

  const PHASES = [
    [0, "Reading the job description…"],
    [4, "Extracting ranking signals and seniority…"],
    [9, "Drafting bullets from your evidence…"],
    [20, "Checking length, verbs and grounding…"],
    [30, "Selecting coursework, projects and skills…"],
  ];

  let progressTimer = null;

  function startProgress() {
    const started = Date.now();
    $("generateProgress").hidden = false;

    const skel = $("generateSkeleton");
    const roles = state.status?.evidence_files?.length || 3;
    skel.innerHTML = Array.from({ length: Math.min(roles, 4) })
      .map(
        () =>
          '<div class="skelGroup"><div class="skelBar head"></div>' +
          '<div class="skelBar"></div><div class="skelBar short"></div></div>'
      )
      .join("");

    progressTimer = setInterval(() => {
      const secs = Math.floor((Date.now() - started) / 1000);
      $("progressElapsed").textContent = `${secs}s`;
      const phase = [...PHASES].reverse().find(([t]) => secs >= t);
      if (phase) $("progressLabel").textContent = phase[1];
    }, 500);
  }

  function stopProgress() {
    clearInterval(progressTimer);
    progressTimer = null;
    $("generateProgress").hidden = true;
  }

  // ── Generate ─────────────────────────────────────────────────────────────

  $("generateBtn").addEventListener("click", async () => {
    const companyName = $("companyName").value.trim();
    const positionName = $("positionName").value.trim();
    const jdText = $("jdText").value.trim();

    $("generateBtn").disabled = true;
    setStatus($("generateStatus"), "", null);
    $("generateHint").hidden = true;
    startProgress();

    try {
      const result = await api("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company_name: companyName,
          position_name: positionName,
          jd_text: jdText,
          model: $("modelSelect").value,
          generation_mode: $("modeSelect").value,
          log_prompts: $("logPrompts").checked,
        }),
      });

      Object.assign(state, {
        bullets: result.bullets,
        selectedCourses: result.selected_courses || [],
        selectedTopics: result.selected_academic_topics || [],
        jdSignals: result.jd_signals || {},
        companyName,
        positionName,
        jdText,
        stage: 2,
        compiled: null,
      });

      stopProgress();
      const seniority = state.jdSignals.seniority;
      setStatus(
        $("generateStatus"),
        `Drafted ${Object.keys(result.bullets).length} roles` +
          (seniority ? ` — pitched at ${seniority} level.` : "."),
        "ok"
      );

      $("card-review").hidden = false;
      $("card-export").hidden = false;
      $("previewBlock").hidden = true;
      $("reportDetails").hidden = true;
      setStatus($("compileStatus"), "", null);

      renderReview();
      // Fold the JD away — you're done with it — and open what you now need.
      setCard("card-jd", { open: false });
      setCard("card-jd", { summary: cardSummary("card-jd") });
      setCard("card-review", { open: true });
      $("card-review").classList.add("reveal");
      renderStepper();
      updateActionBar();
      $("card-review").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      stopProgress();
      setStatus($("generateStatus"), `Generation failed: ${err.message}`, "err");
    } finally {
      $("generateBtn").disabled = false;
      updateReadiness();
    }
  });

  // ── Review ───────────────────────────────────────────────────────────────

  function bulletsFromDom() {
    const out = {};
    Object.keys(state.bullets).forEach((company) => {
      out[company] = Array.from(
        document.querySelectorAll(`textarea[data-company="${company}"]`)
      ).map((a) => a.value);
    });
    return out;
  }

  function paintMeter(row, value) {
    const { min_bullet_chars: MIN, max_bullet_chars: MAX } = state.status;
    const n = value.length;
    const stateName = n < MIN ? "under" : n > MAX ? "over" : "ok";
    row.dataset.state = stateName;
    row.querySelector(".meterFill").style.width =
      `${Math.min(100, (n / MAX) * 100).toFixed(1)}%`;
    row.querySelector(".meterLabel").textContent =
      stateName === "ok"
        ? `${n} · in range`
        : stateName === "under"
        ? `${n} · ${MIN - n} under`
        : `${n} · ${n - MAX} over`;
  }

  function renderReview() {
    const { min_bullet_chars: MIN, max_bullet_chars: MAX } = state.status;
    const editor = $("bulletEditor");
    editor.innerHTML = "";

    Object.entries(state.bullets).forEach(([company, list]) => {
      const display = titleCase(company);
      const block = document.createElement("div");
      block.className = "company";
      block.innerHTML = `
        <div class="companyHead">
          <h3>${display}</h3>
          <span class="meta">${list.length} bullets</span>
          <span class="spacer"></span>
          <button class="btn btnQuiet" data-act="toggle-regen" type="button">
            ${icon("refresh", 14)} Regenerate
          </button>
        </div>
        <div class="regenPanel" data-open="false">
          <input class="input" placeholder="Optional steer — e.g. lead with the Airflow work"
                 aria-label="Regeneration instruction for ${display}" />
          <button class="btn btnGhost" data-act="run-regen" type="button">Run</button>
        </div>
        <div class="bulletList"></div>`;

      const list_ = block.querySelector(".bulletList");
      list.forEach((bullet, i) => {
        const row = document.createElement("div");
        row.className = "bulletRow";
        row.innerHTML = `
          <textarea class="textarea" data-company="${company}" data-idx="${i}"
                    aria-label="${display} bullet ${i + 1}"></textarea>
          <div class="meter">
            <span class="meterTrack">
              <span class="meterFill"></span>
              <span class="meterTick" style="left:${((MIN / MAX) * 100).toFixed(1)}%"></span>
            </span>
            <span class="meterLabel"></span>
          </div>`;
        const ta = row.querySelector("textarea");
        ta.value = bullet;
        paintMeter(row, bullet);
        ta.addEventListener("input", () => {
          paintMeter(row, ta.value);
          renderCombined();
          updateActionBar();
        });
        list_.appendChild(row);
      });

      const panel = block.querySelector(".regenPanel");
      block.querySelector('[data-act="toggle-regen"]').addEventListener("click", () => {
        const open = panel.dataset.open !== "true";
        panel.dataset.open = String(open);
        if (open) panel.querySelector("input").focus();
      });
      block.querySelector('[data-act="run-regen"]').addEventListener("click", (e) =>
        regenerate(company, panel.querySelector("input").value, e.currentTarget)
      );

      editor.appendChild(block);
    });

    renderChips(
      $("courseChips"),
      Array.from(new Set([...state.status.default_courses, ...state.selectedCourses])),
      state.selectedCourses,
      (next) => (state.selectedCourses = next)
    );
    $("courseCaption").textContent =
      `Shown under Education. The model picked ${state.selectedCourses.length} for this posting — add or drop any.`;

    const topics = state.academicProjects
      .map((p) => String(p.Topic || "").trim())
      .filter(Boolean);
    renderChips(
      $("projectChips"),
      Array.from(new Set([...topics, ...state.selectedTopics])),
      state.selectedTopics,
      (next) => (state.selectedTopics = next)
    );
    $("projectCaption").textContent =
      `Shown under Academic Projects. The model picked ${state.selectedTopics.length} — swap in any of your others.`;

    renderCombined();
  }

  function renderChips(wrap, options, selected, onChange) {
    wrap.innerHTML = "";
    options.forEach((name) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.setAttribute("aria-pressed", String(selected.includes(name)));
      chip.innerHTML = `<span class="tick">${icon("check", 13)}</span><span>${name}</span>`;
      chip.addEventListener("click", () => {
        const on = chip.getAttribute("aria-pressed") === "true";
        chip.setAttribute("aria-pressed", String(!on));
        const next = on ? selected.filter((x) => x !== name) : [...selected, name];
        selected = next;
        onChange(next);
      });
      wrap.appendChild(chip);
    });
  }

  async function regenerate(company, instruction, btn) {
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.textContent = "Working…";
    try {
      const others = bulletsFromDom();
      delete others[company];
      const result = await api("/api/regenerate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company,
          jd_text: state.jdText,
          instruction,
          model: $("modelSelect").value,
          other_bullets: others,
          seniority: state.jdSignals.seniority || "",
        }),
      });
      state.bullets = { ...bulletsFromDom(), [company]: result.bullets };
      renderReview();
      updateActionBar();
    } catch (err) {
      setStatus($("compileStatus"), `Regeneration failed: ${err.message}`, "err");
    } finally {
      btn.disabled = false;
      btn.innerHTML = original;
    }
  }

  function renderCombined() {
    const wrap = $("combinedBlocks");
    wrap.innerHTML = "";
    Object.entries(bulletsFromDom()).forEach(([company, list]) => {
      const text = list.filter((b) => b.trim()).map((b) => `• ${b.trim()}`).join("\n");
      const block = document.createElement("div");
      block.className = "copyBlock";
      block.innerHTML = `
        <div class="copyHead">
          <span class="name">${titleCase(company)}</span>
          <button class="btn btnQuiet" type="button">${icon("copy", 14)} Copy</button>
        </div>
        <pre class="copyBody"></pre>`;
      block.querySelector(".copyBody").textContent = text;

      const btn = block.querySelector("button");
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(text);
        } catch (_) {
          const ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          ta.remove();
        }
        btn.innerHTML = `${icon("check", 14)} Copied`;
        setTimeout(() => (btn.innerHTML = `${icon("copy", 14)} Copy`), 1600);
      });
      wrap.appendChild(block);
    });
  }

  // ── Action bar ───────────────────────────────────────────────────────────

  function updateActionBar() {
    const bar = $("actionBar");
    if (!state.bullets) {
      bar.dataset.show = "false";
      return;
    }
    const b = bulletsFromDom();
    const roles = Object.keys(b).length;
    const total = Object.values(b).reduce((a, l) => a + l.length, 0);
    const { min_bullet_chars: MIN, max_bullet_chars: MAX } = state.status;
    const off = Object.values(b)
      .flat()
      .filter((t) => t.length < MIN || t.length > MAX).length;

    $("actionSummary").innerHTML =
      `<strong>${total}</strong> bullets across <strong>${roles}</strong> roles` +
      (off ? ` · <span style="color:var(--warn)">${off} outside the length band</span>` : "");
    bar.dataset.show = "true";
  }

  $("barCompileBtn").addEventListener("click", () => $("compileBtn").click());
  $("barReviewBtn").addEventListener("click", () => {
    setCard("card-review", { open: true });
    setCard("card-review", { summary: "" });
    $("card-review").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // ── Compile ──────────────────────────────────────────────────────────────

  $("compileBtn").addEventListener("click", async () => {
    const btn = $("compileBtn");
    btn.disabled = true;
    $("barCompileBtn").disabled = true;
    setStatus($("compileStatus"), "Injecting bullets, compiling, fitting to one page…", "busy");

    try {
      const result = await api("/api/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company_name: state.companyName,
          position_name: state.positionName,
          jd_text: state.jdText,
          model: $("modelSelect").value,
          bullets: bulletsFromDom(),
          selected_courses: state.selectedCourses,
          selected_academic_topics: state.selectedTopics,
          jd_signals: state.jdSignals,
        }),
      });

      state.compiled = result;
      state.stage = 3;
      setStatus($("compileStatus"), "Compiled to a single page.", "ok");

      $("reportDetails").hidden = false;
      $("reportDetails").open = !result.qa_report_clean;
      $("reportCode").textContent = result.qa_report_text;
      const badge = $("reportBadge");
      badge.textContent = result.qa_report_clean ? "clean" : "issues found";
      badge.dataset.issues = String(!result.qa_report_clean);

      const pdfUrl = `data:application/pdf;base64,${result.pdf_base64}`;
      $("pdfPreview").src = pdfUrl;
      $("downloadPdf").href = pdfUrl;
      $("downloadPdf").download = result.pdf_name;
      $("downloadTex").href = URL.createObjectURL(
        new Blob([result.tex_text], { type: "text/plain" })
      );
      $("downloadTex").download = result.tex_name;
      $("previewBlock").hidden = false;
      $("previewBlock").classList.add("reveal");

      // Reading the PDF is the point now, so give it the screen.
      setCard("card-review", { open: false });
      setCard("card-review", { summary: cardSummary("card-review") });
      renderStepper();
      $("card-export").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setStatus($("compileStatus"), `Compilation failed: ${err.message}`, "err");
    } finally {
      btn.disabled = false;
      $("barCompileBtn").disabled = false;
    }
  });

  // ── Init ─────────────────────────────────────────────────────────────────

  (async function init() {
    renderStepper();
    await Promise.all([loadStatus(), loadJdDefault(), loadAcademicProjects()]);
    updateReadiness();
  })();

  // ── Cover letter ─────────────────────────────────────────────────────────
  // Two calls, deliberately: /api/cover-letter drafts and judges, and only
  // /api/compile-cover-letter produces a file. The judgement is shown either
  // way — a refused draft is displayed so you can see what was wrong with it,
  // never so it can be sent.

  function coverParagraphsFromDom() {
    return Array.from(document.querySelectorAll("textarea[data-cover-para]"))
      .map((a) => a.value.trim())
      .filter(Boolean);
  }

  function renderCoverIssues(issues) {
    const wrap = $("coverIssues");
    const list = $("coverIssueList");
    list.innerHTML = "";
    if (!issues || !issues.length) {
      wrap.hidden = true;
      return;
    }
    issues.forEach((issue) => {
      const li = document.createElement("li");
      li.textContent = issue;
      list.appendChild(li);
    });
    wrap.hidden = false;
  }

  function renderCoverWords() {
    const words = coverParagraphsFromDom().join(" ").split(/\s+/).filter(Boolean).length;
    const limits = state.coverLimits || {};
    const badge = $("coverWords");
    badge.textContent = `${words} words`;
    const under = limits.min_words && words < limits.min_words;
    const over = limits.max_words && words > limits.max_words;
    badge.title = under
      ? `Below the ${limits.min_words}-word minimum`
      : over
        ? `Above the ${limits.max_words}-word maximum`
        : "";
    badge.dataset.state = under || over ? "warn" : "ok";
  }

  function renderCoverDraft(paragraphs) {
    const wrap = $("coverParagraphs");
    wrap.innerHTML = "";
    paragraphs.forEach((text, i) => {
      const area = document.createElement("textarea");
      area.className = "textarea";
      area.rows = Math.max(4, Math.ceil(text.length / 95) + 1);
      area.value = text;
      area.dataset.coverPara = String(i);
      area.style.marginBottom = "10px";
      area.addEventListener("input", renderCoverWords);
      wrap.appendChild(area);
    });
    $("coverEditor").hidden = paragraphs.length === 0;
    renderCoverWords();
  }

  $("coverBtn").addEventListener("click", async () => {
    const btn = $("coverBtn");
    btn.disabled = true;
    setStatus($("coverStatus"), "Drafting from your evidence…", null);
    $("coverPreviewBlock").hidden = true;
    setStatus($("coverCompileStatus"), "", null);
    try {
      const result = await api("/api/cover-letter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company_name: $("companyName").value.trim(),
          position_name: $("positionName").value.trim(),
          jd_text: $("jdText").value.trim(),
          model: $("modelSelect").value,
          log_prompts: $("logPrompts").checked,
        }),
      });
      state.coverLimits = result.limits || {};
      renderCoverDraft(result.paragraphs || []);
      renderCoverIssues(result.issues);
      setStatus(
        $("coverStatus"),
        result.usable
          ? "Draft passed every check. Read it before you send it."
          : "This draft was refused — see the reasons below. Fix them and it will compile.",
        result.usable ? "ok" : "err"
      );
      $("coverCompileBtn").disabled = false;
    } catch (err) {
      setStatus($("coverStatus"), `Drafting failed: ${err.message}`, "err");
    } finally {
      btn.disabled = false;
    }
  });

  $("coverCompileBtn").addEventListener("click", async () => {
    const btn = $("coverCompileBtn");
    const paragraphs = coverParagraphsFromDom();
    if (!paragraphs.length) {
      setStatus($("coverCompileStatus"), "Nothing to compile.", "err");
      return;
    }
    btn.disabled = true;
    setStatus($("coverCompileStatus"), "Compiling…", null);
    try {
      const result = await api("/api/compile-cover-letter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company_name: $("companyName").value.trim(),
          position_name: $("positionName").value.trim(),
          paragraphs,
          jd_text: $("jdText").value.trim(),
        }),
      });
      const url = `data:application/pdf;base64,${result.pdf_base64}`;
      $("coverPreview").src = url;
      const link = $("coverDownload");
      link.href = url;
      link.setAttribute("download", result.pdf_name);
      $("coverPreviewBlock").hidden = false;
      renderCoverIssues([]);
      const warning = result.report && result.report.warning;
      setStatus(
        $("coverCompileStatus"),
        warning || `Compiled — ${result.report.pages} page, ${result.report.word_count} words.`,
        warning ? "err" : "ok"
      );
    } catch (err) {
      // A 422 here is the server refusing an edited draft, which is the check
      // doing its job — show the reason verbatim rather than "compile failed".
      setStatus($("coverCompileStatus"), err.message, "err");
    } finally {
      btn.disabled = false;
    }
  });

})();
