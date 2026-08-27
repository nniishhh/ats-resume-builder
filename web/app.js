(() => {
  "use strict";

  const API = "";
  const state = {
    status: null,
    bullets: null,
    selectedCourses: [],
    selectedTopics: [],
    academicProjects: [],
    jdSignals: {},
    companyName: "",
    positionName: "",
    jdText: "",
    stage: 0, // 0=input, 2=bullets ready, 3=compiled
    lastCompile: null,
  };

  const $ = (id) => document.getElementById(id);

  function titleCase(s) {
    return s.replace(/_/g, " ").replace(/\w\S*/g, (t) => t[0].toUpperCase() + t.slice(1));
  }

  async function api(path, opts) {
    const res = await fetch(API + path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  // ── Stepper ──────────────────────────────────────────────────────────────
  const STEPS = ["Job Description", "Generate", "Review & Polish", "Export"];
  function renderStepper() {
    const el = $("stepper");
    el.innerHTML = "";
    STEPS.forEach((label, i) => {
      const st = i < state.stage ? "done" : i === state.stage ? "active" : "";
      const marker = st === "done" ? "&#10003;" : String(i + 1);
      const step = document.createElement("div");
      step.className = "step " + st;
      step.innerHTML = `<span class="dot">${marker}</span><span class="label">${label}</span>`;
      el.appendChild(step);
      if (i < STEPS.length - 1) {
        const line = document.createElement("div");
        line.className = "stepLine " + (i < state.stage ? "done" : "");
        el.appendChild(line);
      }
    });
  }

  // ── Bootstrapping / sidebar ──────────────────────────────────────────────
  async function loadStatus() {
    const status = await api("/api/status");
    state.status = status;

    const modelSelect = $("modelSelect");
    modelSelect.innerHTML = "";
    status.model_choices.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelSelect.appendChild(opt);
    });
    modelSelect.value = status.default_model;

    $("modeSelect").value = status.default_generation_mode;

    const evEl = $("evidenceList");
    evEl.innerHTML = "";
    if (status.evidence_files.length === 0) {
      evEl.innerHTML = '<p class="caption">No work_*.json files found.</p>';
    } else {
      status.evidence_files.forEach((f) => {
        const chip = document.createElement("div");
        chip.className = "fileChip";
        chip.innerHTML = `<span>${titleCase(f.company)}</span><span class="count">${f.min}–${f.max}</span>`;
        evEl.appendChild(chip);
      });
    }

    const tmpl = $("templateStatus");
    if (status.template_exists) {
      tmpl.textContent = "data/main.tex ✓";
      tmpl.className = "templateStatus ok";
    } else {
      tmpl.textContent = "data/main.tex not found.";
      tmpl.className = "templateStatus bad";
    }
  }

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
      const d = await api("/api/academic-projects");
      state.academicProjects = d.projects || [];
    } catch (_) {
      state.academicProjects = [];
    }
  }

  // ── JD file upload ───────────────────────────────────────────────────────
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
        const body = rest.join("---").trim();
        header.trim().split("\n").forEach((line) => {
          const idx = line.indexOf(":");
          if (idx === -1) return;
          const k = line.slice(0, idx).trim();
          const v = line.slice(idx + 1).trim();
          if (k === "company_name") $("companyName").value = v;
          else if (k === "position_name") $("positionName").value = v;
        });
        $("jdText").value = body;
      } else {
        $("jdText").value = raw;
      }
    }
  });

  // ── Step 2: Generate ─────────────────────────────────────────────────────
  $("generateBtn").addEventListener("click", async () => {
    const companyName = $("companyName").value.trim();
    const positionName = $("positionName").value.trim();
    const jdText = $("jdText").value.trim();
    const statusEl = $("generateStatus");

    if (!companyName || !jdText) {
      statusEl.className = "statusMsg err";
      statusEl.textContent = "Company name and job description are required.";
      return;
    }
    if (!state.status || !state.status.evidence_files.length || !state.status.template_exists) {
      statusEl.className = "statusMsg err";
      statusEl.textContent = "Evidence files or data/main.tex are missing — check the sidebar.";
      return;
    }

    statusEl.className = "statusMsg loading";
    statusEl.textContent = "Analyzing JD and generating bullets — this takes ~30s...";
    $("generateBtn").disabled = true;

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

      state.bullets = result.bullets;
      state.selectedCourses = result.selected_courses || [];
      state.selectedTopics = result.selected_academic_topics || [];
      state.jdSignals = result.jd_signals || {};
      state.companyName = companyName;
      state.positionName = positionName;
      state.jdText = jdText;
      state.stage = 2;
      state.lastCompile = null;

      statusEl.className = "statusMsg ok";
      statusEl.textContent = `Generated bullets for ${Object.keys(result.bullets).length} companies.`;

      renderStepper();
      renderReview();
      $("reviewCard").hidden = false;
      $("exportCard").hidden = false;
      $("previewBlock").hidden = true;
      $("reportDetails").hidden = true;
      $("compileStatus").textContent = "";
    } catch (err) {
      statusEl.className = "statusMsg err";
      statusEl.textContent = "Generation failed: " + err.message;
    } finally {
      $("generateBtn").disabled = false;
    }
  });

  // ── Step 3: Review & Polish ──────────────────────────────────────────────
  function charBadge(n, min, max) {
    if (n >= min && n <= max) return `<span class="charBadge ok">${n} chars</span>`;
    if (n < min) return `<span class="charBadge under">${n} chars — ${min - n} under min</span>`;
    return `<span class="charBadge over">${n} chars — ${n - max} over max</span>`;
  }

  function currentBulletsFromDom() {
    const out = {};
    Object.keys(state.bullets).forEach((company) => {
      const areas = document.querySelectorAll(`textarea[data-company="${company}"][data-role="bullet"]`);
      out[company] = Array.from(areas).map((a) => a.value);
    });
    return out;
  }

  function renderReview() {
    const { min_bullet_chars: MIN, max_bullet_chars: MAX } = state.status;
    const editor = $("bulletEditor");
    editor.innerHTML = "";

    Object.entries(state.bullets).forEach(([company, bulletList]) => {
      const display = titleCase(company);
      const block = document.createElement("div");
      block.className = "companyBlock";

      const regenId = `regen_${company}`;
      block.innerHTML = `
        <h3 class="companyName">${display}</h3>
        <div class="regenRow">
          <input class="input" id="${regenId}" placeholder="Optional instruction — e.g. lead with the Airflow work; drop 'predictive'" />
          <button class="btn btnQuiet" data-action="regen" data-company="${company}">Regenerate</button>
        </div>
        <div class="bulletList" data-company="${company}"></div>
      `;
      editor.appendChild(block);

      const list = block.querySelector(".bulletList");
      bulletList.forEach((bullet, i) => {
        const row = document.createElement("div");
        row.className = "bulletRow";
        row.innerHTML = `
          <textarea class="textarea" data-company="${company}" data-role="bullet" data-idx="${i}"></textarea>
          <div class="badgeHolder"></div>
        `;
        const ta = row.querySelector("textarea");
        ta.value = bullet;
        const badgeHolder = row.querySelector(".badgeHolder");
        const updateBadge = () => {
          badgeHolder.innerHTML = charBadge(ta.value.length, MIN, MAX);
        };
        ta.addEventListener("input", () => {
          updateBadge();
          renderCombined();
        });
        updateBadge();
        list.appendChild(row);
      });
    });

    editor.querySelectorAll('[data-action="regen"]').forEach((btn) => {
      btn.addEventListener("click", () => regenerateCompany(btn.dataset.company));
    });

    renderCourseChips();
    renderProjectChips();
    renderCombined();
  }

  async function regenerateCompany(company) {
    const instrEl = $(`regen_${company}`);
    const btn = document.querySelector(`[data-action="regen"][data-company="${company}"]`);
    const instruction = instrEl.value;
    btn.disabled = true;
    btn.textContent = "Regenerating…";
    try {
      const otherBullets = currentBulletsFromDom();
      delete otherBullets[company];
      const result = await api("/api/regenerate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company,
          jd_text: state.jdText,
          instruction,
          model: $("modelSelect").value,
          other_bullets: otherBullets,
        }),
      });
      state.bullets[company] = result.bullets;
      renderReview();
    } catch (err) {
      alert(`Regeneration failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Regenerate";
    }
  }

  function renderCourseChips() {
    const wrap = $("courseChips");
    wrap.innerHTML = "";
    const options = Array.from(new Set([...state.status.default_courses, ...state.selectedCourses]));
    $("courseCaption").textContent =
      `Listed under Education. AI picked ~${state.status.default_top_course_count} for this role — add or drop any you like.`;
    options.forEach((course) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = course;
      chip.dataset.pressed = state.selectedCourses.includes(course);
      chip.addEventListener("click", () => {
        const on = chip.dataset.pressed === "true";
        chip.dataset.pressed = !on;
        if (on) {
          state.selectedCourses = state.selectedCourses.filter((c) => c !== course);
        } else {
          state.selectedCourses.push(course);
        }
      });
      wrap.appendChild(chip);
    });
  }

  function renderProjectChips() {
    const wrap = $("projectChips");
    wrap.innerHTML = "";
    const topics = state.academicProjects
      .map((p) => String(p.Topic || "").trim())
      .filter(Boolean);
    const options = Array.from(new Set([...topics, ...state.selectedTopics]));
    $("projectCaption").textContent =
      `Listed under Academic Projects. AI picked ~${state.status.default_top_academic_project_count} most relevant — swap in any of your projects.`;
    options.forEach((topic) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = topic;
      chip.dataset.pressed = state.selectedTopics.includes(topic);
      chip.addEventListener("click", () => {
        const on = chip.dataset.pressed === "true";
        chip.dataset.pressed = !on;
        if (on) {
          state.selectedTopics = state.selectedTopics.filter((t) => t !== topic);
        } else {
          state.selectedTopics.push(topic);
        }
      });
      wrap.appendChild(chip);
    });
  }

  function renderCombined() {
    const wrap = $("combinedBlocks");
    wrap.innerHTML = "";
    const bullets = currentBulletsFromDom();
    Object.entries(bullets).forEach(([company, list]) => {
      const display = titleCase(company);
      const combined = list.filter((b) => b.trim()).map((b) => `- ${b.trim()}`).join("\n");
      const block = document.createElement("div");
      block.className = "combinedBlock";
      block.innerHTML = `<p class="label">${display} — combined</p><textarea class="textarea" readonly></textarea>`;
      block.querySelector("textarea").value = combined;
      wrap.appendChild(block);
    });
  }

  // ── Step 4: Export ───────────────────────────────────────────────────────
  $("compileBtn").addEventListener("click", async () => {
    const statusEl = $("compileStatus");
    statusEl.className = "statusMsg loading";
    statusEl.textContent = "Injecting bullets and compiling LaTeX…";
    $("compileBtn").disabled = true;

    try {
      const bullets = currentBulletsFromDom();
      const result = await api("/api/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company_name: state.companyName,
          position_name: state.positionName,
          jd_text: state.jdText,
          model: $("modelSelect").value,
          bullets,
          selected_courses: state.selectedCourses,
          selected_academic_topics: state.selectedTopics,
          jd_signals: state.jdSignals,
        }),
      });

      state.lastCompile = result;
      state.stage = 3;
      renderStepper();

      statusEl.className = "statusMsg ok";
      statusEl.textContent = "Resume compiled successfully!";

      const reportDetails = $("reportDetails");
      const reportSummary = $("reportSummary");
      reportDetails.hidden = false;
      reportDetails.open = !result.qa_report_clean;
      reportSummary.textContent = "Build report" + (result.qa_report_clean ? "" : "  —  issues found");
      $("reportCode").textContent = result.qa_report_text;

      const pdfDataUrl = "data:application/pdf;base64," + result.pdf_base64;
      $("pdfPreview").src = pdfDataUrl;
      $("downloadPdf").href = pdfDataUrl;
      $("downloadPdf").download = result.pdf_name;

      const texBlob = new Blob([result.tex_text], { type: "text/plain" });
      $("downloadTex").href = URL.createObjectURL(texBlob);
      $("downloadTex").download = result.tex_name;

      $("previewBlock").hidden = false;
    } catch (err) {
      statusEl.className = "statusMsg err";
      statusEl.textContent = "Compilation failed: " + err.message;
    } finally {
      $("compileBtn").disabled = false;
    }
  });

  // ── Init ──────────────────────────────────────────────────────────────────
  async function init() {
    renderStepper();
    await Promise.all([loadStatus(), loadJdDefault(), loadAcademicProjects()]);
  }

  init();
})();
