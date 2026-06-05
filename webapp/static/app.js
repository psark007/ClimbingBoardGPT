/*
 * Browser-side controller for the ClimbingBoardGPT demo.
 *
 * The server returns route tokens, board coordinates, and canvas metadata as
 * JSON. This file keeps the current UI state, draws holds as SVG markers over
 * the board image, and serializes clicked holds back into frames strings for
 * grade prediction.
 */
const state = {
  boards: {},
  boardHolds: {},
  activeBoard: "tb2",
  lastResult: null,
  builder: [],
};

// Marker styles are expressed in board-coordinate units because the SVG viewBox
// is calibrated to the same coordinate system as token_metadata.csv.
const roleStyle = {
  start: { fill: "#69f0ae", stroke: "#102022", r: 1.55, shape: "circle" },
  middle: { fill: "#82b1ff", stroke: "#102022", r: 1.55, shape: "circle" },
  finish: { fill: "#ff8a80", stroke: "#102022", r: 2.25, shape: "star" },
  foot: { fill: "#ffd740", stroke: "#102022", r: 1.1, shape: "square" },
  unknown: { fill: "#b0bec5", stroke: "#102022", r: 1.45, shape: "circle" },
};

// The builder uses brighter styling than generated routes so edited holds stand
// out after a user clicks or removes placements.
const builderStyle = {
  start: { fill: "#00e676", stroke: "#000000", r: 1.85, shape: "circle" },
  middle: { fill: "#448aff", stroke: "#000000", r: 1.85, shape: "circle" },
  finish: { fill: "#ff5252", stroke: "#000000", r: 2.6, shape: "star" },
  foot: { fill: "#ffea00", stroke: "#000000", r: 1.25, shape: "square" },
  unknown: { fill: "#cfd8dc", stroke: "#000000", r: 1.85, shape: "circle" },
};

/** Return an element by ID. */
function $(id) {
  return document.getElementById(id);
}

/** Fetch JSON and normalize FastAPI/Pydantic errors into readable exceptions. */
async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { detail: text };
  }
  if (!response.ok) {
    const detail = payload.detail ?? payload;
    let message;
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail.map((item) => item.msg || JSON.stringify(item)).join("\n");
    } else {
      message = JSON.stringify(detail, null, 2);
    }
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return payload;
}

/** Toggle a button's disabled state and working label. */
function setBusy(button, busy) {
  button.disabled = busy;
  button.textContent = busy ? "Working…" : button.dataset.label;
}

/** Load and cache the clickable hold coordinate list for a board. */
async function ensureBoardHolds(boardKey) {
  if (!state.boardHolds[boardKey]) {
    const payload = await fetchJson(`/api/board-holds/${boardKey}`);
    state.boardHolds[boardKey] = payload.holds || [];
  }
  return state.boardHolds[boardKey];
}

/** Switch the displayed board image and overlay coordinate system. */
async function setBoardBackground(boardKey) {
  const board = state.boards[boardKey];
  if (!board) return;

  state.activeBoard = boardKey;
  syncAngleSelectors();
  syncGradeSelector();
  const bg = $("board-bg");
  if (bg.getAttribute("src") !== board.background_url) {
    bg.src = board.background_url;
  }

  const svg = $("overlay");
  const c = board.canvas;
  svg.setAttribute("viewBox", `${c.x_min} ${c.y_min} ${c.width} ${c.height}`);

  await ensureBoardHolds(boardKey);
  redrawCurrentOverlay();
}



/** Pick a sensible angle for the active board when the current value is absent. */
function preferredAngleForBoard(boardKey, currentValue = null) {
  const board = state.boards[boardKey] || {};
  const angles = board.available_angles || [];
  const current = Number(currentValue);

  if (angles.includes(current)) return current;
  if (angles.includes(40)) return 40;
  if (angles.length > 0) return angles[Math.floor(angles.length / 2)];
  return 40;
}

/** Rebuild an angle select using angles available in the processed dataset. */
function populateAngleSelect(selectId, boardKey, selectedValue = null) {
  const select = $(selectId);
  if (!select) return;

  const board = state.boards[boardKey] || {};
  const angles = board.available_angles || [20, 25, 30, 35, 40, 45, 50];
  const selected = preferredAngleForBoard(boardKey, selectedValue ?? select.value);

  select.innerHTML = "";
  for (const angle of angles) {
    const option = document.createElement("option");
    option.value = String(angle);
    option.textContent = `${angle}°`;
    if (Number(angle) === Number(selected)) {
      option.selected = true;
    }
    select.appendChild(option);
  }
}

/** Keep generation and prediction angle selectors in sync. */
function syncAngleSelectors(angleValue = null) {
  const boardKey = state.activeBoard || $("gen-board")?.value || $("pred-board")?.value || "tb2";
  const selected = preferredAngleForBoard(boardKey, angleValue ?? $("gen-angle")?.value ?? $("pred-angle")?.value);

  populateAngleSelect("gen-angle", boardKey, selected);
  populateAngleSelect("pred-angle", boardKey, selected);

  const genAngle = $("gen-angle");
  const predAngle = $("pred-angle");
  if (genAngle) genAngle.value = String(selected);
  if (predAngle) predAngle.value = String(selected);
}


/** Pick a sensible grade for the active board when the current value is absent. */
function preferredGradeForBoard(boardKey, currentValue = null) {
  const board = state.boards[boardKey] || {};
  const grades = board.available_grades || Array.from({ length: 16 }, (_, i) => i);
  const current = Number(currentValue);

  if (grades.includes(current)) return current;
  if (grades.includes(6)) return 6;
  if (grades.length > 0) return grades[Math.floor(grades.length / 2)];
  return 6;
}

/** Rebuild a grade select using grades available in the processed dataset. */
function populateGradeSelect(selectId, boardKey, selectedValue = null) {
  const select = $(selectId);
  if (!select) return;

  const board = state.boards[boardKey] || {};
  const grades = board.available_grades || Array.from({ length: 16 }, (_, i) => i);
  const selected = preferredGradeForBoard(boardKey, selectedValue ?? select.value);

  select.innerHTML = "";
  for (const grade of grades) {
    const option = document.createElement("option");
    option.value = String(grade);
    option.textContent = `V${grade}`;
    if (Number(grade) === Number(selected)) {
      option.selected = true;
    }
    select.appendChild(option);
  }
}

/** Keep the generation grade selector valid for the active board. */
function syncGradeSelector(gradeValue = null) {
  const boardKey = state.activeBoard || $("gen-board")?.value || "tb2";
  const selected = preferredGradeForBoard(boardKey, gradeValue ?? $("gen-grade")?.value);
  populateGradeSelect("gen-grade", boardKey, selected);

  const genGrade = $("gen-grade");
  if (genGrade) genGrade.value = String(selected);
}

/** Mirror the active board between generation and prediction controls. */
function syncBoardSelectors(boardKey) {
  const genBoard = $("gen-board");
  const predBoard = $("pred-board");
  if (genBoard && genBoard.value !== boardKey) genBoard.value = boardKey;
  if (predBoard && predBoard.value !== boardKey) predBoard.value = boardKey;
}

/** Replace the editable builder state with holds from an API result. */
function setBuilderFromRouteResult(result) {
  const board = result.board_key;
  state.builder = (result.holds || []).map((hold) => ({
    board,
    placement_id: hold.placement_id,
    x: hold.x,
    y: hold.y,
    role: hold.role,
  }));
  syncBuilderToFrames();
}

/** Summarize the active board's edited route for headers and validation hints. */
function activeBuilderSummary() {
  const board = $("pred-board")?.value || state.activeBoard;
  const holds = state.builder.filter((hold) => hold.board === board);
  const starts = holds.filter((hold) => hold.role === "start").length;
  const finishes = holds.filter((hold) => hold.role === "finish").length;
  return { holds, starts, finishes };
}

/** Show an editable-route status message while the user is clicking holds. */
function updateEditingHeader(prefix = "Editing climb") {
  const summary = activeBuilderSummary();
  const frameText = $("pred-frames")?.value?.trim() || "";
  $("result-title").textContent = prefix;
  $("result-subtitle").textContent =
    `${summary.holds.length} selected holds | starts=${summary.starts} | finishes=${summary.finishes} | editing`;
  $("raw-json").textContent = JSON.stringify({
    mode: "editable_climb",
    board: $("pred-board")?.value || state.activeBoard,
    n_holds: summary.holds.length,
    starts: summary.starts,
    finishes: summary.finishes,
    frames: frameText,
    note: "Run grade prediction to refresh known-climb status after edits."
  }, null, 2);
}

/** Remove all SVG children from the overlay. */
function clearOverlay() {
  $("overlay").innerHTML = "";
}

/** Create an SVG element with a simple attribute object. */
function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

/** Return polygon points for the finish-hold star marker. */
function starPoints(cx, cy, outerR, innerR, n = 5) {
  const points = [];
  for (let i = 0; i < n * 2; i++) {
    const angle = -Math.PI / 2 + (i * Math.PI) / n;
    const r = i % 2 === 0 ? outerR : innerR;
    points.push(`${cx + Math.cos(angle) * r},${cy + Math.sin(angle) * r}`);
  }
  return points.join(" ");
}

/** Draw one route or builder marker into a transformed SVG group. */
function drawMarker(group, hold, style, className = "route-marker") {
  if (style.shape === "square") {
    const s = style.r * 2;
    group.appendChild(svgEl("rect", {
      x: hold.x - style.r,
      y: hold.y - style.r,
      width: s,
      height: s,
      rx: 0.28,
      fill: style.fill,
      stroke: style.stroke,
      "stroke-width": 0.35,
      opacity: 0.96,
      class: className,
    }));
  } else if (style.shape === "star") {
    group.appendChild(svgEl("polygon", {
      points: starPoints(hold.x, hold.y, style.r, style.r * 0.48),
      fill: style.fill,
      stroke: style.stroke,
      "stroke-width": 0.38,
      opacity: 0.98,
      class: className,
    }));
  } else {
    group.appendChild(svgEl("circle", {
      cx: hold.x,
      cy: hold.y,
      r: style.r,
      fill: style.fill,
      stroke: style.stroke,
      "stroke-width": 0.35,
      opacity: 0.96,
      class: className,
    }));
  }
}

/** Return a group that flips board coordinates into image/SVG screen space. */
function transformedGroup(resultOrBoardKey) {
  const boardKey = typeof resultOrBoardKey === "string" ? resultOrBoardKey : resultOrBoardKey.board_key;
  const board = state.boards[boardKey];
  const c = board.canvas;
  return svgEl("g", {
    transform: `translate(0 ${c.y_min + c.y_max}) scale(1 -1)`,
  });
}

/** Draw transparent click targets over every known hold on the active board. */
function drawSelectableTargets(boardKey, group) {
  const holds = state.boardHolds[boardKey] || [];
  for (const hold of holds) {
    const target = svgEl("circle", {
      cx: hold.x,
      cy: hold.y,
      r: 2.45,
      fill: "transparent",
      stroke: "transparent",
      "stroke-width": 0,
      class: "click-target",
    });

    target.addEventListener("mouseenter", () => {
      const hover = svgEl("circle", {
        cx: hold.x,
        cy: hold.y,
        r: 2.15,
        class: "hover-marker",
        id: "hover-marker",
      });
      group.appendChild(hover);
    });

    target.addEventListener("mouseleave", () => {
      const hover = document.getElementById("hover-marker");
      if (hover) hover.remove();
    });

    target.addEventListener("click", (event) => {
      event.stopPropagation();
      addBuilderHold(hold);
    });

    group.appendChild(target);
  }
}

/** Render an API result and make it editable through the route builder. */
function drawOverlay(result) {
  state.lastResult = result;
  syncBoardSelectors(result.board_key);
  state.activeBoard = result.board_key;
  setBuilderFromRouteResult(result);
  redrawCurrentOverlay();
}

/** Redraw builder markers and click targets from the current state. */
function redrawCurrentOverlay() {
  const svg = $("overlay");
  svg.innerHTML = "";
  const boardKey = state.activeBoard || $("pred-board")?.value || "tb2";

  const group = transformedGroup(boardKey);
  svg.appendChild(group);

  for (const hold of state.builder) {
    if (hold.board !== boardKey) continue;
    const style = builderStyle[hold.role] || builderStyle.unknown;
    drawMarker(group, hold, style, "builder-marker");
  }

  drawSelectableTargets(boardKey, group);
}



/** Find the selected builder hold index for a board placement. */
function findBuilderHoldIndex(board, placementId) {
  return state.builder.findIndex(
    (hold) => hold.board === board && Number(hold.placement_id) === Number(placementId)
  );
}

/** Return true when a board placement is already present in the builder. */
function isBuilderHoldSelected(board, placementId) {
  return findBuilderHoldIndex(board, placementId) >= 0;
}

/** Return builder holds belonging to the currently displayed board. */
function activeBuilderHolds() {
  return state.builder.filter((hold) => hold.board === state.activeBoard);
}

/** Count a role among builder holds on the currently displayed board. */
function roleCountForActiveBoard(role) {
  return activeBuilderHolds().filter((hold) => hold.role === role).length;
}

/** Enforce the same start/finish limits used by the web API. */
function canAddBuilderRole(role) {
  if ((role === "start" || role === "finish") && roleCountForActiveBoard(role) >= 2) {
    showWarnings([
      `You already have two ${role} holds. A climb can have at most two ${role} holds in this builder.`
    ]);
    return false;
  }
  return true;
}

/** Convert the builder route into a frames string. */
function frameStringForBuilder() {
  const board = state.boards[$("pred-board").value];
  if (!board) return "";
  const roleDefinitions = board.role_definitions || {};
  return state.builder
    .filter((hold) => hold.board === $("pred-board").value)
    .map((hold) => `p${hold.placement_id}r${roleDefinitions[hold.role]}`)
    .join("");
}

/** Update the frames textarea and selected-hold list from builder state. */
function syncBuilderToFrames() {
  $("pred-frames").value = frameStringForBuilder();
  const list = $("builder-list");
  list.innerHTML = "";

  const active = state.builder.filter((hold) => hold.board === $("pred-board").value);
  for (const [idx, hold] of active.entries()) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${idx + 1}. ${hold.role}</span><span>p${hold.placement_id}</span>`;
    list.appendChild(li);
  }
  updatePredictButton();
}

/** Add or remove a clicked hold, using the currently selected semantic role. */
function addBuilderHold(hold) {
  const board = $("pred-board").value;
  const role = $("click-role").value;
  if (board !== state.activeBoard) {
    $("pred-board").value = state.activeBoard;
  }

  const existingIndex = findBuilderHoldIndex(state.activeBoard, hold.placement_id);
  if (existingIndex >= 0) {
    state.builder.splice(existingIndex, 1);
    showWarnings([]);
    syncBuilderToFrames();
    redrawCurrentOverlay();
    updateEditingHeader("Editing generated / clicked climb");
    return;
  }

  if (!canAddBuilderRole(role)) {
    return;
  }

  state.builder.push({
    board: state.activeBoard,
    placement_id: hold.placement_id,
    x: hold.x,
    y: hold.y,
    role,
  });

  showWarnings([]);
  syncBuilderToFrames();
  redrawCurrentOverlay();
  updateEditingHeader("Editing generated / clicked climb");
}

/** Remove the most recently added hold for the selected prediction board. */
function undoBuilderHold() {
  const board = $("pred-board").value;
  for (let i = state.builder.length - 1; i >= 0; i--) {
    if (state.builder[i].board === board) {
      state.builder.splice(i, 1);
      break;
    }
  }
  syncBuilderToFrames();
  redrawCurrentOverlay();
  updateEditingHeader("Editing generated / clicked climb");
}

/** Backward-compatible clear handler retained for older markup. */
function clearBuilder() {
  clearEntireBoard();
}


/** Show warning text in the dedicated box, falling back to the subtitle. */
function showWarnings(warnings = []) {
  const box = $("warning-box");
  const normalized = (warnings || []).filter(Boolean).map(String);

  if (!box) {
    if (normalized.length > 0) {
      $("result-subtitle").textContent = normalized.join(" | ");
    }
    return;
  }

  if (normalized.length === 0) {
    box.hidden = true;
    box.textContent = "";
    return;
  }

  box.hidden = false;
  box.textContent = normalized.join("\n");
}


/** Clear builder state without changing result headers or overlay background. */
function clearClickedHoldsOnly() {
  state.builder = [];
  const frames = $("pred-frames");
  if (frames) frames.value = "";
  const list = $("builder-list");
  if (list) list.innerHTML = "";
  updatePredictButton();
}

/** Public clear action for both clear buttons. */
function clearBoard() {
  clearEntireBoard();
}


/** Enable prediction only when a frames string is available. */
function updatePredictButton() {
  const button = $("predict-btn");
  if (!button) return;

  const frames = $("pred-frames").value.trim();
  const hasFrames = frames.length > 0;
  button.disabled = !hasFrames;
  button.title = hasFrames
    ? ""
    : "Paste a frames string or click holds on the board first.";
}

/** Reset route, raw JSON, warnings, and overlay markers for the current board. */
function clearEntireBoard() {
  state.lastResult = null;
  state.builder = [];
  $("pred-frames").value = "";
  const list = $("builder-list");
  if (list) list.innerHTML = "";
  clearOverlay();
  showWarnings([]);
  $("result-title").textContent = "Board cleared";
  $("result-subtitle").textContent = "Click holds to build a climb, paste a frames string, or generate a new one.";
  $("raw-json").textContent = "{}";
  updatePredictButton();
  redrawCurrentOverlay();
}


/** Format exact-match known-climb metadata for the result subtitle. */
function knownClimbSummary(result) {
  const known = result.known_climb;
  if (!known || !known.checked) return null;
  if (!known.is_known) return "not in known-climb database";

  const first = (known.examples || [])[0] || {};
  const name = first.climb_name || "unnamed climb";
  const grade = first.boulder_grade || (first.grouped_v !== undefined && first.grouped_v !== null ? `V${first.grouped_v}` : "unknown grade");
  return `known climb: ${name} (${grade}); ${known.match_count} matching route-angle entr${known.match_count === 1 ? "y" : "ies"}`;
}

/** Render human-readable result text and raw JSON for an API response. */
function summarizeResult(result, mode) {
  if (mode === "generate") {
    const pieces = [
      `${result.board_display_name}`,
      `requested V${result.requested_grouped_v}`,
      `${result.requested_angle}°`,
    ];

    if (result.predicted_grouped_v !== undefined) {
      pieces.push(`predicted V${result.predicted_grouped_v} (${Number(result.predicted_display_difficulty).toFixed(2)})`);
      if (result.critic_v_error !== undefined) {
        const sign = Number(result.critic_v_error) >= 0 ? "+" : "";
        pieces.push(`error ${sign}${result.critic_v_error}V`);
      }
    }

    const knownSummary = knownClimbSummary(result);
    if (knownSummary) pieces.push(knownSummary);
    $("result-title").textContent = "Generated climb";
    $("result-subtitle").textContent = pieces.join(" | ");
  } else {
    const knownSummary = knownClimbSummary(result);
    $("result-title").textContent = "Grade prediction";
    $("result-subtitle").textContent =
      `${result.board_display_name} | ${result.requested_angle}° | predicted V${result.predicted_grouped_v} (${Number(result.predicted_display_difficulty).toFixed(2)})`
      + (knownSummary ? ` | ${knownSummary}` : "");
  }

  showWarnings(result.warnings || []);
  $("raw-json").textContent = JSON.stringify(result, null, 2);
}


/** Convert structured API error JSON into a compact warning message. */
function formatErrorMessage(message) {
  if (!message) return "Unknown error.";
  try {
    const parsed = JSON.parse(message);
    if (parsed.message && Array.isArray(parsed.reasons)) {
      return `${parsed.message}\n- ${parsed.reasons.join("\n- ")}`;
    }
    if (Array.isArray(parsed.reasons)) {
      return parsed.reasons.map((r) => `- ${r}`).join("\n");
    }
    return JSON.stringify(parsed, null, 2);
  } catch {
    return message;
  }
}

/** Submit a generation request and render the sampled route. */
async function generate() {
  const button = $("generate-btn");
  setBusy(button, true);
  try {
    const payload = {
      board: $("gen-board").value,
      angle: Number($("gen-angle").value),
      grade: Number($("gen-grade").value),
      temperature: Number($("gen-temperature").value),
      top_k: 50,
      max_new_tokens: 40,
    };
    const result = await fetchJson("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    drawOverlay(result);
    summarizeResult(result, "generate");
  } catch (err) {
    showWarnings([formatErrorMessage(err.message)]);
  } finally {
    setBusy(button, false);
  }
}

/** Submit a grade-prediction request for the current frames string. */
async function predict() {
  const button = $("predict-btn");
  const frames = $("pred-frames").value.trim();
  if (!frames) {
    showWarnings(["Paste a frames string or click holds on the board before predicting."]);
    updatePredictButton();
    return;
  }

  setBusy(button, true);
  try {
    const payload = {
      board: $("pred-board").value,
      angle: Number($("pred-angle").value),
      frames,
    };
    const result = await fetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    drawOverlay(result);
    summarizeResult(result, "predict");
  } catch (err) {
    showWarnings([formatErrorMessage(err.message)]);
  } finally {
    setBusy(button, false);
    updatePredictButton();
  }
}


/** Initialize collapsible cards: clicking the h2 toggles the .collapsed class. */
function initCollapsibleCards() {
  document.querySelectorAll(".card.collapsible > h2").forEach((heading) => {
    const card = heading.parentElement;
    heading.addEventListener("click", () => {
      card.classList.toggle("collapsed");
    });
  });
}

/** Attach a listener only when optional markup exists. */
function addListenerIfPresent(id, eventName, handler) {
  const element = $(id);
  if (!element) {
    console.warn(`Missing optional element #${id}; skipping ${eventName} listener.`);
    return;
  }
  element.addEventListener(eventName, handler);
}

/** Initialize server state, controls, board background, and event listeners. */
async function init() {
  $("generate-btn").dataset.label = "Generate";
  $("predict-btn").dataset.label = "Predict pasted / clicked climb";

  const health = await fetchJson("/api/health");
  $("health").textContent =
    `device=${health.device} | generator=${health.generator_loaded ? "loaded" : "missing"} | grade=${health.grade_predictor_loaded ? "loaded" : "missing"}`;

  state.boards = await fetchJson("/api/boards");
  syncBoardSelectors("tb2");
  syncAngleSelectors(40);
  syncGradeSelector(6);
  await ensureBoardHolds("tb2");
  await setBoardBackground("tb2");
  syncAngleSelectors(40);

  initCollapsibleCards();

  addListenerIfPresent("gen-board", "change", async (e) => {
    syncBoardSelectors(e.target.value);
    await setBoardBackground(e.target.value);
    syncGradeSelector();
    syncBuilderToFrames();
    updateEditingHeader("Editing climb");
  });

  addListenerIfPresent("pred-board", "change", async (e) => {
    syncBoardSelectors(e.target.value);
    await setBoardBackground(e.target.value);
    syncGradeSelector();
    syncBuilderToFrames();
    updateEditingHeader("Editing climb");
  });

  addListenerIfPresent("gen-angle", "change", (e) => {
    syncAngleSelectors(e.target.value);
  });
  addListenerIfPresent("pred-angle", "change", (e) => {
    syncAngleSelectors(e.target.value);
  });
  addListenerIfPresent("gen-grade", "change", (e) => {
    syncGradeSelector(e.target.value);
  });
  addListenerIfPresent("pred-frames", "input", updatePredictButton);
  addListenerIfPresent("generate-btn", "click", generate);
  addListenerIfPresent("predict-btn", "click", predict);
  addListenerIfPresent("undo-hold-btn", "click", undoBuilderHold);
  addListenerIfPresent("clear-holds-btn", "click", clearBoard);
  addListenerIfPresent("clear-board-btn", "click", clearBoard);
  updatePredictButton();
}

init().catch((err) => {
  const health = $("health");
  if (health) health.textContent = `Error: ${err.message}`;
  showWarnings([`Initialization error: ${err.message}`]);
  console.error(err);
});
