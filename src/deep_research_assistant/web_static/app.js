const state = {
  conversations: [],
  current: null,
  busy: false,
  pollTimer: null,
  pollInFlight: false,
  eventSource: null,
  progressEvents: [],
  lastEventId: 0,
  liveReport: "",
  pendingDelete: null,
};

const processingStatuses = new Set(["queued", "running", "responding"]);

const elements = {
  history: document.querySelector("#history-list"),
  newChatPanel: document.querySelector("#new-chat-panel"),
  title: document.querySelector("#conversation-title"),
  status: document.querySelector("#conversation-status"),
  conversation: document.querySelector("#conversation-view"),
  messages: document.querySelector("#message-list"),
  composerWrap: document.querySelector("#composer-wrap"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  menu: document.querySelector("#menu-button"),
  sidebarClose: document.querySelector("#sidebar-close"),
  sidebarScrim: document.querySelector("#sidebar-scrim"),
  reportOverlay: document.querySelector("#report-overlay"),
  reportClose: document.querySelector("#report-close"),
  reportContent: document.querySelector("#report-content"),
  toast: document.querySelector("#toast"),
  deleteOverlay: document.querySelector("#delete-overlay"),
  deleteName: document.querySelector("#delete-conversation-name"),
  deleteCancel: document.querySelector("#delete-cancel"),
  deleteConfirm: document.querySelector("#delete-confirm"),
};

const statusLabels = {
  new: "等待输入",
  queued: "Run 排队中",
  running: "Agent 正在工作",
  waiting_for_clarification: "等待补充参数",
  waiting_for_outline_confirmation: "等待确认大纲",
  complete: "调研完成",
  responding: "正在回答追问",
  failed: "运行失败",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatTime(value) {
  if (!value) return "";
  const normalized = value.endsWith("Z") ? value : `${value.replace(" ", "T")}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return value;
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  return sameDay
    ? date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 3200);
}

function setBusy(busy, label = "Agent 正在分析并推进调研流程…") {
  state.busy = busy;
  elements.send.disabled = busy;
  elements.input.disabled = busy;
  renderMessages(label);
}

function adoptConversationPayload(payload) {
  state.current = payload;
  state.progressEvents = Array.isArray(payload?.events) ? payload.events : [];
  state.lastEventId = state.progressEvents.reduce(
    (latest, event) => Math.max(latest, Number(event.id) || 0),
    0,
  );
}

function autoResize() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
}

async function refreshConversationList() {
  state.conversations = await api("/api/conversations");
  renderHistory();
}

function renderHistory() {
  if (!state.conversations.length) {
    elements.history.innerHTML = '<div class="history-empty">尚无历史会话。<br>从一次新调研开始。</div>';
    return;
  }
  elements.history.innerHTML = state.conversations
    .map(
      (item) => `
        <div class="history-entry">
          <button class="history-item ${state.current?.conversation.id === item.id ? "active" : ""}" data-id="${escapeHtml(item.id)}">
            <span class="history-title">${escapeHtml(item.title)}</span>
            <span class="history-time">${escapeHtml(formatTime(item.updated_at))} · ${escapeHtml(statusLabels[item.status] || item.status)}</span>
          </button>
          <button class="history-delete" data-id="${escapeHtml(item.id)}" data-title="${escapeHtml(item.title)}" aria-label="删除 ${escapeHtml(item.title)}" ${state.busy ? "disabled" : ""}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" /></svg>
          </button>
        </div>`,
    )
    .join("");
  elements.history.querySelectorAll(".history-item").forEach((button) => {
    button.addEventListener("click", () => selectConversation(button.dataset.id));
  });
  elements.history.querySelectorAll(".history-delete").forEach((button) => {
    button.addEventListener("click", () => openDeleteDialog(button.dataset.id, button.dataset.title));
  });
}

function openDeleteDialog(id, title) {
  if (state.busy || !id) return;
  state.pendingDelete = id;
  elements.deleteName.textContent = `“${title || "未命名会话"}”`;
  elements.deleteOverlay.classList.add("open");
  elements.deleteOverlay.setAttribute("aria-hidden", "false");
  elements.deleteConfirm.focus();
}

function closeDeleteDialog() {
  state.pendingDelete = null;
  elements.deleteOverlay.classList.remove("open");
  elements.deleteOverlay.setAttribute("aria-hidden", "true");
}

async function deleteConversation() {
  const id = state.pendingDelete;
  if (!id) return;
  elements.deleteConfirm.disabled = true;
  try {
    await api(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
    const deletedCurrent = state.current?.conversation.id === id;
    if (deletedCurrent) {
      stopStreaming();
      state.current = null;
      state.progressEvents = [];
      state.liveReport = "";
      localStorage.removeItem("deep-research-assistant-conversation");
    }
    closeDeleteDialog();
    await refreshConversationList();
    if (deletedCurrent) {
      if (state.conversations.length) await selectConversation(state.conversations[0].id);
      else await createConversation();
    }
    showToast("会话已删除");
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.deleteConfirm.disabled = false;
  }
}

async function createConversation() {
  if (state.busy) return;
  try {
    const conversation = await api("/api/conversations", { method: "POST", body: "{}" });
    stopStreaming();
    adoptConversationPayload(conversation);
    state.liveReport = "";
    localStorage.setItem("deep-research-assistant-conversation", conversation.conversation.id);
    await refreshConversationList();
    renderCurrent();
    elements.input.focus();
  } catch (error) {
    showToast(error.message);
  }
}

async function selectConversation(id) {
  if (state.busy || !id) return;
  try {
    stopStreaming();
    adoptConversationPayload(await api(`/api/conversations/${encodeURIComponent(id)}`));
    state.liveReport = "";
    localStorage.setItem("deep-research-assistant-conversation", id);
    renderCurrent();
    renderHistory();
    syncPollingForCurrent();
    closeSidebar();
  } catch (error) {
    showToast(error.message);
  }
}

function stopPolling() {
  if (state.pollTimer) window.clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.pollInFlight = false;
}

function syncPollingForCurrent(label) {
  const status = state.current?.conversation.status;
  if (processingStatuses.has(status)) {
    startStreaming(label);
  } else {
    stopStreaming();
    stopPolling();
    state.busy = false;
  }
}

function stopStreaming() {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = null;
}

function startStreaming(label = "LangGraph Run 正在推进调研流程…") {
  stopStreaming();
  stopPolling();
  setBusy(true, label);
  const id = state.current?.conversation.id;
  if (!id) return;
  const url = `/api/conversations/${encodeURIComponent(id)}/events?after=${state.lastEventId}`;
  const source = new EventSource(url);
  state.eventSource = source;

  source.addEventListener("progress", (event) => {
    if (state.eventSource !== source) return;
    const progress = JSON.parse(event.data);
    const eventId = Number(progress.id) || 0;
    if (!state.progressEvents.some((item) => Number(item.id) === eventId)) {
      state.progressEvents.push(progress);
      state.progressEvents = state.progressEvents.slice(-120);
    }
    state.lastEventId = Math.max(state.lastEventId, eventId);
    renderCurrent();
  });

  source.addEventListener("report_snapshot", (event) => {
    state.liveReport = JSON.parse(event.data).markdown || "";
    renderCurrent();
  });

  source.addEventListener("report_delta", (event) => {
    state.liveReport += JSON.parse(event.data).delta || "";
    renderCurrent();
  });

  source.addEventListener("state", async () => {
    if (state.eventSource !== source) return;
    stopStreaming();
    await reloadCurrent();
    state.busy = processingStatuses.has(state.current?.conversation.status);
    await refreshConversationList();
    renderCurrent();
  });

  source.addEventListener("stream_error", (event) => {
    try {
      showToast(JSON.parse(event.data).message || "过程流暂时中断");
    } catch (_) {
      showToast("过程流暂时中断");
    }
  });

  source.onerror = () => {
    if (state.eventSource !== source) return;
    stopStreaming();
    if (processingStatuses.has(state.current?.conversation.status)) {
      startPolling("实时连接中断，正在同步后台状态…");
    }
  };
}

function startPolling(label = "LangGraph Run 正在推进调研流程…") {
  stopPolling();
  setBusy(true, label);
  state.pollTimer = window.setInterval(pollCurrentConversation, 1500);
}

async function pollCurrentConversation() {
  const id = state.current?.conversation.id;
  if (!id || state.pollInFlight) return;
  state.pollInFlight = true;
  try {
    const payload = await api(`/api/conversations/${encodeURIComponent(id)}`);
    if (state.current?.conversation.id !== id) return;
    adoptConversationPayload(payload);
    const processing = processingStatuses.has(payload.conversation.status);
    state.busy = processing;
    renderCurrent();
    if (!processing) {
      stopStreaming();
      stopPolling();
      await refreshConversationList();
      renderCurrent();
    }
  } catch (error) {
    stopPolling();
    state.busy = false;
    showToast(`刷新任务状态失败：${error.message}`);
    renderCurrent();
  } finally {
    state.pollInFlight = false;
  }
}

function renderCurrent() {
  if (!state.current) return;
  const conversation = state.current.conversation;
  elements.title.textContent = conversation.title;
  elements.status.textContent = statusLabels[conversation.status] || conversation.status;
  const hasMessages = state.current.messages.length > 0 || state.busy;
  elements.conversation.classList.toggle("has-messages", hasMessages);
  const canCompose = conversation.status === "new" || conversation.status === "complete";
  const composerDisabled = state.busy || !canCompose;
  elements.composerWrap.classList.toggle("hidden", !canCompose);
  elements.input.disabled = composerDisabled;
  elements.send.disabled = composerDisabled;
  elements.input.placeholder = conversation.status === "complete"
    ? "继续询问报告内容、证据或结论…"
    : "描述你想调研的问题…";
  renderMessages();
}

function renderMessages(thinkingLabel = "Agent 正在分析并推进调研流程…") {
  if (!state.current) return;
  const messages = state.current.messages || [];
  const reportReadyIndex = messages.findLastIndex((message) => message.kind === "report_ready");
  const processInsertIndex = reportReadyIndex >= 0 ? reportReadyIndex : messages.length;
  elements.messages.innerHTML = messages
    .slice(0, processInsertIndex)
    .map((message, index) => renderMessage(message, index))
    .join("");
  if (state.progressEvents.length || state.busy) {
    elements.messages.insertAdjacentHTML(
      "beforeend",
      `<div class="message assistant">
        <div class="avatar">D</div>
        <div><div class="message-role">DEEP RESEARCH ASSISTANT</div>
          ${renderProcessPanel(thinkingLabel)}
        </div>
      </div>`,
    );
  }
  if (processInsertIndex < messages.length) {
    elements.messages.insertAdjacentHTML(
      "beforeend",
      messages
        .slice(processInsertIndex)
        .map((message, index) => renderMessage(message, processInsertIndex + index))
        .join(""),
    );
  }
  if (state.liveReport && state.busy) {
    elements.messages.insertAdjacentHTML(
      "beforeend",
      `<div class="message assistant live-report-message">
        <div class="avatar">D</div>
        <div><div class="message-role">REPORT GENERATION</div>
          <article class="live-report markdown">${renderMarkdown(state.liveReport)}</article>
        </div>
      </div>`,
    );
  }
  bindInteractionEvents();
  requestAnimationFrame(() => {
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
  });
}

function renderProcessPanel(fallbackLabel) {
  const events = state.progressEvents.slice(-40);
  if (!events.length) {
    return `<div class="thinking-card"><div class="thinking-dots"><span></span><span></span><span></span></div>${escapeHtml(fallbackLabel)}</div>`;
  }
  const rows = events.map((event, index) => {
    const failed = event.event_type?.endsWith("failed");
    const completed = /completed|approved/.test(event.event_type || "");
    const isLast = index === events.length - 1 && state.busy;
    const marker = failed ? "!" : completed ? "✓" : isLast ? "●" : "·";
    const detail = event.detail ? `<div class="process-detail">${escapeHtml(event.detail)}</div>` : "";
    return `<div class="process-row ${failed ? "failed" : ""} ${isLast ? "active" : ""}">
      <span class="process-marker">${marker}</span>
      <div><div class="process-title">${escapeHtml(event.title)}</div>${detail}</div>
    </div>`;
  }).join("");
  return `<details class="process-card" open>
    <summary><span>研究过程</span><span class="process-count">${events.length}</span></summary>
    <div class="process-list">${rows}</div>
  </details>`;
}

function renderMessage(message, index) {
  const isUser = message.role === "user";
  const last = index === state.current.messages.length - 1;
  const interactive = last && !state.busy;
  let extra = "";
  if (message.kind === "clarification") {
    extra = renderClarification(message.payload, interactive);
  } else if (message.kind === "outline_confirmation") {
    extra = renderOutline(message.payload, interactive);
  } else if (message.kind === "report_ready") {
    extra = renderReportReady(message.payload);
  }
  const isMarkdown = message.kind === "follow_up";
  const renderedContent = isMarkdown ? renderMarkdown(message.content) : escapeHtml(message.content);
  return `
    <div class="message ${isUser ? "user" : "assistant"} ${message.kind === "error" ? "error" : ""}">
      <div class="avatar">${isUser ? "你" : "D"}</div>
      <div>
        <div class="message-role">${isUser ? "YOU" : "DEEP RESEARCH ASSISTANT"}</div>
        <div class="message-text ${isMarkdown ? "markdown" : ""}">${renderedContent}</div>
        ${extra}
      </div>
    </div>`;
}

function renderClarification(payload, interactive) {
  const questions = payload.questions || [];
  if (!interactive || state.current.conversation.status !== "waiting_for_clarification") return "";
  return `<form class="interaction-card clarification-form">
    ${questions
      .map(
        (question, index) => `<div class="question-field">
          <label for="answer-${index}">${index + 1}. ${escapeHtml(question)}</label>
          <input id="answer-${index}" name="answer" autocomplete="off" required />
        </div>`,
      )
      .join("")}
    <div class="action-row"><button class="primary-button" type="submit">提交补充信息</button></div>
  </form>`;
}

function renderOutline(payload, interactive) {
  const outline = payload.outline || {};
  const canAct = interactive && state.current.conversation.status === "waiting_for_outline_confirmation";
  return `<div class="interaction-card outline-card">
    <h3 class="outline-title">${escapeHtml(outline.title || "调研大纲")}</h3>
    <p class="outline-thesis">${escapeHtml(outline.thesis || "")}</p>
    ${(outline.sections || [])
      .map(
        (section) => `<section class="outline-section">
          <div class="section-number">${escapeHtml(section.section_id)}</div>
          <div><h4>${escapeHtml(section.title)}</h4><p>${escapeHtml(section.objective)}</p>
          <ul>${(section.research_questions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>
        </section>`,
      )
      .join("")}
    ${canAct ? `<div class="action-row">
      <button class="secondary-button revise-toggle" type="button">提出修改</button>
      <button class="primary-button approve-outline" type="button">确认并开始检索</button>
    </div>
    <div class="revision-box">
      <textarea class="revision-area" placeholder="例如：合并背景章节，增加国内外方法对比…"></textarea>
      <div class="action-row"><button class="primary-button submit-revision" type="button">重新生成大纲</button></div>
    </div>` : ""}
  </div>`;
}

function renderReportReady(payload) {
  const tools = Array.isArray(payload.tools_used) ? payload.tools_used.join("、") : "";
  return `<div class="report-ready-card">
    <div class="metrics">
      <div class="metric"><strong>${escapeHtml(payload.research_unit_count || 0)}</strong><span>并行研究单元</span></div>
      <div class="metric"><strong>${escapeHtml(payload.source_count || 0)}</strong><span>可追溯链接</span></div>
      <div class="metric"><strong>${escapeHtml(tools || "模型直答")}</strong><span>使用工具</span></div>
    </div>
    <div class="action-row"><button class="primary-button view-report" type="button">查看完整报告</button></div>
  </div>`;
}

function bindInteractionEvents() {
  const clarificationForm = elements.messages.querySelector(".clarification-form");
  clarificationForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const answers = [...clarificationForm.querySelectorAll('input[name="answer"]')].map((input) => input.value.trim());
    if (answers.some((answer) => !answer)) return showToast("请回答全部问题");
    await resumeConversation({ kind: "clarification", answers }, "正在重新确认调研意图…");
  });

  elements.messages.querySelector(".revise-toggle")?.addEventListener("click", (event) => {
    const card = event.currentTarget.closest(".outline-card");
    card.querySelector(".revision-box").classList.toggle("open");
    card.querySelector(".revision-area").focus();
  });

  elements.messages.querySelector(".approve-outline")?.addEventListener("click", () =>
    resumeConversation(
      { kind: "outline_confirmation", action: "approve" },
      "正在检索、筛选文献并生成报告，这通常需要一至三分钟…",
    ),
  );

  elements.messages.querySelector(".submit-revision")?.addEventListener("click", async (event) => {
    const feedback = event.currentTarget.closest(".outline-card").querySelector(".revision-area").value.trim();
    if (!feedback) return showToast("请输入大纲修改意见");
    await resumeConversation(
      { kind: "outline_confirmation", action: "revise", feedback },
      "正在根据你的意见重新生成大纲…",
    );
  });

  elements.messages.querySelector(".view-report")?.addEventListener("click", openReport);
}

async function resumeConversation(payload, label) {
  if (state.busy) return;
  setBusy(true, label);
  try {
    const accepted = await api(`/api/conversations/${encodeURIComponent(state.current.conversation.id)}/resume`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.current.conversation.status = accepted.status;
    state.current.conversation.active_run_id = accepted.run_id;
    await refreshConversationList();
    startStreaming(label);
  } catch (error) {
    showToast(error.message);
    await reloadCurrent();
    state.busy = false;
    renderCurrent();
  }
}

async function sendInitialMessage(content) {
  const status = state.current?.conversation.status;
  if (state.busy || !state.current || !["new", "complete"].includes(status)) return;
  const value = content.trim();
  if (status === "new" && value.length < 4) return showToast("请至少描述调研主题和目标");
  if (!value) return showToast("请输入消息");
  elements.input.value = "";
  autoResize();
  state.current.messages.push({ role: "user", content: value, kind: "text", payload: {} });
  elements.conversation.classList.add("has-messages");
  setBusy(
    true,
    status === "complete"
      ? "正在结合当前报告和会话历史回答…"
      : "正在识别调研意图并检查缺失参数…",
  );
  try {
    const accepted = await api(`/api/conversations/${encodeURIComponent(state.current.conversation.id)}/messages`, {
      method: "POST",
      body: JSON.stringify({ content: value }),
    });
    state.current.conversation.status = accepted.status;
    state.current.conversation.active_run_id = accepted.run_id;
    await refreshConversationList();
    startStreaming(
      status === "complete"
        ? "LangGraph Run 正在结合报告回答…"
        : "LangGraph Run 正在推进调研流程…",
    );
  } catch (error) {
    showToast(error.message);
    await reloadCurrent();
    state.busy = false;
    renderCurrent();
  }
}

async function reloadCurrent() {
  const id = state.current?.conversation.id;
  if (!id) return;
  try {
    adoptConversationPayload(await api(`/api/conversations/${encodeURIComponent(id)}`));
  } catch (_) {
    // The visible error is already shown by the caller.
  }
}

function renderInline(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/\[(REF\d{3})\]/g, '<span class="citation">[$1]</span>');
  safe = safe.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a class="report-link" href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  return safe;
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const html = [];
  let paragraph = [];
  let inList = false;
  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (inList) html.push("</ul>");
    inList = false;
  };
  for (const line of lines) {
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const listItem = line.match(/^[-*]\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
    } else if (listItem) {
      flushParagraph();
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${renderInline(listItem[1])}</li>`);
    } else if (!line.trim()) {
      flushParagraph();
      closeList();
    } else {
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  closeList();
  return html.join("");
}

async function openReport() {
  try {
    elements.reportContent.innerHTML = '<div class="thinking-card">正在读取报告…</div>';
    elements.reportOverlay.classList.add("open");
    elements.reportOverlay.setAttribute("aria-hidden", "false");
    const payload = await api(`/api/conversations/${encodeURIComponent(state.current.conversation.id)}/report`);
    elements.reportContent.innerHTML = renderMarkdown(payload.markdown);
  } catch (error) {
    closeReport();
    showToast(error.message);
  }
}

function closeReport() {
  elements.reportOverlay.classList.remove("open");
  elements.reportOverlay.setAttribute("aria-hidden", "true");
}

function openSidebar() {
  document.body.classList.add("sidebar-open");
  elements.menu.setAttribute("aria-expanded", "true");
  elements.menu.setAttribute("aria-label", "收起历史会话");
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
  elements.menu.setAttribute("aria-expanded", "false");
  elements.menu.setAttribute("aria-label", "展开历史会话");
}

function toggleSidebar() {
  if (document.body.classList.contains("sidebar-open")) closeSidebar();
  else openSidebar();
}

elements.newChatPanel.addEventListener("click", createConversation);
elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendInitialMessage(elements.input.value);
});
elements.input.addEventListener("input", autoResize);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});
elements.menu.addEventListener("click", toggleSidebar);
elements.sidebarClose.addEventListener("click", closeSidebar);
elements.sidebarScrim.addEventListener("click", closeSidebar);
elements.reportClose.addEventListener("click", closeReport);
elements.deleteCancel.addEventListener("click", closeDeleteDialog);
elements.deleteConfirm.addEventListener("click", deleteConversation);
elements.reportOverlay.addEventListener("click", (event) => {
  if (event.target === elements.reportOverlay) closeReport();
});
elements.deleteOverlay.addEventListener("click", (event) => {
  if (event.target === elements.deleteOverlay) closeDeleteDialog();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSidebar();
    closeReport();
    closeDeleteDialog();
  }
});

async function initialize() {
  try {
    await refreshConversationList();
    const remembered = localStorage.getItem("deep-research-assistant-conversation");
    const target = state.conversations.find((item) => item.id === remembered)?.id || state.conversations[0]?.id;
    if (target) {
      await selectConversation(target);
    } else {
      await createConversation();
    }
  } catch (error) {
    showToast(`应用初始化失败：${error.message}`);
  }
}

initialize();
