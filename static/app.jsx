/* =========================================================================
   MediCare AI — React frontend
   Built by Pritam

   Loaded directly by the browser via Babel standalone (see templates/chat.html)
   — no npm, no build step, so deployment is unchanged: Flask just serves this
   file as a static asset, exactly like the old script.js. React/ReactDOM/Babel
   /Tailwind all come from CDN <script> tags in chat.html.

   Talks to the exact same Flask API as before: /get/stream (SSE), /feedback,
   /health, /stats. None of that changed — only the rendering layer did.
   ========================================================================= */

const { useState, useEffect, useRef, useCallback, useMemo } = React;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY = "medicare_ai_conversations";
const ACTIVE_KEY = "medicare_ai_active";
const THEME_KEY = "medicare_ai_theme";

// Same-origin ("") unless the page defines window.__MEDICARE_API_BASE__
// (see templates/chat.html) -- only needed when this frontend is hosted
// separately from the Flask API, e.g. a static Netlify/Vercel deploy
// calling a Render-hosted API cross-origin (see DEPLOYMENT.md). The
// default single-service deployment, where Flask serves this file
// itself, is entirely unaffected either way: every fetch below already
// used a plain relative path before this existed, and apiUrl(path) below
// returns that exact same path whenever API_BASE is empty.
const API_BASE = (typeof window !== "undefined" && window.__MEDICARE_API_BASE__) || "";

function apiUrl(path) {
  return API_BASE + path;
}

const SUGGESTIONS = [
  { eyebrow: "Understand a symptom", prompt: "What could cause a persistent headache that lasts for days?" },
  { eyebrow: "Learn about a condition", prompt: "Explain what type 2 diabetes is and how it's usually managed" },
  { eyebrow: "Decode a medical term", prompt: "What does the term \"hypertension\" mean?" },
  { eyebrow: "Prep for a doctor visit", prompt: "What should I ask my doctor about high cholesterol?" },
];

// ---------------------------------------------------------------------------
// Pure helpers (framework-independent — no DOM, no React; easy to unit test)
// ---------------------------------------------------------------------------

function uid() {
  return Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
}

function truncateTitle(text) {
  return text.length > 42 ? text.slice(0, 42) + "…" : text;
}

function createConversation() {
  return { id: uid(), title: "New consultation", messages: [], updatedAt: Date.now() };
}

function stripMarkdown(text) {
  return (text || "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`(.*?)`/g, "$1")
    .replace(/#+\s?/g, "")
    .replace(/\n+/g, ". ");
}

// Splits a growing SSE buffer into complete "data: {...}" events plus
// whatever incomplete tail should be kept for the next chunk.
function parseSSEBuffer(buffer) {
  const events = buffer.split("\n\n");
  const remaining = events.pop();
  const parsed = [];
  for (const evt of events) {
    if (!evt.startsWith("data: ")) continue;
    try {
      parsed.push(JSON.parse(evt.slice(6)));
    } catch {
      // ignore a malformed event rather than crashing the stream
    }
  }
  return { parsed, remaining };
}

// Applies one parsed SSE payload to the in-progress bot message draft.
function applyStreamEvent(draft, payload) {
  if (payload.type === "chunk") {
    return { ...draft, content: draft.content + payload.content };
  }
  if (payload.type === "error") {
    return { ...draft, errorMessage: payload.message || "Something went wrong. Please try again." };
  }
  if (payload.type === "done") {
    return {
      ...draft,
      sources: payload.sources || [],
      emergency: !!payload.emergency,
      cached: !!payload.cached,
      no_info: !!payload.no_info,
    };
  }
  return draft;
}

function loadConversations() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Icons (inline SVG — no icon package needed)
// ---------------------------------------------------------------------------

const Icon = {
  Logo: () => (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <rect x="1" y="1" width="30" height="30" rx="9" fill="#12332C" />
      <path d="M16 9v14M9 16h14" stroke="#5FC3A8" strokeWidth="3.4" strokeLinecap="round" />
    </svg>
  ),
  Plus: (p) => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  ),
  Trash: (p) => (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  Sun: (p) => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M12 3a9 9 0 109 9c0-.46-.04-.92-.1-1.36A5.4 5.4 0 0112 3z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  ),
  Moon: (p) => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" {...p}>
      <circle cx="12" cy="12" r="4.5" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  ),
  PanelToggle: (p) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" {...p}>
      <rect x="3" y="4" width="18" height="16" rx="3" stroke="currentColor" strokeWidth="1.6" />
      <line x1="9.5" y1="4" x2="9.5" y2="20" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  ),
  Alert: (p) => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  ),
  Bot: (p) => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M12 3v4M8 7h8a2 2 0 012 2v9a2 2 0 01-2 2H8a2 2 0 01-2-2V9a2 2 0 012-2z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <circle cx="9.5" cy="13" r="1" fill="currentColor" />
      <circle cx="14.5" cy="13" r="1" fill="currentColor" />
    </svg>
  ),
  Mic: (p) => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M12 15a3 3 0 003-3V6a3 3 0 10-6 0v6a3 3 0 003 3z" stroke="currentColor" strokeWidth="1.8" />
      <path d="M19 11a7 7 0 01-14 0M12 18v3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  ),
  Send: (p) => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M4 12h15M13 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  Speak: (p) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M4 9v6h4l5 5V4L8 9H4z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M17 8a5 5 0 010 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  ),
  Copy: (p) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" {...p}>
      <rect x="9" y="9" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M5 15H4a1 1 0 01-1-1V4a1 1 0 011-1h10a1 1 0 011 1v1" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  ),
  Check: (p) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M5 13l4 4L19 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  ThumbsUp: (p) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M7 11v9H4a1 1 0 01-1-1v-7a1 1 0 011-1h3zm0 0l4-8a2 2 0 012 2v4h5a2 2 0 012 2.2l-1.2 7A2 2 0 0116.8 20H7" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  ),
  ThumbsDown: (p) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M17 13V4h3a1 1 0 011 1v7a1 1 0 01-1 1h-3zm0 0l-4 8a2 2 0 01-2-2v-4H6a2 2 0 01-2-2.2l1.2-7A2 2 0 017.2 4H17" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  ),
  Bolt: (p) => (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" {...p}>
      <path d="M13 2L4 14h6l-1 8 9-12h-6l1-8z" />
    </svg>
  ),
  Paperclip: (p) => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M8 12l6.5-6.5a3.5 3.5 0 015 5L11 19a5.5 5.5 0 01-7.8-7.8L12 2.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  FileText: (p) => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M6 2h9l5 5v15a1 1 0 01-1 1H6a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M14 2v6h6M8 13h8M8 17h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
  X: (p) => (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  ),
  Spinner: (p) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" {...p} className={"animate-spin " + (p.className || "")}>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" strokeOpacity="0.25" />
      <path d="M21 12a9 9 0 00-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  ),
  BarChart: (p) => (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M4 20V10M11 20V4M18 20v-6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  Refresh: (p) => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" {...p}>
      <path
        d="M3.5 12a8.5 8.5 0 0114.5-6M20.5 12a8.5 8.5 0 01-14.5 6M17.5 5.5v3.5H14M6.5 18.5V15H10"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
};

// ---------------------------------------------------------------------------
// Small hooks
// ---------------------------------------------------------------------------

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || "light");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  return [theme, toggle];
}

const SIDEBAR_WIDTH_KEY = "medicare_ai_sidebar_width";
const SIDEBAR_MIN_WIDTH = 240;
const SIDEBAR_MAX_WIDTH = 440;
const SIDEBAR_DEFAULT_WIDTH = 288; // matches the old fixed w-72 (288px)

function clampSidebarWidth(px) {
  return Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, px));
}

function useSidebarWidth() {
  const [width, setWidth] = useState(() => {
    const saved = Number(localStorage.getItem(SIDEBAR_WIDTH_KEY));
    return saved ? clampSidebarWidth(saved) : SIDEBAR_DEFAULT_WIDTH;
  });
  const [resizing, setResizing] = useState(false);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(width));
  }, [width]);

  // Drag-to-resize, like a terminal or IDE panel divider. Tracked with
  // window-level listeners (not just on the handle itself) so the drag
  // keeps working even when the pointer moves faster than the thin
  // handle, or drifts off it entirely mid-drag -- and measured as a
  // delta from the position/width at drag-start rather than
  // accumulating small per-event deltas, so it can't drift out of sync
  // with the pointer over a long drag.
  const startResizing = useCallback(
    (startEvent) => {
      startEvent.preventDefault();
      const startX = startEvent.clientX;
      const startWidth = width;
      setResizing(true);
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";

      const handleMove = (moveEvent) => {
        setWidth(clampSidebarWidth(startWidth + (moveEvent.clientX - startX)));
      };
      const stopResizing = () => {
        setResizing(false);
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
        window.removeEventListener("pointermove", handleMove);
        window.removeEventListener("pointerup", stopResizing);
      };
      window.addEventListener("pointermove", handleMove);
      window.addEventListener("pointerup", stopResizing);
    },
    [width]
  );

  return { width, resizing, startResizing };
}

function useVoiceInput(onResult) {
  const recognitionRef = useRef(null);
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(true);
  const [interimTranscript, setInterimTranscript] = useState("");

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSupported(false);
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.addEventListener("result", (e) => {
      // interimResults=true makes this fire repeatedly while the person
      // is still speaking (each result marked isFinal: false) and once
      // more with the settled text (isFinal: true) when they pause --
      // surfacing the interim text live (below) is what makes voice
      // input visible *as you speak* instead of the composer sitting
      // empty until you stop talking.
      let transcript = "";
      let isFinal = false;
      for (let i = 0; i < e.results.length; i++) {
        transcript += e.results[i][0].transcript;
        if (e.results[i].isFinal) isFinal = true;
      }
      if (isFinal) {
        setInterimTranscript("");
        onResult(transcript);
      } else {
        setInterimTranscript(transcript);
      }
    });
    recognition.addEventListener("end", () => {
      setListening(false);
      setInterimTranscript("");
    });
    recognition.addEventListener("error", () => {
      setListening(false);
      setInterimTranscript("");
    });
    recognitionRef.current = recognition;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggle = useCallback(() => {
    if (!recognitionRef.current) return;
    if (listening) {
      recognitionRef.current.stop();
    } else {
      setInterimTranscript("");
      setListening(true);
      recognitionRef.current.start();
    }
  }, [listening]);

  return { listening, supported, toggle, interimTranscript };
}

// ---------------------------------------------------------------------------
// Presentational components
// ---------------------------------------------------------------------------

function IconButton({ title, onClick, className = "", children, disabled, type = "button" }) {
  return (
    <button
      type={type}
      title={title}
      aria-label={title}
      disabled={disabled}
      onClick={onClick}
      className={
        "flex items-center justify-center rounded-full border border-[var(--border)] bg-[var(--paper-raised)] text-[var(--ink-soft)] transition-all duration-150 hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed active:scale-95 " +
        className
      }
    >
      {children}
    </button>
  );
}

function HistoryItem({ convo, active, onSelect, onDelete }) {
  return (
    <div
      onClick={onSelect}
      className={
        "group flex items-center justify-between gap-2 rounded-lg pl-3 pr-2 py-2.5 mb-0.5 cursor-pointer text-[13.5px] text-[var(--sidebar-text)] border-l-2 transition-colors " +
        (active ? "bg-[var(--sidebar-hover)] border-[var(--accent)]" : "border-transparent hover:bg-[var(--sidebar-hover)]")
      }
    >
      <span className="truncate">{convo.title}</span>
      <button
        title="Delete"
        aria-label="Delete conversation"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="opacity-0 group-hover:opacity-100 text-[var(--sidebar-muted)] hover:text-[var(--alert)] transition-opacity p-0.5 flex-shrink-0"
      >
        <Icon.Trash />
      </button>
    </div>
  );
}

function DocumentItem({ doc, selected, onToggleSelect, onDelete }) {
  return (
    <div className="group flex items-center justify-between gap-2 rounded-lg pl-2 pr-2 py-2 mb-0.5 text-[13px] text-[var(--sidebar-text)]">
      <label className="flex items-center gap-2 min-w-0 cursor-pointer">
        <input
          type="checkbox"
          checked={selected}
          onChange={() => onToggleSelect(doc.id)}
          aria-label={`${selected ? "Exclude" : "Include"} ${doc.filename} when answering`}
          className="flex-shrink-0 w-3.5 h-3.5 rounded accent-[var(--accent)] cursor-pointer"
        />
        <span className="text-[var(--sidebar-muted)] flex-shrink-0">
          <Icon.FileText />
        </span>
        <div className="min-w-0">
          <p className="truncate leading-snug">{doc.filename}</p>
          <p className="text-[10.5px] text-[var(--sidebar-muted)] leading-snug">
            {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
            {doc.page_count != null ? ` · ${doc.page_count} pg` : ""}
          </p>
        </div>
      </label>
      <button
        title="Remove document"
        aria-label={`Remove ${doc.filename}`}
        onClick={() => onDelete(doc.id)}
        className="opacity-0 group-hover:opacity-100 text-[var(--sidebar-muted)] hover:text-[var(--alert)] transition-opacity p-0.5 flex-shrink-0"
      >
        <Icon.Trash />
      </button>
    </div>
  );
}

function DocumentsPanel({ documents, selectedIds, onToggleSelect, onToggleAll, onDelete, onUploadClick, uploading }) {
  const allSelected = documents.length > 0 && documents.every((d) => selectedIds.has(d.id));
  const someSelected = documents.some((d) => selectedIds.has(d.id));
  const selectAllRef = useRef(null);

  useEffect(() => {
    if (selectAllRef.current) selectAllRef.current.indeterminate = someSelected && !allSelected;
  }, [someSelected, allSelected]);

  return (
    <div className="border-t border-[var(--sidebar-border)] pt-3.5 mt-2.5">
      <div className="flex items-center justify-between mx-2 mb-1.5">
        <div className="flex items-center gap-1.5 min-w-0">
          <input
            ref={selectAllRef}
            type="checkbox"
            checked={allSelected}
            onChange={() => onToggleAll(!allSelected)}
            disabled={documents.length === 0}
            title={allSelected ? "Answer using every document" : "Select every document"}
            aria-label="Select all documents"
            className="flex-shrink-0 w-3.5 h-3.5 rounded accent-[var(--accent)] cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
          />
          <p className="text-[11px] uppercase tracking-wider text-[var(--sidebar-muted)] truncate">Documents</p>
        </div>
        <button
          onClick={onUploadClick}
          title="Upload a PDF"
          aria-label="Upload a PDF"
          disabled={uploading}
          className="flex items-center justify-center w-6 h-6 rounded-md text-[var(--sidebar-muted)] hover:bg-[var(--sidebar-hover)] hover:text-[var(--sidebar-text)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0"
        >
          {uploading ? <Icon.Spinner /> : <Icon.Plus />}
        </button>
      </div>
      {documents.length === 0 ? (
        <p className="text-[11px] leading-relaxed text-[var(--sidebar-muted)] px-2">
          Upload a PDF to add it to the knowledge base MediCare AI answers from.
        </p>
      ) : (
        <>
          <div className="max-h-36 overflow-y-auto pr-0.5 sidebar-scroll">
            {documents.map((doc) => (
              <DocumentItem
                key={doc.id}
                doc={doc}
                selected={selectedIds.has(doc.id)}
                onToggleSelect={onToggleSelect}
                onDelete={onDelete}
              />
            ))}
          </div>
          {!allSelected && (
            <p className="text-[10.5px] leading-relaxed text-[var(--accent-strong)] px-2 mt-1.5">
              {selectedIds.size === 0
                ? 'Nothing ticked — MediCare AI will search every document until you tick at least one.'
                : `Answering from ${selectedIds.size} of ${documents.length} document${documents.length === 1 ? "" : "s"}.`}
            </p>
          )}
        </>
      )}
    </div>
  );
}

function Sidebar({ theme, onToggleTheme, conversations, activeId, onSelect, onDelete, onNewChat, open, onClose, width, resizing, onStartResize, documents, selectedDocIds, onToggleSelectDocument, onToggleSelectAllDocuments, onDeleteDocument, onUploadClick, uploadingDocument, view, onOpenDashboard }) {
  const list = useMemo(
    () =>
      Object.values(conversations)
        .filter((c) => c.messages.length > 0)
        .sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations]
  );

  const scrollRef = useRef(null);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);

  const updateScrollFades = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollUp(el.scrollTop > 4);
    setCanScrollDown(el.scrollTop + el.clientHeight < el.scrollHeight - 4);
  }, []);

  useEffect(() => {
    updateScrollFades();
  }, [list, updateScrollFades]);

  return (
    <>
      {open && (
        <div onClick={onClose} className="fixed inset-0 bg-black/40 z-10 md:hidden" aria-hidden="true" />
      )}
      <aside
        style={open ? { width } : undefined}
        className={
          "fixed md:relative z-20 h-full flex-shrink-0 bg-[var(--sidebar-bg)] text-[var(--sidebar-text)] border-r border-[var(--sidebar-border)] overflow-hidden " +
          (resizing ? "" : "transition-all duration-200 ease-in-out ") +
          (open
            ? "translate-x-0 shadow-2xl md:shadow-none"
            : "w-72 md:w-0 -translate-x-full md:translate-x-0 md:border-r-0")
        }
      >
        {/* Fixed-width inner wrapper: the <aside> above animates its width
            down to 0 on desktop collapse, but this stays at the sidebar's
            current width (whatever it's set to below) the whole time, so
            the content slides/clips cleanly instead of squishing and
            reflowing mid-animation. */}
        <div style={{ width }} className="h-full flex flex-col p-5">
          <div className="flex items-center gap-3 px-1 pb-5">
            <Icon.Logo />
            <div className="flex-1 min-w-0">
              <h1 className="font-display text-[21px] font-semibold tracking-tight flex items-center gap-1.5">
                MediCare
                <span className="font-normal italic text-[#4FA890]">AI</span>
                <span className="w-1.5 h-1.5 rounded-full bg-[#4FA890] animate-pulse ml-0.5" title="Ready" />
              </h1>
              <p className="font-mono text-[11px] text-[var(--sidebar-muted)] tracking-wide mt-0.5">by Pritam</p>
            </div>
            <button
              onClick={onClose}
              title="Collapse sidebar"
              aria-label="Collapse sidebar"
              className="hidden md:flex items-center justify-center w-7 h-7 rounded-md text-[var(--sidebar-muted)] hover:bg-[var(--sidebar-hover)] hover:text-[var(--sidebar-text)] transition-colors flex-shrink-0"
            >
              <Icon.PanelToggle />
            </button>
          </div>

          <button
            onClick={onNewChat}
            className="flex items-center justify-center gap-2 w-full py-2.5 px-3.5 rounded-xl border border-[var(--sidebar-border)] bg-[var(--sidebar-bg-raised)] text-sm font-medium hover:bg-[var(--sidebar-hover)] active:scale-[0.98] transition-all"
          >
            <Icon.Plus /> New consultation
          </button>

          <button
            onClick={onOpenDashboard}
            className={
              "flex items-center gap-2 w-full py-2 px-3.5 rounded-xl text-[13px] font-medium mt-1.5 transition-colors " +
              (view === "dashboard"
                ? "bg-[var(--accent-soft)] text-[var(--accent-strong)]"
                : "text-[var(--sidebar-muted)] hover:bg-[var(--sidebar-hover)] hover:text-[var(--sidebar-text)]")
            }
          >
            <Icon.BarChart /> Analytics
          </button>

          <div className="relative flex-1 min-h-0 mt-4">
            <div
              ref={scrollRef}
              onScroll={updateScrollFades}
              className="h-full overflow-y-auto pr-0.5 sidebar-scroll"
            >
              <p className="text-[11px] uppercase tracking-wider text-[var(--sidebar-muted)] mx-2 mb-2 mt-1">Recent</p>
              {list.length === 0 ? (
                <p className="text-[11px] leading-relaxed text-[var(--sidebar-muted)] px-2">
                  Your consultations will show up here.
                </p>
              ) : (
                list.map((c) => (
                  <HistoryItem
                    key={c.id}
                    convo={c}
                    active={c.id === activeId}
                    onSelect={() => onSelect(c.id)}
                    onDelete={() => onDelete(c.id)}
                  />
                ))
              )}
            </div>

            {/* Fade indicators -- only visible when there's actually more
                content to scroll in that direction, so they double as a
                subtle "yes, this list scrolls" affordance rather than a
                static decoration. Color matches --sidebar-bg exactly, so
                it blends seamlessly and adapts automatically with the
                light/dark theme toggle. */}
            <div
              className={
                "pointer-events-none absolute top-0 left-0 right-0 h-6 bg-gradient-to-b from-[var(--sidebar-bg)] to-transparent transition-opacity duration-150 " +
                (canScrollUp ? "opacity-100" : "opacity-0")
              }
            />
            <div
              className={
                "pointer-events-none absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t from-[var(--sidebar-bg)] to-transparent transition-opacity duration-150 " +
                (canScrollDown ? "opacity-100" : "opacity-0")
              }
            />
          </div>

          <DocumentsPanel
            documents={documents}
            selectedIds={selectedDocIds}
            onToggleSelect={onToggleSelectDocument}
            onToggleAll={onToggleSelectAllDocuments}
            onDelete={onDeleteDocument}
            onUploadClick={onUploadClick}
            uploading={uploadingDocument}
          />

          <div className="border-t border-[var(--sidebar-border)] pt-3.5 mt-2.5">
            <button
              onClick={onToggleTheme}
              className="flex items-center gap-2 w-full border border-[var(--sidebar-border)] rounded-lg py-2 px-2.5 text-[13px] hover:bg-[var(--sidebar-hover)] transition-colors mb-3"
            >
              {theme === "dark" ? <Icon.Sun /> : <Icon.Moon />}
              {theme === "dark" ? "Light mode" : "Dark mode"}
            </button>
            <p className="text-[11px] leading-relaxed text-[var(--sidebar-muted)]">
              MediCare AI shares general health information only. It is not a substitute for professional medical
              advice, diagnosis, or treatment.
            </p>
          </div>
        </div>

        {/* Drag-to-resize handle -- like a terminal/IDE panel divider.
            Desktop only: mobile shows the sidebar as a fixed full-width-ish
            overlay (see the "open"/"-translate-x-full" branch above),
            where a user-adjustable width doesn't really apply. The hit
            area (w-2, straddling the visible border) is deliberately
            wider than the thin indicator line inside it, so it's easy to
            grab without looking heavy-handed at rest. */}
        <div
          onPointerDown={onStartResize}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
          title="Drag to resize"
          className="hidden md:flex absolute top-0 -right-1 h-full w-2 cursor-col-resize items-center justify-center touch-none select-none group z-10"
        >
          <span
            className={
              "w-[3px] h-10 rounded-full transition-colors " +
              (resizing ? "bg-[var(--accent)]" : "bg-transparent group-hover:bg-[var(--accent)]")
            }
          />
        </div>
      </aside>
    </>
  );
}

function SuggestionCard({ eyebrow, prompt, onClick }) {
  return (
    <button
      onClick={onClick}
      className="text-left bg-[var(--paper-raised)] border border-[var(--border)] rounded-2xl p-4 shadow-sm hover:border-[var(--accent)] hover:-translate-y-0.5 transition-all duration-150"
    >
      <span className="block font-mono text-[11px] text-[var(--accent)] mb-1.5 tracking-wide">{eyebrow}</span>
      <span className="text-[13.5px] text-[var(--ink)] leading-snug">{prompt}</span>
    </button>
  );
}

function Hero({ onSuggestionClick, onUploadClick }) {
  return (
    <div className="pt-8 md:pt-10 pb-3 px-1 animate-msg-in">
      <h2 className="font-display text-[28px] md:text-[34px] font-semibold leading-tight max-w-md mb-2.5 text-[var(--ink)]">
        What's going on with you today?
      </h2>
      <p className="text-[var(--muted)] text-[14.5px] max-w-md leading-relaxed mb-2">
        Ask a health question in your own words. Every answer is grounded in the PDFs in your knowledge base
        — with the exact source and page underneath the reply.
      </p>
      <button
        onClick={onUploadClick}
        className="inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--accent-strong)] hover:text-[var(--accent)] transition-colors mb-6"
      >
        <Icon.Paperclip /> Upload your own PDF to ask about
      </button>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {SUGGESTIONS.map((s) => (
          <SuggestionCard key={s.eyebrow} eyebrow={s.eyebrow} prompt={s.prompt} onClick={() => onSuggestionClick(s.prompt)} />
        ))}
      </div>
    </div>
  );
}

function SourceChips({ sources, cached }) {
  if ((!sources || sources.length === 0) && !cached) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {(sources || []).map((s, i) => (
        <span
          key={i}
          title={s.score != null ? `Relevance: ${Math.round(s.score * 100)}%` : undefined}
          className="font-mono text-[10.5px] tracking-wide text-[var(--muted)] border border-dashed border-[var(--border)] rounded-md px-2 py-0.5"
        >
          <Icon.FileText className="inline-block w-3 h-3 -mt-0.5 mr-1 opacity-70" />
          {s.source}
          {s.page != null ? ` · p.${s.page}` : ""}
        </span>
      ))}
      {cached && (
        <span className="flex items-center gap-1 font-mono text-[10.5px] text-[var(--accent-strong)] border border-[var(--accent)] rounded-md px-2 py-0.5">
          <Icon.Bolt /> instant (cached)
        </span>
      )}
    </div>
  );
}

function MessageActions({ message, onFeedback }) {
  const [copied, setCopied] = useState(false);
  const [rated, setRated] = useState(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(stripMarkdown(message.content)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  };

  const handleRate = (rating) => {
    setRated(rating);
    onFeedback(message, rating);
  };

  return (
    <div className="flex gap-1 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
      <IconButton title="Read aloud" className="w-7 h-7 border-none bg-transparent" onClick={() => speak(message.content)}>
        <Icon.Speak />
      </IconButton>
      <IconButton title={copied ? "Copied!" : "Copy"} className="w-7 h-7 border-none bg-transparent" onClick={handleCopy}>
        {copied ? <Icon.Check /> : <Icon.Copy />}
      </IconButton>
      {!message.emergency && message.question && (
        <>
          <IconButton
            title="Helpful"
            className={"w-7 h-7 border-none bg-transparent " + (rated === "up" ? "text-[var(--accent)]" : "")}
            onClick={() => handleRate("up")}
            disabled={!!rated}
          >
            <Icon.ThumbsUp />
          </IconButton>
          <IconButton
            title="Not helpful"
            className={"w-7 h-7 border-none bg-transparent " + (rated === "down" ? "text-[var(--alert)]" : "")}
            onClick={() => handleRate("down")}
            disabled={!!rated}
          >
            <Icon.ThumbsDown />
          </IconButton>
        </>
      )}
    </div>
  );
}

function speak(markdownText) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(stripMarkdown(markdownText));
  utter.rate = 1;
  window.speechSynthesis.speak(utter);
}

function Message({ message, onFeedback, isStreaming, onUploadClick }) {
  const isUser = message.role === "user";
  const isEmergency = !!message.emergency;
  const html = window.marked ? window.marked.parse(message.content || "") : message.content;

  if (isUser) {
    return (
      <div className="flex flex-row-reverse gap-3 animate-msg-in">
        <div className="max-w-[86%] md:max-w-[74%] flex flex-col items-end">
          <div className="px-4 py-3 rounded-2xl rounded-tr-sm text-[14.5px] leading-relaxed shadow-sm bg-[var(--amber-soft)] text-[var(--ink)]">
            {message.content}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="group flex gap-3 animate-msg-in">
      <div
        className={
          "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 " +
          (isEmergency ? "bg-[var(--alert-soft)] text-[var(--alert)]" : "bg-[var(--accent-soft)] text-[var(--accent-strong)]")
        }
      >
        {isEmergency ? <Icon.Alert /> : <Icon.Bot />}
      </div>
      <div className="max-w-[86%] md:max-w-[74%] flex flex-col items-start">
        <div
          className={
            "px-4 py-3 rounded-2xl rounded-tl-sm text-[14.5px] leading-relaxed shadow-sm prose-chat " +
            (isEmergency
              ? "bg-[var(--alert-soft)] border border-[var(--alert)]"
              : "bg-[var(--paper-raised)] border border-[var(--border)]")
          }
          dangerouslySetInnerHTML={{ __html: html }}
        />
        {isStreaming && (message.content || "").length > 0 && (
          <span className="inline-block w-[2px] h-[14px] bg-[var(--accent)] animate-pulse -mt-5 ml-4" />
        )}
        {!isStreaming && (
          <>
            <SourceChips sources={message.sources} cached={message.cached} />
            {message.no_info && onUploadClick && (
              <button
                onClick={onUploadClick}
                className="flex items-center gap-1.5 mt-2 text-[12px] font-medium text-[var(--accent-strong)] hover:text-[var(--accent)] transition-colors"
              >
                <Icon.Paperclip /> Upload a PDF about this so I can check it
              </button>
            )}
            <MessageActions message={message} onFeedback={onFeedback} />
          </>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-msg-in">
      <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-[var(--accent-soft)] text-[var(--accent-strong)]">
        <Icon.Bot />
      </div>
      <div className="px-4 py-3.5 rounded-2xl rounded-tl-sm bg-[var(--paper-raised)] border border-[var(--border)] shadow-sm flex gap-1">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--muted)] animate-typing-bounce" />
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--muted)] animate-typing-bounce [animation-delay:150ms]" />
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--muted)] animate-typing-bounce [animation-delay:300ms]" />
      </div>
    </div>
  );
}

function EmergencyBanner({ show }) {
  if (!show) return null;
  return (
    <div className="flex items-center gap-2.5 bg-[var(--alert)] text-white px-4 md:px-6 py-2.5 text-[13.5px] font-medium">
      <Icon.Alert />
      <span>Possible emergency detected — see the guidance below and contact local emergency services if needed.</span>
    </div>
  );
}

function Composer({ value, onChange, onSubmit, voice, onAttachClick, uploading }) {
  // While actively listening, the field previews the in-progress
  // transcript live (see useVoiceInput's interimResults handling) rather
  // than sitting on whatever was last typed until speech recognition
  // settles on a final result -- read-only for that same span so a
  // half-spoken phrase can't get mixed up with manual typing.
  const displayValue = voice.listening ? voice.interimTranscript : value;

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="flex items-center gap-2 max-w-[760px] w-full mx-auto px-4 md:px-6 pb-5 pt-3"
    >
      <IconButton
        title="Attach a PDF"
        onClick={onAttachClick}
        disabled={uploading}
        className="w-11 h-11 flex-shrink-0"
      >
        {uploading ? <Icon.Spinner /> : <Icon.Paperclip />}
      </IconButton>
      {voice.supported && (
        <IconButton
          title={voice.listening ? "Stop voice input" : "Voice input"}
          onClick={voice.toggle}
          className={"w-11 h-11 flex-shrink-0 " + (voice.listening ? "!bg-[var(--alert)] !border-[var(--alert)] !text-white animate-mic-pulse" : "")}
        >
          <Icon.Mic />
        </IconButton>
      )}
      <div className="relative flex-1">
        <input
          type="text"
          value={displayValue}
          onChange={(e) => onChange(e.target.value)}
          readOnly={voice.listening}
          placeholder={voice.listening ? "Listening…" : "Describe your symptom or ask a health question…"}
          autoComplete="off"
          className={
            "w-full h-[46px] rounded-full border bg-[var(--paper-raised)] text-[var(--ink)] pl-5 text-[14.5px] placeholder:text-[var(--muted)] focus:outline-none transition-colors " +
            (voice.listening ? "border-[var(--alert)] pr-11" : "border-[var(--border)] focus:border-[var(--accent)] pr-5")
          }
        />
        {voice.listening && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 flex items-end gap-[3px] h-4" aria-hidden="true">
            <span className="w-[3px] h-full rounded-full bg-[var(--alert)] animate-voice-wave" />
            <span className="w-[3px] h-full rounded-full bg-[var(--alert)] animate-voice-wave [animation-delay:150ms]" />
            <span className="w-[3px] h-full rounded-full bg-[var(--alert)] animate-voice-wave [animation-delay:300ms]" />
          </span>
        )}
        {/* Screen-reader-only announcement -- the waveform above and the
            mic button's own pulse/color change are both purely visual. */}
        <span className="sr-only" role="status" aria-live="polite">
          {voice.listening ? `Listening…${voice.interimTranscript ? ` ${voice.interimTranscript}` : ""}` : ""}
        </span>
      </div>
      <button
        type="submit"
        title="Send"
        disabled={!value.trim()}
        className="w-11 h-11 flex-shrink-0 rounded-full bg-[var(--accent)] text-white flex items-center justify-center hover:bg-[var(--accent-strong)] active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
      >
        <Icon.Send />
      </button>
    </form>
  );
}

function UploadNotice({ notice, onDismiss }) {
  return (
    <div
      className={
        "flex items-center gap-2.5 rounded-xl border px-3.5 py-2 text-[12.5px] leading-snug shadow-sm animate-msg-in " +
        (notice.status === "error"
          ? "border-[var(--alert)] bg-[var(--alert-soft)] text-[var(--ink)]"
          : "border-[var(--border)] bg-[var(--paper-raised)] text-[var(--ink)]")
      }
    >
      {notice.status === "uploading" && <Icon.Spinner className="text-[var(--accent)] flex-shrink-0" />}
      {notice.status === "done" && (
        <span className="text-[var(--accent-strong)] flex-shrink-0">
          <Icon.Check />
        </span>
      )}
      {notice.status === "error" && (
        <span className="text-[var(--alert)] flex-shrink-0">
          <Icon.Alert width="14" height="14" />
        </span>
      )}
      <span className="flex-1 min-w-0 truncate">
        <span className="font-medium">{notice.filename}</span> — {notice.message}
      </span>
      {notice.status !== "uploading" && (
        <button
          onClick={onDismiss}
          aria-label="Dismiss"
          title="Dismiss"
          className="text-[var(--muted)] hover:text-[var(--ink)] flex-shrink-0 transition-colors"
        >
          <Icon.X />
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard — a small analytics view over /stats, built with Chart.js
// (loaded via CDN in templates/chat.html — see the comment there for why
// the UMD build specifically). Kept in its own section since nothing here
// is chat logic: it's just "fetch /stats, render some cards and charts."
// ---------------------------------------------------------------------------

function cssVar(name) {
  // Charts render to a <canvas>, which can't itself follow the page's CSS
  // cascade or a data-theme attribute switch the way a styled DOM element
  // can -- Chart.js needs concrete color values up front. Resolving each
  // color through getComputedStyle at chart-build time (rather than
  // hardcoding hex values here) means a chart still picks up whatever
  // theme.css actually defines for the *current* theme, light or dark,
  // without this file needing to know either palette itself.
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function formatShortDate(isoDate) {
  const d = new Date(isoDate + "T00:00:00");
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function StatCard({ label, value, icon, tone }) {
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--paper-raised)] p-4 flex flex-col gap-2 min-w-0">
      <div className={"flex items-center gap-1.5 " + (tone === "alert" ? "text-[var(--alert)]" : "text-[var(--muted)]")}>
        {icon}
        <span className="text-[11px] uppercase tracking-wide truncate">{label}</span>
      </div>
      <p className="font-display text-[26px] leading-none font-semibold text-[var(--ink)] truncate">{value}</p>
    </div>
  );
}

function ChartEmptyState({ message }) {
  return (
    <div className="h-full flex items-center justify-center text-center px-4">
      <p className="text-[12.5px] text-[var(--muted)] leading-relaxed">{message}</p>
    </div>
  );
}

function QueriesChart({ daily, theme }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart || !daily) return;
    const accent = cssVar("--accent");
    const accentSoft = cssVar("--accent-soft");
    const muted = cssVar("--muted");
    const border = cssVar("--border");

    chartRef.current = new window.Chart(canvasRef.current, {
      type: "line",
      data: {
        labels: daily.map((d) => formatShortDate(d.date)),
        datasets: [
          {
            label: "Queries",
            data: daily.map((d) => d.queries),
            borderColor: accent,
            backgroundColor: accentSoft,
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: muted, maxRotation: 0, autoSkip: true, maxTicksLimit: 7 } },
          y: { beginAtZero: true, ticks: { color: muted, precision: 0 }, grid: { color: border } },
        },
      },
    });

    return () => chartRef.current && chartRef.current.destroy();
  }, [daily, theme]);

  return <canvas ref={canvasRef} role="img" aria-label="Queries per day over the last 14 days" />;
}

function LatencyChart({ avgRetrieval, avgGeneration, theme }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;
    const accent = cssVar("--accent");
    const amber = cssVar("--amber");
    const muted = cssVar("--muted");
    const border = cssVar("--border");
    const ink = cssVar("--ink");

    chartRef.current = new window.Chart(canvasRef.current, {
      type: "bar",
      data: {
        labels: ["Retrieval", "Generation"],
        datasets: [{ data: [avgRetrieval, avgGeneration], backgroundColor: [accent, amber], borderRadius: 6, barThickness: 28 }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => `${Math.round(ctx.parsed.x)}ms avg` } },
        },
        scales: {
          x: { beginAtZero: true, ticks: { color: muted }, grid: { color: border } },
          y: { ticks: { color: ink, font: { size: 12.5 } }, grid: { display: false } },
        },
      },
    });

    return () => chartRef.current && chartRef.current.destroy();
  }, [avgRetrieval, avgGeneration, theme]);

  return <canvas ref={canvasRef} role="img" aria-label="Average retrieval vs generation latency" />;
}

function DonutChart({ segments, theme, ariaLabel }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const total = segments.reduce((sum, s) => sum + s.value, 0);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;
    const muted = cssVar("--muted");

    chartRef.current = new window.Chart(canvasRef.current, {
      type: "doughnut",
      data: {
        labels: segments.map((s) => s.label),
        datasets: [{ data: segments.map((s) => s.value), backgroundColor: segments.map((s) => cssVar(s.colorVar)), borderWidth: 0 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        cutout: "70%",
        plugins: { legend: { position: "bottom", labels: { color: muted, boxWidth: 10, padding: 12, font: { size: 11.5 } } } },
      },
    });

    return () => chartRef.current && chartRef.current.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(segments.map((s) => s.value)), theme]);

  if (total === 0) return <ChartEmptyState message="No data yet." />;
  return <canvas ref={canvasRef} role="img" aria-label={ariaLabel} />;
}

function Dashboard({ theme, onClose }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchStats = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(apiUrl("/stats"))
      .then((res) => {
        if (!res.ok) throw new Error("Request failed with status " + res.status);
        return res.json();
      })
      .then((data) => setStats(data))
      .catch(() => setError("Couldn't load statistics right now. Please try again."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const hasQueries = !!(stats && stats.total_queries > 0);

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-y-auto chat-scroll">
      <div className="flex items-center gap-2.5 px-3 md:px-4 py-2.5 border-b border-[var(--border)]">
        <button
          onClick={onClose}
          className="md:hidden p-1.5 rounded-md text-[var(--ink)] hover:bg-[var(--accent-soft)] transition-colors flex-shrink-0"
          aria-label="Back to chat"
          title="Back to chat"
        >
          <Icon.PanelToggle />
        </button>
        <h1 className="flex-1 font-display text-[15px] font-semibold text-[var(--ink)]">Analytics</h1>
        <button
          onClick={fetchStats}
          disabled={loading}
          title="Refresh"
          aria-label="Refresh statistics"
          className="flex items-center gap-1.5 text-[12px] font-medium text-[var(--muted)] hover:text-[var(--ink)] disabled:opacity-50 transition-colors px-2 py-1 rounded-md hover:bg-[var(--accent-soft)]"
        >
          {loading ? <Icon.Spinner /> : <Icon.Refresh />} Refresh
        </button>
      </div>

      <div className="max-w-[900px] w-full mx-auto px-4 md:px-6 py-5 flex flex-col gap-5">
        {error && (
          <div className="rounded-xl border border-[var(--alert)] bg-[var(--alert-soft)] px-4 py-3 text-[13px] text-[var(--ink)] flex items-center justify-between gap-3">
            {error}
            <button onClick={fetchStats} className="font-medium text-[var(--alert)] hover:underline flex-shrink-0">
              Try again
            </button>
          </div>
        )}

        {!stats && loading && (
          <div className="flex items-center justify-center py-20 text-[var(--muted)] gap-2 text-[13px]">
            <Icon.Spinner /> Loading statistics…
          </div>
        )}

        {stats && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="Total queries" value={stats.total_queries} icon={<Icon.Bot />} />
              <StatCard
                label="Avg response time"
                value={hasQueries ? `${(stats.avg_total_ms / 1000).toFixed(1)}s` : "—"}
                icon={<Icon.Bolt />}
              />
              <StatCard
                label="Cache hit rate"
                value={hasQueries ? `${Math.round(stats.cache_hit_rate * 100)}%` : "—"}
                icon={<Icon.Refresh />}
              />
              <StatCard label="Documents indexed" value={stats.documents_indexed} icon={<Icon.FileText />} />
            </div>

            {stats.emergency_count > 0 && (
              <div className="rounded-xl border border-[var(--alert)] bg-[var(--alert-soft)] px-4 py-2.5 text-[12.5px] text-[var(--ink)] flex items-center gap-2">
                <Icon.Alert width="15" height="15" />
                <span>
                  <strong>{stats.emergency_count}</strong> conversation{stats.emergency_count === 1 ? "" : "s"}{" "}
                  triggered the emergency guardrail and were routed straight to crisis resources.
                </span>
              </div>
            )}

            <div className="rounded-2xl border border-[var(--border)] bg-[var(--paper-raised)] p-4">
              <p className="text-[12px] font-medium text-[var(--muted)] mb-3">Queries, last 14 days</p>
              <div className="h-[200px]">
                <QueriesChart daily={stats.daily} theme={theme} />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="rounded-2xl border border-[var(--border)] bg-[var(--paper-raised)] p-4 md:col-span-1">
                <p className="text-[12px] font-medium text-[var(--muted)] mb-3">Avg latency by stage</p>
                <div className="h-[150px]">
                  {hasQueries ? (
                    <LatencyChart avgRetrieval={stats.avg_retrieval_ms} avgGeneration={stats.avg_generation_ms} theme={theme} />
                  ) : (
                    <ChartEmptyState message="Ask a few questions to see a latency breakdown here." />
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-[var(--border)] bg-[var(--paper-raised)] p-4 md:col-span-1">
                <p className="text-[12px] font-medium text-[var(--muted)] mb-3">Cache performance</p>
                <div className="h-[150px]">
                  {hasQueries ? (
                    <DonutChart
                      theme={theme}
                      ariaLabel="Cache hits vs misses"
                      segments={[
                        { label: "Cache hits", value: stats.cache_hits, colorVar: "--accent" },
                        { label: "Live answers", value: stats.total_queries - stats.cache_hits, colorVar: "--border" },
                      ]}
                    />
                  ) : (
                    <ChartEmptyState message="Cache hits vs. freshly-generated answers will show up here." />
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-[var(--border)] bg-[var(--paper-raised)] p-4 md:col-span-1">
                <p className="text-[12px] font-medium text-[var(--muted)] mb-3">Feedback</p>
                <div className="h-[150px]">
                  {stats.feedback_up + stats.feedback_down > 0 ? (
                    <DonutChart
                      theme={theme}
                      ariaLabel="Thumbs up vs thumbs down feedback"
                      segments={[
                        { label: "Helpful", value: stats.feedback_up, colorVar: "--accent" },
                        { label: "Not helpful", value: stats.feedback_down, colorVar: "--alert" },
                      ]}
                    />
                  ) : (
                    <ChartEmptyState message="👍 / 👎 feedback on answers will show up here." />
                  )}
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-[var(--border)] bg-[var(--paper-raised)] p-4 flex items-center justify-between flex-wrap gap-2">
              <p className="text-[12px] text-[var(--muted)]">
                <span className="font-medium text-[var(--ink)]">{stats.chunks_indexed}</span> chunks indexed across{" "}
                <span className="font-medium text-[var(--ink)]">{stats.documents_indexed}</span> document
                {stats.documents_indexed === 1 ? "" : "s"}.{" "}
                <span className="font-medium text-[var(--ink)]">{stats.cache_size}</span> answers currently cached.
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [conversations, setConversations] = useState(loadConversations);
  const [activeId, setActiveId] = useState(() => localStorage.getItem(ACTIVE_KEY));
  const [theme, toggleTheme] = useTheme();
  const sidebar = useSidebarWidth();
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 768);
  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [streamingDraft, setStreamingDraft] = useState(null); // { convoId, content, sources, emergency, cached, started }
  const [emergencyActive, setEmergencyActive] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedDocIds, setSelectedDocIds] = useState(() => new Set());
  const knownDocIdsRef = useRef(null); // null until the first /documents fetch resolves
  const [uploadingCount, setUploadingCount] = useState(0);
  const [uploadNotices, setUploadNotices] = useState([]); // [{ id, filename, status: 'uploading'|'done'|'error', message }]
  const [view, setView] = useState("chat"); // "chat" | "dashboard"
  const scrollRef = useRef(null);
  const fileInputRef = useRef(null);

  const voice = useVoiceInput((transcript) => setInputValue(transcript));

  const fetchDocuments = useCallback(() => {
    fetch(apiUrl("/documents"))
      .then((res) => res.json())
      .then((data) => {
        const docs = data.documents || [];
        const currentIds = new Set(docs.map((d) => d.id));
        setDocuments(docs);
        setSelectedDocIds((prev) => {
          if (knownDocIdsRef.current === null) {
            // First successful load this session: nothing to reconcile
            // against yet, so default to "answer from every document" --
            // the sensible starting point until someone deliberately
            // narrows it down.
            return currentIds;
          }
          const next = new Set();
          currentIds.forEach((id) => {
            // Stays selected if it already was; a document that's new
            // since the last time we checked (e.g. just uploaded) starts
            // selected too, so it's included without an extra click. A
            // document that existed before and was deliberately unticked
            // stays unticked.
            if (prev.has(id) || !knownDocIdsRef.current.has(id)) next.add(id);
          });
          return next;
        });
        knownDocIdsRef.current = currentIds;
      })
      .catch(() => {
        // Sidebar just keeps showing whatever it last knew about — the
        // upload button itself still works even if this particular
        // refresh failed, so this is quiet-by-design rather than an
        // error the user needs to act on.
      });
  }, []);

  const toggleSelectDocument = useCallback((docId) => {
    setSelectedDocIds((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      return next;
    });
  }, []);

  const toggleSelectAllDocuments = useCallback(
    (selectAll) => {
      setSelectedDocIds(selectAll ? new Set(documents.map((d) => d.id)) : new Set());
    },
    [documents]
  );

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
    else localStorage.removeItem(ACTIVE_KEY);
  }, [activeId]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [conversations, activeId, streamingDraft]);

  const activeConversation = activeId ? conversations[activeId] : null;

  const commitBotMessage = useCallback((convoId, botMsg) => {
    setConversations((prev) => {
      const c = prev[convoId];
      if (!c) return prev;
      return { ...prev, [convoId]: { ...c, messages: [...c.messages, { id: uid(), ...botMsg }], updatedAt: Date.now() } };
    });
  }, []);

  const sendMessage = useCallback(
    async (rawText) => {
      const text = (rawText || "").trim();
      if (!text || isSending) return;

      let convoId = activeId;
      let convo = convoId ? conversations[convoId] : null;
      if (!convo) {
        convo = createConversation();
        convoId = convo.id;
      }

      const isFirstMessage = convo.messages.length === 0;
      const historyPayload = convo.messages.map((m) => ({ role: m.role, content: m.content }));
      const userMsg = { id: uid(), role: "user", content: text };

      setConversations((prev) => ({
        ...prev,
        [convoId]: {
          ...convo,
          title: isFirstMessage ? truncateTitle(text) : convo.title,
          messages: [...convo.messages, userMsg],
          updatedAt: Date.now(),
        },
      }));
      setActiveId(convoId);
      setInputValue("");
      setIsSending(true);
      setStreamingDraft({ convoId, content: "", sources: [], emergency: false, cached: false, started: false });

      // Declared outside the try block on purpose: if the connection drops
      // mid-stream, the catch block below still needs whatever content we'd
      // already accumulated, instead of throwing it away.
      let draft = { content: "", sources: [], emergency: false, cached: false };

      // A stalled connection (server hung, network died silently) would
      // otherwise leave the user staring at a typing indicator forever.
      const controller = new AbortController();
      const stallTimer = setTimeout(() => controller.abort(), 60000);

      try {
        const res = await fetch(apiUrl("/get/stream"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            history: historyPayload,
            // null (not []) when everything is ticked, so this request is
            // indistinguishable from one sent before document selection
            // existed at all -- see _normalize_document_ids in app.py.
            document_ids: selectedDocIds.size > 0 && selectedDocIds.size < documents.length
              ? Array.from(selectedDocIds)
              : null,
          }),
          signal: controller.signal,
        });

        if (res.status === 429) {
          commitBotMessage(convoId, {
            role: "bot",
            content: "You're sending messages a little fast — please wait a moment and try again.",
            sources: [],
            emergency: false,
          });
          return;
        }
        if (!res.ok || !res.body) throw new Error("stream request failed with status " + res.status);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const { parsed, remaining } = parseSSEBuffer(buffer);
          buffer = remaining;
          for (const payload of parsed) {
            draft = applyStreamEvent(draft, payload);
            setStreamingDraft({ convoId, ...draft, started: true });
          }
        }

        if (draft.errorMessage) {
          // The backend caught a failure server-side (e.g. Pinecone/Groq
          // outage). If some good content had already streamed in before
          // the failure, keep it (with a note) instead of throwing away a
          // mostly-complete answer — same principle as the dropped-
          // connection case below. If nothing had streamed yet, just show
          // the clean error message the backend sent — never a stack
          // trace or raw payload either way.
          commitBotMessage(convoId, {
            role: "bot",
            content: draft.content ? draft.content + `\n\n*(${draft.errorMessage})*` : draft.errorMessage,
            sources: draft.content ? draft.sources : [],
            emergency: false,
          });
        } else if (!draft.content) {
          // Nothing usable came back and no explicit error was sent either
          // — fail safe rather than showing an empty bubble.
          commitBotMessage(convoId, {
            role: "bot",
            content: "I didn't get a response that time. Please try asking again.",
            sources: [],
            emergency: false,
          });
        } else {
          commitBotMessage(convoId, { role: "bot", question: text, ...draft });
          if (draft.emergency) setEmergencyActive(true);
        }
      } catch (err) {
        if (draft.content) {
          // We had a partial answer streaming in before the connection
          // dropped — keep it, with a note, instead of discarding it.
          commitBotMessage(convoId, {
            role: "bot",
            content: draft.content + "\n\n*(Connection interrupted — this answer may be incomplete.)*",
            sources: draft.sources,
            emergency: draft.emergency,
          });
        } else if (err && err.name === "AbortError") {
          commitBotMessage(convoId, {
            role: "bot",
            content: "That took longer than expected. Please try again.",
            sources: [],
            emergency: false,
          });
        } else {
          commitBotMessage(convoId, {
            role: "bot",
            content: "I couldn't reach the server. Please check your connection and try again.",
            sources: [],
            emergency: false,
          });
        }
      } finally {
        clearTimeout(stallTimer);
        setIsSending(false);
        setStreamingDraft(null);
      }
    },
    [activeId, conversations, isSending, commitBotMessage, selectedDocIds, documents]
  );

  const triggerUpload = useCallback(() => {
    if (fileInputRef.current) fileInputRef.current.click();
  }, []);

  const uploadOneFile = useCallback((file) => {
    const noticeId = uid();
    setUploadNotices((prev) => [...prev, { id: noticeId, filename: file.name, status: "uploading", message: "Uploading and indexing…" }]);
    setUploadingCount((n) => n + 1);

    const formData = new FormData();
    formData.append("file", file);

    // Embedding a PDF locally can genuinely take a while on a slow
    // free-tier CPU (see README) — a longer timeout than a normal chat
    // turn, so a big-but-valid file isn't cut off mid-index.
    const controller = new AbortController();
    const stallTimer = setTimeout(() => controller.abort(), 120000);

    fetch(apiUrl("/documents/upload"), { method: "POST", body: formData, signal: controller.signal })
      .then(async (res) => {
        if (res.status === 429) {
          throw new Error("Too many uploads right now — please wait a moment and try again.");
        }
        let data = null;
        try {
          data = await res.json();
        } catch {
          // fall through to the generic error below
        }
        if (!res.ok || !data || !data.ok) {
          throw new Error((data && data.message) || "Upload failed. Please try again.");
        }
        return data.document;
      })
      .then((doc) => {
        setDocuments((prev) => [doc, ...prev]);
        const pages = doc.page_count;
        setUploadNotices((prev) =>
          prev.map((n) =>
            n.id === noticeId
              ? {
                  ...n,
                  status: "done",
                  message: `Indexed — ${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}${
                    pages != null ? ` from ${pages} page${pages === 1 ? "" : "s"}` : ""
                  }. Ask away.`,
                }
              : n
          )
        );
        setTimeout(() => setUploadNotices((prev) => prev.filter((n) => n.id !== noticeId)), 5000);
      })
      .catch((err) => {
        const message = err && err.name === "AbortError" ? "That took longer than expected. Please try a smaller file." : (err && err.message) || "Upload failed. Please try again.";
        setUploadNotices((prev) => prev.map((n) => (n.id === noticeId ? { ...n, status: "error", message } : n)));
      })
      .finally(() => {
        clearTimeout(stallTimer);
        setUploadingCount((n) => Math.max(0, n - 1));
      });
  }, []);

  const handleFilesSelected = useCallback(
    (fileList) => {
      const files = Array.from(fileList || []);
      for (const file of files) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          setUploadNotices((prev) => [
            ...prev,
            { id: uid(), filename: file.name, status: "error", message: "Only PDF files are supported." },
          ]);
          continue;
        }
        uploadOneFile(file);
      }
    },
    [uploadOneFile]
  );

  const handleDeleteDocument = useCallback(
    (id) => {
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      setSelectedDocIds((prev) => {
        if (!prev.has(id)) return prev;
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      fetch(apiUrl(`/documents/${id}`), { method: "DELETE" }).then((res) => {
        if (!res.ok) fetchDocuments(); // reconcile the sidebar if the delete didn't actually happen
      }).catch(() => fetchDocuments());
    },
    [fetchDocuments]
  );

  const dismissUploadNotice = useCallback((id) => {
    setUploadNotices((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const closeSidebarOnMobile = () => {
    if (window.innerWidth < 768) setSidebarOpen(false);
  };

  const handleNewChat = () => {
    setActiveId(null);
    setEmergencyActive(false);
    setView("chat");
    closeSidebarOnMobile();
  };

  const handleSelectConversation = (id) => {
    setActiveId(id);
    setEmergencyActive(false);
    setView("chat");
    closeSidebarOnMobile();
  };

  const handleOpenDashboard = () => {
    setView("dashboard");
    closeSidebarOnMobile();
  };

  const handleDeleteConversation = (id) => {
    setConversations((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    if (id === activeId) setActiveId(null);
  };

  const handleFeedback = (message, rating) => {
    fetch(apiUrl("/feedback"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: message.question || "", answer: message.content, rating }),
    }).catch(() => {});
  };

  const messages = activeConversation ? activeConversation.messages : [];
  const isStreamingHere = streamingDraft && streamingDraft.convoId === activeId;
  const showHero = messages.length === 0 && !isStreamingHere;

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--paper)] text-[var(--ink)]">
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        multiple
        hidden
        onChange={(e) => {
          handleFilesSelected(e.target.files);
          e.target.value = ""; // reset so selecting the same file again still fires onChange
        }}
      />

      <Sidebar
        theme={theme}
        onToggleTheme={toggleTheme}
        conversations={conversations}
        activeId={activeId}
        onSelect={handleSelectConversation}
        onDelete={handleDeleteConversation}
        onNewChat={handleNewChat}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        width={sidebar.width}
        resizing={sidebar.resizing}
        onStartResize={sidebar.startResizing}
        documents={documents}
        selectedDocIds={selectedDocIds}
        onToggleSelectDocument={toggleSelectDocument}
        onToggleSelectAllDocuments={toggleSelectAllDocuments}
        onDeleteDocument={handleDeleteDocument}
        onUploadClick={triggerUpload}
        uploadingDocument={uploadingCount > 0}
        view={view}
        onOpenDashboard={handleOpenDashboard}
      />

      {view === "dashboard" ? (
        <Dashboard theme={theme} onClose={() => setView("chat")} />
      ) : (
        <main className="flex-1 flex flex-col min-w-0 relative">
        <div className="flex items-center gap-2.5 px-3 md:px-4 py-2.5 border-b border-[var(--border)]">
          <button
            onClick={() => setSidebarOpen((prev) => !prev)}
            className={
              (sidebarOpen ? "md:hidden " : "") +
              "p-1.5 rounded-md text-[var(--ink)] hover:bg-[var(--accent-soft)] transition-colors flex-shrink-0"
            }
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
          >
            <Icon.PanelToggle />
          </button>
          <p className="flex-1 text-center text-[12.5px] font-medium text-[var(--muted)]">
            <span className="text-[var(--amber)]">Reminder —</span> for general information only. Always consult a
            licensed doctor for diagnosis or treatment.
          </p>
        </div>

        <EmergencyBanner show={emergencyActive} />

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 md:px-6 pt-4 chat-scroll">
          <div className="max-w-[760px] mx-auto">
            {showHero && <Hero onSuggestionClick={sendMessage} onUploadClick={triggerUpload} />}

            <div className="flex flex-col gap-5 pb-4">
              {messages.map((m) => (
                <Message key={m.id} message={m} onFeedback={handleFeedback} onUploadClick={triggerUpload} />
              ))}

              {isStreamingHere &&
                (streamingDraft.started ? (
                  <Message message={{ role: "bot", ...streamingDraft }} onFeedback={handleFeedback} isStreaming />
                ) : (
                  <TypingIndicator />
                ))}
            </div>
          </div>
        </div>

        {uploadNotices.length > 0 && (
          <div className="max-w-[760px] w-full mx-auto px-4 md:px-6 flex flex-col gap-2 pb-1">
            {uploadNotices.map((notice) => (
              <UploadNotice key={notice.id} notice={notice} onDismiss={() => dismissUploadNotice(notice.id)} />
            ))}
          </div>
        )}

        <Composer
          value={inputValue}
          onChange={setInputValue}
          onSubmit={() => sendMessage(inputValue)}
          voice={voice}
          onAttachClick={triggerUpload}
          uploading={uploadingCount > 0}
        />
      </main>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mount
// ---------------------------------------------------------------------------

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
