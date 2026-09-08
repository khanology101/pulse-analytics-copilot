import { useEffect, useRef, useState } from "react";
import { streamQuery } from "./api";
import Chart from "./Chart.jsx";

export default function Chat({ sessionId }) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function patchMessage(id, patch) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m))
    );
  }

  async function submit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;

    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setMessages((prev) => [...prev, { id, question: q, charts: {}, insight: "", status: "streaming", error: null }]);
    setQuestion("");
    setBusy(true);

    try {
      await streamQuery(sessionId, q, (event, payload) => {
        if (event === "chart") {
          patchMessage(id, (m) => ({
            charts: { ...m.charts, [payload.index]: { label: payload.label, figure: payload.figure } },
          }));
        } else if (event === "chart_error") {
          patchMessage(id, (m) => ({
            charts: { ...m.charts, [payload.index]: { label: payload.label, error: payload.message } },
          }));
        } else if (event === "insight") {
          patchMessage(id, (m) => ({ insight: m.insight + payload.token }));
        } else if (event === "error") {
          patchMessage(id, { status: "error", error: payload.message });
        } else if (event === "done") {
          patchMessage(id, (m) => ({ status: m.status === "error" ? "error" : "done" }));
        }
      });
    } catch (err) {
      patchMessage(id, { status: "error", error: err.message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat">
      <div className="chat__log">
        {messages.length === 0 && (
          <div className="chat__empty">Ask a question about your data, e.g. "show me revenue by region for Q4"</div>
        )}
        {messages.map((m) => {
          const chartEntries = Object.entries(m.charts)
            .sort(([a], [b]) => Number(a) - Number(b))
            .map(([, c]) => c);
          const multi = chartEntries.length > 1;

          return (
            <div key={m.id} className="message fade-in-up">
              <div className="message__question">{m.question}</div>

              {chartEntries.length > 0 && (
                <div className={multi ? "message__charts" : undefined}>
                  {chartEntries.map((c, i) => (
                    <div key={i} className="message__chart">
                      {multi && <div className="chart-card__label">{c.label}</div>}
                      {c.error ? <div className="chart-card__error">{c.error}</div> : <Chart figure={c.figure} />}
                    </div>
                  ))}
                </div>
              )}

              {m.status === "error" ? (
                <div className="message__error">{m.error}</div>
              ) : (
                (m.insight || m.status === "streaming") && (
                  <div className="message__insight">
                    <span className="message__insight-label">Insight</span>
                    <p>
                      {m.insight}
                      {m.status === "streaming" && <span className="cursor-blink" />}
                    </p>
                  </div>
                )
              )}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={submit} className="chat__composer">
        <div className="chat__composer-bar">
          <input
            className="chat__input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask about your data..."
            disabled={busy}
          />
          <button type="submit" className="chat__send" disabled={busy || !question.trim()} aria-label="Send">
            {busy ? (
              <span className="pulse-dot" />
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 19V5M12 5L5 12M12 5L19 12" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
