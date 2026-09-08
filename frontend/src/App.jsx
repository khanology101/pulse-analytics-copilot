import { useState } from "react";
import Upload from "./Upload.jsx";
import Chat from "./Chat.jsx";

export default function App() {
  const [session, setSession] = useState(null);

  return (
    <div className="app">
      <header className="app-header fade-in-up">
        <div className="brand">
          <span className="brand__mark">PULSE</span>
          <span className="brand__tagline">BUSINESS ANALYTICS COPILOT</span>
        </div>
        {session && (
          <div className="session-meta">
            <span>
              {session.filename} · {session.rows} rows · {session.columns.length} cols
            </span>
            <button className="btn btn--ghost" onClick={() => setSession(null)}>
              NEW FILE
            </button>
          </div>
        )}
      </header>

      <main className="app-main">
        {session ? <Chat sessionId={session.session_id} /> : <Upload onUploaded={setSession} />}
      </main>
    </div>
  );
}
