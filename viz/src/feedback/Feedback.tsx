// In-app feedback entry point. Shared by BOTH Vite entries (the play page and the
// story page), so it lives outside src/presentations and carries its own
// self-contained, theme-aware styles (feedback.css) that depend only on the CSS
// variables both stylesheets define.
//
// It never talks to the GitHub API (the site is a static Pages deploy with no
// backend / no secret). Instead it composes a prefilled "new issue" URL and opens
// GitHub in a new tab; the reporter submits under their own account. If the issue
// author is the maintainer, a workflow auto-starts the coding agent (see
// .github/workflows/agent-implement.yml); otherwise the maintainer triggers it
// manually with the `agent-go` label.

import { useEffect, useState } from 'react';
import './feedback.css';

const REPO_URL = 'https://github.com/pbaer/neural-chess';

type Kind = 'bug' | 'feature';

const KIND_META: Record<Kind, { label: string; emoji: string; labels: string; heading: string }> = {
  bug: { label: 'Bug', emoji: '🐞', labels: 'bug,feedback', heading: 'Bug report' },
  feature: { label: 'Feature', emoji: '💡', labels: 'enhancement,feedback', heading: 'Feature request' },
};

/**
 * A muted footer-style trigger that opens the feedback dialog.
 *
 * @param getContext optional callback returning extra Markdown appended to the
 *   issue body (e.g. the live model id + FEN for a bug). Called at submit time so
 *   it captures current state, and only used for bug reports.
 */
export function FeedbackLink({ getContext }: { getContext?: () => string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="fb-link" onClick={() => setOpen(true)}>
        Feedback
      </button>
      {open && <FeedbackDialog getContext={getContext} onClose={() => setOpen(false)} />}
    </>
  );
}

function FeedbackDialog({ getContext, onClose }: { getContext?: () => string; onClose: () => void }) {
  const [kind, setKind] = useState<Kind>('bug');
  const [title, setTitle] = useState('');
  const [details, setDetails] = useState('');

  // Close on Escape; lock background scroll while open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const canSubmit = title.trim().length > 0;

  const submit = () => {
    if (!canSubmit) return;
    const url = buildIssueUrl(kind, title.trim(), details.trim(), kind === 'bug' ? getContext : undefined);
    window.open(url, '_blank', 'noopener,noreferrer');
    onClose();
  };

  return (
    <div className="fb-overlay" onMouseDown={onClose}>
      <div
        className="fb-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Send feedback"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="fb-head">
          <h2 className="fb-title">Send feedback</h2>
          <button type="button" className="fb-x" aria-label="Close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="fb-kind" role="radiogroup" aria-label="Feedback type">
          {(Object.keys(KIND_META) as Kind[]).map((k) => (
            <button
              key={k}
              type="button"
              role="radio"
              aria-checked={kind === k}
              className={'fb-kind-btn' + (kind === k ? ' fb-kind-on' : '')}
              onClick={() => setKind(k)}
            >
              <span aria-hidden="true">{KIND_META[k].emoji}</span> {KIND_META[k].label}
            </button>
          ))}
        </div>

        <label className="fb-field">
          <span className="fb-label">Title</span>
          <input
            className="fb-input"
            type="text"
            value={title}
            maxLength={120}
            autoFocus
            placeholder={kind === 'bug' ? 'e.g. Board freezes after castling' : 'e.g. Add a "flip board" button'}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') submit();
            }}
          />
        </label>

        <label className="fb-field">
          <span className="fb-label">
            Details <span className="fb-optional">(optional)</span>
          </span>
          <textarea
            className="fb-textarea"
            value={details}
            rows={5}
            placeholder={
              kind === 'bug'
                ? 'What happened, and what did you expect? Steps to reproduce help a lot.'
                : 'What would you like, and why would it help?'
            }
            onChange={(e) => setDetails(e.target.value)}
          />
        </label>

        <p className="fb-note">
          Opens a prefilled issue on GitHub — review it there and click <strong>Submit</strong>. A free GitHub account
          is required.
        </p>

        <div className="fb-actions">
          <button type="button" className="fb-btn fb-btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="fb-btn fb-btn-primary" disabled={!canSubmit} onClick={submit}>
            Open GitHub issue →
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Compose a prefilled `issues/new` URL. Uses the classic title/body/labels prefill
 * (not the Issue-Form template route) so the auto-captured environment block lands
 * verbatim in the body and prefill is reliable. The `.github/ISSUE_TEMPLATE` forms
 * still serve people who file directly from GitHub's "New issue" button.
 */
export function buildIssueUrl(kind: Kind, title: string, details: string, getContext?: () => string): string {
  const meta = KIND_META[kind];
  const parts = [`### ${meta.heading}`, '', details || '_No additional details provided._'];

  if (getContext) {
    let ctx = '';
    try {
      ctx = getContext();
    } catch {
      ctx = '';
    }
    if (ctx.trim()) {
      parts.push('', '---', ctx.trim());
    }
  }

  const params = new URLSearchParams({
    title,
    body: parts.join('\n'),
    labels: meta.labels,
  });
  return `${REPO_URL}/issues/new?${params.toString()}`;
}
