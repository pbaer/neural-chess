import { describe, expect, it } from 'vitest';
import { buildIssueUrl } from './Feedback.tsx';

describe('buildIssueUrl', () => {
  it('targets the repo new-issue endpoint with encoded title', () => {
    const url = new URL(buildIssueUrl('feature', 'Add a flip-board button', ''));
    expect(url.origin + url.pathname).toBe('https://github.com/pbaer/neural-chess/issues/new');
    expect(url.searchParams.get('title')).toBe('Add a flip-board button');
  });

  it('labels a bug with bug,feedback and a feature with enhancement,feedback', () => {
    expect(new URL(buildIssueUrl('bug', 't', '')).searchParams.get('labels')).toBe('bug,feedback');
    expect(new URL(buildIssueUrl('feature', 't', '')).searchParams.get('labels')).toBe('enhancement,feedback');
  });

  it('uses a heading + the details in the body, with a placeholder when empty', () => {
    const withDetails = new URL(buildIssueUrl('bug', 't', 'It froze')).searchParams.get('body') ?? '';
    expect(withDetails).toContain('### Bug report');
    expect(withDetails).toContain('It froze');

    const noDetails = new URL(buildIssueUrl('feature', 't', '')).searchParams.get('body') ?? '';
    expect(noDetails).toContain('### Feature request');
    expect(noDetails).toContain('_No additional details provided._');
  });

  it('appends context only when provided (bug reports)', () => {
    const ctx = () => '**Environment**\n- Model: v3.1-nano';
    const body = new URL(buildIssueUrl('bug', 't', 'd', ctx)).searchParams.get('body') ?? '';
    expect(body).toContain('---');
    expect(body).toContain('- Model: v3.1-nano');

    const noCtx = new URL(buildIssueUrl('feature', 't', 'd')).searchParams.get('body') ?? '';
    expect(noCtx).not.toContain('---');
  });

  it('never lets a throwing context callback break URL construction', () => {
    const boom = () => {
      throw new Error('nope');
    };
    const body = new URL(buildIssueUrl('bug', 't', 'd', boom)).searchParams.get('body') ?? '';
    expect(body).toContain('### Bug report');
    expect(body).not.toContain('---');
  });
});
