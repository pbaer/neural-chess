// @vitest-environment jsdom
//
// Interaction/UX coverage for the feedback widget: opening the dialog, switching
// type, title validation, the composed GitHub URL on submit, context attachment,
// and every close path. Fully deterministic — no network, no LLM. window.open is
// stubbed so nothing actually navigates.
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { FeedbackLink } from './Feedback.tsx';

// Stub window.open so submits never navigate; a fresh mock per test.
const openMock = vi.fn<(url?: string | URL, target?: string, features?: string) => Window | null>();

beforeEach(() => {
  openMock.mockReset();
  window.open = openMock;
});

/** The URL the widget passed to window.open on the most recent submit. */
function submittedUrl(): URL {
  expect(openMock).toHaveBeenCalledTimes(1);
  return new URL(String(openMock.mock.calls[0][0]));
}

describe('FeedbackLink', () => {
  it('shows the trigger and opens the dialog on click', async () => {
    const user = userEvent.setup();
    render(<FeedbackLink />);
    expect(screen.queryByRole('dialog')).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Feedback' }));
    expect(screen.getByRole('dialog', { name: 'Send feedback' })).toBeInTheDocument();
  });

  it('disables submit until a title is entered', async () => {
    const user = userEvent.setup();
    render(<FeedbackLink />);
    await user.click(screen.getByRole('button', { name: 'Feedback' }));
    const submit = screen.getByRole('button', { name: /Open GitHub issue/ });
    expect(submit).toBeDisabled();
    await user.type(screen.getByRole('textbox', { name: /Title/ }), 'It froze');
    expect(submit).toBeEnabled();
  });

  it('submits a bug: correct labels, title, heading, and appended context', async () => {
    const user = userEvent.setup();
    render(<FeedbackLink getContext={() => '**Environment**\n- Model: test-model'} />);
    await user.click(screen.getByRole('button', { name: 'Feedback' }));
    await user.type(screen.getByRole('textbox', { name: /Title/ }), 'Board freezes');
    await user.type(screen.getByRole('textbox', { name: /Details/ }), 'after castling');
    await user.click(screen.getByRole('button', { name: /Open GitHub issue/ }));

    const url = submittedUrl();
    expect(url.origin + url.pathname).toBe('https://github.com/pbaer/neural-chess/issues/new');
    expect(url.searchParams.get('title')).toBe('Board freezes');
    expect(url.searchParams.get('labels')).toBe('bug,feedback');
    const body = url.searchParams.get('body') ?? '';
    expect(body).toContain('### Bug report');
    expect(body).toContain('after castling');
    expect(body).toContain('- Model: test-model');
    // Dialog closes after submit.
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('submits a feature: enhancement labels and no environment context', async () => {
    const user = userEvent.setup();
    render(<FeedbackLink getContext={() => '**Environment**\n- Model: test-model'} />);
    await user.click(screen.getByRole('button', { name: 'Feedback' }));
    await user.click(screen.getByRole('radio', { name: /Feature/ }));
    await user.type(screen.getByRole('textbox', { name: /Title/ }), 'Flip board button');
    await user.click(screen.getByRole('button', { name: /Open GitHub issue/ }));

    const url = submittedUrl();
    expect(url.searchParams.get('labels')).toBe('enhancement,feedback');
    const body = url.searchParams.get('body') ?? '';
    expect(body).toContain('### Feature request');
    // Feature reports never attach the play-page context, even when provided.
    expect(body).not.toContain('test-model');
  });

  it('closes on Escape without submitting', async () => {
    const user = userEvent.setup();
    render(<FeedbackLink />);
    await user.click(screen.getByRole('button', { name: 'Feedback' }));
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(openMock).not.toHaveBeenCalled();
  });

  it('closes on Cancel and on overlay click without submitting', async () => {
    const user = userEvent.setup();
    const { container } = render(<FeedbackLink />);

    await user.click(screen.getByRole('button', { name: 'Feedback' }));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).toBeNull();

    await user.click(screen.getByRole('button', { name: 'Feedback' }));
    // The backdrop closes on mousedown; clicking the dialog body must NOT.
    fireEvent.mouseDown(screen.getByRole('dialog'));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.mouseDown(container.querySelector('.fb-overlay') as Element);
    expect(screen.queryByRole('dialog')).toBeNull();

    expect(openMock).not.toHaveBeenCalled();
  });
});
