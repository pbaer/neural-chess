// @vitest-environment jsdom
//
// Component/UX coverage for two pure presentational play-panel components. These
// take plain props (no engine, no worker), so they're fully deterministic and
// serve as the pattern for testing the play UI's rendering logic.
import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MoveHistory } from './MoveHistory.tsx';
import { ValueGauge } from './ValueGauge.tsx';

describe('MoveHistory', () => {
  it('shows the empty state with no moves', () => {
    render(<MoveHistory sanHistory={[]} />);
    expect(screen.getByText('No moves yet.')).toBeInTheDocument();
  });

  it('pairs SANs into numbered full-moves, leaving a trailing white move unpaired', () => {
    render(<MoveHistory sanHistory={['e4', 'e5', 'Nf3']} />);
    const rows = screen.getAllByRole('listitem');
    expect(rows).toHaveLength(2);
    // Row 1: "1." e4 e5
    expect(within(rows[0]).getByText('1.')).toBeInTheDocument();
    expect(within(rows[0]).getByText('e4')).toBeInTheDocument();
    expect(within(rows[0]).getByText('e5')).toBeInTheDocument();
    // Row 2: "2." Nf3 (black cell empty)
    expect(within(rows[1]).getByText('2.')).toBeInTheDocument();
    expect(within(rows[1]).getByText('Nf3')).toBeInTheDocument();
  });

  it('shows the model value head per move at its ply, signed in the White(+)/Black(−) frame', () => {
    // ply 2 (black's 1st move) = +0.30 → White ahead; ply 4 = -0.55 → Black ahead.
    render(
      <MoveHistory
        sanHistory={['e4', 'e5', 'Nf3', 'Nc6']}
        valueHistory={[
          { ply: 2, whiteValue: 0.3 },
          { ply: 4, whiteValue: -0.55 },
        ]}
      />,
    );
    expect(screen.getByText('+0.30')).toBeInTheDocument();
    expect(screen.getByText('-0.55')).toBeInTheDocument();
    // A move without a reading shows no value badge.
    expect(screen.queryByText('+0.00')).not.toBeInTheDocument();
    // The badge carries a descriptive tooltip naming the favored side.
    expect(screen.getByTitle('Value head: +0.30 (White better)')).toBeInTheDocument();
    expect(screen.getByTitle('Value head: -0.55 (Black better)')).toBeInTheDocument();
  });

  it('lays each row out as five ordered cells: num, White SAN, White value, Black SAN, Black value', () => {
    render(
      <MoveHistory
        sanHistory={['e4', 'e5']}
        valueHistory={[
          { ply: 1, whiteValue: 0.12 },
          { ply: 2, whiteValue: -0.4 },
        ]}
      />,
    );
    // The value bar lives in its own dedicated grid column — not stuffed inside the SAN
    // cell — so every column (including the two value columns) lines up across rows.
    const row = screen.getAllByRole('listitem')[0];
    const cells = Array.from(row.children);
    expect(cells).toHaveLength(5);
    expect(cells[0]).toHaveClass('move-num');
    expect(cells[1]).toHaveClass('move-san');
    expect(cells[1]).toHaveTextContent('e4');
    expect(cells[2]).toHaveClass('move-value');
    expect(within(cells[2] as HTMLElement).getByText('+0.12')).toBeInTheDocument();
    expect(cells[3]).toHaveClass('move-san');
    expect(cells[3]).toHaveTextContent('e5');
    expect(cells[4]).toHaveClass('move-value');
    expect(within(cells[4] as HTMLElement).getByText('-0.40')).toBeInTheDocument();
  });

  it('keeps the value columns aligned by holding an empty placeholder cell when a move has no reading', () => {
    // Only White's move has a reading; the layout must still emit a value cell for Black so
    // the shared grid columns (and thus every bar's x offset) stay put row to row.
    render(
      <MoveHistory
        sanHistory={['Qxd8+', 'e5']}
        valueHistory={[{ ply: 1, whiteValue: 0.12 }]}
      />,
    );
    const cells = Array.from(screen.getAllByRole('listitem')[0].children);
    expect(cells).toHaveLength(5);
    // White's value cell carries the badge; Black's is an empty (aria-hidden) placeholder.
    expect(cells[2]).toHaveClass('move-value');
    expect(cells[2]).not.toHaveClass('move-value-empty');
    expect(cells[4]).toHaveClass('move-value-empty');
    expect(cells[4]).toBeEmptyDOMElement();
    // A long SAN ("Qxd8+") does not push the value into a different column — the badge is
    // its own cell, so the fixed-width signed number stays a clean 5 characters.
    expect(screen.getByText('+0.12').textContent).toHaveLength(5);
  });

  it('omits value badges entirely when no value history is provided', () => {
    render(<MoveHistory sanHistory={['e4', 'e5']} />);
    // No actual badge (bar + number) renders; the value columns hold only empty placeholders
    // (which keep the table columns aligned) — so no bar and no signed number appear.
    expect(document.querySelector('.move-value-bar')).toBeNull();
    expect(document.querySelector('.move-value-num')).toBeNull();
    expect(document.querySelectorAll('.move-value:not(.move-value-empty)')).toHaveLength(0);
  });
});

describe('ValueGauge', () => {
  it('shows an em dash when the model has not moved (null value)', () => {
    render(<ValueGauge whiteValue={null} />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('renders the White-framed value it is handed', () => {
    // +0.60 → White is ahead → "+0.60".
    const { unmount } = render(<ValueGauge whiteValue={0.6} />);
    expect(screen.getByText('+0.60')).toBeInTheDocument();
    unmount();

    // −0.60 → White is behind → "-0.60".
    render(<ValueGauge whiteValue={-0.6} />);
    expect(screen.getByText('-0.60')).toBeInTheDocument();
  });
});
