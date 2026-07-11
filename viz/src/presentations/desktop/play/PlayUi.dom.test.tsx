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

  it('omits value badges entirely when no value history is provided', () => {
    render(<MoveHistory sanHistory={['e4', 'e5']} />);
    expect(document.querySelector('.move-value')).toBeNull();
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
