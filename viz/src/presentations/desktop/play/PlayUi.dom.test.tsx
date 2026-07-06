// @vitest-environment jsdom
//
// Component/UX coverage for two pure presentational play-panel components. These
// take plain props (no engine, no worker), so they're fully deterministic and
// serve as the pattern for testing the play UI's rendering logic.
import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ModelMoveInfo } from '../../../core/index.ts';
import { MoveHistory } from './MoveHistory.tsx';
import { ValueGauge } from './ValueGauge.tsx';

function modelMove(value: number): ModelMoveInfo {
  return { uci: 'e2e4', san: 'e4', fromIdx: 12, toIdx: 28, value, policyProb: 0.5 };
}

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
});

describe('ValueGauge', () => {
  it('shows an em dash when the model has not moved', () => {
    render(<ValueGauge model={null} modelColor="w" />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('re-frames a side-to-move value into White(+)/Black(−)', () => {
    // Model played as White with +0.60 → White is ahead → "+0.60".
    const { unmount } = render(<ValueGauge model={modelMove(0.6)} modelColor="w" />);
    expect(screen.getByText('+0.60')).toBeInTheDocument();
    unmount();

    // Same raw +0.60 but the model played Black → White is behind → "-0.60".
    render(<ValueGauge model={modelMove(0.6)} modelColor="b" />);
    expect(screen.getByText('-0.60')).toBeInTheDocument();
  });
});
