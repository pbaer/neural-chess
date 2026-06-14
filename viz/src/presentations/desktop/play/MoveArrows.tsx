// MoveArrows — the shared on-board move-arrow renderer. One <g> of SVG <line>s
// with arrowheads, drawn in the board's 0..8 viewBox (cell centers at +0.5).
// Reused by the play Board (policy/MCTS candidate arrows) AND the Model Inspector's
// policy-head board, so every "the model wants to play this move" arrow looks the
// same. Each arrow carries a `rel` weight (0..1, scales thickness + opacity) and an
// optional `hovered` accent; the hovered arrow is drawn last (on top).

export interface ArrowSpec {
  /** Stable key (usually the move's uci). */
  key: string;
  /** Origin cell in display coordinates (top-left origin, 0..7). */
  from: { r: number; c: number };
  /** Destination cell in display coordinates. */
  to: { r: number; c: number };
  /** Relative strength in [0,1] (1 = the top move) — scales thickness + opacity. */
  rel: number;
  /** Draw with the hover accent (color + fixed bold width), on top of the rest. */
  hovered?: boolean;
}

export interface MoveArrowStyle {
  /** Base color. */
  color: string;
  /** Hover/accent color. */
  hoverColor: string;
  /** Stroke width = base + rel·span (px in viewBox units); hovered uses hoverWidth. */
  baseWidth: number;
  relWidth: number;
  hoverWidth: number;
  /** Opacity = baseOpacity + rel·relOpacity; hovered = 1. */
  baseOpacity: number;
  relOpacity: number;
}

export const CANDIDATE_ARROW_STYLE: MoveArrowStyle = {
  color: '#7ec4ff',
  hoverColor: '#ff4d4d',
  baseWidth: 0.05,
  relWidth: 0.09,
  hoverWidth: 0.16,
  baseOpacity: 0.3,
  relOpacity: 0.6,
};

export const SEARCH_ARROW_STYLE: MoveArrowStyle = {
  color: '#54d6a0',
  hoverColor: '#ffd24d',
  baseWidth: 0.04,
  relWidth: 0.12,
  hoverWidth: 0.16,
  baseOpacity: 0.28,
  relOpacity: 0.6,
};

export interface MoveArrowsProps {
  arrows: ArrowSpec[];
  /** Unique id prefix so multiple instances' arrowhead markers never collide. */
  idPrefix: string;
  style?: MoveArrowStyle;
  className?: string;
}

/** Render a set of move arrows. Coordinate space is the board's 0..8 viewBox. */
export function MoveArrows({ arrows, idPrefix, style = CANDIDATE_ARROW_STYLE, className }: MoveArrowsProps) {
  if (arrows.length === 0) return null;
  const headId = `${idPrefix}-arrowhead`;
  const headHoverId = `${idPrefix}-arrowhead-hover`;
  return (
    <g className={className} pointerEvents="none">
      <defs>
        <marker
          id={headId}
          viewBox="0 0 10 10"
          refX={7}
          refY={5}
          markerWidth={0.4}
          markerHeight={0.4}
          markerUnits="userSpaceOnUse"
          orient="auto"
        >
          <path d="M0,1 L9,5 L0,9 z" fill={style.color} />
        </marker>
        <marker
          id={headHoverId}
          viewBox="0 0 10 10"
          refX={7}
          refY={5}
          markerWidth={0.46}
          markerHeight={0.46}
          markerUnits="userSpaceOnUse"
          orient="auto"
        >
          <path d="M0,1 L9,5 L0,9 z" fill={style.hoverColor} />
        </marker>
      </defs>
      {arrows
        .slice()
        // draw the hovered arrow last so it sits on top of the others
        .sort((a, b) => Number(!!a.hovered) - Number(!!b.hovered))
        .map((a) => {
          const x1c = a.from.c + 0.5;
          const y1c = a.from.r + 0.5;
          const x2c = a.to.c + 0.5;
          const y2c = a.to.r + 0.5;
          const len = Math.hypot(x2c - x1c, y2c - y1c) || 1;
          const ux = (x2c - x1c) / len;
          const uy = (y2c - y1c) / len;
          return (
            <line
              key={a.key}
              x1={x1c + ux * 0.3}
              y1={y1c + uy * 0.3}
              x2={x2c - ux * 0.34}
              y2={y2c - uy * 0.34}
              stroke={a.hovered ? style.hoverColor : style.color}
              strokeWidth={a.hovered ? style.hoverWidth : style.baseWidth + style.relWidth * a.rel}
              strokeLinecap="round"
              opacity={a.hovered ? 1 : style.baseOpacity + style.relOpacity * a.rel}
              markerEnd={`url(#${a.hovered ? headHoverId : headId})`}
            />
          );
        })}
    </g>
  );
}
