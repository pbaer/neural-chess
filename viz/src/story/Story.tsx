// The "behind the scenes" story page — a narrative account of the project for a
// general audience (high-school-calculus level), written to give context to the
// Model Inspector and the play settings on the main page. Self-contained: prose
// + inline SVG figures, no external assets. Responsive layout in story.css.

import { useSyncExternalStore } from 'react';
import { FeedbackLink } from '../feedback/Feedback.tsx';
import { REPO_URL } from '../shared/repo.ts';

const BASE = import.meta.env.BASE_URL ?? '/';
const PLAY_URL = `${BASE}`;

export function Story() {
  return (
    <div className="story">
      <header className="story-top">
        <a className="story-back" href={PLAY_URL}>← Back to play</a>
        <ThemeToggle />
      </header>

      <article className="story-body">
        <h1 className="story-title">How a neural network learned chess</h1>
        <p className="story-lede">
          The little model you can play on the main page has about <strong>116,000 numbers</strong> inside it —
          smaller than a single photo on your phone. It never read a chess book, was never told what a good move
          looks like, and was never shown an engine&rsquo;s analysis. It learned the game the way a curious person
          might: by watching millions of human games and trying to guess what comes next. This is the story of how
          it got built, what worked, and — just as interesting — what didn&rsquo;t.
        </p>

        <Section title="The question">
          <p>
            Chess engines have crushed the best humans for decades, but they do it with brute force: searching
            millions of positions a second and scoring them with rules people hand-wrote (&ldquo;a rook is worth
            five pawns,&rdquo; &ldquo;control the center,&rdquo; and so on). That&rsquo;s impressive, but it
            answers a slightly boring question — <em>can a calculator out-calculate a person?</em> Yes; that was
            settled long ago.
          </p>
          <p>
            A more interesting question drives this project: <strong>how much chess can a machine figure out on its
            own, just from watching humans play?</strong> No built-in rules of thumb, no engine to copy. Only the
            raw record of games: the positions, the moves people chose, and who eventually won.
          </p>
        </Section>

        <Section title="The ground rules">
          <p>To keep the question honest, three principles held the whole way through.</p>
          <ol className="story-principles">
            <li>
              <strong>Human games only.</strong> Every example it learns from was played by people. No engine
              games, no letting the model play itself, no copying a stronger program&rsquo;s judgments.
            </li>
            <li>
              <strong>Learn only from what&rsquo;s on the board.</strong> Picture a robot that knows nothing about
              chess, watching games over someone&rsquo;s shoulder. It sees where the pieces are, it sees which
              move gets played, and it sees who wins. From that — and nothing else — it has to work out what
              matters. It is never handed &ldquo;material count&rdquo; or &ldquo;king safety.&rdquo; If those
              ideas matter, it has to <em>discover</em> them.
            </li>
            <li>
              <strong>One look, one move.</strong> By default the model glances at the position once and names its
              move — a single pass through the network, no searching ahead. (An optional &ldquo;think harder&rdquo;
              mode was added later, but even then the only judgment it uses is its own — more on that below.)
            </li>
          </ol>
          <p>
            These constraints are the whole point. Hand the model a pile of chess wisdom and you only learn how
            strong a hand-fed model can get — not the interesting question. The point is what it can figure out by
            watching.
          </p>
        </Section>

        <Section title="How a network &ldquo;sees&rdquo; a board">
          <p>
            A neural network only does arithmetic, so the first job is turning a chessboard into numbers. The
            position is fed in as a stack of small 8&times;8 grids — one grid marking where the pawns are, one for
            the knights, and so on, plus a few for things like castling rights and whose turn it is. Everything in
            that input stack is just what you could write down by looking; no judgments, only facts.
          </p>
          <Figure caption="Every move: the board becomes numbers, flows through the network, and comes out as two things — a ranked list of moves, and a single &ldquo;who&rsquo;s winning&rdquo; score.">
            <BoardToMove />
          </Figure>
          <p>
            Out the other end come two answers. One is a <strong>move preference</strong> — a score for every
            move it could make, highest for the move it likes best. The other is a single number, the{' '}
            <strong>value</strong>, its gut read on who&rsquo;s winning, from <span className="nowrap">+1
            (I&rsquo;m winning)</span> to <span className="nowrap">−1 (I&rsquo;m losing)</span>. The Model
            Inspector on the main page lets you open this machine up and watch those numbers form, square by
            square, for whatever position is on your board.
          </p>
        </Section>

        <Section title="The journey">
          <p>
            The model didn&rsquo;t arrive in one step. It went through several generations, and the most useful
            lessons came from the things that <em>didn&rsquo;t</em> work.
          </p>

          <Figure caption="The steps to a strong browser-based model.">
            <JourneyTimeline />
          </Figure>

          <h3 className="story-h3">First attempts: it actually plays</h3>
          <p>
            The earliest versions were built from the same kind of network that recognizes cats in photos (a
            &ldquo;convolutional&rdquo; network, good at spotting local patterns). The first one learned real
            chess — sensible openings, reasonable moves — which was already a small thrill. But it had blind
            spots: drop an unexpected threat in front of it and it would walk right into trouble. So it was made
            bigger, taught to play both colors, and — importantly — given that second output, the
            who&rsquo;s-winning score, by learning from how each game ended.
          </p>

          <Insight kind="fail" label="What didn&rsquo;t work">
            An early attempt built &ldquo;look a few moves ahead&rdquo; directly into the network&rsquo;s wiring.
            It sounded great. It flopped: taking that same budget and just making the plain network <em>bigger</em>{' '}
            beat it on every measure. The lesson — <strong>don&rsquo;t hand-engineer cleverness the model can learn
            on its own.</strong> Give it capacity and good signal, and get out of the way.
          </Insight>

          <h3 className="story-h3">A better shape: attention</h3>
          <p>
            The next leap was switching to a <strong>transformer</strong> — the same family of model behind modern
            language AI. Its key trick is &ldquo;attention&rdquo;: instead of only looking at nearby squares, every
            square can directly consult every other square in a single step. A rook can glance straight down its
            file; a bishop down its diagonal. This matched chess so well that the new model beat a version{' '}
            <em>twice its size</em> from the previous generation. Smaller and stronger — exactly the direction you
            want to be moving.
          </p>

          <h3 className="story-h3">The real bottleneck wasn&rsquo;t size — it was signal</h3>
          <p>
            Here&rsquo;s where it got interesting. Making the model bigger kept barely moving the needle. Its sense
            of <em>who&rsquo;s winning</em>, in particular, hit a wall. The problem turned out not to be the model
            at all — it was the <strong>data</strong>.
          </p>
          <p>
            Consider how it had been taught: for each position, &ldquo;a human played this one move, and that one
            game ended this way.&rdquo; But a single game&rsquo;s result is noisy — good positions get lost and lost
            positions get saved all the time. So the lesson changed: <em>every</em> game that ever passed through a
            given position was pooled, and the model was handed the <strong>average</strong> outcome and the{' '}
            <strong>full spread of moves</strong> people chose there. Same model, far cleaner signal.
          </p>
          <Figure caption="One game gives a noisy hint. Pooling every game through a position gives a clean target: how it usually turns out, and what people actually play.">
            <SignalDiagram />
          </Figure>
          <Insight kind="win" label="What worked">
            This single change to the <em>data</em> — not the model — was worth as much as <strong>doubling the
            model&rsquo;s size</strong>. A model trained this way matched one with twice as many parameters. It was
            the biggest free lunch of the whole project, and it came from diagnosing a failure (the stalled
            who&rsquo;s-winning score) rather than just piling on more computing power.
          </Insight>

          <Insight kind="fail" label="What didn&rsquo;t work">
            Surely a strong model should train only on the games of the <em>strongest</em> players? Tried — and it
            was <strong>worse.</strong> The variety in a broad mix of human play — the mistakes and all — carried
            more useful signal than a narrow diet of master games. Intuition lost to measurement, which happened
            more than once.
          </Insight>

          <h3 className="story-h3">Shrinking it down so you can see inside</h3>
          <p>
            The model was then taken the other direction — shrunk until it was <em>tiny</em>, small enough that you
            can look at every single number it contains and watch it think. (That&rsquo;s the model on the main
            page, and what the Model Inspector is showing you.) Shrinking it revealed which parts actually earned
            their keep and which were dead weight to throw away, leaving a leaner, cleaner design.
          </p>

          <h3 className="story-h3">A small student with a good teacher</h3>
          <p>
            A model that small can only learn so much from raw human games on its own. So a trick called{' '}
            <strong>distillation</strong> steps in: the biggest, strongest model acts as a <em>teacher</em>, and the
            tiny model learns to imitate the teacher&rsquo;s full judgment rather than just the bare record of human
            games. The student punches well above its weight — especially in its sense of who&rsquo;s winning. (The
            teacher, importantly, only ever learned from human games too, so no outside chess knowledge sneaks in.)
          </p>
          <Insight kind="fail" label="A surprise">
            The obvious recipe would <em>blend</em> the teacher&rsquo;s guidance with the original human data. It
            didn&rsquo;t work: mixing the two actively <strong>hurt</strong>. Learning purely from the teacher won
            cleanly. Another reminder that the obvious-sounding approach often isn&rsquo;t the right one.
          </Insight>

          <h3 className="story-h3">Letting it think harder</h3>
          <p>
            Finally, the model gets an optional way to spend more effort on a move. Instead of answering at a
            glance, it can play out many short &ldquo;what if&rdquo; lines from the current position, spending more
            of them on the moves that look promising, then play whatever those trials support best. Crucially, the
            only compass it uses is its own move preferences and who&rsquo;s-winning scores — no outside chess
            rules. The search just lets the model concentrate its own judgment where it matters. (On the main
            page, this is the <strong>MCTS</strong> mode and the &ldquo;max simulations&rdquo; slider.)
          </p>
          <Insight kind="fail" label="A counter-intuitive result">
            You&rsquo;d think a little thinking is always better than none. Not so: with only a handful of trials,
            the search is shallow and noisy and actually plays <em>worse</em> than the instant, one-glance move.
            It takes a real budget of thinking before searching pays off — after which it climbs steadily.
          </Insight>

          <h3 className="story-h3">All on one desktop machine</h3>
          <p>
            None of this needed a data center. Every model and experiment in this story was trained on a{' '}
            <strong>single consumer graphics card</strong> — an NVIDIA RTX 4080 Super, a typical PC gamer
            GPU card, with 16&nbsp;GB of memory. The largest models took a few hours for each pass through the
            millions of games; the tiny one you&rsquo;re playing trains from scratch in about an hour. A big part
            of the project was finding tricks to make the training fast enough to actually iterate on a setup like
            that — packing the whole dataset into memory so the graphics card never waits on the disk, for
            instance — so that trying a new idea took minutes, not days.
          </p>
        </Section>

        <Section title="So how strong is it, really?">
          <p>
            To put a number on it, the model played hundreds of games against a calibrated opponent (Stockfish,
            dialed to known strength levels), measuring where it held its own.
          </p>
          <p>
            First, a yardstick. Chess strength is measured in <strong>Elo</strong> — a single rating number where
            higher is stronger and a 100-point gap is a noticeable edge. Rough landmarks: a beginner sits around
            800, a casual player about 1,200, a club player 1,600, an &ldquo;expert&rdquo; 2,000, a master near
            2,200&ndash;2,400, and the world champion hovers around 2,800 (Magnus Carlsen peaked at 2,882). These
            numbers come from playing Stockfish, so they track this familiar scale, give or take.
          </p>
          <Figure caption="Where this model lands on the rating ladder — from a casual player up to master strength, depending on how hard it thinks.">
            <EloScale />
          </Figure>
          <p>And exactly where it lands depends entirely on how hard you let it think:</p>
          <Figure caption="Estimated strength versus how much the model thinks. Notice the dip at the far left — a tiny bit of search is worse than none — and the steady climb after.">
            <EloCurve />
          </Figure>
          <p>
            At a single glance it plays around the level of a solid club player. Let it think for a couple of
            seconds and it climbs into strong-amateur, even expert territory — enough to give a serious club or
            tournament player a real game. On the main page, the &ldquo;estimated Elo&rdquo; in the configuration
            panel is reading off exactly this calibration, updating live as you change the settings.
          </p>
        </Section>

        <Section title="What you&rsquo;re looking at on the main page">
          <p>
            Everything above is sitting right there for you to poke at:
          </p>
          <ul className="story-list">
            <li>
              The <strong>Model Inspector</strong> opens the network up so you can watch the real numbers flow —
              the board becoming a stack of grids, attention connecting square to square, and the final move
              scores and who&rsquo;s-winning value forming. It&rsquo;s the picture from &ldquo;how a network sees a
              board,&rdquo; live.
            </li>
            <li>
              <strong>Move variety</strong> dials how adventurous it plays — it sharpens to its very best move when
              it judges it&rsquo;s losing, and loosens up when comfortably ahead, but never plays a move a strong
              player would reject.
            </li>
            <li>
              <strong>One-Shot vs. MCTS</strong> and the <strong>simulations</strong> slider are the
              think-harder trade-off from the strength chart: more thinking, stronger play, a little more waiting.
            </li>
          </ul>
          <p>
            A tiny bundle of numbers, taught only by watching people play, that you can both <em>beat</em> and{' '}
            <em>look inside</em>. Give it a try!
          </p>
          <p className="story-cta">
            <a className="story-cta-link" href={PLAY_URL}>Play Neural Chess →</a>
          </p>
        </Section>

        <Section title="Who made this">
          <p>
            Created by{' '}
            <a href="https://www.linkedin.com/in/peterpbaer/" target="_blank" rel="noopener noreferrer">Peter Baer</a>{' '}
            using Claude Code.
          </p>
        </Section>
      </article>

      <footer className="story-footer">
        <a href={`${REPO_URL}/blob/master/LICENSE`} target="_blank" rel="noopener noreferrer">MIT © Peter Baer</a>
        <span className="story-footer-sep" aria-hidden="true">·</span>
        <a href={REPO_URL} target="_blank" rel="noopener noreferrer">github.com/pbaer/neural-chess</a>
        <span className="story-footer-sep" aria-hidden="true">·</span>
        <FeedbackLink />
      </footer>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="story-section">
      <h2 className="story-h2">{title}</h2>
      {children}
    </section>
  );
}

function Figure({ caption, children }: { caption: string; children: React.ReactNode }) {
  return (
    <figure className="story-figure">
      <div className="story-figure-art">{children}</div>
      <figcaption className="story-figcaption">{caption}</figcaption>
    </figure>
  );
}

function Insight({ kind, label, children }: { kind: 'win' | 'fail'; label: string; children: React.ReactNode }) {
  return (
    <aside className={'story-insight story-insight-' + kind}>
      <span className="story-insight-label">{label}</span>
      <p className="story-insight-text">{children}</p>
    </aside>
  );
}

// ─────────────────────────── inline SVG figures ───────────────────────────

/** Board → network → (move + value). A simple left-to-right concept strip. */
function BoardToMove() {
  return (
    <svg viewBox="0 0 460 140" className="fig" role="img" aria-label="Board to move and value">
      {/* mini board */}
      <g transform="translate(8,30)">
        {Array.from({ length: 8 }, (_, r) =>
          Array.from({ length: 8 }, (_, c) => (
            <rect key={`${r}-${c}`} x={c * 10} y={r * 10} width="10" height="10"
              fill={(r + c) % 2 ? 'var(--panel-2)' : 'var(--line)'} />
          )),
        )}
        <text x="40" y="98" textAnchor="middle" className="fig-label">board</text>
      </g>
      <Arrow x1={96} x2={150} y={70} />
      {/* network box */}
      <g transform="translate(150,34)">
        <rect width="150" height="58" rx="8" fill="var(--accent-soft)" stroke="var(--accent)" />
        {[18, 38, 58, 78, 98, 118, 132].map((x, i) => (
          <line key={i} x1={x} y1={8} x2={x} y2={50} stroke="var(--accent)" strokeWidth="1" opacity="0.5" />
        ))}
        <text x="75" y="33" textAnchor="middle" className="fig-label-strong">the network</text>
        <text x="75" y="78" textAnchor="middle" className="fig-label">~116k numbers</text>
      </g>
      <Arrow x1={300} x2={352} y={70} />
      {/* outputs */}
      <g transform="translate(356,34)">
        <rect width="96" height="26" rx="5" fill="var(--panel-2)" stroke="var(--line)" />
        <text x="48" y="17" textAnchor="middle" className="fig-label">move scores</text>
        <g transform="translate(0,34)">
          <rect width="96" height="22" rx="5" fill="var(--panel-2)" stroke="var(--line)" />
          <rect x="48" y="0" width="34" height="22" rx="0" fill="var(--accent-2)" opacity="0.5" />
          <line x1="48" y1="0" x2="48" y2="22" stroke="var(--muted)" />
          <text x="48" y="35" textAnchor="middle" className="fig-label">who&rsquo;s winning</text>
        </g>
      </g>
    </svg>
  );
}

function Arrow({ x1, x2, y }: { x1: number; x2: number; y: number }) {
  return (
    <g stroke="var(--muted)" strokeWidth="2" fill="none">
      <line x1={x1} y1={y} x2={x2 - 6} y2={y} />
      <path d={`M ${x2 - 10} ${y - 4} L ${x2} ${y} L ${x2 - 10} ${y + 4}`} />
    </g>
  );
}

/** The five generations as a vertical, mobile-friendly timeline. */
function JourneyTimeline() {
  const steps = [
    ['v1', 'A first network that learns to play — with tactical blind spots.'],
    ['v2', 'Bigger; plays both colors; learns a sense of who&rsquo;s winning.'],
    ['v3', 'Switch to attention (a transformer): smaller and stronger.'],
    ['Data + distill', 'Clean up the signal in the data; a tiny student learns from a big teacher.'],
    ['In your browser', 'Shrunk to ~116k numbers — small enough to see inside, with optional &ldquo;think harder&rdquo; search.'],
  ];
  return (
    <ol className="timeline">
      {steps.map(([label, text], i) => (
        <li className="timeline-step" key={i}>
          <span className="timeline-dot" aria-hidden="true" />
          <div className="timeline-text">
            <span className="timeline-label">{label}</span>
            <span className="timeline-desc" dangerouslySetInnerHTML={{ __html: text }} />
          </div>
        </li>
      ))}
    </ol>
  );
}

/** One noisy game vs. many games pooled through a position. */
function SignalDiagram() {
  return (
    <svg viewBox="0 0 460 168" className="fig" role="img" aria-label="One game versus many pooled games">
      <g transform="translate(10,8)">
        <text x="0" y="10" className="fig-label-strong">One game</text>
        <line x1="6" y1="30" x2="180" y2="30" stroke="var(--line)" strokeWidth="2" />
        {[6, 50, 94, 138, 180].map((x, i) => <circle key={i} cx={x} cy={30} r="4" fill="var(--muted)" />)}
        <text x="6" y="52" className="fig-label">one move, one result — noisy</text>
      </g>
      <g transform="translate(10,86)">
        <text x="0" y="10" className="fig-label-strong">Every game through one position</text>
        {/* many paths converging on a node — kept BELOW the title so they don't
            overlap the text */}
        {[0, 12, 24, 36, 48].map((dy, i) => (
          <line key={i} x1="6" y1={24 + dy} x2="150" y2="48" stroke="var(--accent)" strokeWidth="1.5" opacity="0.55" />
        ))}
        <circle cx="150" cy="48" r="6" fill="var(--accent)" />
        <Arrow x1={164} x2={216} y={48} />
        <text x="224" y="44" className="fig-label-strong">average outcome</text>
        <text x="224" y="60" className="fig-label-strong">+ spread of moves</text>
        <text x="224" y="76" className="fig-label">clean signal</text>
      </g>
    </svg>
  );
}

/** A rating ladder with landmark levels and where this model sits. */
function EloScale() {
  const lo = 800, hi = 2900, W = 460, H = 96, x0 = 18, x1 = 442, y = 60;
  const xs = (e: number) => x0 + ((e - lo) / (hi - lo)) * (x1 - x0);
  const marks: [number, string][] = [
    [800, 'beginner'], [1200, 'casual'], [1600, 'club'],
    [2000, 'expert'], [2400, 'master'], [2800, 'champion'],
  ];
  const bandLo = 1300, bandHi = 2474; // model: ~shallow-MCTS up to 300-sim play
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fig" role="img" aria-label="Chess rating ladder and where the model sits">
      {/* the model's range */}
      <rect x={xs(bandLo)} y={y - 13} width={xs(bandHi) - xs(bandLo)} height={13} rx={3}
        fill="var(--accent)" opacity="0.35" stroke="var(--accent)" />
      <text x={(xs(bandLo) + xs(bandHi)) / 2} y={y - 19} textAnchor="middle" className="fig-label-strong">this model</text>
      {/* axis + landmarks */}
      <line x1={x0} y1={y} x2={x1} y2={y} stroke="var(--muted)" strokeWidth="2" />
      {marks.map(([e, l]) => (
        <g key={e}>
          <line x1={xs(e)} y1={y - 4} x2={xs(e)} y2={y + 4} stroke="var(--muted)" strokeWidth="2" />
          <text x={xs(e)} y={y + 18} textAnchor="middle" className="fig-label-strong">{e}</text>
          <text x={xs(e)} y={y + 31} textAnchor="middle" className="fig-label">{l}</text>
        </g>
      ))}
    </svg>
  );
}

/** Estimated Elo vs. amount of thinking (one-shot, then MCTS sims, log x). */
function EloCurve() {
  // Measured Stockfish-anchored points.
  const oneShot = 1572;
  const pts: [number, number][] = [
    [10, 1310], [25, 1692], [50, 1912], [100, 2132], [150, 2300], [200, 2300], [300, 2474],
  ];
  const W = 460, H = 220, padL = 48, padR = 16, padT = 16, padB = 40;
  const yLo = 1250, yHi = 2550;
  const x0 = padL, x1 = W - padR;
  const ySc = (e: number) => padT + (1 - (e - yLo) / (yHi - yLo)) * (H - padT - padB);
  // x: a slot for "one-shot", then log scale over sims 10..300.
  const lmin = Math.log(10), lmax = Math.log(300);
  const simX = (s: number) => x0 + 64 + ((Math.log(s) - lmin) / (lmax - lmin)) * (x1 - x0 - 64);
  const osX = x0 + 24;
  const line = pts.map(([s, e]) => `${simX(s)},${ySc(e)}`).join(' ');
  const yticks = [1400, 1600, 1800, 2000, 2200, 2400];
  const labels: [number, string][] = [[10, '10'], [50, '50'], [100, '100'], [200, '200'], [300, '300']];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fig" role="img" aria-label="Estimated Elo versus thinking">
      {/* y grid + ticks */}
      {yticks.map((t) => (
        <g key={t}>
          <line x1={x0} y1={ySc(t)} x2={x1} y2={ySc(t)} stroke="var(--line)" strokeWidth="1" opacity="0.6" />
          <text x={x0 - 8} y={ySc(t) + 4} textAnchor="end" className="fig-label">{t}</text>
        </g>
      ))}
      <text x={14} y={padT + 6} className="fig-label-strong" transform={`rotate(-90 14 ${(H) / 2})`}>Elo</text>
      {/* one-shot point + label */}
      <circle cx={osX} cy={ySc(oneShot)} r="4.5" fill="var(--muted)" />
      <text x={osX} y={H - padB + 16} textAnchor="middle" className="fig-label">one&#8209;shot</text>
      <line x1={osX} y1={padT} x2={osX} y2={H - padB} stroke="var(--muted)" strokeDasharray="3 3" opacity="0.35" />
      {/* connect one-shot to first sim point to show the dip */}
      <line x1={osX} y1={ySc(oneShot)} x2={simX(10)} y2={ySc(1310)} stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="4 3" />
      {/* sims curve */}
      <polyline points={line} fill="none" stroke="var(--accent)" strokeWidth="2.5" />
      {pts.map(([s, e]) => <circle key={s} cx={simX(s)} cy={ySc(e)} r="3.5" fill="var(--accent)" />)}
      {labels.map(([s, t]) => (
        <text key={s} x={simX(s)} y={H - padB + 16} textAnchor="middle" className="fig-label">{t}</text>
      ))}
      <text x={(simX(50) + simX(300)) / 2} y={H - 6} textAnchor="middle" className="fig-label">simulations (more thinking →)</text>
    </svg>
  );
}

/** Compact theme toggle mirroring the play page (writes nc-theme, sets data-theme). */
function ThemeToggle() {
  const theme = useSyncExternalStore(subscribeTheme, getTheme, () => 'dark');
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem('nc-theme', next);
    } catch {
      /* ignore */
    }
    document.documentElement.dataset.theme = next;
    window.dispatchEvent(new Event('nc-theme-change'));
  };
  return (
    <button className="story-theme" onClick={toggle} aria-label="Toggle light / dark theme">
      {theme === 'dark' ? '☀ Light' : '☾ Dark'}
    </button>
  );
}

function getTheme(): string {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}
function subscribeTheme(cb: () => void): () => void {
  window.addEventListener('nc-theme-change', cb);
  return () => window.removeEventListener('nc-theme-change', cb);
}
