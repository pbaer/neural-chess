# Crash Analysis — FARNSWORTH (workstation running neural-chess training)

**Original analysis:** 2026-05-24 — from Windows System event log, WER, WHEA-Logger, and `nvidia-smi`
**Updated:** 2026-05-24 — hardware identified, power-capped test run
**Updated:** 2026-05-25 — XMP-off test crashed too (RMA verdict); refined diagnosis (build date + low-utilization crash mechanism) added below
**Prepared for:** the agent running the training job

---

## TL;DR (updated)

The machine **blue-screened (`0x1E`) in the early morning of 2026-05-24, mid-training** — not a planned reboot. A cluster of CPU machine-check errors (WHEA-19) and GPU resets (TDRs) began **exactly on 2026-05-17, when heavy GPU training started**, so this is **one load-driven stability problem, not three separate faults.**

The original hypothesis was *"marginal power delivery (PSU / 12VHPWR connector)."* **After identifying the hardware, that is downgraded and the diagnosis re-centred on the CPU:**

- **PSU = be quiet! 1000 W 80+ Gold** — vastly oversized for a ~620 W peak load, quality brand, connectors reported solid. **Undersized PSU is ruled out** (it was the original #1 suspect).
- **Prime suspect is now the CPU: an Intel Core i9-14900K** — the flagship part of the Intel 13th/14th-gen **"Raptor Lake" instability/degradation** issue. The fatal bugcheck (`0x1E` / `STATUS_PRIVILEGED_INSTRUCTION`) is a *corrupted-instruction-stream* signature of an unstable core — **not** the GPU-TDR bugcheck (`0x116/0x117`). The WHEA "Processor Core / internal parity" errors are CPU-internal electrical instability. The GPU TDRs are a concurrent **symptom** of the same system instability, not the killer.
- The Intel microcode fix (**0x12B**, Oct 2024) **is already loaded**, so ongoing degradation should be halted — yet it *still crashed with the fix present*. That points to (a) an over-aggressive **motherboard voltage/power profile**, (b) **DDR5-6400 XMP** stressing the memory controller, and/or (c) damage the chip **already sustained** before the fix landed.

**Mitigation in progress (software, applied now):** GPU capped to **250 W** (`nvidia-smi -pl 250`), **mid-run checkpointing** (≤15 min loss on any crash), and live WHEA / TDR / telemetry monitoring.
**Result (decisive): the cap does NOT stop the CPU errors.** Through ~3 h capped, GPU TDRs stayed at **zero**, but **two WHEA-19 CPU machine-checks recurred — and both fired during a light-GPU / heavy-CPU eval phase** (Stockfish loading the cores while the GPU was nearly idle). That isolates the fault to **CPU load, not GPU power** — it *refutes* the power-delivery theory and *confirms* the 14900K as root cause. The cap may still cut GPU heat/TDRs (worth keeping), but the actual fix is CPU-side (BIOS / RMA).

**VERDICT (2026-05-25):** Intel Defaults were **already set** through every crash (board profile eliminated). **Disabling XMP eliminated the corrected WHEA-19 errors** (0 this run vs a stream before) — so the DDR5-6400/IMC *was* causing those — **but the box STILL hard-crashed** (bugcheck `0x00020001`, no WHEA warning). With PSU, GPU power (capped 250W), board profile (Intel spec), and memory (XMP off) all eliminated, the only remaining variable is the CPU. **Conclusion: the i9-14900K is degraded → RMA it** (covered by Intel's 2-year warranty extension; boxed parts ~5 yrs total). The crash is a CPU-core failure mode separate from the (now-fixed) memory parity errors.

**Web-validated (2026-05-25):** Intel's published root cause is **"Vmin Shift Instability"** — a clock-tree circuit in the IA core that ages under elevated voltage/temperature, raising the chip's minimum stable voltage over time (irreversible; microcode *prevents* but cannot *fix* a degraded chip). Crucially, Intel's 0x12B fix targets *"elevated voltage requests... during **idle and/or light activity**"* — which independently confirms our finding that the crashes hit at low utilization / high single-core boost. Microcode 0x12B already loaded; a newer 0x12F exists. Sources in the conversation log.

**UPDATE 2026-05-26 — CPU cap helps substantially but is NOT a full fix (revised 2026-05-28).** With `powercfg ... PROCTHROTTLEMAX = 90%` (`0x5a`) + GPU 250W, the **mean time to crash extended from ~0.5–3.5 h → ~9–13 h** (got 3 full epochs of v3-37M done in a row) — strong improvement but not a cure. **Corrected WHEA-19 errors kept firing constantly throughout** (36 over ~48 h of runs); the cap just made it take longer before one escalated to uncorrectable → BSOD (`0x1E` again, 2026-05-27 20:27, ~8.5 h into e3).

**2026-05-28 (later) — the % cap was a NO-OP; turbo now actually disabled.** Discovered via live measurement that `PROCTHROTTLEMAX` (90/80/70%) **never reduced the clock** — under Intel Speed Shift / HWP, Windows' "max processor state" % is ignored and the CPU ran full **~5.9 GHz turbo the entire time**. So the earlier "cap extended MTBF to ~9–13 h" conclusion is **retracted** — that was stochastic variation, not the cap. The frequency/voltage mitigation had **never actually been tested.** Fix that works: unhide + set **`PERFBOOSTMODE = 0`** (disable Turbo Boost) → verified busiest core dropped **5.9 GHz → 3.1 GHz** (base), applied live to running training. *This* is the real test of the high-boost-voltage theory. (Most robust = BIOS turbo-disable / multiplier lock, reboot required.) GPU-bound workload, so base-clock CPU costs ~nothing in throughput.

**2026-05-28 — new failure mode: Python segfault (no BSOD).** Cap lowered to **80%** (`0x50`); training resumed; still throwing ~1 WHEA-19 every ~60–90 min. ~9 h into the resumed run, the python process died with a **segmentation fault** while the system stayed up. Most likely cause: a corrected WHEA error corrupted a user-space computation that didn't get caught/corrected by the CPU's correction logic in time → process segfault rather than kernel BSOD. **Less disruptive** (no reboot, session survives, ~17 min lost from `model_latest`) but reinforces that the chip is broken at a level no software cap can fix. RMA urgency unchanged: it's the only cure.

---

## Hardware inventory (identified 2026-05-24)

| Component | Part | Notes |
|---|---|---|
| **CPU** | **Intel Core i9-14900K** | Center of the 13th/14th-gen Raptor Lake instability/degradation issue |
| **Motherboard** | **MSI MAG Z790 TOMAHAWK WIFI (MS-7D91)** | BIOS **H.E0**, dated 2024-09-25 |
| **RAM** | **2 × 32 GB G.Skill DDR5-6400** (64 GB) | XMP 6400 is well above the 14900K's guaranteed memory spec, especially for two dual-rank DIMMs |
| **GPU** | **NVIDIA RTX 4080 SUPER**, driver **581.80** | default power limit 320 W (max 370, min 150) |
| **PSU** | **be quiet! 1000 W 80+ Gold** | quality unit; ~380 W headroom over peak load |
| **Loaded CPU microcode** | **0x12B** | Intel's comprehensive Raptor Lake instability fix (Oct 2024) — **already present** |

---

## What changed in the diagnosis (and why)

### PSU: prime suspect → cleared
A 4080 SUPER under sustained training pulls ~300–320 W; a 14900K peaks ~253–300 W → **~620 W total**, against a **1000 W 80+ Gold** rail. That's ~380 W of headroom, comfortably absorbing even the 40-series' millisecond transient spikes. be quiet! is a reputable brand and the connectors are reported solidly seated. **Wattage/PSU is not the cause.** This matters: it removes the original report's #1 suspect and redirects attention to the CPU.

### CPU (i9-14900K): bystander → prime suspect
| Evidence | Interpretation |
|---|---|
| CPU is an **i9-14900K** | The single most affected SKU in the Raptor Lake instability/degradation crisis |
| Bugcheck **`0x1E` + `STATUS_PRIVILEGED_INSTRUCTION`** | Corrupted instruction stream = unstable core. **A GPU-driver TDR crash is `0x116/0x117`, not `0x1E`** → the *fatal* crash was CPU/system instability |
| WHEA-19 **"Processor Core / Internal parity error"** | CPU-internal *electrical* instability under load (corrected, so individually non-fatal, but a clear stream) |
| Onset **with GPU load**, **hours into overnight runs** | Classic load- and heat-dependent Raptor Lake behavior |

### Memory (DDR5-6400 XMP): co-suspect
Running 2 × 32 GB at DDR5-6400 via XMP is well beyond the 14900K integrated memory controller's guaranteed spec. An unstable IMC also produces WHEA parity errors and `0x1E`-class crashes. Disabling XMP is a free, high-value test.

### Microcode 0x12B already loaded — the key implication
The fix that prevents further degradation is **already active**, so a BIOS *microcode update* is **not** the missing piece. Because it still crashes with 0x12B present, the remaining levers are the **BIOS voltage/power settings** and the possibility that the chip **already degraded** before the fix — which is what makes RMA a live option if settings don't resolve it.

---

## Mitigations APPLIED (2026-05-24)

1. **GPU power cap → 250 W** (`nvidia-smi -pl 250`; needs an elevated shell; resets on reboot). Confirmed enforced; throttle reason reads `0x4 = SW Power Cap` (i.e. power-limited by us, **not** thermal/HW). **Measured cost ≈ +1.2 % epoch time** (2.80 h → ~2.83 h) — essentially free, because the 4080 SUPER sits on the flat part of its power-efficiency curve.
2. **Mid-run checkpointing** — `train.py --save-every-steps 3500` writes `model_latest.pt` every ~15 min, **atomically** (temp file + `os.replace`, so a crash mid-write cannot corrupt it). Auto-resume now **prefers the mid-epoch checkpoint** and restarts the in-progress epoch from its first batch (LR/scheduler are per-epoch, so this is correct). **A crash now costs ≤ 15 min instead of ~2 h.**
3. **Telemetry logging** — `gpu_telem_resume2.csv` records temp / power / throttle reasons / util / clocks every 5 s for post-hoc correlation.
4. **Live monitors during runs** — WHEA-19 recurrence watch, GPU-TDR (Event 153) watch, and a training health/crash watch.
5. **NO automatic relaunch on reboot** — per the operator's explicit choice: the machine is used for other work, so after any reboot the operator retains full manual control over what starts. Step-checkpoints bound the loss; they do **not** auto-restart training.

### Results (capped run, 2026-05-24 ~11:00 → 16:17 crash)
- **WHEA-19: 3 corrected** — incl. 13:51 & 13:52 **during the heavy-CPU eval** (GPU near-idle), same signature (`Processor Core / Internal parity / APIC ID 41`). The errors track **CPU load, not GPU power**.
- **GPU TDR (Event 153): 0** under the cap (the cap does help the GPU side).
- **SECOND HARD CRASH at 16:17**, ~2.3 h into the capped **e6** training run — bugcheck **`0x00020001`** (params 0x11 / 0x210720 / 0x1005 / kernel-ptr), dump `052426-19062-01.dmp`. A **different stop code** than the 05:30 `0x1E`. The 250 W GPU cap did **not** prevent it.
- **Recovery worked as designed:** the mid-run step-checkpoint saved `model_latest.pt` (epoch 6) at 16:13 — 4 min before the crash → **~4 min of training lost** (vs. ~2 h before step-checkpointing). Machine rebooted on its own and sat idle (no auto-relaunch, by choice).
- **Interpretation (decisive):** the cap addresses neither the CPU machine-checks nor the crashes — both track **CPU load**, with the GPU capped or idle. **Two distinct bugchecks (`0x1E`, `0x00020001`) + the 05-20 dump-less hang = three failure modes in a week** → the signature of **hardware electrical instability**, not one driver bug. **Confirms the i9-14900K** (Raptor Lake instability/degradation and/or DDR5-6400 IMC stress). Keep the cap only for GPU heat/TDR reduction; it is not a fix. **Fix = BIOS Intel Defaults + disable XMP; RMA if it persists.**

### 2026-05-25 overnight (XMP OFF, GPU capped 250W) — VERDICT
- Resumed e6 ~23:17; **e6 completed clean** (passed the ~2.3 h mark where the prior run died); **crashed 02:48, ~41 min into e7** (~3.5 h in). Bugcheck **`0x00020001`** — *same code as the 16:17 crash* (recurring signature), dump `052526-14828-01.dmp`. Step-checkpoint limited loss to ~11 min.
- **WHEA-19 this run: 0** (vs a stream with XMP on) → **disabling XMP fixed the corrected memory/parity errors.** But the **hard crash recurred anyway, with no WHEA warning** → the crash is a **CPU-core failure mode independent of memory.**
- **All external variables now eliminated**: PSU (oversized), GPU power (capped 250W + crashes with GPU idle), board profile (Intel Defaults already set), memory (XMP off). **Only the CPU remains → the i9-14900K is degraded → RMA** (Intel 5-year extended warranty). Running it at lower clocks/voltage (BIOS turbo limits or Windows max-processor-state cap) is the only software lever likely to reduce crash frequency enough to limp through a workload before the RMA.

---

## 2026-05-25 — refined diagnosis (build date + low-utilization crashes)

### The chip ran on the fix its WHOLE life — and degraded anyway
System was **built November 2024**. The BIOS (H.E0, Sept 2024) **supplies microcode 0x12B itself** (registry `Previous Update Revision` = 0x12B → BIOS-loaded, not an OS patch), and a 14900K bought new in late 2024 is **post-oxidation-fix production silicon**. So **both of Intel's named root causes are ruled out for this chip**: it never ran under the dangerous early over-volting microcode, and it isn't from the 2023 oxidation batch. It ran in-spec, on the fix, for ~18 months — and degraded to instability anyway.

**Implication:** the microcode fix + Intel Defaults did **not** protect this chip, so they cannot be relied on to guarantee a replacement's safety. Remaining drivers:
- **Cumulative wear** on a hard-run, hot i9 — 0x12B *slows* Vmin-shift degradation but doesn't stop it under sustained high-boost/high-temp use.
- **Heat** — never measured; the 14900K is hard to cool. (Action: measure.)
- **Motherboard voltage delivery** — MSI's "Intel Default" can still feed elevated vcore via Lite Load / loadline. A board over-volting would degrade this chip *and* a replacement. (Action: measure vcore under load.)

### Why it crashes during training despite ~3% CPU utilization
Measured live during training: **all-core utilization 3.3%, but a single core boosting to ~171% of base ≈ 5.5 GHz.** Low utilization is misleading — for a degraded Raptor Lake it's nearly the worst case:
- **The fragile state is high-frequency LIGHT load, not heavy load.** Peak single-core boost = peak voltage; under heavy all-core load the chip runs *lower* clocks/voltage (power-limited). The Vmin-shifted chip faults at the high-voltage boost point — exactly the light-load state. (This is why Raptor Lake users crashed at desktop idle / shader compile / decompression.)
- **"Feeding a GPU" is maximally bursty** — Python + dataloader workers spike individual cores to max boost thousands of times/min, each a high-voltage transition = a dice-roll for a fault. Over hours they accumulate → crash.
- **The uncore/IMC is slammed regardless of core %** — keeping the GPU at 100% pushes GB/s through the memory controller continuously; core utilization says nothing about uncore stress. (The corrected WHEA errors were *memory-controller* parity, fixed by XMP-off.)
- **Correction to earlier wording:** the errors don't track raw "CPU load" — they track the **voltage/frequency state + uncore activity**, both high even at 3% utilization.
- **Testable:** if this mechanism is right, **capping max frequency / disabling turbo** should reduce crashes *disproportionately* (removes the ~5.5 GHz peak-voltage state) — and would confirm the degraded-core-at-max-boost diagnosis.

### Protecting a replacement chip (for when the RMA arrives)
Because the fixes didn't save this chip, treat these as **required**, not optional:
1. **Run it cooler** — verify/improve cooling; target sustained temps well under ~85 °C.
2. **Run it below stock** — a modest **undervolt** (negative Vcore offset / higher Lite Load) and/or a **power limit** below 253 W. For a GPU-bound box this costs nothing real.
3. **Or don't use a flagship i9** — neural-chess is GPU-bound; the CPU only feeds data. A power-limited i9 or a cooler mid-tier CPU does this workload identically with far less degradation risk.
4. **Measure first** — log vcore + core temps under load (HWiNFO64). If the board over-volts or temps run high, fix that or the replacement degrades too.
5. **Treat the RMA as the CPU-vs-board test** — fresh chip, same board: stable long-term ⇒ it was the CPU; degrades again ⇒ suspect the board, replace it next.

---

## Remaining actions NOT yet tried (keep these)

The applied software mitigations only **reduce blast radius and gather evidence**. The items below are the candidate **actual fixes**, ordered by value.

### BIOS / settings (operator) — status updated 2026-05-24 eve
1. **Intel Default Settings — ALREADY SET** (verified in BIOS 2026-05-24 eve). The board was at Intel spec (PL1=PL2=253 W) through *every* crash → an over-aggressive/over-volting board profile is **NOT** the cause. **Suspect eliminated.**
2. **Disable XMP — DONE** (2026-05-24 eve; confirmed RAM now 4800 MHz JEDEC, was 6400). **This is now the deciding test:** if the box is stable under sustained load with XMP off, the DDR5-6400 / memory controller was the culprit; if it still crashes, the 14900K is degraded → RMA (item 4). *Perf cost measured (D:\dev\cpubench): −23.5% memory bandwidth, +28.5% latency; compute unchanged.*
3. *(If XMP-off proves stable, optional)* try a milder memory speed (e.g. 5600/6000) or a manual sub-timing tune to recover some bandwidth while staying stable.

### → It DID still crash (XMP off + Intel Defaults + 250 W cap) → RMA CONFIRMED
4. **RMA the i9-14900K — this is the answer.** Every external variable was eliminated (PSU, GPU power, board profile, memory) and it still hard-crashed (`0x00020001`, 2026-05-25 02:48). Covered by Intel's **2-year warranty extension** for the Raptor Lake "Vmin Shift Instability" issue (boxed parts ~5 yrs total); these symptoms qualify. Settings cannot fix a degraded chip. See **"Protecting a replacement chip"** above for keeping the new one healthy (the fixes did *not* protect this one — and note *undervolting* is a legitimate prevention for a healthy chip, since high voltage is the root cause; on a *degraded* chip pair any undervolt with lower turbo, since its Vmin has risen).

### Other untried items (from the original analysis, still valid)
5. **Reseat the 12VHPWR / 12V-2×6 connector** at both ends until it clicks (reported solid, but cheap to re-verify; a known 40-series sustained-load failure point).
6. **Improve case airflow / verify fan curves** — heat soak lowers stability margins and fits the early-morning crash timing.
7. **Lower the training batch size or add periodic cooldown pauses** if telemetry shows temps climbing toward limits — reduces peak power and heat soak.
8. **Clean NVIDIA driver reinstall (DDU)** of 581.80 *if* GPU TDRs persist after power/thermal/CPU are addressed.
9. **Analyze the crash dump** `C:\WINDOWS\Minidump\052426-19375-01.dmp` with WinDbg `!analyze -v` to name the faulting module (Debugging Tools for Windows not yet installed on the host).
10. **On-demand stability tests** to localize the fault instead of waiting for overnight crashes: Prime95 small-FFT (CPU/heat), OCCT (CPU + power), memtest86 / TestMem5 (RAM/IMC). If any fail quickly, that pinpoints CPU vs memory.

---

## Original forensic record (capture 2026-05-24)

### What happened (the crash)

| Time (2026-05-24) | Event | Meaning |
|---|---|---|
| 05:30:35 | Kernel-Power 41 | Rebooted **without cleanly shutting down** |
| 05:30:51 | BugCheck 1001 | **Crash confirmed**, crash dump written |
| 05:30:51 | EventLog 6008 | Previous shutdown was **unexpected** |

- **Bugcheck code:** `0x1E` — `KMODE_EXCEPTION_NOT_HANDLED`
- **Parameter 1:** `0xFFFFFFFFC0000096` = `STATUS_PRIVILEGED_INSTRUCTION` (a privileged CPU instruction faulted in kernel mode — consistent with a *corrupted instruction stream*, the hallmark of voltage instability / an unstable core)
- **PowerButtonTimestamp:** `0` → not a manual shutdown / power-button hold
- **Crash dump:** `C:\WINDOWS\Minidump\052426-19375-01.dmp` (not yet analyzed)

Distinct from this machine's normal **planned Windows Update reboots** (2026-05-15, 2026-04-18, 2026-03-30: `TrustedInstaller … Operating System: Upgrade (Planned)` + clean shutdown). A second unclean shutdown on **2026-05-20** (down ~05:54, up ~06:18) produced **no bugcheck/dump** — a hard hang or power cut, a *different* failure mode. Two distinct unclean-shutdown signatures in one week reinforce "hardware/stability" over "one bad driver." Both crashes hit in the **early morning (04:58 and 05:54)** — hours into an overnight run, when heat soak and sustained load peak.

### Correlating evidence (all onset 2026-05-17, the start of the training window)

**CPU machine-check errors (WHEA-Logger, Event 19)** — *now read as the lead evidence, not a bystander*
- **Zero before 2026-05-17.** Then 10 events clustered 05-17 → 05-23.
- Detail: `Reported by component: Processor Core` / `Corrected Machine Check` / `Internal parity error` / `APIC ID 41`.
- Per-day: 05-17 ×3, 05-20 ×1, 05-22 ×5, 05-23 ×1. Corrected (non-fatal individually), but a steady stream that appeared with the load = electrical instability, not heavy compute.

**GPU resets (NVIDIA `nvlddmkm`, Event 153)** — *now read as a symptom of the same instability*
- Repeated GPU **TDRs** throughout the training window: 05-17 (multiple), 05-19, 05-23 (multiple).

**Storage — clean (crossed off)**
- Despite heavy SSD read load, **zero disk/NVMe/NTFS errors.** Only `volmgr 162 "Dump file generation succeeded"` (Windows writing the crash dump).

**GPU health at idle (`nvidia-smi`)**
- RTX 4080 SUPER, driver 581.80. Idle 33 °C, ~11 W, no throttling. Power limits default 320 W (max 370, min 150). No problems at idle — the instability is load-only.

### Raw reference data
```
Bugchecks: 05-24 05:30 -> 0x1E (KMODE_EXCEPTION_NOT_HANDLED), arg1 0xC0000096 (STATUS_PRIVILEGED_INSTRUCTION), dump 052426-19375-01.dmp
           05-24 16:17 -> 0x00020001 (params 0x11 / 0x210720 / 0x1005 / ffffe700014059a0), dump 052426-19062-01.dmp   [DIFFERENT signature]
CPU:       Intel Core i9-14900K   | microcode 0x12B loaded
Board:     MSI MAG Z790 TOMAHAWK WIFI (MS-7D91), BIOS H.E0 (2024-09-25)
RAM:       2x 32GB G.Skill DDR5-6400 (XMP)
GPU:       NVIDIA RTX 4080 SUPER, driver 581.80, power limit 320W (max 370 / min 150) -> capped to 250W on 2026-05-24
PSU:       be quiet! 1000W 80+ Gold  (ruled out — ample headroom)

WHEA-Logger Event 19 (Corrected Machine Check, Processor Core, Internal parity error, APIC ID 41):
  2026-05-17 x3, 2026-05-20 x1, 2026-05-22 x5, 2026-05-23 x1   (none before 05-17)
  2026-05-24 x2 at 13:51/13:52 -> RECURRED under the 250W cap, during heavy-CPU/light-GPU eval (refutes GPU-power theory)

nvlddmkm Event 153 (GPU TDR / reset):
  2026-05-17 (multiple), 2026-05-19, 2026-05-23 (multiple)     (none under the 250W cap)

Unclean shutdowns:
  2026-05-27 ~20:27  (BSOD 0x1E,        dump 052726-14781-01.dmp) — CPU cap 90% + XMP off + GPU 250W; ~8.5h into e3 of v3-37M; crash interval extended ~9-13h but NOT eliminated; 36 corrected WHEA-19 throughout the runs
  2026-05-25 ~16:35  (BSOD 0x00020001, dump 052526-17609-01.dmp) — XMP off; only ~29 min into e9 (EARLY crash -> crashes are stochastic, NOT a 3h threshold). WHEA-19=0
  2026-05-25 ~02:48  (BSOD 0x00020001, dump 052526-14828-01.dmp) — XMP OFF + capped 250W; STILL crashed, WHEA-19=0 -> CPU degraded
  2026-05-24 ~16:17  (BSOD 0x00020001, dump 052426-19062-01.dmp) — during the capped e6 training run
  2026-05-24 ~05:30  (BSOD 0x1E, dump 052426-19375-01.dmp)
  2026-05-20 ~05:54  (no dump; hard hang or power loss)

Planned update reboots (for contrast, NOT crashes):
  2026-05-15, 2026-04-18, 2026-03-30  (TrustedInstaller "Operating System: Upgrade (Planned)")
```
