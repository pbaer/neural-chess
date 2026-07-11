# Intel Core i9-14900K — Warranty Replacement (RMA) Request Summary

**Prepared for:** the PC shop handling the warranty claim
**Customer:** Peter Baer
**Date prepared:** 2026-07-10

---

## Executive summary

The Intel Core **i9-14900K** in this workstation has **electrically degraded** and is now unstable
under normal sustained use. The failure signature matches Intel's officially acknowledged hardware
defect in 13th/14th Gen ("Raptor Lake") desktop processors — **"Vmin Shift Instability"** — in which
elevated operating voltage causes **irreversible aging of a clock-tree circuit inside the CPU core**,
raising the chip's minimum stable voltage over time until it crashes. The system has been running the
latest applicable **Intel microcode fix and Intel Default power settings from day one**, yet it
continues to crash — which is exactly the case Intel's own guidance describes as needing a
**replacement**, because microcode *prevents further* degradation but **cannot repair a chip that has
already degraded**.

**The ask:** a **warranty replacement of the i9-14900K** under **Intel's extended warranty** for
affected 13th/14th Gen processors (standard boxed-processor warranty extended by **2 additional years,
to 5 years total from date of purchase**). The i9-14900K is a covered SKU. This document provides the
symptom evidence, the mitigations already tried, and a checklist of what Intel/the shop will need to
process the claim.

---

## System / CPU details

Please complete the `[fill in]` fields from the invoice and the physical CPU markings.

| Item | Value |
|---|---|
| **CPU** | Intel Core **i9-14900K** (14th Gen, Raptor Lake Refresh) |
| CPU **batch number (FPO)** | `[fill in — printed on the CPU heat spreader / on the retail box label]` |
| CPU **serial number (ATPO)** | `[fill in — partial serial on the CPU's outside edge; full serial in the 2D matrix code / on the box]` |
| Motherboard | MSI MAG Z790 TOMAHAWK WIFI (MS-7D91) |
| BIOS version | H.E0 (dated 2024-09-25) — *see note below re: updating* |
| Loaded CPU microcode | **0x12B** (Intel's Raptor Lake instability fix; see mitigations) |
| Memory | 2 × 32 GB G.Skill DDR5-6400 (64 GB) |
| GPU | NVIDIA RTX 4080 SUPER (driver 581.80) |
| PSU | be quiet! 1000 W 80+ Gold |
| System built | November 2024 |
| **Purchase date** | `[fill in]` |
| **Invoice / order number** | `[fill in]` |
| **Place of purchase** | `[fill in — must match the proof of purchase]` |

### Where to find the CPU batch/serial numbers
- **FPO (batch number):** printed on the **metal heat spreader** on the top of the processor, and on
  the **retail box** label.
- **ATPO (serial number):** a **partial serial** (last 3–5 digits) is printed in human-readable text
  on the **outside edge** of the processor; the **full serial** is encoded in the **2D matrix (data
  matrix) code** on the chip and printed on the **original box**.
- Intel provides an online **"Processor FPO / ATPO Serial Number Finder"** tool to help read these.
- **Important:** do **not** clean or wipe the top of the CPU with anything abrasive or with liquid
  metal — if the laser markings/serial become unreadable, Intel can reject the RMA. Photograph the
  markings *before* removing the cooler if possible.

---

## Symptoms observed

Documented from the machine's Windows System event log, WHEA-Logger, Windows Error Reporting, and
crash minidumps. The instability appeared **only under sustained load** and has recurred across many
runs since mid-May 2026.

- **Onset:** a cluster of CPU errors and crashes began **exactly on 2026-05-17**, when a sustained
  multi-day compute workload started. **Zero** such errors before that date.
- **Repeated blue-screen crashes (BSOD)** mid-workload, hours into overnight runs, including:
  - Bugcheck **`0x1E` (KMODE_EXCEPTION_NOT_HANDLED)**, parameter
    **`0xC0000096` = STATUS_PRIVILEGED_INSTRUCTION** — a **corrupted-instruction-stream** signature
    characteristic of an unstable CPU core (not a GPU/driver crash, which would be `0x116/0x117`).
  - Bugcheck **`0x00020001`** — recurred on multiple separate crashes.
  - One **dump-less hard hang** (a different, hardware-consistent failure mode).
- **CPU machine-check errors — WHEA-Logger Event 19:** a steady stream of **"Corrected Machine Check /
  Processor Core / Internal parity error / APIC ID 41"** events (e.g. 36 corrected events over roughly
  48 hours of runs). These are CPU-internal electrical-instability errors.
- **A new failure mode emerged over time — a user-space process segfault** (application crash without a
  BSOD) — consistent with a corrected CPU error corrupting a computation. The appearance of
  **additional/worsening failure modes over weeks** is evidence of **progressive degradation**.
- **Crashes occur at LOW CPU utilization but HIGH single-core boost.** Measured live: all-core
  utilization ~3%, while a single core boosted to ~5.5–5.9 GHz. This "high-frequency, light-load"
  state is the **peak-voltage** condition and is exactly where a Vmin-shifted Raptor Lake chip faults.
- **Crashes are stochastic, not tied to a fixed runtime** — they hit anywhere from ~0.5 h to ~13 h into
  a run, sometimes within half an hour — consistent with marginal electrical stability rather than a
  reproducible software bug.

---

## Why this is the known Intel defect (Vmin Shift Instability)

Intel has **publicly acknowledged** a hardware instability affecting **13th and 14th Gen Core desktop
processors**, and identified the root cause as **"Vmin Shift Instability"**:

> Intel localized the issue to **a clock-tree circuit within the IA core that is vulnerable to
> reliability aging under elevated voltage and temperature**, leading to a duty-cycle shift of the
> clocks and system instability. *(Intel Core 13th/14th Gen Desktop Instability Root Cause Update.)*

Key points that match this machine's symptoms:
1. **Elevated voltage → irreversible aging.** The defect raises the chip's minimum stable voltage over
   its life. It is **cumulative and permanent** — which is why a chip that ran fine can become unstable.
2. **Worst at idle / light activity / high boost.** Intel specifically calls out "**elevated voltage
   requests… during idle and/or light activity periods.**" This box crashes precisely in that state
   (single-core boost to ~5.5–5.9 GHz at ~3% utilization).
3. **The i9-14900K is the most-affected, highest-boost SKU** in the affected family and is covered by
   the program.
4. **Microcode cannot undo existing damage.** Intel's position is that the microcode fixes **prevent
   crashes in unaffected units, while already-degraded CPUs require replacement.** That is the case
   here.

The symptom set — `0x1E`/`STATUS_PRIVILEGED_INSTRUCTION` corrupted-instruction crashes, WHEA-19
processor-core internal-parity machine checks, high-boost/light-load faulting, and worsening over time
— is textbook Vmin Shift Instability, not a power-supply, memory, GPU, or software fault (all of which
were independently ruled out; see below).

---

## Mitigations already applied — and the result

Every reasonable non-replacement fix has been tried. The instability persists, which is why a
**hardware replacement, not a setting change, is required.**

| Mitigation | Applied? | Result |
|---|---|---|
| **Intel Default Settings** (BIOS power profile, PL1=PL2=253 W) | Yes — set through **every** crash | Did **not** stop crashes. Board profile eliminated as cause. |
| **Latest available microcode (0x12B)** loaded (BIOS-supplied) | Yes — present since the machine was built | Crashed anyway **with the fix already active** → the chip degraded despite it. |
| **XMP / DDR5-6400 disabled** (RAM dropped to JEDEC 4800) | Yes | Eliminated the *corrected memory* WHEA errors, but the machine **still hard-crashed** (`0x00020001`) with **zero** WHEA warning → the crash is a **CPU-core** fault independent of memory. |
| **GPU power capped to 250 W** | Yes | Cut GPU-side resets, but CPU machine-checks and BSODs **continued** (some fired while the GPU was idle) → not a GPU/power issue. |
| **PSU headroom check** | N/A (verified) | 1000 W 80+ Gold vs ~620 W peak load — **ruled out**. |
| **Disable Turbo Boost** (`PERFBOOSTMODE 0`; core dropped 5.9→3.1 GHz) | Yes — as a **stopgap only** | Removing the peak-voltage boost state let one long run complete, confirming the **high-voltage/high-boost** root cause — but this is a **crippling workaround** (the chip runs at base clock), **not a cure**. The chip continues to degrade under any load. |

**Conclusion:** With PSU, GPU power, motherboard profile, and memory all eliminated, and with Intel's
own microcode fix and default settings already in place, **the only remaining variable is the CPU
itself, which is degraded.** No BIOS or Windows setting can restore a Vmin-shifted core; **replacement
is the only remedy**, exactly as Intel's guidance states.

> **Optional step to strengthen the claim:** Intel later released microcode **0x12F (May 2025)**, which
> specifically targets **instability on systems running for multiple days on low-activity, lightly-
> threaded workloads** — the exact usage pattern that fails here. This box is on **0x12B**. Updating the
> MSI BIOS to a build carrying 0x12F (or newer) before/while filing demonstrates that **the very latest
> mitigation was applied and the instability still persists** — Intel will not have that as an objection.
> (It will **not** fix the chip, since it is already degraded, but it removes any "you're not on the
> latest microcode" pushback.)

---

## The request

**Replace the Intel Core i9-14900K under Intel's extended warranty** for 13th/14th Gen Vmin Shift
Instability.

- Intel extended the standard boxed-processor limited warranty by **2 additional years — to a total of
  5 years from the date of purchase** — for the affected 13th/14th Gen Core SKUs (a list of ~18 models
  that **includes the i9-14900K**). The extension applies **globally** and to **both newly and
  previously purchased** chips.
- **This CPU should be within warranty** provided the purchase date is within 5 years (it was built in
  November 2024, so it is comfortably inside the window) — `[confirm against the invoice date]`.

**Routing the RMA (choose the path that applies):**
- **If the CPU was bought as a boxed retail processor:** it can be RMA'd **directly with Intel Customer
  Support** (Intel fulfils boxed-processor warranties). The shop can do this on the customer's behalf,
  or the customer can file it themselves.
- **If it was bought as part of a pre-built system, or as a tray/OEM processor:** the claim must go
  **through the system builder / retailer / point of purchase**, not Intel directly.
- `[Shop to confirm which applies for this unit based on the invoice.]`

**Preferred fulfilment:** Intel's **Standard Warranty Replacement (SWR)** — Intel issues a **prepaid
shipping label**, the defective CPU is returned, and after screening (~3–5 business days) a replacement
is shipped. (An advance-replacement option may be available; the shop can ask Intel.)

---

## What the shop needs from the customer (checklist)

- [ ] **Proof of purchase** — a valid dated **invoice/receipt** showing the **original purchaser's
      name** (must match the person claiming the warranty). If lost, the reseller can reissue a copy.
- [ ] **CPU batch (FPO) and serial (ATPO) numbers** — read from the heat spreader / box, or via Intel's
      FPO/ATPO finder tool. Provide clear **photos of the CPU top markings** and the box label.
- [ ] **Purchase date and place of purchase.**
- [ ] **Motherboard + BIOS version** and confirmation that **Intel Default Settings** were used and the
      **latest microcode** was applied (see below).
- [ ] **Symptom description** — this document plus the crash detail (bugcheck codes, WHEA-19 events).
- [ ] **The physical CPU** (once an RMA number is issued) — handled carefully so the markings stay
      legible; ship in anti-static packaging.

---

## Supporting evidence to attach

- [ ] **Invoice / proof of purchase** (PDF or photo).
- [ ] **Photos of the CPU** heat spreader markings (FPO/serial) and the retail box label.
- [ ] **Windows Event Viewer exports:**
  - **WHEA-Logger Event 19** entries (Corrected Machine Check / Processor Core / Internal parity /
    APIC ID 41) — the stream of CPU machine-check errors.
  - **BugCheck / Kernel-Power 41** entries and the crash **minidumps** (e.g. `052426-19375-01.dmp`,
    `052426-19062-01.dmp`, `052526-14828-01.dmp`) from `C:\Windows\Minidump\`.
- [ ] **The detailed crash write-up** kept with the machine (`CrashAnalysis.md`) documenting the full
      elimination of PSU / GPU / memory / board profile and the dates/bugcheck codes.
- [ ] **(Optional but persuasive) a quick stress-test result** — e.g. Prime95 small-FFT, OCCT, or a
      Cinebench/y-cruncher run at **stock Intel Default settings with Turbo enabled** that reproduces an
      error or crash, versus stability only with **Turbo disabled**. This on-demand demonstration is
      often faster/cleaner evidence than waiting for an overnight crash.
- [ ] **Note that instability persists at Intel Default Settings with the latest microcode** — the
      single most important point for approval.

---

## Notes and caveats (things to verify / could not fully confirm)

- **Purchase date, invoice number, place of purchase, and the CPU's FPO/ATPO numbers are unknown to the
  preparer** and must be supplied by the customer from the invoice and the physical chip. No values were
  invented here.
- **Warranty eligibility depends on the purchase date and a valid, name-matching proof of purchase.**
  Based on a November 2024 build, the unit should be within the 5-year window, but this must be
  confirmed against the actual invoice.
- Intel's own support pages could not be fetched directly during research (they returned HTTP 403), so
  the warranty-term and RMA-process details here are corroborated from **multiple reputable secondary
  sources and Intel Community postings** (listed below). The specifics (2-year extension to 5 years
  total; i9-14900K covered; boxed → Intel direct, OEM/tray → point of purchase; FPO/ATPO required;
  proof of purchase required) are consistent across those sources, but the shop should confirm the
  current exact terms on Intel's official warranty pages when filing.
- **RMA experiences vary.** Some 13th/14th-gen claims have been approved smoothly (prepaid label within
  ~2 days, replacement within ~2 weeks); others report friction or requests for additional validation.
  Having complete documentation (this summary + logs + proof of purchase + legible CPU markings) up
  front reduces the chance of delays or denial.
- **The current microcode on this box is 0x12B, not the newer 0x12F.** This does not affect eligibility
  (the chip is already degraded and 0x12F would not repair it), but updating to 0x12F/newer before
  filing removes a possible objection — see the note in the mitigations section.

---

## Sources

- Intel Community — *Intel Core 13th and 14th Gen Desktop Instability Root Cause Update* (official root-cause statement, microcode 0x125/0x129/0x12B):
  https://community.intel.com/t5/Blogs/Tech-Innovation/Client/Intel-Core-13th-and-14th-Gen-Desktop-Instability-Root-Cause/post/1633239
- Intel Support — *Intel Core 13th and 14th Gen Desktop Processor Vmin Shift Instability Issue – Latest Information* (article 000102331):
  https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html
- XDA — *Intel identifies the root cause of 13th and 14th Gen CPU instability, rolls out a new microcode update*:
  https://www.xda-developers.com/intel-identifies-the-root-cause-of-13th-and-14th-gen-cpu-instability/
- Intel Community — *Vmin Shift Instability Update: New Microcode 0x12F* (does not repair degraded units; targets multi-day low-activity workloads):
  https://community.intel.com/t5/Mobile-and-Desktop-Processors/Intel-Core-13th-and-14th-Gen-Vmin-Shift-Instabilty-Update-New/m-p/1686948
- Tom's Hardware — *Raptor Lake instability saga continues as Intel releases 0x12F update*:
  https://www.tomshardware.com/pc-components/cpus/raptor-lake-instability-saga-continues-as-intel-releases-0x12f-update-to-fix-vmin-instability
- Tom's Hardware — *Intel releases extended warranty details for 13th and 14th-gen chips (i5, i7, i9 list)*:
  https://www.tomshardware.com/pc-components/cpus/intel-releases-extended-warranty-details-for-13th-and-14th-gen-chips-list-includes-core-i5-i7-and-i9-processors
- PC Gamer — *Intel extends its warranty support for 13th and 14th Gen CPUs by two years*:
  https://www.pcgamer.com/hardware/processors/intel-extends-its-warranty-support-for-13th-and-14th-gen-cpus-by-two-years-but-its-rma-procedure-has-been-anything-but-straightforward-for-some/
- TopCPU — *Warranty Extension Details for Intel 13th & 14th Gen Core Processors: Now Up to 5 Years for 18 Models*:
  https://www.topcpu.net/en/news/warranty-extension-details-for-intel-13th-14th-gen-core-processors-now-up-to-5-years-for-18-models
- Intel Support — *Intel Boxed Processors Three-Year Limited Warranty Terms and Conditions* (article 000005862):
  https://www.intel.com/content/www/us/en/support/articles/000005862/processors.html
- Intel Support — *What Information Do I Need to Provide When Requesting a Warranty Return?* (article 000060117):
  https://www.intel.com/content/www/us/en/support/articles/000060117/services/warranty.html
- Intel Support — *Where to Find Intel Boxed Processor Serial Numbers (FPO and ATPO) for Warranty Request* (article 000005609):
  https://www.intel.com/content/www/us/en/support/articles/000005609/processors.html
- Intel Support — *Warranty Guide for Intel Processors* (article 000005494):
  https://www.intel.com/content/www/us/en/support/articles/000005494/processors.html
- Intel — *Warranty support / RMA ticket portal*:
  https://supporttickets.intel.com/s/warrantyinfo?language=en_US
