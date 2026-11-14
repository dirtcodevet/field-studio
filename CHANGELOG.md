# Changelog

## 0.2.0-research (2026-11-14)

Second public research release, and the first with code.

- **Code** under Apache-2.0: Field Studio (`engine/fieldstudio/`), the mouth library (`engine/library/mouth/`), the
  light rig (`engine/light/`), the hair engine (`engine/hair/`, mesh-free) and the material library (`engine/materials/profiles/`, sources in `docs/DATA_SOURCES.md`). A source snapshot of the
  project's private working repository at commit `3389050f` (2026-11-14 13:51 PT), with the parts tied to the private
  test model left out or made optional, and comments reworded where they described private material (the README's
  "What the code includes" lists what is left out and why).
- **Status review** (`docs/STATUS_2026-11-14.md`): what changed since 0.1.0, what works with its numbers, what is open,
  the research direction, findings, tooling and methods.
- **Research, 2026-11-14** (`docs/research/2026-11-14/`): the specific research and its findings, measured, with
  images.
- **The methods we changed** (`docs/PATENT_STORY.md`) and **Direction** (`docs/DIRECTION.md`).
- **Patent readings** (`docs/PATENTS.md`), for transparency. Held out of the code on that reading: the earlier line
  tool, Connect along a curved path, and an unused earlier storage layout.
- **The milestone session** and **the hair session** as public tests (`tests/`).
- **Twelve new images** of the tool (`images/v0.2.0/`).
- `LICENSE` is now the Apache License 2.0 for the code; the CC BY 4.0 text that covers the documents and images moved
  to `LICENSE-CC-BY-4.0`. `NOTICE` lists dependencies and data sources with their licences.

## 0.1.0-research (2026-11-12)

First public research release, documents and images only.

- The research write-up (`docs/RESEARCH.md`, also by part in `docs/research/`).
- Technical disclosures by subsystem (`docs/disclosures/`).
- Nine short articles (`articles/`).
- Eighteen images of the tool and the test subject in segments (`images/`).
- The code is not included; it follows separately under Apache-2.0 once parts are redesigned (write-up Part 10).
