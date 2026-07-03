"""Build the synthetic fixture PDF used by all tests (design doc §3, §10).

The content is a hand-made miniature wargame ruleset ("Concordance of
Battle") — original text, NOT Games Workshop content — so CI never touches
GW material. Layout mimics the real target: heading hierarchy by font size,
two-column body text, a boxed commentary sidebar, and cross-references.
"""

from __future__ import annotations

from pathlib import Path

import fitz

H1, H2, H3, BODY = 20, 15, 12, 9.5
MARGIN = 40
PAGE_W, PAGE_H = 595, 842  # A4 points


class _Writer:
    def __init__(self, doc: fitz.Document) -> None:
        self.doc = doc
        self.page = doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN

    def _ensure(self, needed: float) -> None:
        if self.y + needed > PAGE_H - MARGIN:
            self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
            self.y = MARGIN

    def heading(self, text: str, size: float) -> None:
        self._ensure(size * 2 + 10)
        self.y += size * 0.6
        self.page.insert_text((MARGIN, self.y + size), text, fontsize=size, fontname="hebo")
        self.y += size * 1.6

    def body(self, text: str, height: float = 0) -> None:
        rect_h = height or (BODY * 1.35 * (text.count("\n") + len(text) // 85 + 2))
        self._ensure(rect_h + 8)
        rect = fitz.Rect(MARGIN, self.y, PAGE_W - MARGIN, self.y + rect_h)
        # insert_textbox returns the unused height at the bottom of the rect
        leftover = self.page.insert_textbox(rect, text, fontsize=BODY, fontname="helv")
        self.y = rect.y0 + (rect_h - max(leftover, 0)) + 8

    def sidebar(self, text: str) -> None:
        rect_h = BODY * 1.35 * (text.count("\n") + len(text) // 80 + 2) + 16
        self._ensure(rect_h + 12)
        box = fitz.Rect(MARGIN, self.y, PAGE_W - MARGIN, self.y + rect_h)
        self.page.draw_rect(box, color=(0.3, 0.3, 0.3), fill=(0.92, 0.92, 0.92), width=1)
        inner = fitz.Rect(box.x0 + 8, box.y0 + 8, box.x1 - 8, box.y1 - 8)
        self.page.insert_textbox(inner, text, fontsize=BODY, fontname="helv")
        self.y = box.y1 + 12


def build_fixture_pdf(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    w = _Writer(doc)

    w.heading("Core Concepts", H1)
    w.body(
        "This section defines the terms used throughout the Concordance of "
        "Battle. Players should read it before their first game."
    )
    w.heading("Engagement Range", H2)
    w.body(
        "A model is within Engagement Range of an enemy model if it is within "
        "1 inch of that model horizontally and 5 inches vertically. Units cannot "
        "be selected to shoot while any of their models are within Engagement "
        "Range of an enemy unit."
    )
    w.heading("Objective Control", H2)
    w.body(
        "Each objective marker is controlled by the player whose models within "
        "3 inches of it have the highest total control value. A player who "
        "controls an objective marker at the end of their turn scores it. "
        "Battle-shocked units have a control value of zero."
    )

    w.heading("Movement Phase", H1)
    w.body(
        "In your Movement Phase you can move the units in your army. Select one "
        "unit at a time and choose the kind of move it will make."
    )
    w.heading("Moving Units", H2)
    w.body(
        "When a unit moves, every model in it can move up to the Move "
        "characteristic shown on its profile. No model can move through enemy "
        "models or off the battlefield edge."
    )
    w.heading("Normal Move", H3)
    w.body(
        "A unit making a Normal Move can move a distance up to its Move "
        "characteristic. It cannot move within Engagement Range of any enemy "
        "unit while doing so."
    )
    w.heading("Advance", H3)
    w.body(
        "When a unit Advances, roll one six-sided die and add the result to its "
        "Move characteristic for that phase. A unit that Advances cannot shoot "
        "or charge later in the turn."
    )
    w.heading("Fall Back", H3)
    w.body(
        "A unit that starts its Movement Phase within Engagement Range of an "
        "enemy unit can Fall Back. Falling back means the unit moves up to its "
        "Move characteristic, escaping combat, and it cannot shoot or charge "
        "this turn after retreating unless a special rule says otherwise."
    )

    w.heading("Transports", H2)
    w.body(
        "Some vehicles can carry troops. The number of models a transport can "
        "carry is shown on its profile. Units can embark and disembark as "
        "described below."
    )
    w.heading("Embark", H3)
    w.body(
        "A unit can embark aboard a friendly transport in the Movement Phase if "
        "every model ends a move within 3 inches of it, boarding the vehicle and "
        "being removed from the battlefield. A unit cannot embark and disembark "
        "in the same phase."
    )
    w.heading("Disembark", H3)
    w.body(
        "A unit that begins its Movement Phase embarked aboard a transport can "
        "disembark before the transport moves. When getting out of a transport, "
        "every model must be set up within 3 inches of the vehicle and not "
        "within Engagement Range of any enemy unit. A unit that disembarks from "
        "a transport that has not yet moved can then make a Normal Move; if the "
        "transport arrived from Reserves this turn, the unit cannot (see "
        "Reserves, page 3)."
    )
    w.sidebar(
        "Designer's note: exiting a vehicle counts as movement for abilities "
        "that trigger on movement, even if the disembarking unit stays still "
        "afterwards."
    )

    w.heading("Reinforcements", H1)
    w.heading("Reserves", H2)
    w.body(
        "Units placed in Reserves start the battle off the battlefield, waiting "
        "to arrive. In the Reinforcements step of your Movement Phase, a unit "
        "arriving from Reserves is set up wholly within 6 inches of a "
        "battlefield edge and more than 9 inches from all enemy models. Units "
        "embarked in a transport in Reserves arrive with that transport."
    )
    w.heading("Deep Strike", H2)
    w.body(
        "Units with the Deep Strike ability can be set up in orbit instead of "
        "being deployed. In the Reinforcements step they can teleport in, being "
        "set up anywhere on the battlefield more than 9 inches from all enemy "
        "models. Deep Strike arrivals count as arriving from Reserves."
    )

    doc.save(str(path))
    doc.close()
    return path


if __name__ == "__main__":
    out = build_fixture_pdf(Path(__file__).parent / "fixture.pdf")
    print(f"wrote {out}")
