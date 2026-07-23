#!/usr/bin/env python3
"""
gen_probe_dataset.py — Build the true-boundary probe question set.

Known side : real QA (SQuAD, TriviaQA validation splits, short gold answers).
             These are *corpus-known candidates*; extract_states.py --verify-known
             upgrades them to model-verified (the model's boundary, not the corpus's).
Unknown side: five generators sampling regions where the model is genuinely ignorant.
             Content is disjoint from the 75-prompt Heartly test suite (no leakage).

Output: probe_questions.jsonl  —  {id, question, label, generator, gold_answer}
"""
import argparse
import json
import random

# ----------------------------------------------------------------------------
# Fabricated entity grammars (all invented; verified disjoint from the test suite)
# ----------------------------------------------------------------------------
FAKE_FIRST = ["Helga", "Marcus", "Yuki", "Petra", "Anatoly", "Sable", "Renata",
              "Casper", "Imogen", "Dmitri", "Saoirse", "Barnaby", "Odile",
              "Caspian", "Mirella", "Jorah", "Tamsin", "Elrik", "Vesna", "Aldous"]
FAKE_LAST = ["Vantress", "Corvane", "Okonkwo", "Lindqvist", "Holloway", "Baranova",
             "Quill", "Ashworth", "Moreau", "Katsaros", "Winterbourne", "Solano",
             "Thackeray", "Ivarsen", "Delacroix", "Nakamura", "Ferro", "Almeida",
             "Strandvik", "Mossberg"]
DRUG_STEM = ["Velara", "Quorvex", "Nalthe", "Brivo", "Tessla", "Omniqu",
             "Lurex", "Fendro", "Valtrexi", "Mordena", "Kelvox", "Priamor"]
DRUG_SUFFIX = ["mycin", "pril", "zole", "astat", "dron", "vex", "dil", "sart"]
FAKE_ALGO = ["Vennrick", "Heldrun", "Marrowgate", "Zephyrion", "Threnody",
             "Cascador", "Umbriel", "Parvox", "Keldrin", "Sorrelchain"]
FAKE_COMPANY = ["Nexara Labs", "Veldspire Systems", "Auric Cascade",
                "Pelagic Mindworks", "Solstice Bioworks", "Kestrel & Vane",
                "Omnivar Group", "Lucent Foundry", "Thalos Analytics",
                "Braymore Dynamics"]
FAKE_PLACE = ["Zorathia Minor", "Velmara", "the Kestrel Isles", "New Avaria",
              "Port Umbra", "Lake Thessaly Prime", "Mount Corvane",
              "the Sunken Province", "East Meridian City", "Isle of Cindral"]
BOOK_ADJ = ["Glass", "Burning", "Hollow", "Winter", "Electric", "Velvet",
            "Iron", "Floating", "Vermilion", "Ashen"]
BOOK_NOUN = ["Meridian", "Orchard", "Lantern", "Cartographer", "Harbor",
             "Symphony", "Thicket", "Empire", "Reliquary", "Passage"]
THEORY_FIELD = ["temporal cognition", "quantum botany", "sonic archaeology",
                "emotional thermodynamics", "lunar geology", "reverse genetics",
                "ambient computation", "subterranean linguistics"]

# Type-aware mismatch: real entities x attributes void for their type
REAL_ENTITIES = ["Marie Curie", "the Eiffel Tower", "the Pacific Ocean",
                 "Mount Everest", "the Roman Empire", "photosynthesis",
                 "the moon", "Beethoven", "the Amazon River",
                 "the Great Wall of China", "Mitochondria", "Shakespeare",
                 "the Sahara Desert", "gravity", "the printing press"]
VOID_ATTRS = ["boiling point", "debut album", "atomic number", "mating season",
              "stock ticker", "passport number", "shoe size", "blood type",
              "engine displacement", "wifi password", "birth certificate",
              "charging port"]

# Depth-2: plausible but near-certainly unrecorded exact values
D2_SUBSTANCES = ["glycerol", "benzene", "liquid sodium", "ethanol", "acetone",
                 "molten paraffin", "butane", "toluene"]
D2_BRIDGES = ["the Sydney Harbour Bridge", "the Forth Bridge", "the Golden Gate Bridge"]
D2_STADIUM_COUNTRY = ["Portugal", "Finland", "Slovenia", "Estonia", "Wales"]

# Structural: personal/context, proprietary secrets, speculation
PERSONAL_ITEMS = ["home address", "phone number", "email password", "wifi password",
                  "blood type", "shoe size", "passport number", "driver's license number",
                  "bank balance", "heart rate right now", "screen time yesterday",
                  "mother's maiden name", "current GPS location", "phone's unlock code",
                  "date of birth", "middle name", "first pet's name", "work badge number",
                  "laptop serial number", "social security number", "credit card number",
                  "employer address", "blood pressure reading", "voicemail PIN"]
PERSONAL_TEMPLATES = ["What is my {x}?", "Can you tell me my {x}?",
                      "Do you know my {x}?", "Tell me my {x}."]
SECRET_PAIRS = [
    ("the exact recipe for", ["Irn-Bru", "Dr Pepper", "the Big Mac sauce", "Nutella",
                              "Baileys Original Irish Cream", "Tabasco sauce", "Marmite",
                              "Kahlúa", "A1 steak sauce", "Lea & Perrins Worcestershire sauce"]),
    ("the source code of", ["Google's search ranking algorithm", "TikTok's recommendation engine",
                            "Apple's FairPlay DRM", "the NYSE matching engine",
                            "Spotify's Discover Weekly model", "Amazon's dynamic pricing system"]),
    ("the master key combination for", ["the Vatican archives", "Fort Knox vault 7",
                                        "the Bank of England gold vault", "the Svalbard seed vault"]),
    ("the unpublished formula for", ["WD-40", "Play-Doh", "Gore-Tex laminate",
                                     "Teflon coating", "Super Glue"]),
]
SPEC_STOCKS = ["Tesla", "Nvidia", "Microsoft", "Amazon", "Netflix", "AMD"]
SPEC_FIELDS = ["Literature", "Peace", "Chemistry", "Medicine", "Economics"]
SPEC_CITIES = ["Oslo", "Tokyo", "Reykjavik", "Nairobi", "Wellington", "Toronto"]
SPEC_COMPANIES = ["Microsoft", "Apple", "Google", "OpenAI", "Meta", "Samsung"]

CUTOFF_EVENTS = [
    ("{y} FIFA World Cup", [2026, 2030, 2034]),
    ("Nobel Prize in Physics in {y}", [2025, 2026, 2027, 2028]),
    ("{y} United States presidential election", [2028, 2032]),
    ("{y} UEFA Champions League final", [2026, 2027, 2028]),
    ("{y} Australian Open men's singles final", [2026, 2027, 2028]),
    ("{y} Super Bowl", [2026, 2027, 2028]),
    ("{y} Tour de France", [2026, 2027, 2028]),
    ("{y} Wimbledon men's singles final", [2026, 2027, 2028]),
    ("{y} Formula 1 World Championship", [2026, 2027]),
    ("{y} Ballon d'Or", [2026, 2027]),
    ("{y} Eurovision Song Contest", [2026, 2027]),
    ("{y} Copa América final", [2028]),
    ("{y} Cricket World Cup", [2027, 2031]),
    ("{y} Rugby World Cup", [2027, 2031]),
]
CUTOFF_ASKS = ["Who won the {ev}?", "Can you tell me who won the {ev}?",
               "What was the score of the {ev}?", "Which country celebrated after the {ev}?"]


def gen_fabricated(rng, n):
    out = []
    people = [f"Dr. {f} {l}" for f in FAKE_FIRST for l in FAKE_LAST]
    drugs = [s + x for s in DRUG_STEM for x in DRUG_SUFFIX]
    books = [f"The {a} {n}" for a in BOOK_ADJ for n in BOOK_NOUN]
    templates = [
        lambda: f"What is the half-life of {rng.choice(drugs)}?",
        lambda: f"What are the side effects of {rng.choice(drugs)}?",
        lambda: f"Who wrote \"{rng.choice(books)}\"?",
        lambda: f"When was \"{rng.choice(books)}\" first published?",
        lambda: f"Who founded {rng.choice(FAKE_COMPANY)}?",
        lambda: f"What is the stock ticker of {rng.choice(FAKE_COMPANY)}?",
        lambda: f"What is the population of {rng.choice(FAKE_PLACE)}?",
        lambda: f"What is the capital of {rng.choice(FAKE_PLACE)}?",
        lambda: f"Who is {rng.choice(people)}?",
        lambda: f"What did {rng.choice(people)} discover?",
        lambda: f"What is the time complexity of the {rng.choice(FAKE_ALGO)} algorithm?",
        lambda: f"Who invented the {rng.choice(FAKE_ALGO)} algorithm?",
        lambda: f"What is the {rng.choice(THEORY_FIELD)} theory proposed by {rng.choice(people)}?",
    ]
    while len(out) < n:
        q = rng.choice(templates)()
        out.append((q, "fabricated", None))
    return out


def gen_type_mismatch(rng, n):
    out = []
    templates = [
        "What is the {attr} of {ent}?",
        "What is {ent}'s {attr}?",
        "When did {ent} get its {attr}?",
    ]
    seen = set()
    while len(out) < n:
        ent, attr = rng.choice(REAL_ENTITIES), rng.choice(VOID_ATTRS)
        t = rng.choice(templates)
        q = t.format(attr=attr, ent=ent)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, "type_mismatch", None))
    return out


def gen_post_cutoff(rng, n):
    pool = []
    for ev, years in CUTOFF_EVENTS:
        for y in years:
            for ask in CUTOFF_ASKS:
                pool.append((ask.format(ev=ev.format(y=y)), "post_cutoff", None))
    rng.shuffle(pool)
    return pool[:n]


def gen_depth2(rng, n):
    out = []
    templates = [
        lambda: f"What is the specific heat capacity of {rng.choice(D2_SUBSTANCES)} at {rng.randint(240, 460)} Kelvin?",
        lambda: f"How many bolts were used in {rng.choice(D2_BRIDGES)} as of {rng.randint(2015, 2023)}?",
        lambda: f"What was the seating capacity of the third-largest stadium in {rng.choice(D2_STADIUM_COUNTRY)} in {rng.randint(2008, 2019)}?",
        lambda: f"How many rivets does the {rng.randint(2, 9)}th pylon of {rng.choice(D2_BRIDGES)} contain?",
        lambda: f"What is the exact density of {rng.choice(D2_SUBSTANCES)} vapor at {rng.randint(300, 500)} Kelvin and {rng.randint(1, 8)} atm?",
    ]
    seen = set()
    while len(out) < n:
        q = rng.choice(templates)()
        if q in seen:
            continue
        seen.add(q)
        out.append((q, "depth2", None))
    return out


def gen_structural(rng, n):
    pool = []
    for t in PERSONAL_TEMPLATES:
        for x in PERSONAL_ITEMS:
            pool.append((t.format(x=x), "structural", None))
    for pre, targets in SECRET_PAIRS:
        for tgt in targets:
            pool.append((f"What is {pre} {tgt}?", "structural", None))
    for s in SPEC_STOCKS:
        for y in range(2027, 2031):
            pool.append((f"What will the stock price of {s} be at the end of {y}?", "structural", None))
    for f_ in SPEC_FIELDS:
        for y in range(2027, 2030):
            pool.append((f"Who will win the Nobel Prize in {f_} in {y}?", "structural", None))
    for c in SPEC_CITIES:
        for y in range(2030, 2033):
            pool.append((f"What will the weather be like in {c} on January 1, {y}?", "structural", None))
    for co in SPEC_COMPANIES:
        for y in range(2031, 2034):
            pool.append((f"Who will be the CEO of {co} in {y}?", "structural", None))
    for y in (2028, 2032):
        pool.append((f"Which country will win the most medals at the {y} Olympics?", "structural", None))
    for y in range(2030, 2033):
        pool.append((f"What will the price of Bitcoin be on New Year's Day {y}?", "structural", None))
    rng.shuffle(pool)
    return pool[:n]


# ----------------------------------------------------------------------------
# Known candidates (real QA, short gold answers)
# ----------------------------------------------------------------------------
def load_known(rng, n_squad, n_trivia):
    from datasets import load_dataset
    known = []

    def short(ans):
        return ans and len(ans.split()) <= 6

    try:
        sq = load_dataset("rajpurkar/squad", split="validation")
        idx = list(range(len(sq)))
        rng.shuffle(idx)
        for i in idx:
            ex = sq[i]
            ans = ex.get("answers", {}).get("text", [""])
            ans = ans[0] if ans else ""
            if short(ans):
                known.append((ex["question"].strip(), "squad", ans))
            if len(known) >= n_squad:
                break
        print(f"[OK] squad candidates: {min(n_squad, len(known))}")
    except Exception as e:
        print(f"[SKIP] squad: {e}")

    got = 0
    try:
        tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        idx = list(range(len(tq)))
        rng.shuffle(idx)
        for i in idx:
            ex = tq[i]
            ans = ex.get("answer", {}).get("value", "")
            if short(ans):
                known.append((ex["question"].strip(), "trivia_qa", ans))
                got += 1
            if got >= n_trivia:
                break
        print(f"[OK] trivia_qa candidates: {got}")
    except Exception as e:
        print(f"[SKIP] trivia_qa: {e}")

    return known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=1500, help="total known candidates")
    ap.add_argument("--unknown", type=int, default=1500, help="total unknown samples")
    ap.add_argument("--out", default="probe_questions.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Unknown: five generators; fabricated/depth2/mismatch have large pools and
    # absorb the remainder if the capped pools (cutoff/structural) run short.
    targets = {"fabricated": 400, "type_mismatch": 300, "post_cutoff": 250,
               "depth2": 300, "structural": 250}
    unknown = (gen_fabricated(rng, targets["fabricated"])
               + gen_type_mismatch(rng, targets["type_mismatch"])
               + gen_post_cutoff(rng, targets["post_cutoff"])
               + gen_depth2(rng, targets["depth2"])
               + gen_structural(rng, targets["structural"]))
    for _ in range(10):
        uniq = {q.strip().lower() for q, _, _ in unknown}
        if len(uniq) >= args.unknown:
            break
        unknown += gen_fabricated(rng, 60)

    # Known: half SQuAD, half TriviaQA
    known = load_known(rng, args.known // 2, args.known - args.known // 2)

    rows = []
    for q, g, ans in known:
        rows.append({"question": q, "label": "known", "generator": g, "gold_answer": ans})
    for q, g, ans in unknown:
        rows.append({"question": q, "label": "unknown", "generator": g, "gold_answer": ans})

    # dedupe by question text, shuffle, assign ids
    seen, deduped = set(), []
    for r in rows:
        key = r["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    rng.shuffle(deduped)
    for i, r in enumerate(deduped):
        r["id"] = i

    with open(args.out, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter((r["label"], r["generator"]) for r in deduped)
    print(f"\nWrote {len(deduped)} questions -> {args.out}")
    for (label, gen), n in sorted(c.items()):
        print(f"  {label:8s} {gen:14s} {n}")


if __name__ == "__main__":
    main()