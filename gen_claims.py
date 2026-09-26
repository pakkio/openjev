"""Claim attribution and claim verification sets over a made-up scientific literature.

attrib: a claim -> which reference in the bibliography supports it. The bibliography is fixed
(24 invented papers, 3 per topic); two of every topic's papers study the same intervention in
different populations, so a claim that omits the population can only be attributed by knowing
which paper measured which outcome. Every row offers the target, its 2 same-topic siblings and
5 random papers. Finding phrasings and claim wrappers are split 60/20/20, so validation and test
claims use wordings never seen in training. --keys-only shows options as "Surname et al. (year)"
with no title, so the mapping has to be learned rather than read off the title.

verify: an abstract + a claim -> supported / refuted / not enough info, balanced. Refuted claims
flip the direction of a finding, deny it, or misstate its size; unaddressed claims are about an
outcome the abstract never measured. Interventions and phrasings are split 60/20/20, so held-out
rows describe unseen studies in unseen wordings.

usage: gen_claims.py attrib|verify [OUT_DIR] [N_TRAIN] [N_VAL] [N_TEST] [SEED] [--keys-only]
"""
import json, os, random, sys

TOPICS = {
    "diet": (
        ["intermittent fasting", "a Mediterranean diet", "a low-salt diet", "daily fibre supplements", "a plant-based diet",
         "a ketogenic diet", "a high-protein breakfast", "a sugar-free diet", "time-restricted eating", "daily nut consumption"],
        ["systolic blood pressure", "body weight", "LDL cholesterol", "fasting glucose", "waist circumference", "resting heart rate"],
        ["adults over sixty", "office workers", "people with type 2 diabetes", "university students", "postmenopausal women"],
    ),
    "sleep": (
        ["a fixed bedtime routine", "blue-light glasses", "a ban on evening screens", "a weighted blanket", "mindfulness before bed",
         "a later school start time", "a morning light lamp", "a caffeine curfew", "a sleep-tracking app", "a cooler bedroom"],
        ["sleep onset latency", "total sleep time", "night-time awakenings", "daytime sleepiness", "evening melatonin levels", "morning reaction time"],
        ["teenagers", "shift workers", "new parents", "retired adults", "people with insomnia"],
    ),
    "exercise": (
        ["high-intensity interval training", "a daily brisk walk", "resistance training", "yoga classes", "a standing desk",
         "cycling to work", "a stair-climbing programme", "tai chi", "group swimming lessons", "a step-count challenge"],
        ["VO2 max", "lower-back pain", "grip strength", "falls per year", "resting blood pressure", "muscle mass"],
        ["sedentary adults", "nursing-home residents", "cancer survivors", "call-centre staff", "children aged eight to twelve"],
    ),
    "air": (
        ["HEPA air purifiers", "a low-emission zone", "roadside hedges", "a ban on wood stoves", "electric buses",
         "a car-free school street", "indoor plants", "gas-to-induction cooktop swaps", "window filters", "a congestion charge"],
        ["asthma attacks", "indoor PM2.5 concentration", "hospital admissions for bronchitis", "lung function", "nitrogen dioxide exposure", "inhaler use"],
        ["children with asthma", "commuters", "elderly city residents", "pregnant women", "outdoor workers"],
    ),
    "education": (
        ["a phone-free classroom", "peer tutoring", "a four-day school week", "daily reading aloud", "gamified maths apps",
         "smaller class sizes", "outdoor lessons", "a free school breakfast", "homework bans", "coding clubs"],
        ["maths test scores", "absenteeism", "reading fluency", "self-reported anxiety", "classroom disruptions", "dropout rates"],
        ["primary-school pupils", "secondary-school students", "pupils from low-income families", "first-year undergraduates", "adult learners"],
    ),
    "work": (
        ["a four-day work week", "fully remote work", "open-plan offices", "no-meeting Wednesdays", "flexible hours",
         "a right-to-disconnect policy", "on-site childcare", "hot-desking", "quiet rooms", "a monthly wellbeing day"],
        ["staff turnover", "sick days", "self-rated productivity", "burnout scores", "overtime hours", "commute time"],
        ["software engineers", "hospital nurses", "bank employees", "public-sector clerks", "warehouse staff"],
    ),
    "city": (
        ["new urban parks", "street tree planting", "protected bike lanes", "a pedestrianised centre", "green roofs",
         "lower speed limits", "community gardens", "public drinking fountains", "cooling centres", "night-time street lighting"],
        ["summer surface temperature", "traffic injuries", "reported loneliness", "street crime", "cycling trips", "heat-related deaths"],
        ["inner-city neighbourhoods", "older residents", "families with young children", "suburban commuters", "social-housing tenants"],
    ),
    "medicine": (
        ["a vitamin D supplement", "low-dose aspirin", "a probiotic drink", "a flu vaccine booster", "text-message reminders",
         "a nurse-led clinic", "a digital therapy app", "an exercise prescription", "a pharmacist medication review", "group therapy sessions"],
        ["hospital readmissions", "depression scores", "medication adherence", "infection rates", "emergency visits", "bone density"],
        ["patients after heart surgery", "care-home residents", "people with chronic pain", "young adults with depression", "patients with hypertension"],
    ),
}

DOWN = ["{i} reduced {o} by {p}%", "{i} lowered {o} by {p} percent", "{i} was followed by a {p}% fall in {o}",
        "{o} dropped by {p}% with {i}", "{i} cut {o} by {p}%", "{o} was {p}% lower under {i}",
        "{i} led to a {p}% decrease in {o}", "{i} brought {o} down by {p}%", "{i} decreased {o} by {p}%",
        "{o} shrank by {p}% after {i}"]
UP = ["{i} increased {o} by {p}%", "{i} raised {o} by {p} percent", "{i} was followed by a {p}% rise in {o}",
      "{o} climbed by {p}% with {i}", "{i} boosted {o} by {p}%", "{o} was {p}% higher under {i}",
      "{i} led to a {p}% increase in {o}", "{i} pushed {o} up by {p}%", "{i} lifted {o} by {p}%",
      "{o} grew by {p}% after {i}"]
NONE = ["{i} had no measurable effect on {o}", "{o} did not change with {i}", "{i} left {o} unchanged",
        "no difference in {o} was found with {i}", "{i} did not affect {o}", "{o} was similar with and without {i}",
        "{i} made no significant difference to {o}", "{o} stayed the same under {i}", "{i} failed to change {o}",
        "{i} showed no effect on {o}"]
# claims without a number
DOWN_Q = ["{i} reduced {o}", "{i} lowered {o}", "{o} fell with {i}", "{i} was linked to lower {o}", "{i} decreased {o}",
          "{i} cut {o}", "{o} went down under {i}", "{i} brought down {o}", "{i} led to less {o}", "{o} declined after {i}"]
UP_Q = ["{i} increased {o}", "{i} raised {o}", "{o} rose with {i}", "{i} was linked to higher {o}", "{i} boosted {o}",
        "{i} pushed up {o}", "{o} went up under {i}", "{i} lifted {o}", "{i} led to more {o}", "{o} grew after {i}"]
PHRASES = {"down": (DOWN, DOWN_Q), "up": (UP, UP_Q), "none": (NONE, NONE)}
FLIP = {"down": "up", "up": "down"}

ATTR_TEMPLATES = [
    "Claim: {c}. Which reference supports this claim?",
    "Find the citation for this statement: {c}.",
    "A reviewer asks for a source for the sentence \"{c}.\" Pick the reference.",
    "Which paper in the bibliography reports that {c}?",
    "Our draft says: {c}. Which source should we cite here?",
    "Statement needing a citation: {c}. Choose the reference that backs it.",
    "In the related-work section we wrote \"{c}.\" What do we cite?",
    "Attribute this finding to its source: {c}.",
    "Which study found that {c}?",
    "Citation needed: {c}. Select the supporting reference.",
]
VERIFY_TEMPLATES = [
    "Source abstract: {a}\nClaim: {c}.\nDoes the source support the claim?",
    "Abstract: {a}\nStatement: {c}.\nIs the statement supported, refuted, or not addressed by the abstract?",
    "Read the abstract and judge the claim.\nAbstract: {a}\nClaim: {c}.",
    "Paper summary: {a}\nSomeone cites it for: {c}.\nIs that citation accurate?",
    "Evidence: {a}\nClaim to check: {c}.\nWhat does the evidence say about the claim?",
]
VERDICTS = ["supported", "refuted", "not enough info"]
DESIGNS = ["a randomised trial", "a twelve-week study", "a two-year cohort study", "a controlled pilot study", "a cluster trial"]
SURNAMES = ["Rossi", "Okafor", "Lindqvist", "Tanaka", "Moreau", "Kowalski", "Haddad", "Ferreira", "Novak", "Brennan",
            "Castillo", "Sato", "Mbeki", "Andersen", "Petrov", "Duarte", "Fischer", "Nakamura", "Oyelaran", "Varga",
            "Keller", "Iyer", "Mancini", "Holm", "Chowdhury", "Walsh", "Esposito", "Kim", "Delacroix", "Ahmadi"]
JOURNALS = ["Journal of Applied Health", "Public Health Letters", "Annals of Everyday Medicine", "Urban Studies Review",
            "Education Research Quarterly", "Occupational Wellbeing", "Environmental Epidemiology Notes"]
SPLITS = {"train": (0.0, 0.6), "validation": (0.6, 0.8), "test": (0.8, 1.0)}
BIB_SEED = 2026


def piece(pool, split):
    a, b = SPLITS[split]
    return pool[int(len(pool) * a):int(len(pool) * b)]


def cap(s):
    return s[0].upper() + s[1:]


def phrase(rng, split, d, i, o, p=None, with_pct=True):
    pct, qual = PHRASES[d]
    pool = pct if (with_pct and p is not None) or d == "none" else qual
    return rng.choice(piece(pool, split)).format(i=i, o=o, p=p)


def bibliography():
    """24 fixed papers: per topic, A in pop X, A in pop Y, B in pop X; (intervention, outcome) unique per topic."""
    rng = random.Random(BIB_SEED)
    names = rng.sample(SURNAMES, len(SURNAMES))
    years = iter(rng.sample(range(2008, 2026), 18) * 2)
    papers = []
    for topic, (ints, outs, pops) in TOPICS.items():
        a, b = rng.sample(ints, 2)
        x, y = rng.sample(pops, 2)
        o = rng.sample(outs, 5)
        for inter, pop, (o1, o2) in [(a, x, (o[0], o[1])), (a, y, (o[2], o[3])), (b, x, (o[0], o[4]))]:
            authors = [names.pop()] + rng.sample(SURNAMES, 2)
            year = next(years)
            findings = [{"outcome": oc, "dir": rng.choice(["down", "down", "up", "none"]), "pct": rng.randint(5, 45)}
                        for oc in (o1, o2)]
            papers.append({
                "key": f"{authors[0]} et al. ({year})",
                "entry": f"{authors[0]}, {authors[1]} and {authors[2]} ({year}). {cap(inter)} in {pop}: "
                         f"{rng.choice(DESIGNS).split(' ', 1)[1]}. {rng.choice(JOURNALS)}.",
                "topic": topic, "intervention": inter, "population": pop, "findings": findings,
            })
    return papers


def make_attrib(rng, split, bib, keys_only=False):
    target = rng.choice(bib)
    f = rng.choice(target["findings"])
    c = phrase(rng, split, f["dir"], target["intervention"], f["outcome"], f["pct"], rng.random() < .6)
    if rng.random() < .5:
        c += f" in {target['population']}"
    siblings = [p for p in bib if p["topic"] == target["topic"] and p is not target]
    others = rng.sample([p for p in bib if p["topic"] != target["topic"]], 5)
    opts = [target] + siblings + others
    rng.shuffle(opts)
    field = "key" if keys_only else "entry"
    return {"context": rng.choice(piece(ATTR_TEMPLATES, split)).format(c=c),
            "options": [p[field] for p in opts], "label": opts.index(target),
            "key": target["key"], "claim": c}


def make_verify(rng, split):
    topic = rng.choice(list(TOPICS))
    ints, outs, pops = TOPICS[topic]
    inter, pop = rng.choice(piece(ints, split)), rng.choice(pops)
    measured = rng.sample(outs, rng.randint(2, 3))
    findings = [{"outcome": o, "dir": rng.choice(["down", "up", "none"]), "pct": rng.randint(5, 45)} for o in measured]
    sentences = [cap(phrase(rng, split, f["dir"], inter, f["outcome"], f["pct"])) + "." for f in findings]
    abstract = f"We ran {rng.choice(DESIGNS)} with {rng.randint(8, 200) * 10} {pop}. " + " ".join(sentences)

    verdict = rng.choice(VERDICTS)
    f = rng.choice(findings)
    o, d, p = f["outcome"], f["dir"], f["pct"]
    if verdict == "supported":
        c = phrase(rng, split, d, inter, o, p, rng.random() < .5)
    elif verdict == "refuted":
        if d == "none":
            c = phrase(rng, split, rng.choice(["up", "down"]), inter, o, p, rng.random() < .5)
        else:
            kind = rng.choice(["flip", "deny", "size"])
            if kind == "flip":
                c = phrase(rng, split, FLIP[d], inter, o, p, rng.random() < .5)
            elif kind == "deny":
                c = phrase(rng, split, "none", inter, o)
            else:
                wrong = rng.choice([x for x in range(5, 91) if abs(x - p) >= 15 and max(x, p) / min(x, p) >= 1.8])
                c = phrase(rng, split, d, inter, o, wrong)
    else:
        o = rng.choice([x for x in outs if x not in measured])
        c = phrase(rng, split, rng.choice(["up", "down", "none"]), inter, o, rng.randint(5, 45), rng.random() < .5)

    opts = VERDICTS[:]
    rng.shuffle(opts)
    return {"context": rng.choice(VERIFY_TEMPLATES).format(a=abstract, c=c), "options": opts,
            "label": opts.index(verdict), "verdict": verdict, "claim": c}


if __name__ == "__main__":
    KEYS = "--keys-only" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--keys-only"]
    task = sys.argv[1] if len(sys.argv) > 1 else "attrib"
    default = {"attrib": "claims_attrib_keys" if KEYS else "claims_attrib", "verify": "claims_verify"}[task]
    out = sys.argv[2] if len(sys.argv) > 2 else f"data/synthetic/{default}"
    sizes = [int(x) for x in sys.argv[3:6]] or [2000, 300, 300]
    rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 41)
    bib = bibliography()
    os.makedirs(out, exist_ok=True)
    if task == "attrib":
        with open(f"{out}/bibliography.json", "w") as f:
            json.dump(bib, f, ensure_ascii=False, indent=1)
    for split, n in zip(SPLITS, sizes):
        with open(f"{out}/{split}.jsonl", "w") as f:
            for _ in range(n):
                row = make_attrib(rng, split, bib, KEYS) if task == "attrib" else make_verify(rng, split)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {n} samples")
