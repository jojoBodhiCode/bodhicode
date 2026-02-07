"""
Buddhist technical terms glossary with cross-tradition equivalents.

Provides lookup for key Buddhist terms across Pali, Sanskrit, Tibetan,
Chinese, and English, with brief definitions and tradition context.
"""

from typing import List, Optional


# Each entry: {
#   "english": str,
#   "pali": str,
#   "sanskrit": str,
#   "tibetan": str,  (Wylie transliteration)
#   "chinese": str,
#   "definition": str,
#   "traditions": list of str,
# }

GLOSSARY = [
    # ─── Core Doctrinal Terms ─────────────────────────────────────────────
    {
        "english": "suffering / unsatisfactoriness",
        "pali": "dukkha",
        "sanskrit": "duhkha",
        "tibetan": "sdug bsngal",
        "chinese": "苦 (ku)",
        "definition": "The first Noble Truth. Not merely pain, but the "
                      "pervasive unsatisfactoriness of conditioned existence, "
                      "including impermanence and lack of inherent self.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "non-self / selflessness",
        "pali": "anatta",
        "sanskrit": "anatman",
        "tibetan": "bdag med",
        "chinese": "無我 (wuwo)",
        "definition": "The teaching that no permanent, independent self exists "
                      "in the five aggregates. In Mahayana, extended to the "
                      "selflessness of all phenomena (dharma-nairatmya).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "impermanence",
        "pali": "anicca",
        "sanskrit": "anitya",
        "tibetan": "mi rtag pa",
        "chinese": "無常 (wuchang)",
        "definition": "All conditioned phenomena are impermanent. One of the "
                      "three marks of existence (tilakkhana/trilakshana).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "emptiness",
        "pali": "sunnata",
        "sanskrit": "sunyata",
        "tibetan": "stong pa nyid",
        "chinese": "空 (kong)",
        "definition": "In Theravada: emptiness of self in the aggregates. "
                      "In Madhyamaka: the lack of inherent existence (svabhava) "
                      "of ALL phenomena. Not nihilism — emptiness enables "
                      "dependent origination.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "dependent origination",
        "pali": "paticcasamuppada",
        "sanskrit": "pratityasamutpada",
        "tibetan": "rten cing 'brel bar 'byung ba",
        "chinese": "緣起 (yuanqi)",
        "definition": "All phenomena arise in dependence upon causes and "
                      "conditions. The 12-link chain explains the arising of "
                      "suffering; in Madhyamaka, equivalent to emptiness.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "nirvana / liberation",
        "pali": "nibbana",
        "sanskrit": "nirvana",
        "tibetan": "mya ngan las 'das pa",
        "chinese": "涅槃 (niepan)",
        "definition": "The cessation of suffering and the cycle of rebirth. "
                      "In Mahayana, non-abiding nirvana (apratisthita-nirvana) "
                      "means neither remaining in samsara nor resting in "
                      "individual liberation.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "mindfulness / awareness",
        "pali": "sati",
        "sanskrit": "smrti",
        "tibetan": "dran pa",
        "chinese": "念 (nian)",
        "definition": "Present-moment awareness. The seventh factor of the "
                      "Noble Eightfold Path (samma sati / samyak smrti).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "concentration / meditative absorption",
        "pali": "samadhi",
        "sanskrit": "samadhi",
        "tibetan": "ting nge 'dzin",
        "chinese": "三昧 / 定 (sanmei / ding)",
        "definition": "One-pointed mental absorption. The eighth factor of "
                      "the Noble Eightfold Path. Basis for jhana/dhyana states.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "wisdom / insight",
        "pali": "panna",
        "sanskrit": "prajna",
        "tibetan": "shes rab",
        "chinese": "般若 (bore / hannya)",
        "definition": "Direct insight into the nature of reality. The "
                      "perfection of wisdom (prajnaparamita) is central to "
                      "Mahayana. Distinct from mere intellectual knowledge.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "compassion",
        "pali": "karuna",
        "sanskrit": "karuna",
        "tibetan": "snying rje",
        "chinese": "悲 (bei)",
        "definition": "The wish for beings to be free from suffering. "
                      "Great compassion (mahakaruna) is a defining quality "
                      "of bodhisattvas in Mahayana.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    # ─── Path & Practice ──────────────────────────────────────────────────
    {
        "english": "awakening mind / mind of enlightenment",
        "pali": "",
        "sanskrit": "bodhicitta",
        "tibetan": "byang chub kyi sems",
        "chinese": "菩提心 (putixin)",
        "definition": "The aspiration to attain Buddhahood for the benefit "
                      "of all sentient beings. Has two aspects: aspiring "
                      "(pranidhana) and engaging (prasthana).",
        "traditions": ["Mahayana", "Vajrayana"],
    },
    {
        "english": "perfections",
        "pali": "parami",
        "sanskrit": "paramita",
        "tibetan": "pha rol tu phyin pa",
        "chinese": "波羅蜜 (boluomi)",
        "definition": "Transcendent virtues practiced by bodhisattvas. "
                      "Six in Mahayana (generosity, ethics, patience, effort, "
                      "concentration, wisdom); ten in Theravada.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "calm abiding / tranquility meditation",
        "pali": "samatha",
        "sanskrit": "shamatha",
        "tibetan": "zhi gnas",
        "chinese": "止 (zhi)",
        "definition": "Meditation focused on developing mental stability "
                      "and single-pointed concentration. Paired with "
                      "vipashyana for the complete path.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "insight meditation",
        "pali": "vipassana",
        "sanskrit": "vipashyana",
        "tibetan": "lhag mthong",
        "chinese": "觀 (guan)",
        "definition": "Analytical or insight meditation that develops "
                      "direct understanding of impermanence, suffering, "
                      "and non-self (Theravada) or emptiness (Mahayana).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "meditative absorption / jhana",
        "pali": "jhana",
        "sanskrit": "dhyana",
        "tibetan": "bsam gtan",
        "chinese": "禪 (chan)",
        "definition": "States of deep meditative absorption. Four form "
                      "jhanas and four formless attainments in the "
                      "Theravada system.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    # ─── Philosophical Terms ──────────────────────────────────────────────
    {
        "english": "inherent existence / own-nature",
        "pali": "sabhava",
        "sanskrit": "svabhava",
        "tibetan": "rang bzhin",
        "chinese": "自性 (zixing)",
        "definition": "Intrinsic, independent existence. Nagarjuna's "
                      "Madhyamaka demonstrates that all phenomena lack "
                      "svabhava — this IS emptiness.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "conventional truth",
        "pali": "sammuti sacca",
        "sanskrit": "samvriti satya",
        "tibetan": "kun rdzob bden pa",
        "chinese": "俗諦 (sudi)",
        "definition": "The level of ordinary, everyday reality. Valid "
                      "within its own framework but not ultimately real. "
                      "Part of the two truths doctrine.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "ultimate truth",
        "pali": "paramattha sacca",
        "sanskrit": "paramartha satya",
        "tibetan": "don dam bden pa",
        "chinese": "真諦 (zhendi)",
        "definition": "The level of reality as it truly is — emptiness "
                      "of inherent existence. The two truths are not "
                      "separate realities but two aspects of one reality.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "aggregates / heaps",
        "pali": "khandha",
        "sanskrit": "skandha",
        "tibetan": "'phung po",
        "chinese": "蘊 (yun)",
        "definition": "The five components of psychophysical experience: "
                      "form (rupa), feeling (vedana), perception (sanna), "
                      "formations (sankhara), consciousness (vinnana). "
                      "The self is not found in any of them.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "consciousness",
        "pali": "vinnana",
        "sanskrit": "vijñana",
        "tibetan": "rnam shes",
        "chinese": "識 (shi)",
        "definition": "Awareness of objects through the six sense doors. "
                      "Yogacara posits eight types including the "
                      "storehouse consciousness (alaya-vijñana).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "karma / intentional action",
        "pali": "kamma",
        "sanskrit": "karma",
        "tibetan": "las",
        "chinese": "業 (ye)",
        "definition": "Intentional actions of body, speech, and mind "
                      "that produce results (vipaka). The Buddha defined "
                      "karma as cetana (intention).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "buddha-nature / tathagata-essence",
        "pali": "",
        "sanskrit": "tathagatagarbha",
        "tibetan": "de gshegs snying po",
        "chinese": "佛性 (foxing)",
        "definition": "The innate potential for Buddhahood present in all "
                      "sentient beings. A Mahayana concept not accepted "
                      "in Theravada. Interpreted variously across schools.",
        "traditions": ["Mahayana", "Vajrayana"],
    },
    {
        "english": "storehouse consciousness",
        "pali": "",
        "sanskrit": "alaya-vijñana",
        "tibetan": "kun gzhi rnam shes",
        "chinese": "阿賴耶識 (alaiyeshi)",
        "definition": "The eighth consciousness in Yogacara philosophy. "
                      "Stores karmic seeds (bija) and serves as the basis "
                      "for the other seven consciousnesses.",
        "traditions": ["Mahayana", "Vajrayana"],
    },
    # ─── Vajrayana-Specific ───────────────────────────────────────────────
    {
        "english": "great perfection",
        "pali": "",
        "sanskrit": "mahasandhi",
        "tibetan": "rdzogs chen",
        "chinese": "大圓滿 (dayuanman)",
        "definition": "The highest teaching of the Nyingma school. "
                      "Direct recognition of the nature of mind as "
                      "primordially pure (ka dag) and spontaneously "
                      "present (lhun grub).",
        "traditions": ["Vajrayana"],
    },
    {
        "english": "great seal",
        "pali": "",
        "sanskrit": "mahamudra",
        "tibetan": "phyag rgya chen po",
        "chinese": "大手印 (dashouyin)",
        "definition": "The central meditation practice of the Kagyu school. "
                      "Direct recognition of the nature of mind. Shares "
                      "the same ultimate view as Dzogchen.",
        "traditions": ["Vajrayana"],
    },
    {
        "english": "mandala",
        "pali": "",
        "sanskrit": "mandala",
        "tibetan": "dkyil 'khor",
        "chinese": "曼荼羅 (mantuoluo)",
        "definition": "Sacred geometric diagram representing a deity's "
                      "pure realm. Used in visualization practices and "
                      "initiation rituals.",
        "traditions": ["Vajrayana"],
    },
    {
        "english": "spiritual teacher",
        "pali": "",
        "sanskrit": "guru",
        "tibetan": "bla ma (lama)",
        "chinese": "上師 (shangshi)",
        "definition": "The qualified teacher who transmits teachings and "
                      "empowerments. The guru-disciple relationship is "
                      "central to Vajrayana practice.",
        "traditions": ["Vajrayana"],
    },
    {
        "english": "skillful means",
        "pali": "",
        "sanskrit": "upaya",
        "tibetan": "thabs",
        "chinese": "方便 (fangbian)",
        "definition": "The ability to adapt teachings to the capacity "
                      "of the student. Paired with wisdom (prajna) as "
                      "the two wings of the Mahayana path.",
        "traditions": ["Mahayana", "Vajrayana"],
    },
    # ─── Textual/Scholarly Terms ──────────────────────────────────────────
    {
        "english": "discourse / teaching",
        "pali": "sutta",
        "sanskrit": "sutra",
        "tibetan": "mdo",
        "chinese": "經 (jing)",
        "definition": "A discourse attributed to the Buddha or an "
                      "authorized disciple. The Sutta/Sutra Pitaka is "
                      "one of the three baskets (Tripitaka).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "monastic discipline",
        "pali": "vinaya",
        "sanskrit": "vinaya",
        "tibetan": "'dul ba",
        "chinese": "律 (lu)",
        "definition": "The rules governing monastic life. The Vinaya "
                      "Pitaka is one of the three baskets of the canon.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "higher teaching / phenomenology",
        "pali": "abhidhamma",
        "sanskrit": "abhidharma",
        "tibetan": "mngon pa",
        "chinese": "論 (lun)",
        "definition": "Systematic analysis of mind, mental factors, "
                      "and phenomena. The Theravada Abhidhamma and "
                      "Sarvastivada Abhidharma differ significantly.",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "commentary",
        "pali": "atthakatha",
        "sanskrit": "bhashya",
        "tibetan": "'grel pa",
        "chinese": "疏 (shu)",
        "definition": "Scholarly commentary on canonical texts. Major "
                      "commentators include Buddhaghosa (Theravada), "
                      "Candrakirti (Madhyamaka), Sthiramati (Yogacara).",
        "traditions": ["Theravada", "Mahayana", "Vajrayana"],
    },
    {
        "english": "treatise / philosophical text",
        "pali": "",
        "sanskrit": "shastra",
        "tibetan": "bstan bcos",
        "chinese": "論 (lun)",
        "definition": "An independent philosophical treatise by a "
                      "qualified scholar, as distinct from commentary "
                      "on a sutra.",
        "traditions": ["Mahayana", "Vajrayana"],
    },
]


def search_glossary(query: str, tradition: Optional[str] = None) -> List[dict]:
    """
    Search the glossary for matching terms.

    Searches across all language fields and definitions.
    Optionally filter by tradition.

    Returns list of matching entries.
    """
    query_lower = query.lower()
    results = []

    for entry in GLOSSARY:
        # Search all text fields
        searchable = " ".join([
            entry.get("english", ""),
            entry.get("pali", ""),
            entry.get("sanskrit", ""),
            entry.get("tibetan", ""),
            entry.get("chinese", ""),
            entry.get("definition", ""),
        ]).lower()

        if query_lower in searchable:
            if tradition and tradition not in entry.get("traditions", []):
                continue
            results.append(entry)

    return results


def format_entry(entry: dict) -> str:
    """Format a glossary entry for display."""
    lines = []
    lines.append(f"  {entry['english'].upper()}")

    terms = []
    if entry.get("pali"):
        terms.append(f"Pali: {entry['pali']}")
    if entry.get("sanskrit"):
        terms.append(f"Skt: {entry['sanskrit']}")
    if entry.get("tibetan"):
        terms.append(f"Tib: {entry['tibetan']}")
    if entry.get("chinese"):
        terms.append(f"Ch: {entry['chinese']}")
    if terms:
        lines.append(f"  {' | '.join(terms)}")

    lines.append(f"  {entry['definition']}")
    lines.append(f"  Traditions: {', '.join(entry['traditions'])}")
    return "\n".join(lines)


def format_all_terms() -> str:
    """List all glossary terms in a compact format."""
    lines = []
    for entry in GLOSSARY:
        terms = []
        if entry.get("pali"):
            terms.append(entry["pali"])
        if entry.get("sanskrit") and entry["sanskrit"] != entry.get("pali"):
            terms.append(entry["sanskrit"])
        term_str = " / ".join(terms) if terms else ""
        lines.append(f"  {entry['english']}: {term_str}")
    return "\n".join(lines)
