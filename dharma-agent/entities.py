"""
Known-entities database for Buddhist text verification.

A curated database of verified Buddhist entities used to catch potential
fabrications in generated content. Entities include:
  - Sutta/sutra names
  - Teacher/author names
  - School/tradition names
  - Canonical collection names

The database is populated from a built-in seed + metadata from ingested texts.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ─── Built-in seed data ─────────────────────────────────────────────────────
# These are verified entities that we know exist, even before ingesting texts.

KNOWN_TEXTS = {
    # Pali Canon / Theravada
    "Dhammapada", "Sutta Nipata", "Digha Nikaya", "Majjhima Nikaya",
    "Samyutta Nikaya", "Anguttara Nikaya", "Khuddaka Nikaya",
    "Udana", "Itivuttaka", "Theragatha", "Therigatha", "Jataka",
    "Visuddhimagga", "Milindapanha", "Abhidhamma Pitaka",
    "Vinaya Pitaka", "Patisambhidamagga", "Petavatthu", "Vimanavatthu",
    "Khuddakapatha", "Niddesa", "Patthana", "Dhammasangani",
    "Vibhanga", "Dhatukatha", "Puggalapannatti", "Kathavatthu",
    "Yamaka", "Nettipakarana", "Atthasalini",
    # Common Pali sutta references
    "MN 1", "MN 2", "MN 10", "MN 22", "MN 26", "MN 28", "MN 38",
    "MN 43", "MN 44", "MN 63", "MN 72", "MN 109", "MN 117", "MN 118",
    "MN 131", "MN 140", "MN 141",
    "DN 1", "DN 2", "DN 9", "DN 15", "DN 16", "DN 22",
    "SN 12.2", "SN 22.59", "SN 35.28", "SN 45.8", "SN 56.11",
    "AN 3.65", "AN 4.170", "AN 10.60",
    # Mahayana sutras
    "Heart Sutra", "Diamond Sutra", "Lotus Sutra",
    "Avatamsaka Sutra", "Vimalakirti Sutra", "Lankavatara Sutra",
    "Samdhinirmocana Sutra", "Tathagatagarbha Sutra",
    "Surangama Sutra", "Sukhavativyuha Sutra",
    "Prajnaparamita", "Astasahasrika Prajnaparamita",
    "Pancavimsatisahasrika Prajnaparamita",
    # Indian philosophical texts
    "Mulamadhyamakakarika", "Vigrahavyavartani", "Yuktisastika",
    "Vaidalyaprakarana", "Sunyatasaptati",
    "Catuhsataka", "Madhyamakavatara", "Prasannapada",
    "Bodhicaryavatara", "Siksasamuccaya",
    "Abhidharmasamuccaya", "Abhidharmakosa", "Trimsika",
    "Vimsatika", "Madhyantavibhaga", "Mahayanasutralamkara",
    "Uttaratantra", "Ratnagotravibhaga",
    "Pramanavarttika", "Pramansamuccaya", "Nyayabindu",
    "Hetubindu", "Nyayamukha",
    # Tibetan texts
    "Lam Rim Chen Mo", "Lamrim Chenmo",
    "Kunzang Lame Zhalung", "Words of My Perfect Teacher",
    "Treasury of Precious Qualities", "Guhyagarbha Tantra",
    "Kalachakra Tantra", "Hevajra Tantra", "Guhyasamaja Tantra",
    "Chakrasamvara Tantra",
    # Chinese/Japanese texts
    "Platform Sutra", "Shobogenzo", "Mumonkan",
    "Blue Cliff Record", "Biyan Lu",
}

KNOWN_TEACHERS = {
    # Historical Buddha
    "Buddha", "Shakyamuni", "Siddhartha Gautama", "Gotama",
    # Early Buddhist
    "Sariputta", "Moggallana", "Ananda", "Mahakassapa",
    "Mahapajapati", "Khema", "Uppalavanna",
    # Theravada commentators
    "Buddhaghosa", "Dhammapala", "Buddhadatta",
    "Anuruddha", "Ledi Sayadaw", "Mahasi Sayadaw",
    # Indian Mahayana
    "Nagarjuna", "Aryadeva", "Buddhapalita", "Bhavaviveka",
    "Candrakirti", "Chandrakirti", "Santideva", "Shantideva",
    "Asanga", "Vasubandhu", "Dignaga", "Dharmakirti",
    "Sthiramati", "Dharmapala", "Haribhadra",
    "Santaraksita", "Kamalasila", "Atisha",
    # Tibetan
    "Tsongkhapa", "Longchenpa", "Mipham", "Sakya Pandita",
    "Milarepa", "Marpa", "Gampopa", "Padmasambhava",
    "Patrul Rinpoche", "Jamgon Kongtrul", "Gorampa",
    "Jigme Lingpa", "Dolpopa", "Buton Rinchen Drub",
    "Khedrup Je", "Gyaltsab Je",
    "Dalai Lama", "Karmapa", "Panchen Lama",
    # Chinese/Japanese/Korean
    "Zhiyi", "Fazang", "Xuanzang", "Kumarajiva",
    "Bodhidharma", "Huineng", "Linji", "Dogen",
    "Shinran", "Honen", "Nichiren", "Kukai",
    "Wonhyo", "Chinul",
    # Modern scholars
    "Thanissaro Bhikkhu", "Bhikkhu Bodhi", "Bhikkhu Sujato",
    "Bhikkhu Analayo", "Bhikkhu Nanamoli",
    "Walpola Rahula", "Nyanaponika Thera",
    "T.R.V. Murti", "Edward Conze", "Richard Robinson",
    "Paul Williams", "Rupert Gethin", "Jay Garfield",
    "Mark Siderits", "Jan Westerhoff", "Tom Tillemans",
    "Dan Lusthaus", "Karl Brunnholzl",
    "Robert Thurman", "Jeffrey Hopkins", "Elizabeth Napper",
    "Georges Dreyfus", "Matthew Kapstein",
}

KNOWN_SCHOOLS = {
    # Major traditions
    "Theravada", "Mahayana", "Vajrayana",
    # Indian schools
    "Madhyamaka", "Yogacara", "Cittamatra",
    "Svatantrika", "Prasangika",
    "Sarvastivada", "Sautrantika", "Vaibhasika",
    "Mahasamghika", "Pudgalavada",
    # Tibetan schools
    "Gelug", "Kagyu", "Nyingma", "Sakya", "Jonang", "Bon",
    "Rime",
    # East Asian
    "Chan", "Zen", "Soto", "Rinzai",
    "Pure Land", "Jodo Shinshu", "Jodo Shu",
    "Tiantai", "Tendai", "Huayan", "Kegon",
    "Shingon", "Nichiren",
    # Southeast Asian
    "Thai Forest Tradition", "Burmese Vipassana",
}

KNOWN_COLLECTIONS = {
    "Pali Canon", "Tipitaka", "Tripitaka",
    "Sutta Pitaka", "Vinaya Pitaka", "Abhidhamma Pitaka",
    "Khuddaka Nikaya",
    "Chinese Buddhist Canon", "Taisho Tripitaka",
    "Tibetan Canon", "Kangyur", "Tengyur",
    "BDK English Tripitaka",
}


class EntityDatabase:
    """Database of verified Buddhist entities for hallucination detection."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "entities_db.json"
        )
        self.texts: Set[str] = set(KNOWN_TEXTS)
        self.teachers: Set[str] = set(KNOWN_TEACHERS)
        self.schools: Set[str] = set(KNOWN_SCHOOLS)
        self.collections: Set[str] = set(KNOWN_COLLECTIONS)

        # Load any additional entities from disk
        self._load()

    def _load(self):
        """Load additional entities from the JSON database."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.texts.update(data.get("texts", []))
                self.teachers.update(data.get("teachers", []))
                self.schools.update(data.get("schools", []))
                self.collections.update(data.get("collections", []))
            except Exception:
                pass

    def save(self):
        """Save the current database to disk."""
        data = {
            "texts": sorted(self.texts - KNOWN_TEXTS),  # Only save additions
            "teachers": sorted(self.teachers - KNOWN_TEACHERS),
            "schools": sorted(self.schools - KNOWN_SCHOOLS),
            "collections": sorted(self.collections - KNOWN_COLLECTIONS),
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_from_rag_metadata(self, metadatas: List[dict]):
        """Populate the database from RAG chunk metadata."""
        added = 0
        for meta in metadatas:
            text_id = meta.get("text_id", "")
            if text_id and text_id not in self.texts:
                self.texts.add(text_id)
                added += 1

            translator = meta.get("translator", "")
            if translator and translator not in self.teachers:
                self.teachers.add(translator)
                added += 1

            collection = meta.get("canonical_collection", "")
            if collection and collection not in self.collections:
                self.collections.add(collection)
                added += 1

        if added > 0:
            self.save()
        return added

    def verify_entity(self, name: str) -> Tuple[str, float]:
        """
        Check if an entity name is known.

        Returns (entity_type, confidence):
          - entity_type: "text", "teacher", "school", "collection", or "unknown"
          - confidence: 1.0 for exact match, 0.5-0.9 for fuzzy, 0.0 for unknown
        """
        name_clean = name.strip()

        # Exact matches
        if name_clean in self.texts:
            return "text", 1.0
        if name_clean in self.teachers:
            return "teacher", 1.0
        if name_clean in self.schools:
            return "school", 1.0
        if name_clean in self.collections:
            return "collection", 1.0

        # Case-insensitive match
        name_lower = name_clean.lower()
        for s in self.texts:
            if s.lower() == name_lower:
                return "text", 0.95
        for s in self.teachers:
            if s.lower() == name_lower:
                return "teacher", 0.95
        for s in self.schools:
            if s.lower() == name_lower:
                return "school", 0.95

        # Partial/fuzzy match — name is contained in a known entity or vice versa
        for s in self.texts:
            if name_lower in s.lower() or s.lower() in name_lower:
                return "text", 0.7
        for s in self.teachers:
            if name_lower in s.lower() or s.lower() in name_lower:
                return "teacher", 0.7

        return "unknown", 0.0

    @property
    def total_count(self) -> int:
        return len(self.texts) + len(self.teachers) + len(self.schools) + len(self.collections)
