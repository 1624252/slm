"""Golden set — a small, held-out, human-labeled reference set for quality regression.

A golden set is the trusted subset you run to confirm the agent still behaves: each item pairs an
**input** (a scenario: language, level, theme, TARGET/KNOWN words) with a **reference output** (a
hand-authored, spec-passing story) and **metadata** (tone, keywords, target tier, and the
deterministic validator metrics). It is distinct from the training seed (`datagen/seed.py`), so
scoring a model on it never leaks training data — the PRD's "small human-labeled gold set reserved
for eval" (PRD §7, §14.1).

Every reference story is validated on build; only spec-passing items are written. Run:

    python -m islm.eval.golden --out evals/golden        # build golden.jsonl + report
    python -m islm.eval.golden --out evals/golden --stats  # just print the breakdown

The output `golden.jsonl` shares the training record schema (id/language/level/theme/target_words/
messages/metadata) so the eval harness and loaders read it unchanged; `metadata` additionally
carries `tone`, `keywords`, `target_tier`, coverage tags (`category`/`subcategory`/`difficulty`),
and `source: golden-authored`. Currently 55 items (en 39 / zh 8 / ja 8). See docs/GOLDEN_SET.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..config import EVALS_DIR
from ..datagen.generate import Example
from ..datagen.scenarios import Scenario
from ..datagen.seed import _compact_known
from ..validators import validate_story
from ..vocab.lemmatize import get_analyzer
from ..vocab.wordlists import VOCAB_DIR, Vocabulary

# Gold items: language -> list of (targets, tone, keywords, story). Stories are NEW (not in the
# training seed) and use simple, common vocabulary so the target is the only new word.
GOLD: dict[str, list[tuple[list[str], str, list[str], str]]] = {
    "en": [
        (["beacon"], "hopeful", ["sea", "night", "help"],
         "It is night by the sea. The boat is lost in the dark. "
         "Then the man sees a beacon on the hill. The beacon is a bright light. "
         "The beacon shows the way home. The boat follows the beacon to the shore. "
         "Now the man loves the little beacon."),
        (["harbor"], "calm", ["boat", "sea", "home"],
         "The little boat is tired. It sails all day on the big sea. "
         "At night it comes to a harbor. The harbor is calm and safe. "
         "Many boats sleep in the harbor. The boat is happy in the quiet harbor. "
         "Every night it comes back to the harbor."),
        (["meadow"], "gentle", ["grass", "flower", "walk"],
         "The girl walks up the hill. At the top she finds a meadow. "
         "The meadow is green and full of flowers. A little bird sings in the meadow. "
         "She sits in the soft meadow all day. The meadow is her secret place. "
         "She comes to the meadow every morning."),
        (["comet"], "curious", ["sky", "night", "light"],
         "The boy looks up at the night sky. He sees a comet fly high above. "
         "The comet has a long bright tail. The comet is fast and white. "
         "He wants to follow the comet. But the comet is far away in the sky. "
         "He waits for the comet to come again."),
        (["glacier"], "cold", ["ice", "mountain", "blue"],
         "High on the mountain is a glacier. The glacier is old ice, big and blue. "
         "The cold wind comes off the glacier. A little fox walks near the glacier. "
         "The glacier is quiet and white. The girl looks at the great glacier. "
         "She never forgets the blue glacier."),
        (["sculptor"], "proud", ["stone", "art", "hand"],
         "The man works with stone all day. He is a sculptor. "
         "The sculptor makes a cat from cold stone. His hands are strong and slow. "
         "The sculptor smiles at his work. People come to see what the sculptor makes. "
         "Now the sculptor makes a bird too."),
        (["orchard"], "warm", ["tree", "apple", "garden"],
         "Behind the house is an orchard. The orchard is full of apple trees. "
         "In the orchard the apples are red and sweet. The girl and her dog play in the orchard. "
         "They eat apples under the orchard trees. The orchard is best in the warm sun. "
         "They love the little orchard."),
        (["lantern"], "cozy", ["light", "night", "warm"],
         "It is dark in the old house. The woman lights a lantern. "
         "The lantern gives a warm yellow light. She carries the lantern up the stairs. "
         "The lantern makes the room feel safe. Her cat sleeps by the lantern. "
         "The little lantern burns all night."),
        (["compass"], "adventurous", ["direction", "walk", "help"],
         "The boy walks into the big forest. He has a small compass in his hand. "
         "The compass shows him the way. When he is lost, he looks at the compass. "
         "The compass always points home. He trusts the little compass. "
         "The compass helps him find the way back."),
        (["canyon"], "awed", ["rock", "big", "echo"],
         "They walk to the edge of a canyon. The canyon is deep and wide. "
         "Red rock falls down into the canyon. The boy calls out, and the canyon calls back. "
         "The canyon is very old. They sit and look at the great canyon. "
         "The canyon makes them feel small."),
        (["cottage"], "homey", ["house", "small", "warm"],
         "In the woods is a little cottage. The cottage is small and warm. "
         "An old woman lives in the cottage. The cottage has one door and two windows. "
         "Smoke comes from the cottage roof. The girl knocks on the cottage door. "
         "Inside the cottage it is warm and kind."),
        (["waterfall"], "lively", ["water", "loud", "river"],
         "The river runs fast down the hill. Then it becomes a waterfall. "
         "The waterfall is loud and white. Water falls from the high waterfall. "
         "A bird flies through the waterfall. The children laugh by the waterfall. "
         "They love the cold, loud waterfall."),
        (["telescope"], "wondering", ["star", "night", "look"],
         "The girl has a telescope. At night she looks through the telescope. "
         "The telescope makes the moon big and near. She sees stars in the telescope. "
         "The telescope shows her a new world. Her brother wants the telescope too. "
         "They share the little telescope every night."),
        (["vigilant"], "watchful", ["guard", "careful", "night"],
         "The old dog guards the house. At night he is very vigilant. "
         "A vigilant dog hears every small sound. He stays vigilant by the door. "
         "Nothing gets past the vigilant dog. The family sleeps because the dog is vigilant. "
         "They love their brave, vigilant friend."),
        (["ravenous"], "funny", ["hungry", "eat", "food"],
         "The little cat did not eat all day. Now the cat is ravenous. "
         "A ravenous cat wants food fast. She looks at the fish with ravenous eyes. "
         "The ravenous cat eats it all. After the food, she is not ravenous. "
         "A happy cat is never ravenous."),
        (["luminous"], "dreamy", ["light", "glow", "night"],
         "At night the little fish swims deep. Its body is luminous. "
         "A luminous fish glows soft and blue. The luminous light fills the dark water. "
         "Other fish follow the luminous glow. The sea is beautiful with the luminous fish. "
         "She shines, small and luminous."),
        (["nocturnal"], "quiet", ["night", "sleep", "animal"],
         "The owl sleeps all day. The owl is nocturnal. "
         "A nocturnal bird wakes at night. It hunts in the dark, quiet and nocturnal. "
         "Many forest animals are nocturnal. When the sun goes down, the nocturnal owl flies. "
         "The nocturnal world is full of soft sounds."),
        (["tranquil"], "peaceful", ["calm", "water", "quiet"],
         "The small lake is very still. The water is tranquil. "
         "A tranquil lake is like glass. The girl sits by the tranquil water. "
         "Everything is tranquil in the morning. Birds rest on the tranquil lake. "
         "She feels calm by the tranquil water."),
        (["meander"], "slow", ["river", "walk", "bend"],
         "The little river does not run fast. It likes to meander. "
         "The river meanders through the green fields. It meanders left and right. "
         "The boy walks and watches the river meander. Slow water loves to meander. "
         "They meander home together."),
        (["furtive"], "sneaky", ["quiet", "hide", "cat"],
         "The cat wants the fish on the table. She takes a furtive step. "
         "A furtive cat moves without a sound. With furtive eyes she looks around. "
         "The furtive cat hides behind the chair. No one sees the furtive little cat. "
         "Then, quick and furtive, she is gone."),
        (["ominous"], "tense", ["dark", "cloud", "storm"],
         "The sky turns dark in the afternoon. A big ominous cloud comes near. "
         "The ominous cloud is black and low. The wind feels ominous and cold. "
         "The birds go quiet before the ominous storm. Everyone runs home from the ominous sky. "
         "Then the ominous cloud opens with rain."),
        (["voracious"], "hungry", ["eat", "read", "much"],
         "The boy loves books more than food. He is a voracious reader. "
         "A voracious reader finishes a book in one day. He reads with voracious eyes. "
         "His voracious mind wants more and more. The library feeds his voracious love of books. "
         "He is happy and voracious."),
        (["fossil"], "curious", ["stone", "old", "find"],
         "The girl digs in the dry ground. She finds a fossil in the rock. "
         "The fossil is a very old bone. A fossil can be older than the hills. "
         "She cleans the fossil with care. The fossil tells a story from long ago. "
         "She keeps the little fossil safe."),
        (["hedgehog"], "sweet", ["small", "animal", "garden"],
         "In the garden lives a small hedgehog. The hedgehog is round and brown. "
         "The hedgehog has soft feet and sharp hair. At night the hedgehog looks for food. "
         "The girl gives the hedgehog some water. The happy hedgehog comes back every night. "
         "She loves the little hedgehog."),
        (["shimmer"], "magical", ["light", "water", "glow"],
         "The sun touches the sea in the morning. The water begins to shimmer. "
         "A shimmer of gold is on every wave. The light and water shimmer together. "
         "The girl watches the bright shimmer. Even the sand seems to shimmer. "
         "She loves the morning shimmer on the sea."),
        (["serpentine"], "winding", ["road", "hill", "long"],
         "The road up the hill is not straight. It is serpentine. "
         "A serpentine road bends like a snake. The car climbs the serpentine hill slowly. "
         "The serpentine path goes left, then right. From the top the serpentine road looks small. "
         "They love the long serpentine drive."),
        (["forlorn"], "sad", ["lost", "alone", "rain"],
         "The little dog waits in the rain. He looks forlorn. "
         "A forlorn dog has lost his friend. His forlorn eyes watch the road. "
         "The forlorn dog does not move. Then a girl finds the forlorn dog. "
         "Now he is not forlorn anymore."),
        (["blizzard"], "wild", ["snow", "cold", "wind"],
         "In winter the snow comes fast. Soon it is a blizzard. "
         "A blizzard is white and cold and loud. The blizzard hides the whole town. "
         "The family stays warm inside the blizzard. No one walks in the blizzard. "
         "In the morning the blizzard is gone."),
        (["cellar"], "mysterious", ["under", "dark", "old"],
         "Under the old house is a cellar. The cellar is dark and cool. "
         "The girl goes down into the cellar. In the cellar she finds old boxes. "
         "The cellar smells of stone and dust. A little mouse lives in the cellar. "
         "She is not afraid of the quiet cellar."),
        (["riddle"], "playful", ["puzzle", "question", "think"],
         "The old man tells the children a riddle. A riddle is a hard question. "
         "They think about the riddle all day. The riddle has a clever answer. "
         "At last the girl solves the riddle. Everyone laughs at the funny riddle. "
         "They want another good riddle."),
        (["anchor"], "steady", ["boat", "sea", "heavy"],
         "The boat stops in the calm sea. The man drops the anchor. "
         "The anchor is heavy and holds the boat. The anchor keeps the boat safe. "
         "In the storm, the anchor does not move. The boat trusts the strong anchor. "
         "At night they pull up the anchor and go home."),
        (["blaze"], "dramatic", ["fire", "bright", "warm"],
         "It is cold, so they make a fire. Soon there is a big blaze. "
         "The blaze is bright and warm. The blaze lights up the whole camp. "
         "Everyone sits close to the blaze. The blaze keeps the animals away. "
         "They watch the happy blaze until night."),
        (["pebble"], "small", ["stone", "river", "little"],
         "By the river the girl finds a pebble. The pebble is small and smooth. "
         "The gray pebble sits in her hand. She throws the pebble into the water. "
         "The pebble makes a little sound. She finds one more pretty pebble. "
         "She keeps the small pebble in her pocket."),
        (["gorge"], "grand", ["rock", "deep", "river"],
         "The river runs through a deep gorge. The gorge has high rock walls. "
         "Down in the gorge the water is fast. A bird flies over the gorge. "
         "The boy looks into the wide gorge. The gorge is dark and cool. "
         "They walk along the great gorge all day."),
        (["barn"], "rustic", ["farm", "animal", "big"],
         "On the farm there is a big barn. The barn is red and old. "
         "Animals sleep in the warm barn. The horse lives in the barn at night. "
         "Hay fills the top of the barn. The children play in the barn. "
         "The barn is the best place on the farm."),
        # --- ambiguous: two new words (pace both, one per sentence) ---
        (["meadow", "beacon"], "hopeful", ["walk", "light", "hill"],
         "The girl walks up the green hill. At the top she finds a meadow. "
         "The meadow is soft and full of flowers. She rests in the quiet meadow. "
         "At night she sees a beacon far away. The beacon is a small bright light. "
         "The beacon shows the way home. She leaves the meadow and walks to the beacon."),
        (["lantern", "cellar"], "mysterious", ["dark", "under", "night"],
         "The old house is dark at night. The boy takes a lantern. "
         "The lantern gives a warm light. He goes down into the cellar. "
         "The cellar is cold and full of boxes. With the lantern he sees the whole cellar. "
         "The lantern keeps the cellar from being scary. He likes the quiet cellar now."),
        (["harbor", "voyage"], "adventurous", ["boat", "sea", "long"],
         "The little boat sleeps in the harbor. The harbor is calm and safe. "
         "One morning the boat leaves the harbor. It begins a long voyage. "
         "The voyage takes the boat far over the sea. After the voyage, the boat is tired. "
         "It comes back to the harbor. The harbor is home after every voyage."),
        # --- edge: three new words to introduce and pace ---
        (["orchard", "ravenous", "fossil"], "curious", ["find", "eat", "old"],
         "Behind the house is an orchard. The orchard is full of apple trees. "
         "The boy is ravenous, so he eats a sweet apple. A ravenous dog wants an apple too. "
         "Under a tree they find a fossil. The fossil is a very old bone. "
         "The ravenous dog forgets the fossil. The boy keeps the old fossil from the orchard."),
    ],
    "zh": [
        (["灯塔"], "hopeful", ["sea", "light"],
         "晚上，海很黑。小船看不见路。这时，小船看见灯塔。"
         "灯塔的光很亮。灯塔告诉小船回家的路。小船跟着灯塔回家。现在小船很喜欢灯塔。"),
        (["彩虹"], "joyful", ["rain", "sky", "color"],
         "下雨了，然后太阳出来了。天上有一道彩虹。彩虹很美，有很多颜色。"
         "孩子们看着彩虹笑。彩虹在山的上面。大家都喜欢这道彩虹。"),
        (["蝴蝶"], "gentle", ["flower", "garden"],
         "花园里有很多花。一只蝴蝶飞过来。蝴蝶很小，很漂亮。"
         "蝴蝶在花上面飞。女孩看着蝴蝶笑。蝴蝶也喜欢这个花园。"),
        (["火山"], "awed", ["mountain", "fire", "big"],
         "很远的地方有一座火山。火山很大，很高。火山里面有火。"
         "人们看着火山，不说话。火山很老了。这座火山很有名。"),
        (["钢琴"], "warm", ["music", "play", "home"],
         "家里有一台钢琴。女孩每天弹钢琴。钢琴的声音很美。"
         "妈妈喜欢听钢琴。晚上，钢琴的声音很轻。这台钢琴是家里的朋友。"),
        (["沙漠"], "vast", ["sand", "hot", "dry"],
         "很远的地方有一片沙漠。沙漠又大又热。沙漠里没有水。"
         "沙漠的沙子是黄色的。晚上，沙漠很冷。人们走过大大的沙漠。"),
        (["镜子"], "curious", ["look", "see", "room"],
         "房间里有一面镜子。女孩看着镜子。镜子里有一个女孩。"
         "她笑，镜子里的人也笑。镜子很干净，很亮。她喜欢这面镜子。"),
        (["帐篷"], "cozy", ["camp", "night", "sleep"],
         "他们去山里玩。晚上，他们住在帐篷里。帐篷不大，可是很暖和。"
         "帐篷外面有星星。孩子们在帐篷里睡觉。这个帐篷是他们的小家。"),
    ],
    "ja": [
        (["灯台"], "hopeful", ["sea", "light"],
         "夜、海は暗いです。船は道が見えません。その時、船は灯台を見ます。"
         "灯台の光は明るいです。灯台は帰る道を教えます。船は灯台の光で帰ります。今、船は灯台が好きです。"),
        (["虹"], "joyful", ["rain", "sky"],
         "雨が降って、それから太陽が出ました。空に虹があります。虹はとても綺麗です。"
         "子供たちは虹を見て笑います。虹は山の上にあります。みんなこの虹が好きです。"),
        (["蝶"], "gentle", ["flower", "garden"],
         "庭にたくさんの花があります。蝶が飛んできます。蝶は小さくて綺麗です。"
         "蝶は花の上を飛びます。女の子は蝶を見て笑います。蝶もこの庭が好きです。"),
        (["火山"], "awed", ["mountain", "fire"],
         "遠い所に火山があります。火山は大きくて高いです。火山の中に火があります。"
         "人々は火山を見て、何も言いません。この火山はとても古いです。この火山は有名です。"),
        (["ピアノ"], "warm", ["music", "home"],
         "家にピアノがあります。女の子は毎日ピアノを弾きます。ピアノの音は綺麗です。"
         "お母さんはピアノを聞くのが好きです。夜、ピアノの音は静かです。このピアノは家族の友達です。"),
        (["砂漠"], "vast", ["sand", "hot"],
         "遠い所に砂漠があります。砂漠は大きくて暑いです。砂漠には水がありません。"
         "砂漠の砂は黄色です。夜、砂漠は寒いです。人々は大きい砂漠を歩きます。"),
        (["鏡"], "curious", ["look", "room"],
         "部屋に鏡があります。女の子は鏡を見ます。鏡の中に女の子がいます。"
         "彼女が笑うと、鏡の中の人も笑います。鏡は綺麗で明るいです。彼女はこの鏡が好きです。"),
        (["星"], "dreamy", ["night", "sky"],
         "夜、空に星がたくさんあります。星はとても綺麗です。子供は星を見ます。"
         "星は小さくて明るいです。一つの星が動きます。子供はこの星が好きです。"),
    ],
}


def load_golden_scenarios(path: Path | str, language: str | None = None) -> list[Scenario]:
    """Load golden records as `Scenario`s so a model can be *run* on the golden inputs.

    The golden set stores each item in the training record schema (its assistant message is the
    reference story). To evaluate a model on the golden set, we reconstruct the input `Scenario`:
    `target_words` is a top-level field; `KNOWN_WORDS`/`Level` are parsed from the user message
    (they round-trip exactly, since `generation_prompt` wrote them). `language` filters if given.
    """
    scenarios: list[Scenario] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if language and rec["language"] != language:
            continue
        user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        known = _field(user, "KNOWN_WORDS")
        scenarios.append(
            Scenario(
                id=rec["id"],
                language=rec["language"],
                level=rec.get("level", "baseline"),
                theme=rec.get("theme", ""),
                target_words=rec["target_words"],
                known=known,
            )
        )
    return scenarios


def _field(text: str, label: str) -> list[str]:
    """Parse a comma-separated `LABEL: a, b, c` line from a rendered prompt (KNOWN/TARGET_WORDS)."""
    import re

    m = re.search(rf"^{label}:\s*(.+)$", text, flags=re.MULTILINE)
    return [w.strip() for w in m.group(1).split(",") if w.strip()] if m else []


def _tier_index(language: str) -> dict[str, str]:
    """word -> tier (CEFR/HSK/JLPT/exam) from the committed curated + exam lists, for metadata."""
    index: dict[str, str] = {}
    for name in ("advanced.csv", "exam.csv", "baseline.csv"):
        path = VOCAB_DIR / language / name
        if path.exists():
            index.update(Vocabulary.from_csv(path).levels)
    return index


# Coarse category per tier label, for behavioral-coverage tagging (Layer 2).
_TIER_CATEGORY = {
    "A1": "cefr", "A2": "cefr", "B1": "cefr", "B2": "cefr", "C1": "cefr", "C2": "cefr",
    "GRE": "exam", "SAT": "exam", "ACT": "exam",
}
for _hsk in ("HSK1", "HSK2", "HSK3", "HSK4", "HSK5", "HSK6", "HSK7-9"):
    _TIER_CATEGORY[_hsk] = "hsk"
for _jlpt in ("N1", "N2", "N3", "N4", "N5"):
    _TIER_CATEGORY[_jlpt] = "jlpt"


def _coverage_tags(targets: list[str], tiers: list[str]) -> dict:
    """Behavioral-coverage tags (PDF Layer 2): category, subcategory, difficulty.

    category = the graded scheme (cefr/hsk/jlpt/exam), or 'core' for concrete words not in any
    graded list. subcategory = the exact tier(s). difficulty scales with how many new words the
    story must introduce and pace: 1 target = straightforward, 2 = ambiguous, 3+ = edge.
    """
    cats = sorted({_TIER_CATEGORY.get(t, "core") for t in tiers})
    category = cats[0] if len(cats) == 1 else "mixed"
    difficulty = {1: "straightforward", 2: "ambiguous"}.get(len(targets), "edge")
    return {
        "category": category,
        "subcategory": "+".join(tiers),
        "difficulty": difficulty,
    }


def build_items(languages: list[str]) -> tuple[list[dict], dict]:
    """Validate every gold story; return (kept records, stats). Failing stories are dropped."""
    kept: list[dict] = []
    stats: dict = {}
    for lang in languages:
        analyzer = get_analyzer(lang)
        tiers = _tier_index(lang)
        passing, failures = 0, []
        for i, (targets, tone, keywords, story) in enumerate(GOLD.get(lang, [])):
            known = _compact_known(story, targets, lang, analyzer)
            scenario = Scenario(
                id=f"{lang}-gold-{i:03d}",
                language=lang,
                level="baseline",
                theme=tone,
                target_words=targets,
                known=known,
            )
            report = validate_story(story, set(known), set(targets), analyzer, language=lang)
            if not report.hard_pass:
                failures.append({"id": scenario.id, "reasons": report.failures()})
                continue
            record = Example(scenario, story, report, 0, kept=True).to_record(split="golden")
            target_tiers = [tiers.get(t.lower(), "core") for t in targets]
            record["metadata"]["tone"] = tone
            record["metadata"]["keywords"] = keywords
            record["metadata"]["target_tier"] = target_tiers
            record["metadata"]["source"] = "golden-authored"
            record["metadata"].update(_coverage_tags(targets, target_tiers))
            kept.append(record)
            passing += 1
        stats[lang] = {"authored": len(GOLD.get(lang, [])), "kept": passing, "failures": failures}
    stats["total"] = len(kept)
    return kept, stats


def build(out_dir: Path, languages: list[str]) -> dict:
    kept, stats = build_items(languages)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "golden.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Human-readable report: counts by language, tier, and tone.
    by_tier = Counter(t for r in kept for t in r["metadata"]["target_tier"])
    by_tone = Counter(r["metadata"]["tone"] for r in kept)
    fails = {lang: stats[lang]["failures"] for lang in languages if stats[lang]["failures"]}
    report = {
        "total": len(kept),
        "by_language": {lang: stats[lang]["kept"] for lang in languages},
        "by_target_tier": dict(by_tier),
        "by_tone": dict(by_tone),
        "failures": fails,
    }
    with open(out_dir / "golden_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build the held-out golden reference set.")
    p.add_argument("--out", type=Path, default=EVALS_DIR / "golden")
    p.add_argument("--language", default="all", help="en, zh, ja, or all.")
    p.add_argument("--stats", action="store_true", help="Only print the breakdown, don't write.")
    args = p.parse_args()
    languages = list(GOLD) if args.language == "all" else [args.language]

    if args.stats:
        _, stats = build_items(languages)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    report = build(args.out, languages)
    print(f"golden set: {report['total']} items -> {args.out / 'golden.jsonl'}")
    print(f"  by language: {report['by_language']}")
    print(f"  by target tier: {report['by_target_tier']}")
    for fails in report["failures"].values():
        for fail in fails:
            print(f"  DROPPED {fail['id']}: {fail['reasons']}")


if __name__ == "__main__":
    main()
