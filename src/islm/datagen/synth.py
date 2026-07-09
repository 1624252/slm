"""Programmatic i+1 story generator for building a large dataset (first pass).

Teacher distillation is the quality path but can't reach 100k in a day. This module generates
spec-passing i+1 stories *programmatically*: it composes short stories from large pools of common
words + story-arc templates, introduces one target word (from the graded advanced/exam lists),
and recurs it. Because each story's KNOWN_WORDS is scoped to the words the story actually uses
(minus the target), coverage passes by construction and the target is the only "new" word — so
the deterministic validators certify every kept story as spec-compliant i+1 data.

Diversity comes from: ~7k target words per language × many arcs × randomized characters / settings
/ adjectives / verbs / sentence frames. Duplicates and low-diversity stories are removed by the
second pass (`islm.datagen.curate`). Provenance is recorded as `source: synthetic-v1`.

    python -m islm.datagen.synth --n 150000 --language en --out data/generated/synth_en
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..validators import validate_story
from ..vocab.lemmatize import get_analyzer
from .generate import Example
from .scenarios import Scenario

# ------------------------------------------------------------------ English building blocks ---
# All frame words are common English; each story's known set = exactly the words it uses (minus the
# target), so everything is guaranteed in-vocabulary. Targets are POS-typed so each slots into a
# grammatically correct frame (a noun after "a/an", an adjective after "is", a verb after "to").
EN = {
    "char": [
        "cat", "dog", "boy", "girl", "man", "woman", "bird", "mouse", "rabbit", "duck",
        "hen", "bear", "fish", "frog", "child", "friend", "baby", "horse", "goat", "cow", "pig",
        "old man", "old woman", "little girl", "little boy", "kind man", "small dog", "big cat",
    ],
    "place": [
        "park", "garden", "house", "forest", "sea", "hill", "room", "farm", "river", "tree",
        "town", "road", "field", "beach", "cave", "sky", "home", "market", "school", "boat",
    ],
    "adj": [
        "big", "small", "little", "happy", "sad", "cold", "warm", "old", "new", "red", "blue",
        "green", "quick", "slow", "kind", "good", "funny", "quiet", "soft", "bright", "wet",
        "dark", "hot", "pretty", "clean", "brave", "sweet", "tall", "long",
    ],
    "move": ["runs", "walks", "jumps", "plays", "sits", "sings", "goes", "climbs"],
}


def _a(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _en_noun_arcs(c, t, p, a1, a2, mv, c2):
    """Target is a concrete noun -> 'a/an {t}', 'the {t}'. Introduced once, recurs >=3x."""
    at = _a(t)
    return [
        [f"The {c} {mv} in the {p}.", f"It is a {a1} day.", f"Then the {c} sees {at} {t}.",
         f"The {t} is {a2} and {a1}.", f"The {c} looks at the {t}.",
         f"The {t} is a good friend.", f"Now the {c} and the {t} play in the {p}."],
        [f"The {c} has a {c2}.", f"They live near the {p}.", f"One day they find {at} {t}.",
         f"The {t} is {a1}.", f"The {c} keeps the {t} safe.", f"The {c2} wants the {t} too.",
         f"They love the little {t}."],
        [f"The {c} is {a1}.", f"The {c} looks in the {p}.", f"The {c} needs {at} {t}.",
         f"At last the {c} finds the {t}.", f"The {t} is {a2}.", f"The {t} helps the {c}.",
         f"Now the {c} is happy with the {t}."],
        [f"It is night in the {p}.", f"The {c} can not sleep.",
         f"The {c} sees {at} {t} in the dark.",
         f"The {t} is {a1} and {a2}.", f"The {c} is not afraid of the {t}.",
         f"The {t} sits by the {c}.", f"They sleep by the {t}."],
        [f"The {c} {mv} up the {p}.", "The way is long.", f"On the way the {c} sees {at} {t}.",
         f"The {t} is {a1}.", f"The {c} takes the {t} home.", f"The {t} is now with the {c}.",
         f"The {c} likes the new {t}."],
    ]


def _en_adj_arcs(c, t, p, a1, a2, mv, c2):
    """Target is an adjective -> '{c} is {t}'. Describes the character/scene; recurs >=3x."""
    at = _a(t)  # article agrees with the adjective it precedes ("a brave cat", "an ominous hen")
    return [
        [f"The {c} {mv} in the {p}.", f"The {c} is very {t}.",
         f"{at.capitalize()} {t} {c} is not {a1}.",
         f"The {c2} sees the {t} {c}.", f"The {c2} likes the {t} {c}.",
         f"They play in the {p}.", f"Now both are {t} and {a2}."],
        [f"The {c} has a {c2}.", f"The {c2} is {t} today.",
         f"{at.capitalize()} {t} {c2} is a good friend.",
         f"The {c} helps the {t} {c2}.", "The day is warm.", f"They rest by the {p}.",
         f"The {t} {c2} is happy."],
        [f"It is a {a1} day in the {p}.", f"The {c} feels {t}.", f"To be {t} is good.",
         f"{at.capitalize()} {t} {c} sings.", f"The {c2} is {t} too.", f"They are {t} together.",
         f"The {p} is full of {t} friends."],
    ]


def _en_verb_arcs(c, t, p, a1, a2, mv, c2):
    """Target is a base-form verb -> 'to {t}', 'they {t}' (no conjugation). Recurs >=3x."""
    return [
        [f"The {c} likes to {t}.", f"Every {a1} day the {c} wants to {t}.",
         f"The {c} and the {c2} {t} in the {p}.", f"They {t} all day.",
         f"It is good to {t}.", f"To {t} is fun.", f"Now they {t} together."],
        [f"The {c} is in the {p}.", f"The {c} starts to {t}.", f"The {p} is a good place to {t}.",
         f"The {c2} sees the {c} {t}.", f"Soon they both {t}.", f"They {t} and {t} again.",
         f"To {t} makes them happy."],
    ]


_EN_ARC_BY_POS = {"noun": _en_noun_arcs, "adj": _en_adj_arcs, "verb": _en_verb_arcs}


def _gen_en(rng: random.Random, target: str, pos: str) -> str:
    c, c2 = rng.sample(EN["char"], 2)
    p = rng.choice(EN["place"])
    a1, a2 = rng.sample(EN["adj"], 2)
    mv = rng.choice(EN["move"])
    arcs = _EN_ARC_BY_POS[pos](c, target, p, a1, a2, mv, c2)
    return " ".join(rng.choice(arcs))


# ------------------------------------------------------------------ CJK building blocks --------
# Parameterized frames; {t} = target. Common words only; known set is scoped per story.
ZH = {
    "char": ["小猫", "小狗", "女孩", "男孩", "小鸟", "老人", "朋友", "孩子"],
    "place": ["花园", "森林", "山上", "河边", "家里", "外面", "房间"],
    "adj": ["大", "小", "红", "蓝", "冷", "暖和", "旧", "新", "快", "高兴"],
    "arcs": [
        ["{c}在{p}。", "它看见一个{t}。", "{t}很{a}。", "{c}很喜欢这个{t}。", "现在{c}每天看{t}。"],
        ["{c}和朋友在{p}。", "他们找到一个{t}。", "这个{t}很{a}。", "{c}把{t}拿回家。",
         "他们都喜欢{t}。"],
        ["晚上，{c}在{p}。", "{c}看见{t}。", "{t}很{a}。", "{c}不怕{t}。", "{t}和{c}是朋友。"],
    ],
}
JA = {
    "char": ["猫", "犬", "女の子", "男の子", "鳥", "おじいさん", "友達", "子供"],
    "place": ["庭", "森", "山", "川", "家", "外", "部屋"],
    "adj": ["大きい", "小さい", "赤い", "青い", "寒い", "暖かい", "古い", "新しい", "速い"],
    "arcs": [
        ["{c}は{p}にいます。", "{t}を見ます。", "{t}は{a}です。", "{c}は{t}が好きです。",
         "今、{c}は毎日{t}を見ます。"],
        ["{c}と友達は{p}にいます。", "{t}を見つけます。", "この{t}は{a}です。",
         "{c}は{t}を家に持って帰ります。", "みんな{t}が好きです。"],
        ["夜、{c}は{p}にいます。", "{c}は{t}を見ます。", "{t}は{a}です。",
         "{c}は{t}が怖くないです。", "{t}と{c}は友達です。"],
    ],
}


def _gen_cjk(rng: random.Random, target: str, pos: str, pack: dict) -> str:
    # CJK targets are nouns only (safest for grammaticality); pos is accepted for a uniform API.
    c = rng.choice(pack["char"])
    p = rng.choice(pack["place"])
    a = rng.choice(pack["adj"])
    arc = rng.choice(pack["arcs"])
    return "".join(s.format(c=c, p=p, a=a, t=target) for s in arc)


_GENERATORS = {
    "en": _gen_en,
    "zh": lambda r, t, pos: _gen_cjk(r, t, pos, ZH),
    "ja": lambda r, t, pos: _gen_cjk(r, t, pos, JA),
}

# POS-typed target pools — sensible, mostly graded (CEFR/HSK/JLPT/exam) to-learn words, each placed
# in a grammatically correct frame for its part of speech. CJK is noun-only for safety.
TARGET_POOLS: dict[str, dict[str, list[str]]] = {
    "en": {
        "noun": [
            "beacon", "lantern", "treasure", "feather", "castle", "dragon", "meadow", "harbor",
            "comet", "glacier", "cottage", "canyon", "waterfall", "telescope", "fossil", "hedgehog",
            "riddle", "anchor", "pebble", "gorge", "barn", "robot", "balloon", "pumpkin", "whistle",
            "ribbon", "candle", "basket", "whisker", "mustache", "ghost", "wizard", "kitten",
            "puzzle", "umbrella", "shadow", "secret", "clue", "mystery", "creature", "thicket",
            "orchard", "scroll", "cellar", "blaze", "torch", "cliff", "cave", "well", "gate",
            "bridge", "tower", "lamp", "clock", "mirror", "moon", "star", "cloud", "flower",
            "apple", "cake", "ball", "kite", "drum", "bell", "key", "box", "cup", "hat", "coat",
            "shoe",
            "map", "ship", "wheel", "nest", "egg", "wing", "tail", "horn", "leaf", "seed", "rock",
            "shell", "wave", "crown", "flag", "coin", "ring", "rope", "net", "cart", "sled", "raft",
            "owl", "otter", "swan", "eagle", "wolf", "deer", "seal", "whale", "dolphin", "camel",
            "snail", "spider", "beetle", "butterfly", "dove", "crow", "lizard", "turtle", "pony",
            "cave", "meadow", "harbor", "island", "valley", "pond", "brook", "trail", "peak",
            "storm", "rainbow", "snowman", "fountain", "statue", "market", "temple", "palace",
        ],
        "adj": [
            "radiant", "weary", "gloomy", "fierce", "timid", "vast", "immense", "gentle", "curious",
            "brave", "silent", "shiny", "enormous", "tiny", "frosty", "somber", "luminous",
            "nocturnal", "tranquil", "ominous", "ravenous", "furtive", "forlorn", "lush", "barren",
            "clumsy", "graceful", "cunning", "fragile", "sturdy", "bold", "shy", "eager", "calm",
            "restless", "cheerful", "grumpy", "clever", "gentle", "swift", "sleepy", "hungry",
            "curious", "playful", "loyal", "proud", "humble", "merry", "weary", "bright",
        ],
        "verb": [
            "wander", "vanish", "soar", "tremble", "shiver", "drift", "glisten", "flicker",
            "crumble", "scatter", "gallop", "pounce", "prowl", "meander", "plunge", "linger",
            "wobble", "tumble", "glide", "creep", "dash", "roam", "leap", "float", "sparkle",
            "rustle", "shimmer", "sway", "whirl", "bounce",
        ],
    },
    "zh": {"noun": [
        "灯塔", "彩虹", "蝴蝶", "火山", "钢琴", "沙漠", "镜子", "帐篷", "森林", "城堡", "宝物",
        "秘密", "影子", "机器人", "月亮", "星星", "花朵", "苹果", "气球", "风筝", "钥匙", "小船",
        "海洋", "山洞", "河流", "桥", "塔", "钟", "灯", "帽子", "小鸟", "小猫", "小狗", "老虎",
        "兔子", "乌龟", "蜘蛛", "青蛙", "鱼", "礼物",
    ]},
    "ja": {"noun": [
        "灯台", "虹", "蝶", "火山", "ピアノ", "砂漠", "鏡", "星", "森", "城", "宝物", "秘密",
        "影", "ロボット", "月", "花", "りんご", "風船", "凧", "鍵", "船", "海", "洞窟", "川",
        "橋", "塔", "時計", "ランプ", "帽子", "小鳥", "猫", "犬", "うさぎ", "亀", "蜘蛛",
        "かえる", "魚", "贈り物", "山", "雪",
    ]},
}


def _compact_known(story: str, target: str, analyzer) -> list[str]:
    """Known set = the story's own content words, minus the target (coverage passes by build)."""
    known = set()
    tl = target.lower()
    for tok in analyzer.analyze(story):
        if tok.is_word and tok.lemma != tl and tok.surface.lower() != tl:
            known.add(tok.lemma)
            known.add(tok.surface.lower())
    return sorted(known)


def generate(language: str, n: int, out_dir: Path, seed: int = 0) -> dict:
    """Generate n spec-passing synthetic stories; write train/val/test (80/10/10). Returns stats."""
    analyzer = get_analyzer(language)
    # POS-typed targets so each is placed in a grammatical frame; (pos, target) pairs.
    pools = TARGET_POOLS[language]
    typed = [(pos, w) for pos, words in pools.items() for w in words]
    gen = _GENERATORS[language]
    rng = random.Random(seed)

    records, kept, failed = [], 0, 0
    attempts = 0
    while kept < n and attempts < n * 4:
        attempts += 1
        pos, target = rng.choice(typed)
        story = gen(rng, target, pos)
        known = _compact_known(story, target, analyzer)
        report = validate_story(story, set(known), {target.lower()}, analyzer, language=language)
        if not report.hard_pass:
            failed += 1
            continue
        scenario = Scenario(
            id=f"{language}-synth-{kept:06d}", language=language, level="baseline",
            theme="synthetic", target_words=[target], known=known,
        )
        rec = Example(scenario, story, report, 0, kept=True).to_record()
        rec["metadata"]["source"] = "synthetic-v1"
        rec["metadata"]["target_pos"] = pos
        records.append(rec)
        kept += 1

    rng.shuffle(records)
    n_tr, n_va = int(kept * 0.8), int(kept * 0.1)
    splits = {
        "train": records[:n_tr],
        "val": records[n_tr:n_tr + n_va],
        "test": records[n_tr + n_va:],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rs in splits.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"language": language, "requested": n, "kept": kept, "attempts": attempts,
             "failed": failed, "splits": {k: len(v) for k, v in splits.items()}}
    with open(out_dir / "synth_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Programmatically generate spec-passing i+1 stories.")
    p.add_argument("--n", type=int, default=1000, help="Target number of kept stories.")
    p.add_argument("--language", default="en")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    stats = generate(args.language, args.n, args.out, args.seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
