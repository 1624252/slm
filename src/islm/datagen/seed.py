"""Human-authored seed dataset of genuinely good i+1 stories (English, Chinese, Japanese).

Where no teacher API is available, the stories here are authored to the Behavior Spec by hand so
there is real, spec-passing data to train on and to run the second-pass curation over. Each story
uses only baseline (known) vocabulary plus a single target word, introduces the target once, and
repeats it for spaced repetition. Every story is validated on build; only spec-passing stories are
written. Requires the downloaded vocab lists (``python -m islm.vocab.download``).

    python -m islm.datagen.seed --out data/generated/seed
    python -m islm.datagen.curate --in data/generated/seed --out data/curated/seed
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..config import DATA_DIR
from ..validators import validate_story
from ..vocab.lemmatize import get_analyzer
from ..vocab.wordlists import VOCAB_DIR, Vocabulary
from .generate import Example
from .scenarios import Scenario


def _compact_known(story: str, targets: list[str], language: str, analyzer) -> list[str]:
    """A small, story-scoped KNOWN_WORDS list: the curated baseline sample plus the story's own
    content words (minus the target), so the prompt is a few hundred tokens instead of >10k.

    The full baseline is ~2.3k words, which renders each training record to 5k-12k tokens; since
    the story (the completion we train on) sits at the end, that forces truncation that drops
    TARGET_WORDS and the rules. Scoping the known list to what the story actually uses keeps the
    whole record inside a small window and matches the eval's `--curated` setup.
    """
    base = VOCAB_DIR / language / "baseline.csv"
    known = Vocabulary.from_csv(base).lemmas if base.exists() else set()
    target_set = {t.lower() for t in targets}
    # Add every content word the story uses (except the target). Include tokens the lemmatizer
    # tags as proper nouns too: a sentence-initial function word ("Every", "Who") is mis-tagged
    # proper, and we still need it covered so the story's own vocabulary never reads as OOV.
    for tok in analyzer.analyze(story):
        if tok.is_word and tok.lemma not in target_set:
            known.add(tok.lemma)
    return sorted(known)

# language -> list of (target_words, story). Stories are simple by design (comprehensible input).
SEED: dict[str, list[tuple[list[str], str]]] = {
    "en": [
        (
            ["shadow"],
            "The girl has a little cat. The cat is black. "
            "At night, the cat sees a shadow on the box. "
            "The shadow is very big. The cat looks at the shadow. "
            "The shadow looks at the cat too. "
            "But the shadow is only the little cat! "
            "Now the girl and the cat play with the shadow every night.",
        ),
        (
            ["treasure"],
            "The girl and her dog go to the garden. "
            "The dog looks and looks in the garden. Then the dog finds a little box. "
            "In the box is a treasure! The treasure is old and red. "
            "The girl is so happy with the treasure. She takes the treasure home. "
            "The dog wants the treasure too.",
        ),
        (
            ["secret"],
            "The boy has a secret. He does not tell his friend. The secret is very big. "
            "The boy wants to tell the secret, but he waits. "
            "At night, he tells the cat the secret. The cat likes the secret. "
            "Now the cat and the boy have a secret.",
        ),
        (
            ["umbrella"],
            "It is a cold day. The rain comes down. "
            "The girl has a new umbrella. The umbrella is blue. "
            "She walks in the rain with the umbrella. A little bird is cold. "
            "The girl gives the bird her umbrella. Now the bird is happy under the umbrella.",
        ),
        (
            ["feather"],
            "A little bird sits in the tree. The bird has one red feather. "
            "The feather falls down to the garden. A cat sees the feather and looks up. "
            "The bird wants the feather back. The cat gives the feather to the bird.",
        ),
        (
            ["lantern"],
            "It is night and the house is dark. The old man has a lantern. "
            "He looks for the cat with the lantern. The lantern is not very big, but it helps. "
            "With the lantern, the man finds the little cat. The cat sleeps by the warm lantern.",
        ),
        (
            ["clue"],
            "The girl looks for her lost key. Her cat wants to help. "
            "The cat finds a small clue by the door. The clue is a red hair. "
            "Then the cat finds one more clue on the bed. The last clue is under the box. "
            "The clue takes the girl to the key.",
        ),
        (
            ["forest"],
            "The boy walks in the forest. The forest is big and green. "
            "In the forest, he sees many trees. A bird sings in the forest. "
            "The boy is very happy. He likes the forest. He wants to come back to the forest.",
        ),
        (
            ["dragon"],
            "The girl reads a book. In the book is a dragon. The dragon is big and red. "
            "The dragon can fly. But this dragon is not bad. The dragon helps the girl. "
            "Now the girl likes the dragon in her book.",
        ),
        (
            ["castle"],
            "By the sea is an old castle. The castle is very big. "
            "A boy and his dog go to the castle. They look in every room of the castle. "
            "The castle is cold and dark. But the castle has a warm fire. "
            "Now the castle is their home.",
        ),
        (
            ["mystery"],
            "There is a mystery in the house. Every night, the food is gone. "
            "Who eats the food? The girl wants to know the mystery. She looks and looks. "
            "The mystery is a little mouse! Now the girl knows the mystery.",
        ),
        (
            ["whisper"],
            "The two friends want to be quiet. So they whisper. "
            "The girl whispers to the boy. The boy whispers back. They whisper and laugh. "
            "Do not whisper too much! But the friends like to whisper.",
        ),
        (
            ["balloon"],
            "She has a red balloon. The balloon is big and light. "
            "The balloon goes up in the sky. But now the balloon is in the tree. "
            "The boy gets the balloon for her. Now the girl is happy with her balloon.",
        ),
        (
            ["robot"],
            "The boy makes a small robot. The robot is blue. The robot can walk and talk. "
            "The robot helps the boy. They play all day. At night, the robot sleeps too. "
            "The boy likes his new robot.",
        ),
        (
            ["pumpkin"],
            "In the garden is a big pumpkin. The pumpkin is red. "
            "The girl looks at the pumpkin. The pumpkin is very big. "
            "She takes the pumpkin home. She makes a face on the pumpkin. "
            "Now the pumpkin has a happy face.",
        ),
        (
            ["whistle"],
            "The man has a little whistle. The whistle is old. He looks for his dog. "
            "So the man uses the whistle. The dog hears the whistle and runs home. "
            "Now the dog knows the whistle.",
        ),
    ],
    "zh": [
        (
            ["秘密"],
            "小猫有一个秘密。它不告诉小狗。这个秘密很大。"
            "小猫想说，但是它不说。晚上，它把秘密告诉朋友。"
            "朋友也喜欢这个秘密。现在他们有一个秘密。",
        ),
        (
            ["影子"],
            "外面有很多太阳光。小狗在外面走。它看见自己的影子。"
            "影子很长。小狗跑，影子也跑。小狗不走，影子也不走。影子就是小狗。",
        ),
        (
            ["森林"],
            "小鸟住在森林里。森林很大，也很美。森林里有很多树。"
            "小鸟很喜欢这个森林。它每天在森林里玩。森林是它的家。",
        ),
        (
            ["城堡"],
            "山上有一个城堡。城堡很旧，也很大。女孩和小狗去城堡。"
            "他们看城堡里的房间。城堡里很冷。但是他们很高兴。现在城堡是他们的家。",
        ),
        (
            ["机器人"],
            "男孩做了一个机器人。机器人是蓝色的。机器人会走，也会说话。"
            "机器人帮助男孩。他们一起玩。晚上机器人也睡觉。男孩很喜欢机器人。",
        ),
        (
            ["发现"],
            "小狗在花园里玩。它发现一个东西。这是一个大发现。"
            "小狗把发现给女孩。女孩也很喜欢这个发现。这真是一个好发现。",
        ),
    ],
    "ja": [
        (
            ["秘密"],
            "猫は秘密があります。犬に言いません。この秘密はとても大きいです。"
            "猫は言いたいですが、言いません。夜、友達に秘密を話します。"
            "友達もこの秘密が好きです。今、猫と友達は秘密があります。",
        ),
        (
            ["冒険"],
            "男の子は本を読みます。本の中に大きい冒険があります。"
            "男の子は冒険が好きです。友達と一緒に冒険をします。"
            "冒険はとても楽しいです。今日も冒険をします。",
        ),
        (
            ["城"],
            "山の上に古い城があります。城はとても大きいです。"
            "男の子と犬が城に行きます。城の中は広いです。"
            "でも、城は暖かいです。今、城は男の子と犬の家です。",
        ),
        (
            ["発見"],
            "女の子は庭で遊びます。庭で何かを見つけます。大きい発見です。"
            "女の子は先生に発見を見せます。先生も発見が好きです。これはすごい発見です。",
        ),
        (
            ["宝物"],
            "女の子と犬が庭に行きます。犬が小さいものを見つけます。"
            "その中に宝物があります。宝物は古いです。"
            "女の子は宝物が好きです。犬も宝物が欲しいです。",
        ),
        (
            ["影"],
            "今日はいい天気です。犬が外を歩きます。犬は影を見ます。"
            "影はとても長いです。犬が走ると、影も走ります。"
            "犬が歩くと、影も歩きます。影は犬です。",
        ),
    ],
}

# One deliberate duplicate (of the first English story) to demonstrate curation dedup.
_DUPLICATE = ("en", 0)


def _build_records(languages: list[str]) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    stats: dict = {}
    for lang in languages:
        analyzer = get_analyzer(lang)
        passing, failures = 0, []
        stories = list(SEED[lang])
        if _DUPLICATE[0] == lang:
            stories.append(SEED[lang][_DUPLICATE[1]])  # planted duplicate
        for i, (targets, story) in enumerate(stories):
            compact = _compact_known(story, targets, lang, analyzer)
            scenario = Scenario(
                id=f"{lang}-seed-{i:03d}",
                language=lang,
                level="baseline",
                theme="seed",
                target_words=targets,
                known=compact,
            )
            report = validate_story(story, set(compact), set(targets), analyzer, language=lang)
            if report.hard_pass:
                kept.append(Example(scenario, story, report, 0, kept=True).to_record())
                passing += 1
            else:
                failures.append({"id": scenario.id, "reasons": report.failures()})
        stats[lang] = {"authored": len(stories), "spec_passing": passing, "failures": failures}
    return kept, stats


def build(out_dir: Path, languages: list[str], seed: int = 0) -> dict:
    kept, stats = _build_records(languages)
    random.Random(seed).shuffle(kept)
    n = len(kept)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits = {
        "train": kept[:n_train],
        "val": kept[n_train : n_train + n_val],
        "test": kept[n_train + n_val :],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["total_spec_passing"] = n
    stats["split_counts"] = {k: len(v) for k, v in splits.items()}
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build the human-authored seed dataset.")
    p.add_argument("--out", type=Path, default=DATA_DIR / "generated" / "seed")
    p.add_argument("--language", default="all", help="en, zh, ja, or all.")
    args = p.parse_args()
    languages = list(SEED) if args.language == "all" else [args.language]
    stats = build(args.out, languages)
    for lang in languages:
        s = stats[lang]
        print(f"{lang}: authored={s['authored']} spec_passing={s['spec_passing']}")
        for fail in s["failures"]:
            print("  FAILED", fail["id"], fail["reasons"])
    print(f"total kept={stats['total_spec_passing']} splits={stats['split_counts']}")


if __name__ == "__main__":
    main()
