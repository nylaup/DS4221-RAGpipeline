"""Regenerate data/eval_questions.jsonl: the 4 proposal questions + 28 DRAFTS.

DRAFT questions were written by an AI assistant from the corpus as a starting
point. The proposal promises human-written questions, so before the
evaluation counts them, a team member must review each one (rewrite it in
their own words if they like, check the labels) and put their name in
`author`. Drafts keep author="DRAFT" until then.

Ingredient and constrained labels come from a rule over the corpus (recorded
in `label_rule`), so they are complete rather than hand-picked:
  ingredient:  docs in the wiki category 'Recipes using <x>'
  constrained: docs whose infobox category matches and whose parsed time fits

    uv run python scripts/draft_questions.py && uv run python -m recipe_rag.build --questions-only
"""

import json
from pathlib import Path

from recipe_rag.db import connect

ROOT = Path(__file__).resolve().parent.parent
PROP = "Team (proposal example)"
DRAFT = "DRAFT"

# (q_id, question, q_type, relevant ids | None, label_rule | None, author)
QUESTIONS = [
    # ---------------- lookup: find the recipe (all variants count)
    ("prop-01", "How do I make Nigerian mashed taro?", "lookup", ["Ebiripo_(Nigerian_Mashed_Taro)"], None, PROP),
    ("d-l01", "How do you make carrot cake?", "lookup", ["Carrot_Cake_I", "Carrot_Cake_II"], None, DRAFT),
    ("d-l02", "Is there a recipe for a dense chocolate cake made without any flour?", "lookup",
     ["Flourless_Chocolate_Cake", "Bête_Noire_(Flourless_Chocolate_Cake)"], None, DRAFT),
    ("d-l03", "How do I make the Malaysian dish arisa?", "lookup", ["Arisa_(Malaysian_Blended_Chicken)"], None, DRAFT),
    ("d-l04", "What's the Austrian version of meatloaf?", "lookup", ["Austrian_Meatloaf_(Faschierter_Braten)"], None, DRAFT),
    ("d-l05", "How do I make pancakes that taste like gingerbread?", "lookup",
     ["Gingerbread_Pancakes_I", "Gingerbread_Pancakes_II"], None, DRAFT),
    ("d-l06", "How do I make homemade eggnog with brandy and rum?", "lookup", ["Eggnog"], None, DRAFT),
    ("d-l07", "How do you make shakshuka?", "lookup",
     ["Shakshuka_I", "Shakshuka_II", "Shakshuka_bil_Lahm_(Libyan_Meat_Shakshuka)"], None, DRAFT),
    # ---------------- procedure: a specific fact inside one recipe
    ("prop-02", "What oven temperature does the corn soufflé use?", "procedure", ["Corny_Soufflé"], None, PROP),
    ("d-p01", "What internal temperature should a Cajun burger be cooked to?", "procedure", ["Cajun_Burger"], None, DRAFT),
    ("d-p02", "How long do thumbprint jam cookies bake, and at what temperature?", "procedure",
     ["Thumbprint_Jam_Cookies"], None, DRAFT),
    ("d-p03", "How many minutes do I microwave salmon wrapped in parchment paper?", "procedure",
     ["Microwave_Salmon_in_a_Parchment_Envelope"], None, DRAFT),
    ("d-p04", "What are the saffron strands soaked in for shirmal flatbread?", "procedure",
     ["Shirmal_(Persian_Saffron_Flatbread)"], None, DRAFT),
    ("d-p05", "For breaded fried oysters, what do you dip the oysters in before the crumbs?", "procedure",
     ["Breaded_Fried_Oysters"], None, DRAFT),
    ("d-p06", "How long should I whip the eggs when making eggnog?", "procedure", ["Eggnog"], None, DRAFT),
    ("d-p07", "What do I soak the gelatin sheets in for the strawberry fromage frais dessert?", "procedure",
     ["Strawberry_Fromage_Frais"], None, DRAFT),
    # ---------------- ingredient: what uses X (wiki 'Recipes using X' category)
    ("prop-03", "What can I make with cassava flakes?", "ingredient",
     ["Abacha", "Abacha_Mmiri_(Soaked_Cassava_Flakes)", "African_Salad_II", "Eba_(Cassava_Flake_Swallow)",
      "Eba_with_Efo_Gbure_(Cassava_Flake_Swallow_with_Waterleaf_Soup)"], "find 'cassava flake' + abacha, hand-checked", PROP),
    ("d-i01", "What recipes use agar?", "ingredient", None, "Recipes using agar", DRAFT),
    ("d-i02", "I have some chicken livers. What can I cook with them?", "ingredient", None, "Recipes using chicken liver", DRAFT),
    ("d-i03", "What can I make with kefir?", "ingredient", None, "Recipes using kefir", DRAFT),
    ("d-i04", "Which dishes use paneer?", "ingredient", None, "Recipes using paneer", DRAFT),
    ("d-i05", "What can I cook with venison?", "ingredient", None, "Recipes using venison", DRAFT),
    ("d-i06", "Recipes that use fresh figs?", "ingredient", None, "Recipes using fig", DRAFT),
    ("d-i07", "What can I make with a butternut squash?", "ingredient", None, "Recipes using butternut squash", DRAFT),
    # ---------------- constrained: category + time (+ difficulty)
    ("prop-04", "Show me soup under 30 minutes", "constrained", None, ("soup", "<", 30, None), PROP),
    ("d-c01", "What desserts can I make in under 20 minutes?", "constrained", None, ("dessert", "<", 20, None), DRAFT),
    ("d-c02", "Cookies I can bake in under 30 minutes", "constrained", None, ("cookie", "<", 30, None), DRAFT),
    ("d-c03", "A salad that takes less than 15 minutes", "constrained", None, ("salad", "<", 15, None), DRAFT),
    ("d-c04", "Bread recipes I can finish within an hour", "constrained", None, ("bread", "<=", 60, None), DRAFT),
    ("d-c05", "Quick pasta dishes under 20 minutes", "constrained", None, ("pasta", "<", 20, None), DRAFT),
    ("d-c06", "Easy breakfast that takes 15 minutes or less", "constrained", None, ("breakfast", "<=", 15, 2), DRAFT),
    ("d-c07", "A drink I can make in under 10 minutes", "constrained", None, ("beverage", "<", 10, None), DRAFT),
]


def main():
    con = connect(read_only=True)
    known = {d for (d,) in con.execute("SELECT doc_id FROM documents").fetchall()}
    out = []
    for q_id, text, q_type, rel, rule, author in QUESTIONS:
        rule_txt = None
        if q_type == "ingredient" and rel is None:
            rel = [d for (d,) in con.execute(
                "SELECT doc_id FROM documents WHERE list_contains(categories, ?) ORDER BY 1", [rule]).fetchall()]
            rule_txt = f"wiki category '{rule}'"
        elif q_type == "constrained":
            cat, op, mins, diff = rule
            sql = (f"SELECT doc_id FROM documents WHERE category ILIKE ? AND total_minutes {op} ?"
                   + (" AND difficulty <= ?" if diff else "") + " ORDER BY 1")
            rel = [d for (d,) in con.execute(sql, [f"%{cat}%", mins] + ([diff] if diff else [])).fetchall()]
            rule_txt = (f"category ILIKE '%{cat}%' AND total_minutes {op} {mins}"
                        + (f" AND difficulty <= {diff}" if diff else ""))
        elif rule:
            rule_txt = rule
        missing = [d for d in rel if d not in known]
        assert rel and not missing, (q_id, rel, missing)
        row = {"q_id": q_id, "question": text, "q_type": q_type, "relevant_doc_ids": rel, "author": author}
        if rule_txt:
            row["label_rule"] = rule_txt
        out.append(row)
        print(f"{q_id:8s} {q_type:11s} {len(rel):3d} relevant  {text}")
    path = ROOT / "data" / "eval_questions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} questions -> {path}")


if __name__ == "__main__":
    main()
