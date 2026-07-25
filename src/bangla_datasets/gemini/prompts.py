"""System-prompt templates. Conversation script + system-prompt language are
persona-driven (bengali / banglish script; bangla / english system prompt). The
tool/code layer is always ASCII English regardless of script/language."""
import json

from bangla_datasets.schema import Persona

# --- Language boundary rules, per conversation script -------------------------
# The invariant in BOTH variants: tool names, parameter keys, and enum values
# stay ASCII (English/code layer). Only the conversation script differs.
_BOUNDARY_RULE_BN = (
    "ভাষা নিয়ম (Language boundary rule — STRICT):\n"
    "- আপনার কথা, ব্যাখ্যা এবং চূড়ান্ত উত্তর অবশ্যই বাংলা ভাষায় এবং বাংলা লিপিতে (Bengali script) হতে হবে।\n"
    "  দেবনাগরী লিপি (नमस्ते) বা রোমান লিপিতে লিখবেন না — শুধু বাংলা লিপি (নমস্কার)।\n"
    "- টুলের নাম, প্যারামিটার কী, এনাম ভ্যালু এবং কোড অবশ্যই English/ASCII হবে — এগুলো অনুবাদ করবেন না।\n"
    "- ব্র্যান্ড/প্রযুক্তি নাম (WhatsApp, Google, URL, নম্বর) বাংলায় মিশিয়ে বলতে পারেন।\n"
    "- প্যারামিটার ভ্যালু যদি বাস্তব ডেটা হয় (যেমন স্টেশনের নাম 'ঢাকা') তবে বাংলায় রাখুন।"
)

_BOUNDARY_RULE_BANGLISH = (
    "ভাষা নিয়ম (Language boundary rule — STRICT):\n"
    "- আপনার কথা, ব্যাখ্যা এবং চূড়ান্ত উত্তর অবশ্যই banglish বা romanized bangla হতে হবে "
    "(bangla bhasha kintu roman lipite, jemon 'ami dhaka jete chai')।\n"
    "- বাংলা লিপি (বাংলা) বা দেবনাগরী (नमस्ते) ব্যবহার করবেন না — শুধু roman letter (a-z)।\n"
    "- টুলের নাম, প্যারামিটার কী, এনাম ভ্যালু এবং কোড অবশ্যই English/ASCII হবে — এগুলো অনুবাদ করবেন না।\n"
    "- ব্র্যান্ড/প্রযুক্তি নাম (bKash, Pathao, URL) banglish এ মিশিয়ে বলতে পারেন।\n"
    "- প্যারামিটার ভ্যালু যদি বাস্তব ডেটা হয় (যেমন স্টেশনের নাম) তবে banglish এ রাখুন।"
)

# The boundary rule for the English *system prompt* variant, expressed in
# English so the model reads the rule in the same language as the instructions.
_BOUNDARY_RULE_EN = (
    "Language boundary rule (STRICT):\n"
    "- Your conversation, explanations, and final answer must be in BANGLA. Use the "
    "script the persona speaks (Bengali script or Romanized/Banglish) — follow the "
    "user's turns. Never answer in English prose; the user speaks Bangla.\n"
    "- Tool names, parameter keys, enum values, and code MUST stay English/ASCII. "
    "Never translate these.\n"
    "- Brand/technology names (bKash, Pathao, WhatsApp, URLs) may be kept as-is.\n"
    "- Parameter values that are real data (e.g. station name 'ঢাকা' / 'dhaka') "
    "stay in the user's script."
)

LANGUAGE_BOUNDARY_RULE = _BOUNDARY_RULE_BN  # backward-compat public constant


def _boundary_rule(script: str, language: str) -> str:
    if language == "english":
        return _BOUNDARY_RULE_EN
    return _BOUNDARY_RULE_BANGLISH if script == "banglish" else _BOUNDARY_RULE_BN


# --- Formality descriptors, per register + script -----------------------------
# Three Bangla speech registers (honorific levels):
#   tui   - intimate   (তুই/তোর): very close friends, younger siblings.
#   tumi  - familiar   (তুমি/তোমাকে): friends, colleagues, peers.
#   apni  - formal     (আপনি/আপনার): elders, strangers, officials, service.
# Two maps so the register phrasing matches the conversation script.
_FORMALITY_BN = {
    "tui": (
        "তুই/তোর সম্বোধন (intimate)। খুব কাছের বন্ধু বা ছোট ভাইবোনের সাথে যেমন আলগা "
        "ভাষায় কথা বলিস। ছোট বাক্য, খুব সাবলীল ও অনৌপচারিক টোন। কথোপকথন সংক্ষিপ্ত।"
    ),
    "tumi": (
        "তুমি/তোমাকে সম্বোধন (familiar)। বন্ধু বা পরিচিত সহকর্মীর সাথে যেমন বন্ধুত্বপূর্ণ "
        "ভাষায় কথা বলো। স্বাভাবিক বাক্য, বন্ধুত্বপূর্ণ কিন্তু শালীন টোন।"
    ),
    "apni": (
        "আপনি সম্বোধন, অত্যন্ত আনুষ্ঠানিক ও শ্রদ্ধাশীল টোন (formal)। "
        "বয়োজ্যেষ্ঠ বা অপরিচিত কারো সাথে যেমন কথা বলেন। সম্পূর্ণ বাক্য, নম্রভাবে।"
    ),
}

_FORMALITY_BANGLISH = {
    "tui": (
        "tui/tor shombodhon (intimate). khub kacher bondhu ba choto bhaiboner sathe "
        "jemon alga bhashay kotha bolis. choto bakya, khub shobil o onoupochrik tone. "
        "kothopokkhop shonkhipto."
    ),
    "tumi": (
        "tumi/tomake shombodhon (familiar). bondhu ba porichito shokormir sathe jemon "
        "bondhuttopurno bhashay kotha bolo. shadharon bakya, bondhuttopurno kintu shalim tone."
    ),
    "apni": (
        "apni shombodhon, ottonto anusthanik o shroddhashil tone (formal). "
        "boyojyeshtho ba oporichito karor sathe jemon kotha balen. shompurno bakya, nomrobhave."
    ),
}


def build_persona_prompt(persona: Persona, domain: str, goal: str) -> str:
    """Persona (simulated user) system prompt. Script is persona-driven.

    The conversation script (bengali | banglish) only changes how the persona is
    told to write — the role/structure is identical.
    """
    script = getattr(persona, "script", "bengali") or "bengali"
    formality_map = _FORMALITY_BANGLISH if script == "banglish" else _FORMALITY_BN
    formality_desc = formality_map.get(persona.register, formality_map["apni"])
    rule = _boundary_rule(script, "bangla")

    if script == "banglish":
        return (
            "apni ekjon shadharon byaboharokari — kono shahajjokari ba agent nom. "
            "apni ekjon AI shohogarir sathe kotha balchen ebong apnar ekta nirdisto "
            "kach ache. apnar bhromika shahajjo kora noi — apnar bhromika shahajjo chao.\n\n"
            f"profile: boyosh {persona.age}, obosthan {persona.location}, "
            f"peshka {persona.profession}, prodouktik dakkota {persona.tech_literacy}.\n"
            f"bhashar dhoron: {formality_desc}\n"
            f"apnar lokkho: {goal}\n\n"
            "niyom:\n"
            "- banglish (romanized bangla) te kotha balun — bangla lipite noi.\n"
            "- apni byaboharokari — tai prosn korun, onurodh korun, totto din. "
            "nijey theke kichu shomadhan ba uttor diben na.\n"
            "- shongkhipto thakun — ek ba dui bakye kotha balun."
        )
    return (
        f"আপনি একজন সাধারণ ব্যবহারকারী — কোনো সাহায্যকারী বা এজেন্ট নন। "
        f"আপনি একজন এআই সহকারীর সাথে কথা বলছেন এবং আপনার একটি নির্দিষ্ট কাজ আছে। "
        f"আপনার ভূমিকা সাহায্য করা নয় — আপনার ভূমিকা সাহায্য চাওয়া।\n\n"
        f"প্রোফাইল: বয়স {persona.age}, অবস্থান {persona.location}, পেশা {persona.profession}, "
        f"প্রযুক্তিগত দক্ষতা {persona.tech_literacy}।\n"
        f"ভাষার ধরন: {formality_desc}\n"
        f"আপনার লক্ষ্য: {goal}\n\n"
        f"{rule}\n\n"
        "নিয়ম:\n"
        "- প্রাকৃতিক বাংলায় (বাংলা লিপিতে, দেবনাগরী নয়) কথা বলুন।\n"
        "- আপনি ব্যবহারকারী — তাই প্রশ্ন করুন, অনুরোধ করুন, তথ্য দিন। "
        "নিজে থেকে কিছু সমাধান বা উত্তর দেবেন না।\n"
        "- সংক্ষিপ্ত থাকুন — এক বা দুই বাক্যে কথা বলুন।"
    )


def build_assistant_prompt(tools: list[dict], language: str = "bangla") -> str:
    """Assistant system prompt. ``language`` = the language the PROMPT is in.

    The conversation script is persona-driven and learned from the user's turns;
    the assistant prompt only needs to enforce 'answer in Bangla (the user's
    script), tools stay English'.
    """
    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    rule = _boundary_rule("bengali", language)

    if language == "english":
        return (
            "You are a helpful AI assistant. Solve the user's problem using the "
            "available tools.\n\n"
            f"Available tools (English schemas):\n{tools_str}\n\n"
            f"{rule}\n\n"
            "Important rules:\n"
            "- Each turn: either make a tool call OR give the final answer.\n"
            "- Do not repeat the same tool with the same arguments. Once you have a "
            "result, use it.\n"
            "- When you receive a tool result, analyze it and answer the user in "
            "Bangla (matching the script the user writes in).\n"
            "- Think briefly, then call a tool only if needed."
        )
    return (
        "আপনি একজন সহায়ক এআই সহকারী। ব্যবহারকারীর সমস্যা সমাধানে টুল ব্যবহার করুন।\n\n"
        f"উপলব্ধ টুলসমূহ (English schemas):\n{tools_str}\n\n"
        f"{rule}\n\n"
        "গুরুত্বপূর্ণ নিয়ম:\n"
        "- প্রতিটি টার্নে আপনি হয় একটি টুল কল করবেন অথবা চূড়ান্ত উত্তর দেবেন।\n"
        "- একই টুল একই আর্গুমেন্ট নিয়ে বারবার কল করবেন না। একবার ফলাফল পেলে সেটি ব্যবহার করুন।\n"
        "- টুলের ফলাফল পেলে সেটি বিশ্লেষণ করে ব্যবহারকারীকে বাংলায় উত্তর দিন।\n"
        "- প্রথমে বাংলায় সংক্ষেপে ভাবুন, তারপর প্রয়োজনে টুল ডাকুন।"
    )


def build_judge_prompt(script: str = "bengali") -> str:
    """Judge rubric. Dimension NAMES are fixed (judge.py enforces len==5); only
    the ``bangla_fluency`` description is rephrased for Banglish so legitimate
    Romanized turns are not penalized."""
    if script == "banglish":
        fluency = (
            "3. bangla_fluency — banglish (romanized bangla) ki shadharon, "
            "swabhabik ebom shobhab-anugul? bangla bhashar ortho thik thakche ki?"
        )
        switching = (
            "4. code_switching — bhasha niyom ki manano hoyeche "
            "(code/schema English, kothopokkhop banglish)?"
        )
        return (
            "apni ekjon kothor muloyanokari (judge). nichor trajectory ti 5 ti "
            "manodonde 1-5 scale e muloyan korun.\n\n"
            "manodondosomuhor jonno score 1-5 ebom ek line reason:\n"
            "1. task_completion — shohogari ki byaboharokarir lokkho tool bebohar "
            "kore shomadhan korche?\n"
            "2. tool_correctness — tool call gulo ki shothik o yuktiyukto chilo?\n"
            f"{fluency}\n"
            f"{switching}\n"
            "5. coherence — multi-turn provab ki shongot, kono shobirodh nei?\n\n"
            "nirdesh: pass korte hole sob manodonde >=4 proyojon. uttor JSON e din."
        )
    return (
        "আপনি একজন কঠোর মূল্যায়নকারী (judge)। নিচের ট্রাজেক্টরিটি ৫টি মানদণ্ডে ১–৫ স্কেলে মূল্যায়ন করুন।\n\n"
        "মানদণ্ডসমূহ (প্রতিটির জন্য score ১–৫ এবং এক লাইন reason):\n"
        "1. task_completion — সহকারী কি ব্যবহারকারীর লক্ষ্য টুল ব্যবহার করে সমাধান করেছে?\n"
        "2. tool_correctness — টুল কলগুলো কি সঠিক ও যুক্তিযুক্ত ছিল (অতি/অব্যবহার নয়)?\n"
        "3. bangla_fluency — বাংলা কি স্বাভাবিক, ব্যাকরণসম্মত, প্রোফাইলের উপযুক্ত?\n"
        "4. code_switching — ভাষা নিয়ম কি মানা হয়েছে (কোড/স্কিমা English, কথোপকথন বাংলা)?\n"
        "5. coherence — মাল্টি-টার্ন প্রবাহ কি সঙ্গত, কোনো স্ববিরোধ নেই?\n\n"
        "নির্দেশ: পাস করতে হলে সব মানদণ্ডে ≥৪ প্রয়োজন। উত্তর JSON-এ দিন।"
    )


# Brief requires a SYSTEM_PROMPTS dict produced by the gemini package.
SYSTEM_PROMPTS = {
    "persona": build_persona_prompt,
    "assistant": build_assistant_prompt,
    "judge": build_judge_prompt,
}
