import os, sys, json, re, time, warnings
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# ignore Streamlit warnings if imported helpers use Streamlit context
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# ensure parent dir (where app.py lives) is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

# import helpers from main app
from app import docx_to_text, sanitize_input

load_dotenv()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
GOLDEN_JSON_PATH = Path("evals/golden/standard_nda_reference.json")
TEST_FOLDER = Path("evals/test_ndas")
RESULTS_FILE = Path("evals/eval_results.json")

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

if not all([endpoint, api_key, chat_deployment]):
    raise EnvironmentError("❌ Missing Azure environment variables in .env file.")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview"
)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def extract_json(output: str):
    """Extract first JSON object from model output."""
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        print("⚠️ No JSON detected in model output.")
        return {"issues": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON structure, skipping.")
        return {"issues": []}

def compare_to_golden(pred_json, gold_json):
    """Compare predicted NDA analysis with golden reference JSON."""
    gold_issues = gold_json.get("issues", [])
    pred_issues = pred_json.get("issues", [])

    # normalize and compare by section + type
    gold_pairs = {(i["section"].lower(), i.get("type", "").lower()) for i in gold_issues}
    pred_pairs = {(i["section"].lower(), i.get("type", "").lower()) for i in pred_issues}

    intersection = gold_pairs & pred_pairs
    precision = len(intersection) / max(1, len(pred_pairs))
    recall = len(intersection) / max(1, len(gold_pairs))
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # risk agreement where sections match
    matched_sections = [
        (p, g) for g in gold_issues for p in pred_issues
        if g["section"].lower() == p["section"].lower()
    ]
    risk_matches = sum(1 for p, g in matched_sections if p.get("risk", "").lower() == g.get("risk", "").lower())
    risk_accuracy = risk_matches / max(1, len(matched_sections))

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "risk_accuracy": round(risk_accuracy, 2),
        "predicted_issues": len(pred_issues),
        "golden_issues": len(gold_issues),
    }

# -------------------------------------------------
# LOAD GOLDEN BASELINE
# -------------------------------------------------
if not GOLDEN_JSON_PATH.exists():
    raise FileNotFoundError(f"❌ Missing golden reference JSON: {GOLDEN_JSON_PATH}")

with open(GOLDEN_JSON_PATH) as f:
    gold_json = json.load(f)
print(f"✅ Loaded golden reference: {GOLDEN_JSON_PATH} ({len(gold_json.get('issues', []))} issues)")

# -------------------------------------------------
# RUN EVALUATION
# -------------------------------------------------
results = []

for nda_file in TEST_FOLDER.glob("*.docx"):
    print(f"\nEvaluating: {nda_file.name}")
    nda_text = sanitize_input(docx_to_text(nda_file))

    # --- system prompt enforcing JSON only ---
    system_prompt = (
        "You are NDA Review Copilot. Return ONLY a valid JSON object with keys "
        "'document_title', 'acceptability_score', and 'issues'. "
        "Each issue must include section, type, risk, finding, recommendation, and playbook_ref. "
        "Do not include explanations or text outside the JSON object."
    )

    # --- run model ---
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=chat_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nda_text},
            ],
            temperature=0
        )
    except Exception as e:
        print(f"❌ API error for {nda_file.name}: {e}")
        continue

    runtime = time.time() - start
    output = response.choices[0].message.content
    print("Raw model output (first 400 chars):\n", output[:400], "\n---")

    pred_json = extract_json(output)
    metrics = compare_to_golden(pred_json, gold_json)

    result = {
        "file": nda_file.name,
        **metrics,
        "runtime_sec": round(runtime, 2),
    }
    results.append(result)
    print(json.dumps(result, indent=2))

# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Evaluation complete — results saved to {RESULTS_FILE}")