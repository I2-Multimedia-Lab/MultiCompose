import os
import io
import glob
import json
import time
import base64
import random
import requests
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ==========================================
# 1. Core config
# ==========================================
API_KEY = os.getenv("VLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
API_BASE_URL = os.getenv(
    "VLM_ENDPOINT",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
MODEL_NAME = os.getenv("VLM_MODEL", os.getenv("VLM_MODEL_NAME", "qwen3-vl-plus"))

RESULTS_ROOT = os.getenv("RESULTS_ROOT", "/path/to/generated-images")
REF_IMAGE_DIRS_RAW = os.getenv("REF_IMAGE_DIRS", os.getenv("REF_IMAGES_DIR", "/path/to/reference-images:/path/to/datasets"))
REF_IMAGE_DIRS = [p for p in REF_IMAGE_DIRS_RAW.split(":") if p]

RESULTS_DIR_NAME = os.path.basename(RESULTS_ROOT.rstrip(os.sep))
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(RESULTS_ROOT), f"{RESULTS_DIR_NAME}_评测")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
ONLY_FUSION = os.getenv("ONLY_FUSION", "1") == "1"
GROUP_FILTER = os.getenv("GROUP_FILTER", "").strip().lower()
SUBJECT_FILTER = os.getenv("SUBJECT_FILTER", "").strip().lower()
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
RESUME_WITH_FILTER = os.getenv("RESUME_WITH_FILTER", "0") == "1"


# ==========================================
# 2. Encode + VLM call
# ==========================================
def encode_reference_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((768, 768))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_generated_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def evaluate_with_vlm(ref1_path, ref2_path, bg_ref_path, gen_path, prompt_text, subject_a_name, subject_b_name, bg_name):
    ref1_base64 = encode_reference_image(ref1_path)
    ref2_base64 = encode_reference_image(ref2_path)
    bg_ref_base64 = encode_reference_image(bg_ref_path)
    gen_base64 = encode_generated_image(gen_path)

    system_prompt = f"""# 多概念个性化生成定向评估
请评估生成图对提示词和参考图的还原度。参考图为主体A({subject_a_name})、主体B({subject_b_name})和背景({bg_name})。
提示词：{prompt_text}

基准分100分，请直接在脑内据此扣分，并输出最终分：
1. 概念召回(核心，扣完为止)：漏掉主A或主B各扣20分；缺背景扣10分；缺一关键属性（衣服/道具）扣10分。若主A和主B都没了，总分直接0分。
2. 特征解耦与属性绑定(重要，满分30分)：出现了严重的跨主体特征融合扣15分；提示词要求给主A的配饰穿到了主B身上（张冠李戴）扣15分。
3. 身份保真度(满分30分)：主A长得不像参考图扣10分；主B不像扣10分；背景严重偏离参考图景扣10分。

无需输出长篇推理，只需返回如下纯JSON格式：
{{
  "reasoning": "一句话概括最主要的扣分原因，若完美则写完美",
  "missing_entities": ["在此列出所有未生成的实体或属性，没有则为空数组"],
  "scores_breakdown": {{
      "recall_deduction": "概念召回扣掉的分数(0至-50)",
      "binding_deduction": "解耦与绑定扣掉的分数(0至-30)",
      "fidelity_deduction": "身份保真扣掉的分数(0至-30)"
  }},
  "score": 最终剩余的总分(整数0-100)
}}"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref1_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref2_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{bg_ref_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gen_base64}"}},
                ],
            }
        ],
        "max_tokens": 800,
        "temperature": 0.2,
    }

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.3, 1.0))
            response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

            if response.status_code == 429:
                print(f"⚠️ 429 限流，第 {attempt + 1}/{max_retries} 次重试")
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
                continue

            result = response.json()
            if "error" in result:
                err = result["error"]
                err_code = str(err.get("code", ""))
                if "rate" in err_code.lower() or "quota" in err_code.lower():
                    print(f"⚠️ API 流控/额度限制: {err}")
                    time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
                    continue
                return {
                    "reasoning": f"API Error: {err}",
                    "missing_entities": [],
                    "scores_breakdown": {},
                    "score": 0,
                }

            content = result["choices"][0]["message"]["content"].strip()
            content = content.replace("```json", "").replace("```", "").strip()

            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx >= start_idx:
                content = content[start_idx : end_idx + 1]

            parsed = json.loads(content)
            parsed["score"] = safe_float(parsed.get("score", 0), 0)
            if "missing_entities" not in parsed or not isinstance(parsed["missing_entities"], list):
                parsed["missing_entities"] = []
            if "scores_breakdown" not in parsed or not isinstance(parsed["scores_breakdown"], dict):
                parsed["scores_breakdown"] = {}
            if "reasoning" not in parsed:
                parsed["reasoning"] = ""
            return parsed

        except requests.exceptions.ReadTimeout:
            print(f"⚠️ 请求超时，第 {attempt + 1}/{max_retries} 次重试")
            time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            print(f"❌ 请求解析失败: {e}")
            return {
                "reasoning": f"内部错误: {e}",
                "missing_entities": [],
                "scores_breakdown": {},
                "score": 0,
            }

    return {
        "reasoning": "Maximum retries reached",
        "missing_entities": [],
        "scores_breakdown": {},
        "score": 0,
    }


# ==========================================
# 3. Parse task + organize results
# ==========================================
def get_ref_image(entity_name):
    for root in REF_IMAGE_DIRS:
        p0 = os.path.join(root, entity_name, "0.jpg")
        if os.path.exists(p0):
            return p0

        candidates = glob.glob(os.path.join(root, entity_name, "*.*"))
        candidates = [
            f
            for f in candidates
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
        ]
        if candidates:
            return candidates[0]

    return None


def subject_kind(subj_name):
    s = str(subj_name).lower()
    if s.startswith("pet_cat"):
        return "cat"
    if s.startswith("pet_dog"):
        return "dog"
    return "other"


def split_cat_dog_pair(subj_a, subj_b):
    ka = subject_kind(subj_a)
    kb = subject_kind(subj_b)
    if ka == "cat" and kb == "dog":
        return subj_a, subj_b
    if ka == "dog" and kb == "cat":
        return subj_b, subj_a
    return None, None


def infer_group_from_subjects(subj_a, subj_b):
    ka = subject_kind(subj_a)
    kb = subject_kind(subj_b)

    if ka == "other" and kb == "other":
        return "plushie_plushie"
    if (ka == "other" and kb == "cat") or (ka == "cat" and kb == "other"):
        return "plushie_cat"
    if (ka == "other" and kb == "dog") or (ka == "dog" and kb == "other"):
        return "plushie_dog"
    if (ka == "cat" and kb == "dog") or (ka == "dog" and kb == "cat"):
        return "catdog"
    return "theme"


def find_layout_anchor_window(parts):
    # Supports:
    # anchor_x/attr_a__b/partner_y/bg_scene/.../img
    # and prefixed roots like non_vs_non/anchor_x/...
    for i in range(0, max(0, len(parts) - 4)):
        if (
            parts[i].startswith("anchor_")
            and parts[i + 1].startswith("attr_")
            and parts[i + 2].startswith("partner_")
            and parts[i + 3].startswith("bg_")
        ):
            return i
    return None

def find_layout_theme12_window(parts):
    # Supports:
    # theme1_x/attr_a__b/theme2_y/bg_scene/.../img
    # and prefixed roots like non_vs_non/theme1_x/... or non_vs_catdog/cat/theme1_x/...
    for i in range(0, max(0, len(parts) - 4)):
        if (
            parts[i].startswith("theme1_")
            and parts[i + 1].startswith("attr_")
            and parts[i + 2].startswith("theme2_")
            and parts[i + 3].startswith("bg_")
        ):
            return i
    return None


def parse_task_from_image(image_path):
    rel = os.path.relpath(image_path, RESULTS_ROOT)
    parts = rel.split(os.sep)

    if len(parts) < 5:
        return None
    use_theme12_auto_order = False

    # Layout A:
    # cat_pet_cat2/attr_hat__tie/dog_pet_dog1/bg_scene_castle/g0.90_t0.20/xxx.png
    if (
        len(parts) >= 5
        and parts[0].startswith("cat_")
        and parts[1].startswith("attr_")
        and parts[2].startswith("dog_")
        and parts[3].startswith("bg_")
    ):
        cat_id = parts[0][len("cat_") :]
        attr_raw = parts[1][len("attr_") :]
        dog_id = parts[2][len("dog_") :]
        bg_id = parts[3][len("bg_") :]
        group_dir = "catdog"
        theme_raw = f"{cat_id}__{dog_id}"

    # Layout B:
    # attr_x__y/theme_a__b/bg_scene_xxx/g.../xxx.png
    # attr_x__y/<group>/a__b/bg_scene_xxx/g.../xxx.png
    elif parts[0].startswith("attr_"):
        attr_raw = parts[0][len("attr_") :]
        if len(parts) >= 6 and parts[3].startswith("bg_"):
            group_dir = parts[1]
            theme_raw = parts[2]
            bg_id = parts[3][len("bg_") :]
        elif len(parts) >= 5 and parts[2].startswith("bg_"):
            group_dir = "theme"
            theme_raw = parts[1]
            bg_id = parts[2][len("bg_") :]
        else:
            return None

    # Layout C:
    # theme1_pet_cat1/attr_hat__tie/theme2_pet_dog2/bg_scene_castle/g.../xxx.png
    # and prefixed roots like non_vs_non/theme1_... or non_vs_catdog/cat/theme1_...
    elif (theme_i := find_layout_theme12_window(parts)) is not None:
        theme1_id = parts[theme_i][len("theme1_") :]
        attr_raw = parts[theme_i + 1][len("attr_") :]
        theme2_id = parts[theme_i + 2][len("theme2_") :]
        bg_id = parts[theme_i + 3][len("bg_") :]

        prefix_parts = [p for p in parts[:theme_i] if p]
        if prefix_parts:
            if len(prefix_parts) >= 2 and prefix_parts[0] == "non_vs_catdog" and prefix_parts[1] in ("cat", "dog"):
                group_dir = f"{prefix_parts[0]}/{prefix_parts[1]}"
            else:
                group_dir = prefix_parts[0]
        else:
            group_dir = "theme12"

        theme_raw = f"{theme1_id}__{theme2_id}"
        use_theme12_auto_order = True
    else:
        # Layout D:
        # anchor_x/attr_a__b/partner_y/bg_scene_xxx/g.../xxx.png
        # non_vs_non/anchor_x/attr_a__b/partner_y/bg_scene_xxx/g.../xxx.png
        # non_vs_catdog/anchor_x/attr_a__b/partner_y/bg_scene_xxx/g.../xxx.png
        anchor_i = find_layout_anchor_window(parts)
        if anchor_i is None:
            return None

        subj_a = parts[anchor_i][len("anchor_") :]
        attr_raw = parts[anchor_i + 1][len("attr_") :]
        subj_b = parts[anchor_i + 2][len("partner_") :]
        bg_id = parts[anchor_i + 3][len("bg_") :]
        group_dir = infer_group_from_subjects(subj_a, subj_b)
        theme_raw = f"{subj_a}__{subj_b}"

    if theme_raw.startswith("theme_"):
        theme_raw = theme_raw[len("theme_") :]

    if "__" not in attr_raw or "__" not in theme_raw:
        return None

    attr_a, attr_b = attr_raw.split("__", 1)
    subj_a, subj_b = theme_raw.split("__", 1)

    # For theme1/theme2 layout:
    # - mask2 images are cat-first in prompt semantics
    # - box-only images usually follow theme1->theme2 order
    # We infer order from filename pattern to keep attribute binding consistent.
    if use_theme12_auto_order:
        fname = os.path.basename(image_path).lower()
        cat_id, dog_id = split_cat_dog_pair(subj_a, subj_b)

        if ("mask2" in fname or "box_resample_tome_no_latex_simple_mask2" in fname) and cat_id and dog_id:
            subj_a, subj_b = cat_id, dog_id
        elif "photo of a dog wearing" in fname and " and a cat wearing " in fname and cat_id and dog_id:
            subj_a, subj_b = dog_id, cat_id
        elif "photo of a cat wearing" in fname and " and a dog wearing " in fname and cat_id and dog_id:
            subj_a, subj_b = cat_id, dog_id

    clean_attr_combo = f"{attr_a}+{attr_b}"
    clean_subject_combo = f"{subj_a}+{subj_b}"
    subject_key = f"{group_dir}:{clean_subject_combo}"

    ref1_path = get_ref_image(subj_a)
    ref2_path = get_ref_image(subj_b)
    bg_ref_path = get_ref_image(bg_id)

    if not ref1_path or not ref2_path:
        return None
    if not bg_ref_path:
        bg_ref_path = ref1_path

    prompt_text = f"photo of a {subj_a} wearing {attr_a} and a {subj_b} wearing {attr_b}, {bg_id} background"

    parent_dir = os.path.dirname(image_path)
    txt_files = sorted(glob.glob(os.path.join(parent_dir, "*.txt")))
    if txt_files:
        try:
            with open(txt_files[0], "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if txt:
                prompt_text = txt
        except Exception:
            pass

    return {
        "img_file": image_path,
        "refs": (ref1_path, ref2_path, bg_ref_path),
        "prompt": prompt_text,
        "meta": (clean_attr_combo, subject_key, bg_id, group_dir),
        "subjects": (subj_a, subj_b),
    }


def is_candidate_image(path):
    low = path.lower()
    if not low.endswith((".png", ".jpg", ".jpeg")):
        return False

    name = os.path.basename(low)
    if "tweedie" in name:
        return False
    if any(x in name for x in ["heatmap", "attn", "debug"]):
        return False
    return True


def iter_candidate_images(results_root):
    patterns = [
        os.path.join(results_root, "attr_*", "**", "*.*"),
        os.path.join(results_root, "cat_*", "attr_*", "dog_*", "bg_*", "**", "*.*"),
        os.path.join(results_root, "theme1_*", "attr_*", "theme2_*", "bg_*", "**", "*.*"),
        os.path.join(results_root, "**", "theme1_*", "attr_*", "theme2_*", "bg_*", "**", "*.*"),
        os.path.join(results_root, "anchor_*", "attr_*", "partner_*", "bg_*", "**", "*.*"),
        os.path.join(results_root, "**", "anchor_*", "attr_*", "partner_*", "bg_*", "**", "*.*"),
    ]

    all_imgs = []
    for p in patterns:
        all_imgs.extend(glob.glob(p, recursive=True))
    all_imgs = sorted(set(all_imgs))
    all_imgs = [p for p in all_imgs if is_candidate_image(p)]

    if not ONLY_FUSION:
        return all_imgs

    by_dir = defaultdict(list)
    for p in all_imgs:
        by_dir[os.path.dirname(p)].append(p)

    candidates = []
    for _, files in by_dir.items():
        fusion_files = [f for f in files if "fusion" in os.path.basename(f).lower()]
        candidates.extend(fusion_files if fusion_files else files)

    return sorted(candidates)


def process_single_image(task_info):
    img_file = task_info["img_file"]
    ref1_path, ref2_path, bg_ref_path = task_info["refs"]
    prompt_text = task_info["prompt"]
    clean_attr_combo, subject_key, scene_name, group_name = task_info["meta"]
    subjects = task_info["subjects"]

    short_img_name = os.path.basename(img_file)
    print(
        f"\n🔍 [请求中] 组[{group_name}] 属性[{clean_attr_combo}] -> 主体[{subject_key}] -> 背景[{scene_name}]\n🖼️: {short_img_name}"
    )

    eval_result = evaluate_with_vlm(
        ref1_path,
        ref2_path,
        bg_ref_path,
        img_file,
        prompt_text,
        subject_a_name=subjects[0],
        subject_b_name=subjects[1],
        bg_name=scene_name,
    )

    score = safe_float(eval_result.get("score", 0), 0)
    missing_ents = eval_result.get("missing_entities", [])
    scores_breakdown = eval_result.get("scores_breakdown", {})

    print(f"✅ [已返回] 🖼️: {short_img_name}")
    print(f"   ⭐️ 得分: {score} | 丢失实体: {missing_ents}")
    print(f"   📊 扣分明细: {json.dumps(scores_breakdown, ensure_ascii=False)}")

    record = {
        "image": img_file.replace(RESULTS_ROOT, ""),
        "prompt": prompt_text,
        "group": group_name,
        "score": score,
        "scores_breakdown": scores_breakdown,
        "missing_entities": missing_ents,
        "reasoning": eval_result.get("reasoning", ""),
    }

    return {
        "attr": clean_attr_combo,
        "subj": subject_key,
        "scene": scene_name,
        "record": record,
    }


def task_matches_filters(task_info):
    group_name = str(task_info["meta"][3]).lower()
    subj_a, subj_b = task_info["subjects"]
    subject_text = f"{subj_a} {subj_b}".lower()

    if GROUP_FILTER and GROUP_FILTER not in group_name and GROUP_FILTER not in subject_text:
        return False
    if SUBJECT_FILTER and SUBJECT_FILTER not in subject_text:
        return False
    return True


def build_attr_report(attr_name, subj_dict):
    attr_scores = []
    report = {"attr_name": attr_name, "subject_groups": {}}

    for subj, scene_dict in subj_dict.items():
        subj_scores = []
        report["subject_groups"][subj] = {"backgrounds": {}}

        for scene, records in scene_dict.items():
            valid_scores = [safe_float(r.get("score", 0), 0) for r in records]
            avg_bg = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            subj_scores.extend(valid_scores)
            attr_scores.extend(valid_scores)

            report["subject_groups"][subj]["backgrounds"][scene] = {
                "average_score": round(avg_bg, 2),
                "details": records,
            }

        avg_subj = sum(subj_scores) / len(subj_scores) if subj_scores else 0
        report["subject_groups"][subj]["average_subject_score"] = round(avg_subj, 2)

    avg_attr = sum(attr_scores) / len(attr_scores) if attr_scores else 0
    report["average_attr_score"] = round(avg_attr, 2)
    report["total_images"] = len(attr_scores)
    return report


def collect_processed_images_and_warm_tree(results_tree):
    processed_images = set()
    if not os.path.isdir(OUTPUT_DIR):
        return processed_images

    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    for j_file in json_files:
        if os.path.basename(j_file) == "summary.json":
            continue

        try:
            with open(j_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            attr_name = data.get("attr_name")
            if not attr_name:
                attr_name = os.path.splitext(os.path.basename(j_file))[0].replace("__PAIR__", "+")

            subj_groups = data.get("subject_groups", {})
            for subj, subj_data in subj_groups.items():
                bg_map = subj_data.get("backgrounds", {})
                for scene, scene_data in bg_map.items():
                    records = scene_data.get("details", [])
                    for r in records:
                        img_rel = r.get("image", "")
                        if "tweedie" in str(img_rel).lower():
                            continue
                        if img_rel:
                            processed_images.add(img_rel)
                    filtered_records = [r for r in records if "tweedie" not in str(r.get("image", "")).lower()]
                    if filtered_records:
                        results_tree[attr_name][subj][scene].extend(filtered_records)
        except Exception as e:
            print(f"⚠️ 读取旧报告失败，跳过: {j_file} | {e}")

    return processed_images


def attr_json_filename(attr_name):
    safe = attr_name.replace("+", "__PAIR__").replace("/", "_")
    return f"{safe}.json"


def save_attr_report(attr_name, subj_dict):
    out_json_path = os.path.join(OUTPUT_DIR, attr_json_filename(attr_name))
    report = build_attr_report(attr_name, subj_dict)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)


def save_global_summary(results_tree):
    summary = {
        "results_root": RESULTS_ROOT,
        "output_dir": OUTPUT_DIR,
        "total_attrs": 0,
        "total_images": 0,
        "global_average_score": 0,
        "attrs": {},
    }

    all_scores = []
    for attr, subj_dict in results_tree.items():
        report = build_attr_report(attr, subj_dict)
        summary["attrs"][attr] = {
            "average_attr_score": report["average_attr_score"],
            "total_images": report["total_images"],
        }
        summary["total_images"] += report["total_images"]

        for subj_data in report["subject_groups"].values():
            for bg_data in subj_data["backgrounds"].values():
                all_scores.extend([safe_float(r.get("score", 0), 0) for r in bg_data.get("details", [])])

    summary["total_attrs"] = len(summary["attrs"])
    summary["global_average_score"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)


# ==========================================
# 4. Main
# ==========================================
def main():
    if not API_KEY:
        print("❌ 未检测到 VLM_API_KEY（也兼容 DASHSCOPE_API_KEY），请先设置后再运行。")
        return

    if not os.path.isdir(RESULTS_ROOT):
        print(f"❌ 找不到结果目录: {RESULTS_ROOT}")
        return

    existing_ref_roots = [p for p in REF_IMAGE_DIRS if os.path.isdir(p)]
    if not existing_ref_roots:
        print(f"❌ 找不到参考图目录，当前 REF_IMAGE_DIRS={REF_IMAGE_DIRS}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 开始扫描目录: {RESULTS_ROOT}")
    images = iter_candidate_images(RESULTS_ROOT)
    print(f"🧭 检测到候选图片: {len(images)}")

    results_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if GROUP_FILTER or SUBJECT_FILTER:
        if RESUME_WITH_FILTER:
            processed_images = collect_processed_images_and_warm_tree(results_tree)
            if processed_images:
                print(f"♻️ 过滤模式断点续跑：已加载历史记录 {len(processed_images)} 张")
            else:
                print("🆕 过滤模式断点续跑已开启，但未发现历史记录。")
        else:
            processed_images = set()
            print("🆕 检测到过滤模式，已禁用历史报告预热，按当前过滤条件重新统计。")
    else:
        processed_images = collect_processed_images_and_warm_tree(results_tree)
        if processed_images:
            print(f"♻️ 断点续跑：已加载历史记录 {len(processed_images)} 张")

    all_tasks = []
    skipped_no_ref = 0
    skipped_done = 0
    skipped_filter = 0

    for img in images:
        rel_img = img.replace(RESULTS_ROOT, "")
        if rel_img in processed_images:
            skipped_done += 1
            continue

        task = parse_task_from_image(img)
        if task is None:
            skipped_no_ref += 1
            continue

        if not task_matches_filters(task):
            skipped_filter += 1
            continue

        all_tasks.append(task)

    if MAX_IMAGES > 0 and len(all_tasks) > MAX_IMAGES:
        all_tasks = all_tasks[:MAX_IMAGES]
        print(f"✂️ MAX_IMAGES={MAX_IMAGES}，仅评测前 {len(all_tasks)} 张")

    print(
        f"📦 待评估任务: {len(all_tasks)} | 已跳过(已处理): {skipped_done} | 已跳过(无法解析/缺参考): {skipped_no_ref} | 已跳过(过滤): {skipped_filter}"
    )
    if GROUP_FILTER or SUBJECT_FILTER:
        print(f"🔎 过滤条件: GROUP_FILTER={GROUP_FILTER or '<空>'}, SUBJECT_FILTER={SUBJECT_FILTER or '<空>'}")
    if not all_tasks:
        save_global_summary(results_tree)
        print("✅ 无新任务，已输出 summary.json")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_image, task) for task in all_tasks]

        for future in as_completed(futures):
            try:
                res = future.result()
                attr = res["attr"]
                subj = res["subj"]
                scene = res["scene"]
                rec = res["record"]

                results_tree[attr][subj][scene].append(rec)
                save_attr_report(attr, results_tree[attr])
            except Exception as exc:
                print(f"⚠️ 并发任务异常: {exc}")

    for attr, subj_dict in results_tree.items():
        save_attr_report(attr, subj_dict)

    save_global_summary(results_tree)

    all_scores = []
    for attr, subj_dict in results_tree.items():
        report = build_attr_report(attr, subj_dict)
        for subj_data in report["subject_groups"].values():
            for bg_data in subj_data["backgrounds"].values():
                all_scores.extend([safe_float(r.get("score", 0), 0) for r in bg_data.get("details", [])])

    global_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    print("\n==========================================")
    print(f"🏆 评测大盘总基准分: {global_avg} 分 (共评估 {len(all_scores)} 张)")
    print("==========================================")
    print(f"✅ 评测完成，结果位于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
