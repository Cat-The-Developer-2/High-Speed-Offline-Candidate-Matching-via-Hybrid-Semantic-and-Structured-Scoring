#!/usr/bin/env python3
"""
Local End-to-End Experiment & Ablation Runner for SoAMSCML-2026 Poster
Compares:
  1. Keyword & Rule-Based Scoring Only (0% Semantic)
  2. Pure Semantic Similarity Only (100% all-MiniLM-L6-v2)
  3. Hybrid Ranker — Lightweight Bi-Encoder (30% all-MiniLM-L6-v2 + 70% Rules/Signals)
  4. Hybrid Ranker — Stronger Bi-Encoder (30% BAAI/bge-large-en-v1.5 + 70% Rules/Signals)

Exports:
  - result.csv (Summary metrics table for your abstract & poster)
  - top100_model_comparison.csv (Side-by-side rank shifts between MiniLM and BGE-Large)
  - submission_bge.csv (Top 100 shortlist from the stronger BGE-Large hybrid model)

Usage:
  pip install numpy scikit-learn pandas sentence-transformers python-docx scipy
  python run_local_experiment.py --candidates ./candidates.jsonl
"""

import argparse
import csv
import gzip
import json
import os
import time
from datetime import date

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# --- Domain Constants (matches rank.py) ---
CONSULTING_FIRMS = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
PRODUCT_LOCATIONS = {
    'pune', 'noida', 'hyderabad', 'mumbai', 'delhi',
    'gurgaon', 'gurugram', 'bengaluru', 'bangalore'
}
RELEVANT_TITLE_KEYWORDS = [
    'ai engineer', 'ml engineer', 'machine learning', 'data scientist',
    'nlp', 'search', 'ranking', 'recommendation', 'applied scientist',
    'backend engineer', 'data engineer', 'software engineer'
]
VECTOR_DB_SKILLS = {
    'elasticsearch', 'faiss', 'milvus', 'opensearch', 'pinecone',
    'qdrant', 'weaviate', 'pgvector', 'vector search'
}
RETRIEVAL_SKILLS = {'embeddings', 'information retrieval', 'rag', 'vector search'}
EVAL_SKILLS = {'ndcg', 'mrr', 'map', 'a/b testing', 'evaluation', 'learning to rank'}
FRAMEWORK_ONLY_SKILL = {'langchain'}
TODAY = date(2026, 6, 20)


# --- Data Loading Helpers ---
def load_candidates(path):
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        path = path + '.gz'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find candidates file at '{path}'. "
            "Pass --candidates /path/to/candidates.jsonl"
        )
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jd_text(jd_docx_path):
    if not os.path.exists(jd_docx_path):
        raise FileNotFoundError(
            f"Could not find '{jd_docx_path}'. Place job_description.docx in this folder "
            "or pass --jd_docx /path/to/job_description.docx"
        )
    from docx import Document
    doc = Document(jd_docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def build_candidate_text(c):
    profile = c['profile']
    parts = [
        profile.get('headline', ''),
        profile.get('summary', ''),
        f"Current role: {profile.get('current_title', '')} at {profile.get('current_company', '')}",
    ]
    for job in c.get('career_history', []):
        parts.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
    skill_names = [s['name'] for s in c.get('skills', [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))
    return " | ".join(p for p in parts if p)


# --- Scoring & Signal Functions (matches rank.py) ---
def title_relevance_score(c):
    titles = [c['profile']['current_title'].lower()]
    titles += [job['title'].lower() for job in c.get('career_history', [])]
    matches = sum(1 for t in titles for kw in RELEVANT_TITLE_KEYWORDS if kw in t)
    return min(matches / 3, 1.0)


def is_consulting_only(c):
    companies = [job['company'].lower() for job in c.get('career_history', [])]
    if not companies:
        return False
    return all(any(firm in comp for firm in CONSULTING_FIRMS) for comp in companies)


def is_title_chaser(c):
    history = c.get('career_history', [])
    if len(history) < 3:
        return False
    short_stints = sum(1 for job in history if job.get('duration_months', 999) < 18)
    return short_stints >= len(history) - 1


def location_fit(c):
    loc = c['profile']['location'].lower()
    return any(city in loc for city in PRODUCT_LOCATIONS)


def experience_fit(c):
    yoe = c['profile']['years_of_experience']
    if 5 <= yoe <= 9:
        return 1.0
    elif 3 <= yoe < 5 or 9 < yoe <= 12:
        return 0.6
    return 0.2


def skills_score(c):
    skill_names = {s['name'].lower() for s in c.get('skills', [])}
    score = 0.0
    if skill_names & VECTOR_DB_SKILLS:
        score += 0.4
    if skill_names & RETRIEVAL_SKILLS:
        score += 0.4
    if skill_names & EVAL_SKILLS:
        score += 0.2
    return min(score, 1.0)


def is_framework_enthusiast(c):
    skills = c.get('skills', [])
    skill_names = {s['name'].lower() for s in skills}
    has_only_langchain = (skill_names & FRAMEWORK_ONLY_SKILL) and not (
        skill_names & (VECTOR_DB_SKILLS | RETRIEVAL_SKILLS | EVAL_SKILLS)
    )
    for s in skills:
        if s['name'].lower() in FRAMEWORK_ONLY_SKILL and s.get('duration_months', 99) < 12:
            return True
    return has_only_langchain


def honeypot_flag(c):
    for skill in c.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            return True
    total_months = sum(job.get('duration_months', 0) for job in c.get('career_history', []))
    if c['profile']['years_of_experience'] * 12 > total_months + 24:
        return True
    return False


def signal_multiplier(c):
    s = c['redrob_signals']
    mult = 1.0
    if not s.get('open_to_work_flag', True):
        mult *= 0.7
    if s.get('recruiter_response_rate', 1.0) < 0.2:
        mult *= 0.7
    try:
        last_active = date.fromisoformat(s['last_active_date'])
        days_inactive = (TODAY - last_active).days
        if days_inactive > 180:
            mult *= 0.5
        elif days_inactive > 90:
            mult *= 0.8
    except Exception:
        pass
    if s.get('notice_period_days', 0) > 60:
        mult *= 0.85
    return mult


def generate_reasoning(c, score_info):
    p = c['profile']
    s = c['redrob_signals']
    parts = [f"{p['current_title']} at {p['current_company']} with {p['years_of_experience']:.1f} years experience."]

    skill_names = [sk['name'] for sk in c.get('skills', [])]
    relevant_found = [
        sk for sk in skill_names
        if sk.lower() in (VECTOR_DB_SKILLS | RETRIEVAL_SKILLS | EVAL_SKILLS)
    ]
    if relevant_found:
        parts.append(f"Has hands-on {', '.join(relevant_found[:3])} experience.")

    if location_fit(c):
        parts.append(f"Based in {p['location']}, matching preferred locations.")
    else:
        parts.append(f"Located in {p['location']} (outside preferred cities).")

    if score_info['sig_mult'] < 0.8:
        concerns = []
        if not s.get('open_to_work_flag', True):
            concerns.append("not marked open to work")
        if s.get('recruiter_response_rate', 1) < 0.2:
            concerns.append(f"low recruiter response rate ({s.get('recruiter_response_rate', 0):.0%})")
        if s.get('notice_period_days', 0) > 60:
            concerns.append(f"{s.get('notice_period_days')}-day notice period")
        if concerns:
            parts.append(f"Some concerns: {', '.join(concerns)}.")
    else:
        parts.append(f"Active on platform with {s.get('recruiter_response_rate', 0):.0%} recruiter response rate.")

    if score_info['consulting_penalty'] < 1.0:
        parts.append("Entire career at consulting firms — limited product company exposure.")

    return " ".join(parts)


# --- Embedding Computation / Caching ---
def get_or_compute_embeddings(
    model_name,
    cand_emb_path,
    jd_emb_path,
    candidates,
    jd_text,
    batch_size=64,
    force_recompute=False
):
    if not force_recompute and os.path.exists(cand_emb_path) and os.path.exists(jd_emb_path):
        print(f"[Cache Hit] Loading existing embeddings for {model_name} from {cand_emb_path}...")
        cand_emb = np.load(cand_emb_path)
        jd_emb = np.load(jd_emb_path)
        size_mb = (os.path.getsize(cand_emb_path) + os.path.getsize(jd_emb_path)) / (1024 * 1024)
        return cand_emb, jd_emb, 0.0, size_mb

    print(f"[Encoding] Computing embeddings locally using {model_name} (this may take a few minutes on CPU)...")
    from sentence_transformers import SentenceTransformer

    candidate_texts = [build_candidate_text(c) for c in candidates]
    t0 = time.time()
    model = SentenceTransformer(model_name)
    jd_emb = model.encode([jd_text], show_progress_bar=False, convert_to_numpy=True)
    cand_emb = model.encode(
        candidate_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embed_sec = time.time() - t0

    cand_fp16 = cand_emb.astype(np.float16)
    jd_fp16 = jd_emb.astype(np.float16)
    np.save(cand_emb_path, cand_fp16)
    np.save(jd_emb_path, jd_fp16)

    size_mb = (os.path.getsize(cand_emb_path) + os.path.getsize(jd_emb_path)) / (1024 * 1024)
    print(f"[Saved] {cand_emb_path} ({size_mb:.1f} MB total) computed in {embed_sec:.1f}s")
    return cand_fp16, jd_fp16, embed_sec, size_mb


# --- Ranking Evaluator ---
def evaluate_configuration(
    config_name,
    model_label,
    candidates,
    static_features,
    cand_emb=None,
    jd_emb=None,
    mode="hybrid"
):
    """
    mode:
      - 'rules_only': 100% rule-based + behavioral scoring (0% semantic)
      - 'semantic_only': 100% normalized cosine similarity (0% rules/penalties)
      - 'hybrid': 30% semantic + 70% rule-based + penalties + behavioral multipliers
    """
    t0 = time.time()

    if cand_emb is not None and jd_emb is not None:
        raw_sim = cosine_similarity(jd_emb.astype(np.float32), cand_emb.astype(np.float32))[0]
        sim_min, sim_max = float(raw_sim.min()), float(raw_sim.max())
        sim_mean, sim_std = float(raw_sim.mean()), float(raw_sim.std())
        semantic_norm = (raw_sim - sim_min) / (sim_max - sim_min + 1e-12)
        emb_dim = int(cand_emb.shape[1])
    else:
        raw_sim = np.zeros(len(candidates), dtype=np.float32)
        sim_min = sim_max = sim_mean = sim_std = 0.0
        semantic_norm = raw_sim
        emb_dim = 0

    results = []
    for i, c in enumerate(candidates):
        f = static_features[i]
        sem = float(semantic_norm[i])

        if mode == "rules_only":
            # Re-normalize the 70% structured rule weights to sum to 1.0
            base_score = (
                (0.25 / 0.70) * f['title_score'] +
                (0.20 / 0.70) * f['sk_score'] +
                (0.15 / 0.70) * f['exp_score'] +
                (0.10 / 0.70) * f['loc_score']
            )
            final_score = base_score * f['mult_total']
        elif mode == "semantic_only":
            final_score = sem
        else:  # 'hybrid' (exact rank.py formula)
            base_score = (
                0.30 * sem +
                0.25 * f['title_score'] +
                0.20 * f['sk_score'] +
                0.15 * f['exp_score'] +
                0.10 * f['loc_score']
            )
            final_score = base_score * f['mult_total']

        if f['honeypot'] and mode != "semantic_only":
            final_score = -1.0

        results.append({
            'candidate_id': c['candidate_id'],
            'final_score': final_score,
            'consulting_penalty': f['consulting_penalty'],
            'sig_mult': f['sig_mult'],
            'title_score': f['title_score'],
            'sk_score': f['sk_score'],
            'idx': i
        })

    top_100 = sorted(results, key=lambda r: -r['final_score'])[:100]
    top_100_sorted = sorted(
        top_100,
        key=lambda r: (-round(r['final_score'], 4), candidates[r['idx']]['candidate_id'])
    )
    cpu_rank_sec = time.time() - t0

    top_ids = [r['candidate_id'] for r in top_100_sorted]
    rank_map = {r['candidate_id']: idx + 1 for idx, r in enumerate(top_100_sorted)}

    summary = {
        'configuration': config_name,
        'embedding_model': model_label,
        'embedding_dim': emb_dim,
        'total_candidates': len(candidates),
        'honeypots_filtered': sum(1 for f in static_features if f['honeypot']),
        'cpu_ranking_time_sec': round(cpu_rank_sec, 3),
        'raw_cosine_mean': round(sim_mean, 4),
        'raw_cosine_std': round(sim_std, 4),
        'raw_cosine_min': round(sim_min, 4),
        'raw_cosine_max': round(sim_max, 4),
        'top100_mean_final_score': round(float(np.mean([r['final_score'] for r in top_100_sorted])), 4),
        'top100_avg_skills_score': round(float(np.mean([r['sk_score'] for r in top_100_sorted])), 4),
        'top100_avg_title_score': round(float(np.mean([r['title_score'] for r in top_100_sorted])), 4),
        'top100_non_ai_title_count': sum(1 for r in top_100_sorted if r['title_score'] == 0.0),
    }
    return summary, top_100_sorted, top_ids, rank_map


def main():
    parser = argparse.ArgumentParser(description="Run local candidate ranking comparison and export result.csv")
    parser.add_argument('--candidates', default='./candidates.jsonl', help='Path to candidates.jsonl or .jsonl.gz')
    parser.add_argument('--jd_docx', default='./job_description.docx', help='Path to job_description.docx')
    parser.add_argument('--minilm_cand_emb', default='./candidate_embeddings.npy', help='MiniLM candidate embeddings')
    parser.add_argument('--minilm_jd_emb', default='./jd_embedding.npy', help='MiniLM JD embedding')
    parser.add_argument('--bge_cand_emb', default='./candidate_embeddings_bge.npy', help='BGE-Large candidate embeddings')
    parser.add_argument('--bge_jd_emb', default='./jd_embedding_bge.npy', help='BGE-Large JD embedding')
    parser.add_argument('--result_csv', default='./result.csv', help='Summary results CSV for poster/abstract')
    parser.add_argument('--comparison_csv', default='./top100_model_comparison.csv', help='Detailed Top-100 shift CSV')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for local embedding encoding')
    parser.add_argument('--recompute_all', action='store_true', help='Force re-encoding even if .npy files exist')
    args = parser.parse_args()

    print("1. Loading candidates and job description...")
    candidates = load_candidates(args.candidates)
    jd_text = load_jd_text(args.jd_docx)
    print(f"   Loaded {len(candidates):,} candidates and JD ({len(jd_text)} chars).")

    print("2. Precomputing static rule/behavioral features...")
    static_features = []
    for c in candidates:
        c_pen = 0.4 if is_consulting_only(c) else 1.0
        ch_pen = 0.6 if is_title_chaser(c) else 1.0
        fr_pen = 0.5 if is_framework_enthusiast(c) else 1.0
        s_mult = signal_multiplier(c)
        static_features.append({
            'title_score': title_relevance_score(c),
            'exp_score': experience_fit(c),
            'loc_score': 1.0 if location_fit(c) else 0.3,
            'sk_score': skills_score(c),
            'consulting_penalty': c_pen,
            'sig_mult': s_mult,
            'mult_total': c_pen * ch_pen * fr_pen * s_mult,
            'honeypot': honeypot_flag(c)
        })

    print("3. Loading/computing embeddings for MiniLM (384d) and BGE-Large (1024d)...")
    minilm_cand, minilm_jd, minilm_emb_sec, minilm_mb = get_or_compute_embeddings(
        'sentence-transformers/all-MiniLM-L6-v2',
        args.minilm_cand_emb,
        args.minilm_jd_emb,
        candidates,
        jd_text,
        batch_size=args.batch_size,
        force_recompute=args.recompute_all
    )

    bge_cand, bge_jd, bge_emb_sec, bge_mb = get_or_compute_embeddings(
        'BAAI/bge-large-en-v1.5',
        args.bge_cand_emb,
        args.bge_jd_emb,
        candidates,
        jd_text,
        batch_size=args.batch_size,
        force_recompute=args.recompute_all
    )

    print("4. Evaluating all 4 ranking configurations...")
    configs = [
        ("1_Rules_and_Signals_Only", "None (0% Semantic)", None, None, "rules_only", 0.0, 0.0),
        ("2_Pure_Semantic_MiniLM", "all-MiniLM-L6-v2 (100% Semantic)", minilm_cand, minilm_jd, "semantic_only", minilm_emb_sec, minilm_mb),
        ("3_Hybrid_MiniLM_384d", "all-MiniLM-L6-v2 (30% Semantic + 70% Rules)", minilm_cand, minilm_jd, "hybrid", minilm_emb_sec, minilm_mb),
        ("4_Hybrid_BGE_Large_1024d", "BAAI/bge-large-en-v1.5 (30% Semantic + 70% Rules)", bge_cand, bge_jd, "hybrid", bge_emb_sec, bge_mb),
    ]

    summaries = []
    top_sets = {}
    rank_maps = {}
    top_lists = {}

    for cfg_name, m_label, c_emb, j_emb, mode, emb_sec, size_mb in configs:
        summary, top_100, top_ids, r_map = evaluate_configuration(
            cfg_name, m_label, candidates, static_features, c_emb, j_emb, mode=mode
        )
        summary['embed_compute_time_sec'] = round(emb_sec, 2)
        summary['embedding_storage_mb'] = round(size_mb, 2)
        summaries.append(summary)
        top_sets[cfg_name] = set(top_ids)
        rank_maps[cfg_name] = r_map
        top_lists[cfg_name] = top_100

    # Compare every configuration against the strongest model (4_Hybrid_BGE_Large_1024d)
    ref_name = "4_Hybrid_BGE_Large_1024d"
    ref_set = top_sets[ref_name]
    ref_ranks = rank_maps[ref_name]

    for summary in summaries:
        cfg_name = summary['configuration']
        shared = sorted(top_sets[cfg_name] & ref_set)
        summary['top100_overlap_vs_bge_hybrid_pct'] = len(shared)
        if len(shared) >= 2:
            r1 = [rank_maps[cfg_name][cid] for cid in shared]
            r2 = [ref_ranks[cid] for cid in shared]
            rho, _ = spearmanr(r1, r2)
            summary['spearman_rho_vs_bge_hybrid'] = round(float(rho), 4)
        else:
            summary['spearman_rho_vs_bge_hybrid'] = 1.0 if cfg_name == ref_name else 0.0

    # 5. Export result.csv
    fieldnames = [
        'configuration',
        'embedding_model',
        'embedding_dim',
        'total_candidates',
        'honeypots_filtered',
        'embed_compute_time_sec',
        'embedding_storage_mb',
        'cpu_ranking_time_sec',
        'raw_cosine_mean',
        'raw_cosine_std',
        'raw_cosine_min',
        'raw_cosine_max',
        'top100_mean_final_score',
        'top100_avg_skills_score',
        'top100_avg_title_score',
        'top100_non_ai_title_count',
        'top100_overlap_vs_bge_hybrid_pct',
        'spearman_rho_vs_bge_hybrid'
    ]
    with open(args.result_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    # 6. Export detailed MiniLM vs BGE-Large Top-100 rank shift comparison
    minilm_map = {r['candidate_id']: (idx + 1, r['final_score']) for idx, r in enumerate(top_lists['3_Hybrid_MiniLM_384d'])}
    with open(args.comparison_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bge_rank', 'minilm_rank', 'rank_change', 'candidate_id',
            'current_title', 'current_company', 'years_exp',
            'bge_score', 'minilm_score', 'reasoning'
        ])
        for bge_rank, r in enumerate(top_lists[ref_name], start=1):
            c = candidates[r['idx']]
            cid = c['candidate_id']
            m_rank, m_score = minilm_map.get(cid, ('>100', ''))
            rank_change = (m_rank - bge_rank) if isinstance(m_rank, int) else 'NEW_IN_TOP_100'
            writer.writerow([
                bge_rank,
                m_rank,
                rank_change,
                cid,
                c['profile']['current_title'],
                c['profile']['current_company'],
                c['profile']['years_of_experience'],
                round(r['final_score'], 4),
                round(m_score, 4) if isinstance(m_score, float) else '',
                generate_reasoning(c, r)
            ])

    # 7. Also export submission_bge.csv in exact competition format
    with open('./submission_bge.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, r in enumerate(top_lists[ref_name], start=1):
            c = candidates[r['idx']]
            writer.writerow([c['candidate_id'], rank, round(r['final_score'], 4), generate_reasoning(c, r)])

    print(f"\nDone! Exported:")
    print(f"  1. {args.result_csv} (Summary metrics for your abstract & poster)")
    print(f"  2. {args.comparison_csv} (Detailed candidate rank shifts)")
    print(f"  3. ./submission_bge.csv (Top 100 shortlist from BAAI/bge-large-en-v1.5)")


if __name__ == '__main__':
    main()
