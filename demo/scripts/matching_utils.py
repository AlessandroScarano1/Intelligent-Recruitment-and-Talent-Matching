# shared matching utilities for CV-job matching system
# contains all text processing, skill extraction, and scoring logic used across the app and CLI scripts

# functions:
# - load_skill_dictionary: load filtered skill set from parquet
# - extract_skills_from_text: find skills in text using PhraseMatcher
# - reformat_cv_for_matching: restructure raw CV to "I am a..." format
# - extract_job_fields: parse job posting into structured fields
# - build_job_embedding_string: create job embedding text
# - prepare_for_biencoder: add model prefixes for bi-encoder
# - logistic_percentage: convert scores to percentages

import re
import math
from pathlib import Path
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import torch

# project root for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_device():
    """Pick the best available torch device: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# module-level singletons, lazy-loaded on first use
_nlp = None
_matcher = None
_skill_set = None

# stopwords to filter out from skill dictionary
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
             'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
             'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
             'might', 'must', 'shall', 'can', 'need', 'our', 'we', 'you', 'your', 'they',
             'their', 'its', 'it', 'that', 'this', 'which', 'what', 'who', 'whom', 'any',
             'all', 'some', 'more', 'most', 'other', 'each', 'few', 'many', 'such', 'no',
             'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now'}

# generic resume words, appear in every job description but aren't real skills
GENERIC_RESUME_WORDS = {'experience', 'years', 'strong', 'team', 'work', 'working', 'position',
                        'ability', 'able', 'excellent', 'good', 'great', 'skills', 'knowledge',
                        'understanding', 'familiarity', 'proficiency', 'expertise', 'demonstrated',
                        'proven', 'equivalent', 'gain', 'various', 'different', 'multiple', 'new',
                        'within', 'across', 'using', 'including', 'related', 'required', 'preferred',
                        'minimum', 'ideal', 'desirable', 'essential', 'degree', 'bachelor', 'master',
                        'diploma', 'intern', 'internship', 'job', 'role', 'opportunity', 'employment',
                        'company', 'organization', 'business', 'environment', 'requirements',
                        'responsibilities', 'status', 'seeking', 'level', 'current', 'support'}

# ultra-generic single words that are NOT skills when alone
# keep multi-word versions like 'data analysis' but filter single 'data'
ULTRA_GENERIC_SINGLE = {'data', 'information', 'process', 'systems', 'solutions',
                        'pull', 'building', 'tools', 'supporting', 'identification',
                        'preparation', 'methodology', 'investigations', 'engagement',
                        'developer', 'engineer', 'analyst', 'scientist', 'manager',
                        'coordinator', 'director', 'supervisor', 'administrator',
                        'assistant', 'associate', 'officer', 'executive', 'remote',
                        'exposure', 'issues', 'structure', 'structures', 'oversight',
                        'computer', 'lead', 'maintenance'}


def is_quality_skill(skill):
    """filter out stopwords, generic resume words, and ultra-generic single words"""
    skill_lower = skill.lower().strip()

    # length check
    if len(skill_lower) < 3:
        return False

    # stopwords
    if skill_lower in STOPWORDS:
        return False

    # generic resume words
    if skill_lower in GENERIC_RESUME_WORDS:
        return False

    # for single words only, filter ultra-generic terms
    # keep multi-word phrases like 'data analysis', 'project management'
    if ' ' not in skill_lower and skill_lower in ULTRA_GENERIC_SINGLE:
        return False

    return True


def load_skill_dictionary(min_count=100, filter_stopwords=True):
    """
    Load filtered skill dictionary from parquet files.
    Default min_count=100 reduces 3.3M raw skills to ~22K quality skills.

    Returns:
        list of skill strings (lowercase)
    """
    skill_path = PROJECT_ROOT / 'ingest_job_postings' / 'output' / 'skill_dictionary' / 'all_skills'

    if not skill_path.exists():
        print(f'WARNING: skill dictionary not found at {skill_path}')
        return []

    # read parquet files
    skill_df = pd.read_parquet(skill_path)

    # filter by count threshold
    if min_count > 0:
        skill_df = skill_df[skill_df['count'] >= min_count]

    print(f'loaded {len(skill_df):,} skills (min {min_count} occurrences)')

    # filter out stopwords and generic terms
    if filter_stopwords:
        raw_skills = skill_df['skill'].tolist()
        skills_list = [s.lower() for s in raw_skills if is_quality_skill(s)]
        print(f'after filtering stopwords/generic: {len(skills_list):,} skills')
        print(f'removed {len(raw_skills) - len(skills_list):,} noisy entries')
        return skills_list
    else:
        return [s.lower() for s in skill_df['skill'].tolist()]


def get_skill_matcher():
    """
    Lazy-load spacy PhraseMatcher with skill patterns.
    Returns tuple: (nlp, matcher, skill_set)
    Singleton - only loads once per process.
    """
    global _nlp, _matcher, _skill_set

    if _nlp is not None:
        return _nlp, _matcher, _skill_set

    print('initializing skill matcher (one-time setup)...')

    # load skills
    skills_list = load_skill_dictionary()
    _skill_set = set(skills_list)

    # initialize spacy blank model (just tokenizer, no heavy NLP)
    _nlp = spacy.blank('en')
    print(f'spacy model loaded: {_nlp.lang}')

    # create PhraseMatcher with case-insensitive matching
    _matcher = PhraseMatcher(_nlp.vocab, attr='LOWER')

    # build patterns in batches (more efficient for large sets)
    print('building phrase patterns...')
    batch_size = 10000
    for i in range(0, len(skills_list), batch_size):
        batch = skills_list[i:i+batch_size]
        patterns = [_nlp.make_doc(skill) for skill in batch]
        _matcher.add('SKILLS', patterns)
        if i % 50000 == 0 and i > 0:
            print(f'  processed {i:,} skills...')

    print(f'matcher ready with {len(_skill_set):,} skills')

    return _nlp, _matcher, _skill_set


def extract_skills_from_text(text):
    """
    Extract skills from text using PhraseMatcher.
    Returns list of unique skills found (case-insensitive deduplication).
    """
    if not text or not str(text).strip():
        return []

    nlp, matcher, _ = get_skill_matcher()

    doc = nlp(text.lower())
    matches = matcher(doc)

    # collect skills
    skills = []
    for match_id, start, end in matches:
        skill = doc[start:end].text
        if len(skill) > 1:
            skills.append(skill)

    # deduplicate (case-insensitive)
    seen = set()
    unique_skills = []
    for s in skills:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique_skills.append(s)

    return unique_skills


def strip_prefixes(text):
    """Remove common prefixes from text: query:, Query:, passage:"""
    if not text:
        return ""
    clean = text.replace("query: ", "").replace("Query: ", "").replace("passage: ", "")
    return clean.strip()


def is_cv_already_structured(text):
    """
    Check if CV text is already in structured format.
    Returns True if starts with "I am a..." (after stripping prefixes).
    """
    clean = strip_prefixes(text)
    return clean.startswith("I am a") or clean.startswith("I am an")


def reformat_cv_for_matching(raw_text):
    """
    Reformat raw CV/PDF text into query-friendly format for the model.
    Model expects: "I am a [Role] with [X] years experience. My skills include: ..."

    If already structured, returns as-is.
    Otherwise extracts info from raw text and builds template.
    Uses PhraseMatcher for skill extraction (not hardcoded list).
    """
    # strip prefixes
    clean = strip_prefixes(raw_text)

    # if already structured, use as is
    if is_cv_already_structured(raw_text):
        return clean

    # extract structured info from raw text
    lines = [l.strip() for l in clean.split("\n") if l.strip()]

    # extract skills using PhraseMatcher
    skills_found = extract_skills_from_text(clean)

    # look for role/title
    role = ""
    role_patterns = [
        r'(?:worked as|work as|working as|position:?|role:?)\s+(?:a |an )?([^\n.]{5,50})',
        r'([A-Za-z ]+(?:Developer|Engineer|Analyst|Scientist|Manager|Designer|Architect|Consultant|Student|Graduate|Intern))',
    ]
    for pat in role_patterns:
        matches_found = re.findall(pat, clean)
        for candidate in matches_found:
            candidate = candidate.strip()
            # skip section headers
            if 3 < len(candidate) < 50 and candidate.lower() not in ('experience', 'education', 'contact', 'skills'):
                role = candidate
                break
        if role:
            break

    # look for experience years
    exp_match = re.search(r'(\d+)\s+(?:years?|yr)', clean, re.IGNORECASE)
    if exp_match:
        yrs = exp_match.group(1)
        exp_str = f"{yrs} year{'s' if int(yrs) != 1 else ''} of experience"
    else:
        exp_str = ""

    # look for seniority level
    level = ""
    level_match = re.search(r"(entry|junior|mid|senior|lead|principal)\s+level", clean, re.IGNORECASE)
    if level_match:
        level = level_match.group(1) + " level"

    # look for education
    edu_match = re.search(r"\b((?:Master|Bachelor|PhD|BSc|MSc|B\.?S|M\.?S|B\.?E|M\.?E|B\.?Tech|M\.?Tech|MBA)\b'?s?\s*(?:degree|of science|of arts|of engineering)?(?:\s*,\s*|\s+in\s+)?[A-Za-z ]{0,40})", clean, re.IGNORECASE)
    edu_str = edu_match.group(1).strip().rstrip(',') if edu_match else ""

    # look for company
    company = ""
    company_match = re.search(r'(?:at|@)\s+([A-Z][A-Za-z0-9 &,\.]{2,40})(?:\s|,|\.|$)', clean)
    if company_match:
        company = company_match.group(1).strip()

    # build structured query text following training format
    parts = []

    if role:
        text = f"I am a {role}"
        if exp_str:
            text += f" with {exp_str}"
        if level:
            text += f", {level}"
        parts.append(text + ".")

    if skills_found:
        # limit to top 15 skills
        parts.append(f"My skills include: {', '.join(sorted(skills_found)[:15])}.")

    if edu_str:
        parts.append(f"I studied {edu_str}.")

    if company and role:
        parts.append(f"I worked as {role} at {company}.")

    if parts:
        return ' '.join(parts)

    # if we couldn't extract anything, just return cleaned text
    return clean


def extract_job_fields(job_text):
    """
    Extract structured fields from job posting text.
    Returns dict with keys: title, company, location, skills, experience_years,
    salary_min, salary_max, remote_status, seniority
    """
    fields = {
        'title': '',
        'company': '',
        'location': '',
        'skills': [],
        'experience_years': '',
        'salary_min': None,
        'salary_max': None,
        'remote_status': '',
        'seniority': 'mid'
    }

    if not job_text:
        return fields

    # extract skills using PhraseMatcher
    fields['skills'] = extract_skills_from_text(job_text)

    # title - first non-header line
    lines = job_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith(('company', 'location', 'salary', 'type')):
            fields['title'] = line[:100]
            break

    # company
    company_match = re.search(r'company[:\s]+([^\n]+)', job_text, re.I)
    if company_match:
        fields['company'] = company_match.group(1).strip()[:100]

    # location
    location_match = re.search(r'location[:\s]+([^\n]+)', job_text, re.I)
    if location_match:
        fields['location'] = location_match.group(1).strip()[:100]

    # salary range
    salary_match = re.search(r'\$?([\d,]+)\s*[-\u2013]\s*\$?([\d,]+)', job_text)
    if salary_match:
        try:
            fields['salary_min'] = int(salary_match.group(1).replace(',', ''))
            fields['salary_max'] = int(salary_match.group(2).replace(',', ''))
        except ValueError:
            pass

    # experience years
    exp_match = re.search(r'(\d+)\+?\s*years?', job_text, re.I)
    if exp_match:
        fields['experience_years'] = exp_match.group(1) + '+'

    # remote status
    if re.search(r'\bremote\b', job_text, re.I):
        fields['remote_status'] = 'remote'
    elif re.search(r'\bhybrid\b', job_text, re.I):
        fields['remote_status'] = 'hybrid'
    else:
        fields['remote_status'] = 'onsite'

    # seniority from title (priority order: intern > principal > lead > senior > junior > mid)
    title_lower = fields['title'].lower()
    if any(w in title_lower for w in ['intern', 'internship', 'trainee']):
        fields['seniority'] = 'intern'
    elif any(w in title_lower for w in ['principal', 'staff', 'distinguished']):
        fields['seniority'] = 'principal'
    elif any(w in title_lower for w in ['lead', 'head of', 'director', 'vp', 'chief']):
        fields['seniority'] = 'lead'
    elif any(w in title_lower for w in ['senior', 'sr.', 'sr ']):
        fields['seniority'] = 'senior'
    elif any(w in title_lower for w in ['junior', 'jr.', 'jr ', 'entry']):
        fields['seniority'] = 'junior'
    else:
        fields['seniority'] = 'mid'

    return fields


def build_job_embedding_string(fields):
    """
    Build embedding string from extracted job fields.
    Follows training template format.
    Returns structured string WITHOUT prefix (caller adds prefix).
    """
    parts = []

    title = fields.get('title', 'Unknown Position')
    company = fields.get('company', 'a company')
    location = fields.get('location', '')

    # role part
    role_part = f"Role of {title} at {company}"
    if location:
        role_part += f" in {location}"
    parts.append(role_part + ".")

    # skills - limit to top 10
    skills = fields.get('skills', [])
    if skills:
        skills_str = ', '.join(skills[:10])
        parts.append(f"Required skills: {skills_str}.")

    # seniority with expanded descriptions
    seniority = fields.get('seniority', 'mid')
    seniority_map = {
        'intern': 'Intern level, entry position',
        'junior': 'Junior level, 0-2 years experience',
        'mid': 'Mid-level, 3-5 years experience',
        'senior': 'Senior level, 5+ years experience',
        'lead': 'Lead level, 7+ years experience with leadership',
        'principal': 'Principal level, expert with technical leadership'
    }
    level_desc = seniority_map.get(seniority, seniority)
    parts.append(f"Experience level: {level_desc}.")

    # salary range
    salary_min = fields.get('salary_min')
    salary_max = fields.get('salary_max')
    if salary_min and salary_max:
        parts.append(f"Salary range: ${salary_min:,} to ${salary_max:,}.")

    # remote status
    remote = fields.get('remote_status', '')
    if remote:
        remote_map = {
            'remote': 'Remote work available',
            'hybrid': 'Hybrid work, partially remote',
            'onsite': 'Onsite work'
        }
        work_type = remote_map.get(remote, remote)
        parts.append(f"Work type: {work_type}.")

    return ' '.join(parts)


def prepare_for_biencoder(text, mode):
    """
    Add lowercase prefix for bi-encoder model.
    mode='cv' -> adds 'query: '
    mode='job' -> adds 'passage: '
    Strips existing prefixes first.
    """
    clean = strip_prefixes(text)

    if mode == 'cv':
        return "query: " + clean
    elif mode == 'job':
        return "passage: " + clean
    else:
        return clean


def logistic_percentage(score):
    """
    Convert raw score to percentage using logistic function.
    Formula: 100 / (1 + exp(-0.5 * score))
    """
    return round(100 / (1 + math.exp(-0.5 * score)))


def parse_cv_summary(cv_text):
    """
    Extract role, experience, and skills preview from structured CV text.
    Returns tuple: (role, experience, skills_preview)
    """
    clean = strip_prefixes(cv_text)

    role = "Unknown"
    experience = ""
    skills_preview = ""

    # extract role from "I am a [Role]"
    role_match = re.search(r"I am (?:a |an )?(.+?)(?:\s+with\s+|\s*,|\s*\.)", clean)
    if role_match:
        role = role_match.group(1).strip()

    # extract experience years
    exp_match = re.search(r"(\d+)\s+years?\s+of\s+experience", clean)
    if exp_match:
        experience = f"{exp_match.group(1)}y exp"

    # extract level
    level_match = re.search(r"(entry|junior|mid|senior|lead|principal)\s+level", clean, re.IGNORECASE)
    if level_match:
        experience = f"{level_match.group(1)} / {experience}" if experience else level_match.group(1)

    # extract skills preview
    skills_match = re.search(r"skills include:\s*(.+?)(?:\.|I studied|I worked)", clean)
    if skills_match:
        skills_text = skills_match.group(1).strip()
        skill_list = [s.strip() for s in skills_text.split(",")]
        skills_preview = ", ".join(skill_list[:5])

    return role, experience, skills_preview
