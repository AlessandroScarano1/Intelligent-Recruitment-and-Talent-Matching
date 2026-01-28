# configuration script for feedback signals
# the values represent how much each action is worth

# positive signals = user likes the match
# negative signals = user doesn't like the match

EVENT_SIGNALS = {
    "candidate_apply": 0.5,      # candidate applies to job
    "candidate_save": 0.3,       # candidate saves job for later
    "candidate_view": 0.1,       # candidate views job details
    "recruiter_shortlist": 0.4,  # recruiter adds to shortlist
    "recruiter_interview": 0.7,  # recruiter schedules interview
    "recruiter_hire": 1.0,       # recruiter hires candidate (best signal)
    "recruiter_reject": -0.3,    # recruiter rejects candidate
    "skip": -0.1,                # user skips without action
}

# default weights for the scoring formula
# final_score = alpha*cosine + beta*jaccard + gamma1*title_cat + gamma2*title_sim + delta*feedback
DEFAULT_WEIGHTS = {
    'alpha': 1.0,    # cosine similarity weight
    'beta': 0.5,     # jaccard skill overlap weight
    'gamma1': 0.1,   # title category match weight
    'gamma2': 0.1,   # title similarity weight
    'delta': 0.0,    # feedback weight (starts at 0, then learn)
}
