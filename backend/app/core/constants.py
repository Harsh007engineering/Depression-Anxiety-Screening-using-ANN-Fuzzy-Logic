"""
Clinical Rubric Constants & DASS-21 Questionnaire Definitions
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

DASS21_QUESTIONS = [
    {"id": 1, "subscale": "Stress", "text": "I found it hard to wind down"},
    {"id": 2, "subscale": "Anxiety", "text": "I was aware of dryness of my mouth"},
    {"id": 3, "subscale": "Depression", "text": "I couldn't seem to experience any positive feeling at all"},
    {"id": 4, "subscale": "Anxiety", "text": "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)"},
    {"id": 5, "subscale": "Depression", "text": "I found it difficult to work up the initiative to do things"},
    {"id": 6, "subscale": "Stress", "text": "I tended to over-react to situations"},
    {"id": 7, "subscale": "Anxiety", "text": "I experienced trembling (e.g. in the hands)"},
    {"id": 8, "subscale": "Stress", "text": "I felt that I was using a lot of nervous energy"},
    {"id": 9, "subscale": "Anxiety", "text": "I was worried about situations in which I might panic and make a fool of myself"},
    {"id": 10, "subscale": "Depression", "text": "I felt that I had nothing to look forward to"},
    {"id": 11, "subscale": "Stress", "text": "I found myself getting agitated"},
    {"id": 12, "subscale": "Stress", "text": "I found it difficult to relax"},
    {"id": 13, "subscale": "Depression", "text": "I felt down-hearted and blue"},
    {"id": 14, "subscale": "Stress", "text": "I was intolerant of anything that kept me from getting on with what I was doing"},
    {"id": 15, "subscale": "Anxiety", "text": "I felt I was close to panic"},
    {"id": 16, "subscale": "Depression", "text": "I was unable to become enthusiastic about anything"},
    {"id": 17, "subscale": "Depression", "text": "I felt I wasn't worth much as a person"},
    {"id": 18, "subscale": "Stress", "text": "I felt that I was rather touchy"},
    {"id": 19, "subscale": "Anxiety", "text": "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase)"},
    {"id": 20, "subscale": "Anxiety", "text": "I felt scared without any good reason"},
    {"id": 21, "subscale": "Depression", "text": "I felt that life was meaningless"}
]

SEVERITY_LEVELS = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

CUTOFFS = {
    "Depression": [
        (0, 9, "Normal", 0),
        (10, 13, "Mild", 1),
        (14, 20, "Moderate", 2),
        (21, 27, "Severe", 3),
        (28, 42, "Extremely Severe", 4),
    ],
    "Anxiety": [
        (0, 7, "Normal", 0),
        (8, 9, "Mild", 1),
        (10, 14, "Moderate", 2),
        (15, 19, "Severe", 3),
        (20, 42, "Extremely Severe", 4),
    ],
    "Stress": [
        (0, 14, "Normal", 0),
        (15, 18, "Mild", 1),
        (19, 25, "Moderate", 2),
        (26, 33, "Severe", 3),
        (34, 42, "Extremely Severe", 4),
    ]
}
