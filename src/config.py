BUDGET_HOURS = 20.0
MIN_ALLOCATION_HOURS = 0.5
MAX_OBJECTS = 50
OUTPUT_DIR = "outputs"
SENTRY_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"
CAD_URL = "https://ssd-api.jpl.nasa.gov/cad.api"
PARAMETER_SETS = {
    "test1": {
        "a0": 0.75,
        "alpha_q": 0.75,
        "alpha_c": 0.25,
        "u0": 1.0,
        "beta_q": 3.0,
        "beta_r": 1.0
    },
    "baseline": {
        "a0": 0.75,
        "alpha_q": 1.25,
        "alpha_c": 0.50,
        "u0": 1.0,
        "beta_q": 4.0,
        "beta_r": 1.5
    }
}
ACTIVE_PARAMETER_SET = "baseline"
