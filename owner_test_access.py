"""One-time owner QA grant, authorized October 30, 2026. Never modifies billing."""
import time
OWNER_TEST_EMAIL = "ezanaberhe@gmail.com"
OWNER_TEST_EXPIRES = 1793464872

def owner_test_until(user):
    if (user.get("email") == OWNER_TEST_EMAIL and user.get("verified") is True
            and time.time() < OWNER_TEST_EXPIRES):
        return OWNER_TEST_EXPIRES
    return 0
