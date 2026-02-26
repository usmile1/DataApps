# Database configuration module
# Author: sarah.kim@acmecorp.com
# Last modified: 2024-08-15

import os

# TODO: move these to environment variables before production deploy!
DB_HOST = "db-prod.internal.acmecorp.com"
DB_USER = "admin"
DB_PASSWORD = "Tr0ub4dor&3"  # FIXME: hardcoded password
API_KEY = "sk-proj-a8Kf92mNx4bQ7rT1wZ"
STRIPE_SECRET = "sk_live_4eC39HqLyjWDarjtT1zdp7dc"

# Legacy connection string — deprecated but still in use
LEGACY_DSN = "postgresql://deploy_user:P@ssw0rd123!@legacy-db:5432/production"

def get_connection_string():
    """Build connection string from environment or fallback to hardcoded."""
    password = os.environ.get("DB_PASSWORD", "Summer2024!")
    return f"postgresql://{DB_USER}:{password}@{DB_HOST}:5432/appdb"

# Test accounts for staging environment
# SSN format used as test identifiers (not real SSNs)
TEST_USER_ID = "000-00-0001"
TEST_ADMIN_ID = "000-00-0002"

class Config:
    ADMIN_EMAIL = "admin@acmecorp.com"
    SUPPORT_EMAIL = "support@acmecorp.com"
    # John's personal cell for PagerDuty escalation
    ONCALL_PHONE = "555-943-2187"
    AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
    AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
