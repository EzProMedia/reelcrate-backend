import copy
import owner_test_access as access
from test_regressions import client, isolated
import auth, billing

def test_owner_only_and_exact_expiry(monkeypatch):
    monkeypatch.setattr(access.time, 'time', lambda: access.OWNER_TEST_EXPIRES - 1)
    owner = {'email': access.OWNER_TEST_EMAIL, 'verified': True,
             'subscription_status': 'canceled', 'subscription_current_period_end': 1}
    original = copy.deepcopy(owner)
    auth._save_users({owner['email']: owner})
    assert billing.is_paying(owner['email'])
    assert auth._public(owner)['owner_test_access_until'] == access.OWNER_TEST_EXPIRES
    assert auth._public(owner)['subscription_status'] == 'canceled'
    assert auth._load_users()[owner['email']] == original
    assert access.owner_test_until({**owner, 'verified': False}) == 0
    assert access.owner_test_until({**owner, 'email': 'other@example.test'}) == 0
    monkeypatch.setattr(access.time, 'time', lambda: access.OWNER_TEST_EXPIRES)
    assert not billing.is_paying(owner['email'])
    assert auth._public(owner)['owner_test_access_until'] == 0

def test_cancel_webhook_preserves_real_billing_and_qa_expiry(monkeypatch):
    monkeypatch.setattr(access.time, 'time', lambda: access.OWNER_TEST_EXPIRES - 1)
    email = access.OWNER_TEST_EMAIL
    auth._save_users({email: {'email': email, 'verified': True, 'stripe_customer_id': 'cus_owner'}})
    billing._apply_subscription({'id': 'sub_owner', 'customer': 'cus_owner',
        'metadata': {'reelcrate_email': email}, 'status': 'canceled', 'current_period_end': 1})
    assert auth._load_users()[email]['subscription_status'] == 'canceled'
    assert billing.is_paying(email)
    monkeypatch.setattr(access.time, 'time', lambda: access.OWNER_TEST_EXPIRES + 1)
    assert not billing.is_paying(email)
