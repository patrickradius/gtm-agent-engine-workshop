import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "test-only-key")

import gtm_agent.gtm_agent as agent_module
from gtm_agent import data_service


PROSPECT_ID = "LEAD-TEST"
OFFERING_ID = "OFFER-TEST"
SENSITIVE_VALUES = ("tax-id-test", "birth-date-test", "card-number-test")


def prospect_record():
    return {
        "name": "Test Prospect",
        "email": "prospect@example.com",
        "disqualified": False,
        "annual_revenue": 75000000,
        "enrichment_source": "Test Source",
        "billing_qualification": {
            "tax_id": SENSITIVE_VALUES[0],
            "date_of_birth": SENSITIVE_VALUES[1],
            "card_on_file": SENSITIVE_VALUES[2],
        },
        "engagement_history": [],
        "account_details": [{"industry": "Software"}],
        "tech_stack": ["Python", "PostgreSQL"],
    }


def serialized(value):
    return json.dumps(value, sort_keys=True)


class SensitiveDataRedactionTests(unittest.TestCase):
    def assert_no_sensitive_data(self, value):
        serialized_value = serialized(value)
        for sensitive_value in SENSITIVE_VALUES:
            self.assertNotIn(sensitive_value, serialized_value)
        for sensitive_key in ("billing_qualification", "tax_id", "date_of_birth", "card_on_file"):
            self.assertNotIn(sensitive_key, serialized_value)

    def test_get_prospect_returns_only_allowlisted_contact_fields(self):
        with patch.object(agent_module.data_service, "get_prospect_record", return_value=prospect_record()):
            result = agent_module.get_prospect.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(result["prospect"], {
            "prospect_id": PROSPECT_ID,
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
            "annual_revenue": 75000000,
            "enrichment_source": "Test Source",
        })
        self.assert_no_sensitive_data(result)

    def test_new_and_cached_profiles_exclude_billing_data(self):
        record = prospect_record()
        with (
            patch.object(agent_module.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(agent_module.data_service, "get_prospect_record", return_value=record),
            patch.object(agent_module.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(agent_module.data_service, "fetch_account_details", return_value=record["account_details"]),
            patch.object(agent_module.data_service, "fetch_tech_stack", return_value=record["tech_stack"]),
            patch.object(agent_module.data_service, "save_profile_to_db") as save_profile,
        ):
            newly_built = agent_module.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(newly_built["prospect_profile"]["annual_revenue"], 75000000)
        self.assertEqual(newly_built["prospect_profile"]["tech_stack"], record["tech_stack"])
        self.assert_no_sensitive_data(newly_built)
        self.assert_no_sensitive_data(save_profile.call_args.args[1])

        cached_profile = {"prospect_id": PROSPECT_ID, **record}
        with (
            patch.object(agent_module.data_service, "get_profile_from_db", return_value={"prospect_profile": cached_profile}),
            patch.object(agent_module.data_service, "save_profile_to_db") as save_cached,
        ):
            cached = agent_module.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(cached["prospect_profile"]["annual_revenue"], 75000000)
        self.assert_no_sensitive_data(cached)
        self.assert_no_sensitive_data(save_cached.call_args.args[1])

    def test_profile_cache_is_sanitized_before_traceable_result(self):
        data_service._PROFILES[PROSPECT_ID] = {"prospect_id": PROSPECT_ID, **prospect_record()}
        try:
            cached = data_service.get_profile_from_db(PROSPECT_ID)
        finally:
            data_service._PROFILES.pop(PROSPECT_ID, None)

        self.assert_no_sensitive_data(cached)

    def test_scoring_accepts_ids_and_serializes_only_sanitized_profile(self):
        record = prospect_record()
        offering = {
            "offering_id": OFFERING_ID,
            "name": "Test Offering",
            "target_segment": "Enterprise",
            "required_tech_stack": ["PostgreSQL"],
            "min_annual_revenue": 50000000,
            "description": "Test offering",
        }
        result = SimpleNamespace(model_dump=lambda: {"score": 91})
        scoring_llm = SimpleNamespace(invoke=Mock(return_value=result))
        with (
            patch.object(agent_module.data_service, "get_offering", return_value=offering),
            patch.object(agent_module.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(agent_module.data_service, "get_prospect_record", return_value=record),
            patch.object(agent_module.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(agent_module.data_service, "fetch_account_details", return_value=record["account_details"]),
            patch.object(agent_module.data_service, "fetch_tech_stack", return_value=record["tech_stack"]),
            patch.object(agent_module.data_service, "save_profile_to_db"),
            patch.object(agent_module, "_scoring_llm", scoring_llm),
        ):
            scored = agent_module.score_prospect.invoke({
                "prospect_id": PROSPECT_ID,
                "offering_id": OFFERING_ID,
            })

        self.assertEqual(scored, {"score": 91})
        self.assertEqual(set(agent_module.score_prospect.args_schema.model_fields), {"prospect_id", "offering_id"})
        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assert_no_sensitive_data(prompt)
        self.assertIn('"annual_revenue": 75000000', prompt)
        self.assertIn('"tech_stack": [', prompt)
        self.assertIn('"required_tech_stack": [', prompt)


if __name__ == "__main__":
    unittest.main()
