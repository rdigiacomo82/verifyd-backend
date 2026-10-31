"""Static safety contract for the isolated Render IDLE worker Blueprint."""
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class WorkerBlueprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = yaml.safe_load(
            (ROOT / "render-circle-mail-idle.yaml").read_text())

    def test_only_one_new_private_worker_and_no_existing_services(self):
        services = self.document["services"]
        self.assertEqual(len(services), 1)
        worker = services[0]
        self.assertEqual(worker["type"], "worker")
        self.assertEqual(worker["name"], "verifyd-circle-mail-idle")
        self.assertEqual(worker["region"], "ohio")
        self.assertEqual(worker["plan"], "starter")
        self.assertEqual(worker["branch"], "main")
        self.assertEqual(worker["buildCommand"],
                         "pip install -r requirements-circle-mail.txt")
        self.assertEqual(worker["startCommand"],
                         "python trust_voice_circle_email_idle.py")
        self.assertEqual(
            worker["repo"], "https://github.com/rdigiacomo82/verifyd-backend")

    def test_secrets_are_never_hardcoded(self):
        values = {x["key"]: x for x in self.document["services"][0]["envVars"]}
        expected = [
            "DATABASE_URL",
            "VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER",
            "VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD",
            "VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED",
            "VERIFYD_CIRCLE_EMAIL_APEX_GATEWAY_ENABLED",
            "VERIFYD_CIRCLE_EMAIL_APEX_TRUSTED_RECEIVED_BY",
            "VERIFYD_CIRCLE_EMAIL_APEX_ROTATING_MX_ENABLED",
        ]
        for key in expected:
            self.assertNotIn("value", values[key])
            self.assertEqual(values[key]["fromService"], {
                "type": "cron", "name": "verifyd-circle-hostinger-inbound",
                "envVarKey": key,
            })
        self.assertEqual(values["VERIFYD_CIRCLE_EMAIL_IDLE_ENABLED"]["value"],
                         "true")
        self.assertEqual(values["VERIFYD_CIRCLE_EMAIL_IDLE_EXPECTED_USER"]["value"],
                         "circleinbound@vfvid.com")
        self.assertEqual(
            values["VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED"]["value"],
            "false")

    def test_no_other_blueprint_resources(self):
        self.assertEqual(set(self.document.keys()), {"services"})


if __name__ == "__main__":
    unittest.main()
