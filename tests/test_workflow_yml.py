"""
Unit tests for .github/workflows/azure-deploy.yml and .github/workflows/ci.yml.

Validates:
  1. YAML parses correctly (regression guard for syntax errors like unindented continuations)
  2. Expected jobs exist with correct dependencies
  3. Trigger events, environment variables, runner OS
  4. Health check & smoke-test shell commands (including the fix for the echo line)
  5. Docker build / push commands
"""

import json
import sys
import yaml
from pathlib import Path

import pytest
import requests
from jsonschema import validate as validate_against_schema, ValidationError

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "azure-deploy.yml"
CI_WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


# ==============================================================================
# ── Helper ────────────────────────────────────────────────────────────────────
# ==============================================================================


def normalize_needs(needs):
    """GitHub Actions allows `needs: test` (string) or `needs: [test]` (list).

    Normalize to a list for consistent comparison.
    """
    if needs is None:
        return []
    if isinstance(needs, str):
        return [needs]
    return list(needs)


def get_triggers(workflow) -> dict:
    """Extract the 'on' trigger configuration from the parsed YAML.

    PyYAML treats the bare YAML key 'on' as a boolean, so it's stored
    under the Python key `True` rather than the string 'on'. This function
    checks both possibilities.
    """
    on = workflow.get("on") or workflow.get(True)
    if on is None:
        return {}
    return on


# ==============================================================================
# ── Fixtures ──────────────────────────────────────────────────────────────────
# ==============================================================================


# ==============================================================================
# ── Schema validation helpers ─────────────────────────────────────────────────
# ==============================================================================


SCHEMA_CACHE_PATH = Path(__file__).resolve().parent / ".github_workflow_schema.json"
SCHEMA_URL = "https://json.schemastore.org/github-workflow.json"


def _normalize_for_schema(raw_data):
    """Replace boolean-True YAML key 'on' with the string 'on'.

    PyYAML interprets the unquoted YAML key 'on' as Python boolean True.
    The GitHub JSON schema expects the string key 'on'. This function
    walks the parsed data structure and fixes this.
    """
    def _walk(d):
        if isinstance(d, dict):
            return {("on" if k is True else k): _walk(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_walk(i) for i in d]
        return d
    return _walk(raw_data)


@pytest.fixture(scope="session")
def github_actions_schema():
    """Fetch the GitHub Actions JSON schema from SchemaStore.

    Cached locally to avoid repeated downloads. The cache file
    is gitignored and can be cleared by deleting it.
    """
    if SCHEMA_CACHE_PATH.is_file():
        with open(SCHEMA_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    try:
        resp = requests.get(SCHEMA_URL, timeout=15)
        resp.raise_for_status()
        schema = resp.json()
    except (requests.ConnectionError, requests.Timeout) as e:
        pytest.skip(f"Cannot reach SchemaStore ({e}) — skipping schema validation")

    with open(SCHEMA_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f)

    return schema


# ==============================================================================
# ── Fixtures ──────────────────────────────────────────────────────────────────
# ==============================================================================


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Parse the YAML workflow file once for all tests."""
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def jobs(workflow) -> dict:
    """Shortcut to the jobs section."""
    return workflow.get("jobs", {})


# ==============================================================================
# ── YAML validity ─────────────────────────────────────────────────────────────
# ==============================================================================


class TestYamlValidity:
    """The workflow must be valid YAML — the most critical test after the fix."""

    def test_file_exists(self):
        """Workflow file exists at the expected path."""
        assert WORKFLOW_PATH.is_file(), f"Workflow file not found: {WORKFLOW_PATH}"

    def test_parses_as_valid_yaml(self, workflow):
        """yaml.safe_load returns a dict (not None) — i.e. no syntax error."""
        assert isinstance(workflow, dict), "YAML did not parse to a dict"

    def test_workflow_name(self, workflow):
        """The workflow name is 'Azure Deploy'."""
        assert workflow.get("name") == "Azure Deploy"

    def test_no_parse_errors_since_last_fix(self):
        """Re-parse the file to catch any remaining YAML issues.

        This is the test that would have caught the original unindented
        continuation-line bug at lines 312-313.
        """
        with open(WORKFLOW_PATH, encoding="utf-8") as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"YAML parse error: {e}")


# ==============================================================================
# ── Trigger events ────────────────────────────────────────────────────────────
# ==============================================================================


class TestTriggers:
    """The workflow must trigger on the correct events."""

    def test_triggers_on_push_to_main(self, workflow):
        """Push to main triggers the workflow."""
        on = get_triggers(workflow)
        assert isinstance(on, dict), (
            f"Expected 'on' to be a dict, got {type(on)}: {on}. "
            "(YAML parser may have converted 'on' to boolean True)"
        )
        push = on.get("push", None)
        assert push is not None, f"Missing 'push' key in triggers: {json.dumps(on, default=str)}"
        branches = push.get("branches", [])
        assert "main" in branches, f"Expected 'main' in push branches, got {branches}"

    def test_triggers_on_workflow_dispatch(self, workflow):
        """Manual trigger via GitHub UI is allowed."""
        on = get_triggers(workflow)
        assert isinstance(on, dict), (
            f"Expected 'on' to be a dict, got {type(on)}: {on}. "
            "(YAML parser may have converted 'on' to boolean True)"
        )
        assert "workflow_dispatch" in on, (
            f"workflow_dispatch trigger missing in triggers: {json.dumps(on, default=str)}"
        )


# ==============================================================================
# ── Job structure ─────────────────────────────────────────────────────────────
# ==============================================================================


class TestJobStructure:
    """All expected jobs exist with the correct names."""

    EXPECTED_JOBS = [
        "test",
        "build-and-push-api",
        "build-and-push-dashboard",
        "deploy-api",
        "deploy-dashboard",
        "post-deploy-health-check",
    ]

    def test_all_expected_jobs_present(self, jobs):
        """All 6 expected jobs are defined."""
        for job_name in self.EXPECTED_JOBS:
            assert job_name in jobs, f"Missing expected job: {job_name}"

    def test_no_unexpected_jobs(self, jobs):
        """No extra jobs beyond the expected 6."""
        extra = set(jobs.keys()) - set(self.EXPECTED_JOBS)
        assert not extra, f"Unexpected jobs: {extra}"

    def test_all_jobs_run_on_ubuntu(self, jobs):
        """Every job runs on ubuntu-latest."""
        for job_name, job_config in jobs.items():
            assert job_config.get("runs-on") == "ubuntu-latest", (
                f"{job_name}: expected ubuntu-latest, got {job_config.get('runs-on')}"
            )


# ==============================================================================
# ── Job dependencies ──────────────────────────────────────────────────────────
# ==============================================================================


class TestJobDependencies:
    """The needs chains must be correct for sequential deployment."""

    @pytest.mark.parametrize(
        "job_name, expected_needs",
        [
            ("test", []),                      # no needs
            ("build-and-push-api", ["test"]),
            ("build-and-push-dashboard", ["test"]),
            ("deploy-api", ["build-and-push-api"]),
            ("deploy-dashboard", ["build-and-push-dashboard", "deploy-api"]),
            ("post-deploy-health-check", ["deploy-api", "deploy-dashboard"]),
        ],
    )
    def test_job_dependencies(self, jobs, job_name, expected_needs):
        """Each job has the correct dependencies.

        GitHub Actions accepts `needs: test` (string) or `needs: [test]` (list).
        We normalize both to a list for comparison.
        """
        job = jobs[job_name]
        actual = normalize_needs(job.get("needs"))
        assert actual == expected_needs, (
            f"{job_name}: expected needs={expected_needs}, got {actual}"
        )


# ==============================================================================
# ── Step names ────────────────────────────────────────────────────────────────
# ==============================================================================


class TestStepNames:
    """Key step names are correct (regression guard for accidental renames)."""

    def test_test_job_steps(self, jobs):
        """The test job has the expected step names (some steps use 'uses' without 'name')."""
        steps = jobs["test"]["steps"]
        # Steps without explicit 'name' key (just 'uses') don't have a name field
        names = set()
        for s in steps:
            if "name" in s:
                names.add(s["name"])
        assert "Cache pip" in names
        assert "Install dependencies" in names
        assert "Lint" in names
        assert "Unit tests" in names

    def test_deploy_api_steps(self, jobs):
        """The deploy-api job has the expected step names."""
        steps = jobs["deploy-api"]["steps"]
        step_names = [s["name"] for s in steps]
        assert "Azure login" in step_names
        assert "Update App Service image" in step_names
        assert "Update environment variables" in step_names
        assert "Wait for deployment and smoke test" in step_names

    def test_post_deploy_steps(self, jobs):
        """The post-deploy job has the expected step names."""
        steps = jobs["post-deploy-health-check"]["steps"]
        step_names = [s["name"] for s in steps]
        assert "Check API health endpoint" in step_names
        assert "Check Dashboard health endpoint" in step_names
        assert "Smoke test — API query" in step_names
        assert "Post-deployment summary" in step_names


# ==============================================================================
# ── Environment variables ─────────────────────────────────────────────────────
# ==============================================================================


class TestEnvironmentVariables:
    """Key env vars are set at the workflow and job levels."""

    def test_workflow_level_env_vars(self, workflow):
        """Workflow-level env vars include resource group and app names."""
        env = workflow.get("env", {})
        assert env.get("RESOURCE_GROUP") == "healthcare-rag-rg"
        assert env.get("APP_NAME") == "healthcare-rag-app"
        assert env.get("DASHBOARD_NAME") == "healthcare-rag-dashboard"

    def test_workflow_level_has_acr_name(self, workflow):
        """ACR_NAME references the AZURE_ACR_NAME secret."""
        env = workflow.get("env", {})
        acr = env.get("ACR_NAME", "")
        assert "${{" in str(acr) and "AZURE_ACR_NAME" in str(acr), (
            f"ACR_NAME should reference AZURE_ACR_NAME secret, got: {acr}"
        )


# ==============================================================================
# ── Health check shell commands ───────────────────────────────────────────────
# ==============================================================================


class TestHealthCheckShell:
    """Validate the shell commands used in health checks."""

    def test_api_health_check_retry_count(self, jobs):
        """The API health check uses 20 retries (10 min window)."""
        run = jobs["deploy-api"]["steps"][3]["run"]
        assert "{1..20}" in run, "Expected 20 retries in API health check for-loop"

    def test_dashboard_health_check_retry_count(self, jobs):
        """The dashboard health check uses 12 retries (6 min window)."""
        run = jobs["deploy-dashboard"]["steps"][4]["run"]
        assert "{1..12}" in run, (
            "Expected 12 retries in dashboard health check for-loop"
        )

    def test_api_health_check_sleep_and_exit(self, jobs):
        """API health check sleeps 30s between retries and exits 1 on failure."""
        run = jobs["deploy-api"]["steps"][3]["run"]
        assert "sleep 30" in run
        assert "exit 1" in run

    def test_api_smoke_test_parses_json(self, jobs):
        """The API smoke test parses JSON responses for answer/category/sources."""
        # Smoke test step is index 2 in post-deploy-health-check
        run = jobs["post-deploy-health-check"]["steps"][2]["run"]
        assert "json.load" in run, "Smoke test should parse JSON response"
        assert "answer" in run, "Should check for answer field"
        assert "category" in run, "Should check for category field"
        assert "source_citations" in run, "Should check for source_citations field"

    def test_api_smoke_test_curl_command(self, jobs):
        """The smoke test POSTs a medical question to /query."""
        run = jobs["post-deploy-health-check"]["steps"][2]["run"]
        assert "POST" in run, "Should use POST method"
        assert "/query" in run, "Should POST to /query endpoint"
        assert "Does aspirin reduce cardiovascular risk" in run, (
            "Should include the default smoke-test question"
        )


# ==============================================================================
# ── Docker build / push commands ──────────────────────────────────────────────
# ==============================================================================


class TestDockerCommands:
    """Docker build and push commands must reference the correct Dockerfiles."""

    def test_api_dockerfile(self, jobs):
        """API image uses Dockerfile (not Dockerfile.dashboard)."""
        run = jobs["build-and-push-api"]["steps"][3]["run"]
        assert "docker/Dockerfile" in run
        assert "healthcare-rag" in run
        assert "docker/Dockerfile.dashboard" not in run

    def test_dashboard_dockerfile(self, jobs):
        """Dashboard image uses Dockerfile.dashboard."""
        run = jobs["build-and-push-dashboard"]["steps"][3]["run"]
        assert "docker/Dockerfile.dashboard" in run
        assert "healthcare-rag-dashboard" in run


class TestDockerPushCommands:
    """Docker push commands exist for both images."""

    def test_api_docker_push(self, jobs):
        """API build step pushes both :sha and :latest tags."""
        run = jobs["build-and-push-api"]["steps"][3]["run"]
        assert "docker push" in run
        assert "latest" in run
        assert "github.sha" in run or "${{ github.sha }}" in run


# ==============================================================================
# ── Shell syntax validation (the echo fix) ────────────────────────────────────
# ==============================================================================


class TestPostDeploySummaryShell:
    """Validate the post-deployment summary echo command.

    This specifically guards against the YAML continuation-line bug
    that was fixed (lines 312-313).
    """

    def test_post_deploy_summary_echo_is_single_line(self, jobs):
        """The post-deployment summary echo for API query status should be a
        single echo statement (no YAML continuation lines that could break)."""
        run = jobs["post-deploy-health-check"]["steps"][3]["run"]
        lines = [l.strip() for l in run.split("\n")]
        query_lines = [l for l in lines if "API query" in l and "answer=" in l]
        assert len(query_lines) == 1, (
            f"Expected exactly 1 echo line for API query status, got {len(query_lines)}.\n"
            f"This would indicate the YAML continuation-line bug regressed.\n"
            f"Lines found: {query_lines}"
        )

    def test_post_deploy_summary_echo_has_all_fields(self, jobs):
        """The single echo line contains answer, category, and sources fields."""
        run = jobs["post-deploy-health-check"]["steps"][3]["run"]
        lines = [l.strip() for l in run.split("\n")]
        query_line = next((l for l in lines if "API query" in l and "answer=" in l), "")
        assert "answer=" in query_line, "Missing answer field in echo"
        assert "category=" in query_line, "Missing category field in echo"
        assert "sources=" in query_line, "Missing sources field in echo"

    def test_post_deploy_summary_echo_no_trailing_backslash_quote(self, jobs):
        """The echo line does not contain trailing escaped quotes from a
        broken multi-line continuation (the original bug)."""
        run = jobs["post-deploy-health-check"]["steps"][3]["run"]
        lines = [l.strip() for l in run.split("\n")]
        query_line = next((l for l in lines if "API query" in l and "answer=" in l), "")
        assert not query_line.rstrip().endswith('\\"'), (
            f"Echo line ends with trailing \\\" — this is the old continuation-line bug:\n"
            f"  {query_line}"
        )

    def test_post_deploy_summary_echo_uses_if_then_else(self, jobs):
        """The echo is inside an if/elif/else block that handles ok, connection-failed, and fallback."""
        run = jobs["post-deploy-health-check"]["steps"][3]["run"]
        assert 'if' in run and 'then' in run and 'elif' in run and 'else' in run, (
            "Expected if/elif/else branching for API query status"
        )


# ==============================================================================
# ── post-deploy-health-check behavior ─────────────────────────────────────────
# ==============================================================================


class TestPostDeployJobBehavior:
    """The post-deploy job must always run (even when upstream jobs fail)."""

    def test_post_deploy_always_runs(self, jobs):
        """post-deploy-health-check uses if: always() so dashboard warnings
        don't block the final health report."""
        post = jobs["post-deploy-health-check"]
        assert post.get("if") == "always()", (
            f"Post-deploy health check should use if: always(), got {post.get('if')}"
        )


# ==============================================================================
# ── Azure App Settings ────────────────────────────────────────────────────────
# ==============================================================================


class TestAzureAppSettings:
    """App Service environment variable settings for API and Dashboard."""

    def test_api_has_required_app_settings(self, jobs):
        """API App Service config has the required environment variables."""
        run = jobs["deploy-api"]["steps"][2]["run"]
        assert "GROQ_API_KEY" in run
        assert "HF_TOKEN" in run
        assert "AZURE_APP_URL" in run
        assert "WEBSITES_PORT" in run
        assert '8000' in run, "API must specify WEBSITES_PORT=8000"

    def test_dashboard_has_required_app_settings(self, jobs):
        """Dashboard App Service config has the required environment variables."""
        run = jobs["deploy-dashboard"]["steps"][3]["run"]
        assert "AZURE_APP_URL" in run
        assert "DASHBOARD_URL" in run
        assert "WEBSITES_PORT" in run
        assert '8501' in run, "Dashboard must specify WEBSITES_PORT=8501"


# ==============================================================================
# ── Build & push outputs ──────────────────────────────────────────────────────
# ==============================================================================


class TestBuildOutputs:
    """build-and-push-api outputs the ACR login server for use by deploy-api."""

    def test_api_build_outputs_login_server(self, jobs):
        """build-and-push-api defines an output 'acr-login-server'."""
        outputs = jobs["build-and-push-api"].get("outputs", {})
        assert "acr-login-server" in outputs, (
            "Missing acr-login-server output needed by deploy-api"
        )

    def test_deploy_api_uses_acr_output(self, jobs):
        """deploy-api references the acr-login-server output from build-and-push-api."""
        run = jobs["deploy-api"]["steps"][1]["run"]
        assert "acr-login-server" in run, (
            "deploy-api should reference needs.build-and-push-api.outputs.acr-login-server"
        )


# ==============================================================================
# ── GitHub Actions JSON Schema validation ─────────────────────────────────────
# ==============================================================================


class TestSchemaValidation:
    """Validate all workflow YAMLs against GitHub's official JSON schema.

    Uses the SchemaStore-hosted schema at:
    https://json.schemastore.org/github-workflow.json

    This catches structural issues that YAML-only parsing would miss —
    e.g., invalid property names, wrong types, or missing required fields
    defined by the GitHub Actions workflow specification.
    """

    def _check_schema(self, name, raw_data, schema):
        """Validate a single workflow dict against the schema."""
        normalized = _normalize_for_schema(raw_data)
        try:
            validate_against_schema(instance=normalized, schema=schema)
        except ValidationError as e:
            path = " → ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            pytest.fail(
                f"Schema validation failed for {name} at {path}:\n"
                f"  {e.message}\n"
                f"  Expected: {e.validator_value if hasattr(e, 'validator_value') else 'N/A'}\n"
                f"  Got: {e.instance}"
            )

    @pytest.mark.network
    def test_azure_deploy_validates(self, workflow, github_actions_schema):
        """Azure Deploy workflow validates against GitHub's schema."""
        self._check_schema("azure-deploy.yml", workflow, github_actions_schema)

    @pytest.mark.network
    def test_ci_workflow_validates(self, ci_workflow, github_actions_schema):
        """CI workflow validates against GitHub's schema."""
        self._check_schema("ci.yml", ci_workflow, github_actions_schema)


# ==============================================================================
# ── Workflow documentation header ─────────────────────────────────────────────
# ==============================================================================


class TestWorkflowDocumentation:
    """Workflow file has the required secrets documentation in its header comments."""

    def test_header_documents_required_secrets(self):
        """The workflow file header documents all 4 required secrets."""
        with open(WORKFLOW_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "GROQ_API_KEY" in content, "Missing GROQ_API_KEY doc"
        assert "HF_TOKEN" in content, "Missing HF_TOKEN doc"
        assert "AZURE_ACR_NAME" in content, "Missing AZURE_ACR_NAME doc"
        assert "AZURE_CREDENTIALS" in content, "Missing AZURE_CREDENTIALS doc"


# ==============================================================================
# ── CI workflow ───────────────────────────────────────────────────────────────
# ==============================================================================


# ==============================================================================
# ── Workflow coverage / discovery ─────────────────────────────────────────────
# ==============================================================================


class TestWorkflowCoverage:
    """Ensure every workflow file in .github/workflows/ has corresponding tests.

    This is a discovery/coverage guard: if a new workflow file is added but no
    tests are written for it, this test fails with a clear message listing the
    untested file(s).
    """

    WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    COVERED = {"azure-deploy.yml", "ci.yml"}

    def test_all_workflow_files_are_covered(self):
        """Every .yml/.yaml file in .github/workflows/ must be in COVERED."""
        discovered = set()
        for f in self.WORKFLOWS_DIR.iterdir():
            if f.suffix in (".yml", ".yaml"):
                discovered.add(f.name)

        untested = discovered - self.COVERED
        assert not untested, (
            f"Found {len(untested)} untested workflow file(s):\n"
            + "\n".join(f"  - {name}" for name in sorted(untested)) + "\n"
            f"Add tests for them and include their name(s) in "
            f"TestWorkflowCoverage.COVERED."
        )

        # Warn if a covered file has been deleted
        missing = self.COVERED - discovered
        if missing:
            pytest.fail(
                f"Covered workflow file(s) no longer exist:\n"
                + "\n".join(f"  - {name}" for name in sorted(missing)) + "\n"
                f"Remove them from COVERED if intentionally deleted."
            )


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    """Parse the CI workflow file once for all tests."""
    with open(CI_WORKFLOW_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def ci_jobs(ci_workflow) -> dict:
    """Shortcut to the jobs section of the CI workflow."""
    return ci_workflow.get("jobs", {})


# ==============================================================================
# ── CI: YAML validity ─────────────────────────────────────────────────────────
# ==============================================================================


class TestCiYamlValidity:
    """The CI workflow must be valid YAML."""

    def test_ci_file_exists(self):
        """CI workflow file exists at the expected path."""
        assert CI_WORKFLOW_PATH.is_file(), f"CI workflow file not found: {CI_WORKFLOW_PATH}"

    def test_ci_parses_as_valid_yaml(self, ci_workflow):
        """yaml.safe_load returns a dict (not None) — i.e. no syntax error."""
        assert isinstance(ci_workflow, dict), "CI YAML did not parse to a dict"

    def test_ci_workflow_name(self, ci_workflow):
        """The workflow name is 'CI'."""
        assert ci_workflow.get("name") == "CI"

    def test_ci_no_parse_errors(self):
        """Re-parse to catch any YAML issues."""
        with open(CI_WORKFLOW_PATH, encoding="utf-8") as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"CI YAML parse error: {e}")


# ==============================================================================
# ── CI: Trigger events ────────────────────────────────────────────────────────
# ==============================================================================


class TestCiTriggers:
    """The CI workflow must trigger on the correct events."""

    def test_ci_triggers_on_push_to_main(self, ci_workflow):
        """Push to main triggers the workflow."""
        on = get_triggers(ci_workflow)
        assert isinstance(on, dict), (
            f"Expected 'on' to be a dict, got {type(on)}: {on}"
        )
        push = on.get("push", {})
        branches = push.get("branches", [])
        assert "main" in branches, f"Expected 'main' in push branches, got {branches}"

    def test_ci_triggers_on_push_to_develop(self, ci_workflow):
        """Push to develop triggers the workflow."""
        on = get_triggers(ci_workflow)
        push = on.get("push", {})
        branches = push.get("branches", [])
        assert "develop" in branches, f"Expected 'develop' in push branches, got {branches}"

    def test_ci_triggers_on_pull_request_to_main(self, ci_workflow):
        """Pull request to main triggers the workflow."""
        on = get_triggers(ci_workflow)
        pr = on.get("pull_request", {})
        branches = pr.get("branches", [])
        assert "main" in branches, f"Expected 'main' in pull_request branches, got {branches}"


# ==============================================================================
# ── CI: Job structure ─────────────────────────────────────────────────────────
# ==============================================================================


class TestCiJobStructure:
    """All expected CI jobs exist with the correct names."""

    EXPECTED_JOBS = ["lint-and-test", "docker-build"]

    def test_ci_all_expected_jobs_present(self, ci_jobs):
        """Both expected CI jobs are defined."""
        for job_name in self.EXPECTED_JOBS:
            assert job_name in ci_jobs, f"Missing expected CI job: {job_name}"

    def test_ci_no_unexpected_jobs(self, ci_jobs):
        """No extra jobs beyond the expected 2."""
        extra = set(ci_jobs.keys()) - set(self.EXPECTED_JOBS)
        assert not extra, f"Unexpected CI jobs: {extra}"

    def test_ci_all_jobs_run_on_ubuntu(self, ci_jobs):
        """Every CI job runs on ubuntu-latest."""
        for job_name, job_config in ci_jobs.items():
            assert job_config.get("runs-on") == "ubuntu-latest", (
                f"{job_name}: expected ubuntu-latest, got {job_config.get('runs-on')}"
            )


# ==============================================================================
# ── CI: Job dependencies ──────────────────────────────────────────────────────
# ==============================================================================


class TestCiJobDependencies:
    """The CI job dependency chain must be correct."""

    @pytest.mark.parametrize(
        "job_name, expected_needs",
        [
            ("lint-and-test", []),
            ("docker-build", ["lint-and-test"]),
        ],
    )
    def test_ci_job_dependencies(self, ci_jobs, job_name, expected_needs):
        """Each CI job has the correct dependencies."""
        job = ci_jobs[job_name]
        actual = normalize_needs(job.get("needs"))
        assert actual == expected_needs, (
            f"{job_name}: expected needs={expected_needs}, got {actual}"
        )


# ==============================================================================
# ── CI: Step names ────────────────────────────────────────────────────────────
# ==============================================================================


class TestCiStepNames:
    """Key step names in the CI workflow are correct."""

    def test_ci_lint_and_test_job_steps(self, ci_jobs):
        """The lint-and-test job has the expected step names."""
        steps = ci_jobs["lint-and-test"]["steps"]
        names = set()
        for s in steps:
            if "name" in s:
                names.add(s["name"])
        assert "Cache pip" in names
        assert "Install dependencies" in names
        assert "Lint (flake8)" in names, "Expected Lint (flake8) step"
        assert "Unit tests — API & preprocessing" in names
        assert "Unit tests — Data modules & metrics" in names
        assert "Unit tests — Pipeline orchestration" in names
        assert "Unit tests — RAG modules (fully mocked)" in names
        assert "Coverage gate (unit tests only)" in names

    def test_ci_docker_build_job_steps(self, ci_jobs):
        """The docker-build job has the expected step."""
        steps = ci_jobs["docker-build"]["steps"]
        names = [s["name"] for s in steps if "name" in s]
        assert "Build Docker image (smoke test)" in names


# ==============================================================================
# ── CI: Docker build command ──────────────────────────────────────────────────
# ==============================================================================


class TestCiDockerBuild:
    """Docker build in CI must reference the correct Dockerfile and tag."""

    def test_ci_docker_build_uses_correct_dockerfile(self, ci_jobs):
        """docker-build uses docker/Dockerfile (not Dockerfile.dashboard)."""
        run = ci_jobs["docker-build"]["steps"][1]["run"]
        assert "docker/Dockerfile" in run, "Should use docker/Dockerfile"
        assert "docker/Dockerfile.dashboard" not in run, "Should not use dashboard Dockerfile"

    def test_ci_docker_build_tags_with_commit_sha(self, ci_jobs):
        """docker-build tags the image with the commit SHA."""
        run = ci_jobs["docker-build"]["steps"][1]["run"]
        assert "github.sha" in run, "Should tag with github.sha"
        assert "ci-" in run, "Tag should include ci- prefix"

    def test_ci_docker_build_image_name(self, ci_jobs):
        """docker-build uses the correct image name."""
        run = ci_jobs["docker-build"]["steps"][1]["run"]
        assert "healthcare-rag:" in run, "Should use healthcare-rag image name"


# ==============================================================================
# ── CI: Branch condition ──────────────────────────────────────────────────────
# ==============================================================================


class TestCiBranchCondition:
    """The docker-build job must only run on the main branch."""

    def test_ci_docker_build_only_on_main(self, ci_jobs):
        """docker-build is conditional on github.ref == refs/heads/main."""
        job = ci_jobs["docker-build"]
        condition = job.get("if", "")
        assert "refs/heads/main" in condition, (
            f"Expected docker-build to be conditional on main branch, got: {condition}"
        )


# ==============================================================================
# ── Actionlint — static analysis ─────────────────────────────────────────────
# ==============================================================================


_BIN_DIR = Path(__file__).resolve().parents[1] / ".bin"
ACTIONLINT_BIN = _BIN_DIR / ("actionlint.exe" if sys.platform == "win32" else "actionlint")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestActionlint:
    """Run actionlint static analysis on all GitHub Actions workflow files.

    Actionlint is a linter for GitHub Actions workflow files that catches
    issues beyond YAML syntax — incorrect action references, shell injection
    risks, invalid property usage, wrong event types, and more.

    See: https://github.com/rhysd/actionlint
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_actionlint(self):
        """Skip all tests in this class if actionlint binary is not found."""
        if not ACTIONLINT_BIN.is_file():
            pytest.skip(
                f"actionlint not found at {ACTIONLINT_BIN}. "
                "Install via:\n"
                "  pip install actionlint-py  (recommended)\n"
                "  or download from https://github.com/rhysd/actionlint/releases"
            )

    @pytest.mark.actionlint
    @pytest.mark.parametrize(
        "workflow_file",
        [
            "azure-deploy.yml",
            "ci.yml",
        ],
    )
    def test_workflow_passes_actionlint(self, workflow_file):
        """Run actionlint on a single workflow file and fail on any issues."""
        import subprocess

        workflow_path = PROJECT_ROOT / ".github" / "workflows" / workflow_file
        assert workflow_path.is_file(), f"Workflow file not found: {workflow_path}"

        result = subprocess.run(
            [str(ACTIONLINT_BIN), str(workflow_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )

        if result.returncode != 0:
            header = f"❌ actionlint found issues in {workflow_file}"
            details = result.stdout.strip() or result.stderr.strip() or "(no output)"
            pytest.fail(f"{header}:\n{details}")


