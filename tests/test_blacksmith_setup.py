from pathlib import Path


def test_ci_uses_blacksmith_for_openclaw_with_fork_fallback():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "blacksmith-4vcpu-ubuntu-2404" in workflow
    assert "ubuntu-latest" in workflow
    assert "github.repository_owner == 'openclaw'" in workflow


def test_testbox_workflow_hydrates_secrets_and_dotfiles():
    workflow = Path(".github/workflows/ci-check-testbox.yml").read_text(encoding="utf-8")

    assert "useblacksmith/begin-testbox@v2" in workflow
    assert "useblacksmith/run-testbox@v2" in workflow
    assert "scripts/ci-hydrate-testbox-env.sh" in workflow
    assert "HF_TOKEN" in workflow
    assert "OPENCLAW_CODEX_AUTH_JSON" in workflow
    assert "CLAWBENCH_CODEX_AUTH_JSON" in workflow


def test_crabbox_config_uses_actions_hydration():
    config = Path(".crabbox.yaml").read_text(encoding="utf-8")

    assert "profile: clawbench-check" in config
    assert "provider: aws" in config
    assert "workflow: .github/workflows/crabbox-hydrate.yml" in config
    assert "job: hydrate" in config
    assert "baseRef: main" in config
    assert "- clawbench" in config
    assert "- CLAWBENCH_*" in config
    assert "- OPENCLAW_*" in config


def test_crabbox_workflow_hydrates_secrets_dotfiles_and_ready_marker():
    workflow = Path(".github/workflows/crabbox-hydrate.yml").read_text(encoding="utf-8")

    assert "crabbox_id:" in workflow
    assert "crabbox_runner_label:" in workflow
    assert 'runs-on: [self-hosted, "${{ inputs.crabbox_runner_label }}"]' in workflow
    assert "actions/setup-python@v7.0.0" in workflow
    assert "python -m pip install -e ." in workflow
    assert "scripts/ci-hydrate-testbox-env.sh" in workflow
    assert "HF_TOKEN" in workflow
    assert "OPENCLAW_CODEX_AUTH_JSON" in workflow
    assert "CLAWBENCH_CODEX_AUTH_JSON" in workflow
    assert "/usr/local/bin/clawbench-testbox-env" in workflow
    assert "CRABBOX_ID: ${{ inputs.crabbox_id }}" in workflow
    assert "Invalid crabbox_id" in workflow
    assert "$HOME/.crabbox/actions/${CRABBOX_ID}.env" in workflow
    assert "crabbox_keep_alive_minutes" in workflow


def test_crabbox_skill_documents_clawbench_flow():
    skill = Path(".agents/skills/crabbox/SKILL.md").read_text(encoding="utf-8")

    assert "openclaw/crabbox" in skill
    assert ".crabbox.yaml" in skill
    assert "Default backend: `blacksmith-testbox`" in skill
    assert "separate `blacksmith-testbox` skill" in skill
    assert "crabbox run --provider blacksmith-testbox" in skill
    assert "crabbox actions hydrate" in skill
    assert "clawbench-testbox-env" in skill
    assert ".github/workflows/crabbox-hydrate.yml" in skill
    assert "blacksmith testbox warmup ci-check-testbox.yml" in skill


def test_blacksmith_rules_live_in_crabbox_skill():
    assert not Path(".agents/skills/blacksmith-testbox/SKILL.md").exists()


def test_testbox_helper_sources_hydrated_profile():
    script = Path("scripts/ci-hydrate-testbox-env.sh").read_text(encoding="utf-8")

    assert ".clawbench-testbox-live.profile" in script
    assert "clawbench-testbox-env" in script
    assert "source \"$profile_path\"" in script


def test_hf_sync_ensures_space_before_push():
    workflow = Path(".github/workflows/sync-to-hf-space.yml").read_text(encoding="utf-8")

    assert "Ensure HF Space exists" in workflow
    assert "api.create_repo(" in workflow
    assert "space_sdk=\"docker\"" in workflow
    assert "steps.hf.outputs.username" in workflow
