# tests/test_docker.py
import subprocess
import pytest

def test_dockerfile_builds():
    """Test that Dockerfile builds successfully."""
    result = subprocess.run(
        ["docker", "build", "-t", "rag-test", "."],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Docker build failed: {result.stderr}"

@pytest.mark.skip(reason="Requires Docker daemon")
def test_docker_compose_up():
    """Test docker-compose can start services."""
    result = subprocess.run(
        ["docker-compose", "config"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Docker compose config invalid: {result.stderr}"