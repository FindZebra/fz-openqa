import subprocess


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()


def get_git_branch_name() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("ascii")
        .strip()
    )
