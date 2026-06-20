"""Submit an archive to the ladder — with the 5/day quota guard.

Submissions are the scarcest resource (5/team/day). This refuses to submit if
you've already used the daily quota (unless --force), so a fat-fingered upload
never burns a slot.

    python tools/submit.py --file submissions/submission_rule_based_*.tar.gz -m "baseline v1"
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys

COMPETITION = "pokemon-tcg-ai-battle"
DAILY_LIMIT = 5


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def submissions_today(api) -> int:
    today = dt.datetime.now(dt.timezone.utc).date()
    count = 0
    for s in api.competition_submissions(COMPETITION):
        date = getattr(s, "date", None)
        if isinstance(date, dt.datetime) and date.date() == today:
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="archive path (globs ok; newest used)")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--force", action="store_true", help="bypass the daily-quota guard")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.file))
    if not matches:
        sys.exit(f"no file matching {args.file}")
    archive = matches[-1]

    try:
        api = _api()
    except Exception as e:
        sys.exit(f"Kaggle auth failed ({e}). Put credentials at ~/.kaggle/kaggle.json")

    used = submissions_today(api)
    print(f"submissions used today: {used}/{DAILY_LIMIT}")
    if used >= DAILY_LIMIT and not args.force:
        sys.exit("Daily quota reached. Re-run with --force only if you're sure.")

    print(f"submitting {archive} ...")
    api.competition_submit(archive, args.message, COMPETITION)
    print("submitted. Check the leaderboard for the evaluation result.")


if __name__ == "__main__":
    main()
