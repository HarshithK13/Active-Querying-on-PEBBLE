#!/usr/bin/env python3
import wandb
import time

ENTITY_PROJECT = "kharshi-university-of-michigan/BPref_Active_Querying"  # entity/project
TARGET_GROUP = "DUO sampling"   # exact group name to delete
DRY_RUN = False              # set False to actually delete
SLEEP_BETWEEN = 0.2         # polite pause between delete calls

api = wandb.Api()

print("Fetching runs from project:", ENTITY_PROJECT)
runs = api.runs(ENTITY_PROJECT)

to_delete = []
for run in runs:
    # run.group can be None, so guard it
    if getattr(run, "group", None) == TARGET_GROUP:
        to_delete.append(run)

print(f"Found {len(to_delete)} runs with group='{TARGET_GROUP}'")
for r in to_delete:
    print(f"  {r.id}  name={r.name}  state={r.state}  group={r.group}")

if not to_delete:
    print("Nothing to delete. Exiting.")
    exit(0)

if DRY_RUN:
    print("\nDRY RUN: no runs were deleted. Set DRY_RUN = False to actually delete.")
    exit(0)

# ask for final confirmation
confirm = input("Proceed to delete these runs? Type 'yes' to confirm: ").strip().lower()
if confirm != "yes":
    print("Aborted by user.")
    exit(0)

# delete loop
for r in to_delete:
    try:
        print("Deleting", r.id, r.name)
        r.delete()
        time.sleep(SLEEP_BETWEEN)
    except Exception as e:
        print("Failed to delete", r.id, ":", e)

print("Done.")
