from skills.skill_loader import load_skills_from_folder

SAVE_DIR = "data/02-26/models"
DATA_DIR = "data/02-26/refs"

skills = load_skills_from_folder(DATA_DIR, mode="3d")

for s in skills:
    print(f"Training {s.name}...")
    s.train_gp(k=10)

    s.save(SAVE_DIR)

print("Done.")
