from skills.skill_loader import load_skills_from_folder


SAVE_DIR = "data/04-21/models/6d"
DATA_DIR = "data/04-21/refs/processed"

skills = load_skills_from_folder(DATA_DIR, mode="6d")

for s in skills:
    print(f"Training {s.name}...")
    s.train_gp(k=10)

    s.save(SAVE_DIR)

print("Done.")
