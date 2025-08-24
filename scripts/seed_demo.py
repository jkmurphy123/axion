"""
Optional: seed a demo set by calling the orchestrator directly.
"""
from app.services.orchestrator import generate_frontpage

if __name__ == "__main__":
    generate_frontpage(n=6)
    print("Seeded demo front page.")
