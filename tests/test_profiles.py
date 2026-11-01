from advisor import server
from fastapi.testclient import TestClient


def test_ten_named_profiles_cover_batches_semesters_and_minors():
    profiles = list(server.PROFILES.values())
    names = {profile["label"].split(" – ")[0] for profile in profiles}

    assert len(profiles) == 10
    assert names == {
        "Aarav Sharma",
        "Ananya Iyer",
        "Rohan Mehta",
        "Diya Nair",
        "Ishaan Kapoor",
        "Kabir Rao",
        "Meera Krishnan",
        "Priya Desai",
        "Arjun Patel",
        "Aditi Menon",
    }
    assert {profile["batch"] for profile in profiles} == {2022, 2023, 2024, 2025, 2026}
    assert len({profile["current_semester"] for profile in profiles}) >= 5
    assert {profile["minor"] for profile in profiles} == {
        "Design",
        "Economics",
        "Finance",
        "Law",
        "Marketing",
        "Psychology",
        "Start-up",
    }


def test_profile_list_api_returns_all_ten_named_profiles():
    response = TestClient(server.app).get("/api/profiles")

    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) == 10
    assert profiles[0]["label"].startswith("Aarav Sharma")
