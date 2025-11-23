# utils/confirm_rules.py

"""
Utility module for mapping each mineral class to its
quick chemical or physical verification method.
"""

def get_verification_rule(mineral_name: str) -> str:
    """
    Returns a simple test or method for confirming the mineral identity.
    """

    rules = {
        "Azurite": "Deep blue color; effervesces in HCl and turns black upon heating.",
        "Baryte": "High density; white streak; insoluble in HCl.",
        "Beryl": "Hardness ~7.5–8; hexagonal crystals; no reaction with acid.",
        "Calcite": "Fizz test with dilute HCl — strong CO₂ effervescence.",
        "Cerussite": "High density; white streak; effervesces in warm acid.",
        "Copper": "Metallic red-orange color; conducts electricity; malleable.",
        "Fluorite": "Hardness = 4; cubic cleavage; often fluoresces under UV light.",
        "Gypsum": "Soft (scratched by fingernail); white streak; no acid reaction.",
        "Hematite": "Reddish-brown streak; metallic or dull luster; non-magnetic.",
        "Malachite": "Green color; effervesces in HCl; streak = light green.",
        "Pyrite": "Brassy gold color; cubic crystals; no acid fizz; 'fool’s gold'.",
        "Pyromorphite": "Bright green/yellow; high density; brittle; lead mineral.",
        "Quartz": "Hardness = 7; conchoidal fracture; no reaction with HCl.",
        "Smithsonite": "Effervesces in warm HCl; smooth botryoidal form; light green-blue color.",
        "Wulfenite": "Bright orange-red; tabular crystals; soft (~3 Mohs); no acid fizz."
    }

    # Default rule if the mineral name isn't found
    return rules.get(mineral_name, "No verification rule found — use optical or density tests manually.")


