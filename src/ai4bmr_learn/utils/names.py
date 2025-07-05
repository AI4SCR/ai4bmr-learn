scientists = [
    # FEMALE
    "curie",        # Marie Curie – physics, chemistry (radioactivity)
    "noether",      # Emmy Noether – mathematics (abstract algebra, physics)
    "meitner",      # Lise Meitner – nuclear physics (fission)
    "franklin",     # Rosalind Franklin – biology (DNA structure)
    "lovelace",     # Ada Lovelace – mathematics, computer science
    "leavitt",      # Henrietta Leavitt – astronomy (Cepheid variables)
    "wu",           # Chien-Shiung Wu – experimental physics (parity violation)
    "yonath",       # Ada Yonath – chemistry (ribosome structure)
    "montalcini",   # Rita Levi-Montalcini – biology (nerve growth factor)
    "hodgkin",      # Dorothy Hodgkin – chemistry (protein crystallography)
    "mirzakhani",   # Maryam Mirzakhani – mathematics (geometry, dynamical systems)
    "elion",        # Gertrude Elion – pharmacology, drug development
    "cori",         # Gerty Cori – biochemistry (glucose metabolism)
    "tereshkova",   # Valentina Tereshkova – cosmonaut and physicist
    # MALE
    "einstein",     # Albert Einstein – physics (relativity)
    "newton",       # Isaac Newton – physics, mathematics
    "darwin",       # Charles Darwin – biology (evolution)
    "turing",       # Alan Turing – mathematics, computer science
    "gauss",        # Carl Friedrich Gauss – mathematics, physics
    "bohr",         # Niels Bohr – atomic structure
    "planck",       # Max Planck – quantum theory
    "fermi",        # Enrico Fermi – nuclear physics
    "pasteur",      # Louis Pasteur – microbiology, chemistry
    "raman",        # C. V. Raman – optics, scattering
    "heisenberg",   # Werner Heisenberg – quantum mechanics
    "maxwell",      # James Clerk Maxwell – electromagnetism
    "leibniz",      # Gottfried Leibniz – calculus, logic
    "hawking",      # Stephen Hawking – cosmology, black holes
    "watt",         # James Watt – thermodynamics, engineering
    "kepler",       # Johannes Kepler – astronomy, optics
    "archimedes",   # Archimedes – mathematics, mechanics
    "pauling",      # Linus Pauling – chemistry, molecular biology
    "sanger",       # Frederick Sanger – DNA sequencing
    "boltzmann",    # Ludwig Boltzmann – thermodynamics
]

adjectives = [
    "curious", "brilliant", "precise", "modular", "scalable", "latent", "robust",
    "systemic", "elegant", "adaptive", "dynamic", "causal", "open", "rigorous",
    "neural", "fractal", "quantum", "logical", "atomic", "structured", "bold"
]

import random
def generate_name():
    return f"{random.choice(adjectives)}-{random.choice(scientists)}"