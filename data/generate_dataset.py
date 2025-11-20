"""
Generate a realistic Pokemon card dataset for ML learning.

This script creates a synthetic dataset with realistic distributions and
some intentional data quality issues for educational purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Define Pokemon types with realistic distributions
TYPES = [
    'Fire', 'Water', 'Grass', 'Electric', 'Psychic',
    'Fighting', 'Dark', 'Dragon', 'Steel', 'Fairy',
    'Normal', 'Bug', 'Ghost', 'Rock', 'Ground', 'Ice', 'Flying', 'Poison'
]

TYPE_PROBABILITIES = np.array([
    0.10, 0.12, 0.11, 0.09, 0.08,
    0.07, 0.06, 0.05, 0.05, 0.06,
    0.08, 0.06, 0.04, 0.04, 0.04, 0.03, 0.05, 0.03
])
TYPE_PROBABILITIES = TYPE_PROBABILITIES / TYPE_PROBABILITIES.sum()  # Normalize to sum to 1.0

RARITIES = ['Common', 'Uncommon', 'Rare', 'Ultra Rare', 'Secret Rare']
RARITY_PROBABILITIES = np.array([0.40, 0.30, 0.20, 0.08, 0.02])
RARITY_PROBABILITIES = RARITY_PROBABILITIES / RARITY_PROBABILITIES.sum()  # Normalize

# Pokemon name generators by type
FIRE_NAMES = ['Blazeon', 'Flamara', 'Ignithor', 'Pyroar', 'Emberix', 'Scorchling', 'Infernus', 'Magmoth']
WATER_NAMES = ['Aquaeon', 'Tidalus', 'Wavern', 'Marinox', 'Splashon', 'Hydrox', 'Tsunamon', 'Coralith']
GRASS_NAMES = ['Leafeon', 'Fernix', 'Vinegor', 'Floravia', 'Sproutling', 'Thornax', 'Mossling', 'Petalia']
ELECTRIC_NAMES = ['Voltix', 'Sparkeon', 'Thundera', 'Zaptor', 'Boltund', 'Shockwave', 'Amperix', 'Chargeon']
PSYCHIC_NAMES = ['Psywave', 'Mentalis', 'Cerebryx', 'Mindora', 'Telekine', 'Astralith', 'Zenith', 'Mysticor']
FIGHTING_NAMES = ['Bruison', 'Combatix', 'Striker', 'Machamp', 'Fistina', 'Karaton', 'Brawlix', 'Puncheon']
DARK_NAMES = ['Shadowix', 'Nightshade', 'Grimora', 'Duskull', 'Eclipsor', 'Voidling', 'Obsidian', 'Nyxara']
DRAGON_NAMES = ['Draconus', 'Wyvernix', 'Scaledor', 'Dragonite', 'Serpentis', 'Drakkari', 'Fangora', 'Drakeon']
STEEL_NAMES = ['Ironix', 'Metallix', 'Steelgard', 'Titanor', 'Forgeron', 'Alloyus', 'Chromax', 'Ferrous']
FAIRY_NAMES = ['Pixelia', 'Fairyeon', 'Sparklix', 'Enchanta', 'Mystique', 'Glimmer', 'Faewyn', 'Luminara']
NORMAL_NAMES = ['Normaleon', 'Plainix', 'Averagon', 'Commonus', 'Regulon', 'Standardix', 'Typicus', 'Basicus']
BUG_NAMES = ['Beetlix', 'Buzzwing', 'Larvitar', 'Antennix', 'Mantidor', 'Chitinor', 'Hivelix', 'Swarmling']
GHOST_NAMES = ['Phantomix', 'Spectreon', 'Haunter', 'Spiritus', 'Ethereon', 'Wraith', 'Bansheon', 'Polter']
ROCK_NAMES = ['Boulderix', 'Stonegar', 'Granitus', 'Pebblor', 'Quarryx', 'Geolith', 'Cragmon', 'Bedrock']
GROUND_NAMES = ['Terranix', 'Eartheon', 'Muddor', 'Sandstor', 'Diglett', 'Gaiax', 'Claymon', 'Soilent']
ICE_NAMES = ['Frosteon', 'Glaciar', 'Snowflakix', 'Blizzara', 'Chillix', 'Cryonus', 'Iciclon', 'Frozeon']
FLYING_NAMES = ['Skyeon', 'Winglord', 'Aerion', 'Galeforce', 'Avianix', 'Zephyrus', 'Soarling', 'Cloudor']
POISON_NAMES = ['Toxeon', 'Venomix', 'Poisona', 'Acidius', 'Sludgor', 'Virusix', 'Noxion', 'Biohazard']

TYPE_NAMES = {
    'Fire': FIRE_NAMES,
    'Water': WATER_NAMES,
    'Grass': GRASS_NAMES,
    'Electric': ELECTRIC_NAMES,
    'Psychic': PSYCHIC_NAMES,
    'Fighting': FIGHTING_NAMES,
    'Dark': DARK_NAMES,
    'Dragon': DRAGON_NAMES,
    'Steel': STEEL_NAMES,
    'Fairy': FAIRY_NAMES,
    'Normal': NORMAL_NAMES,
    'Bug': BUG_NAMES,
    'Ghost': GHOST_NAMES,
    'Rock': ROCK_NAMES,
    'Ground': GROUND_NAMES,
    'Ice': ICE_NAMES,
    'Flying': FLYING_NAMES,
    'Poison': POISON_NAMES
}

def generate_stats_by_type(pokemon_type: str, is_legendary: bool) -> dict:
    """Generate stats that correlate with Pokemon type."""

    # Base stats by type (mean values)
    type_stat_profiles = {
        'Fire': {'hp': 75, 'attack': 85, 'defense': 65, 'sp_attack': 80, 'sp_defense': 70, 'speed': 85},
        'Water': {'hp': 85, 'attack': 70, 'defense': 75, 'sp_attack': 75, 'sp_defense': 80, 'speed': 70},
        'Grass': {'hp': 70, 'attack': 65, 'defense': 75, 'sp_attack': 80, 'sp_defense': 75, 'speed': 70},
        'Electric': {'hp': 65, 'attack': 70, 'defense': 60, 'sp_attack': 90, 'sp_defense': 70, 'speed': 95},
        'Psychic': {'hp': 70, 'attack': 55, 'defense': 60, 'sp_attack': 95, 'sp_defense': 85, 'speed': 80},
        'Fighting': {'hp': 80, 'attack': 95, 'defense': 70, 'sp_attack': 50, 'sp_defense': 65, 'speed': 75},
        'Dark': {'hp': 75, 'attack': 85, 'defense': 70, 'sp_attack': 75, 'sp_defense': 70, 'speed': 80},
        'Dragon': {'hp': 85, 'attack': 90, 'defense': 80, 'sp_attack': 85, 'sp_defense': 80, 'speed': 85},
        'Steel': {'hp': 70, 'attack': 80, 'defense': 95, 'sp_attack': 65, 'sp_defense': 85, 'speed': 60},
        'Fairy': {'hp': 75, 'attack': 60, 'defense': 70, 'sp_attack': 85, 'sp_defense': 90, 'speed': 75},
        'Normal': {'hp': 75, 'attack': 70, 'defense': 65, 'sp_attack': 65, 'sp_defense': 65, 'speed': 70},
        'Bug': {'hp': 60, 'attack': 70, 'defense': 65, 'sp_attack': 55, 'sp_defense': 60, 'speed': 75},
        'Ghost': {'hp': 65, 'attack': 70, 'defense': 60, 'sp_attack': 85, 'sp_defense': 75, 'speed': 85},
        'Rock': {'hp': 75, 'attack': 80, 'defense': 95, 'sp_attack': 60, 'sp_defense': 75, 'speed': 55},
        'Ground': {'hp': 80, 'attack': 85, 'defense': 85, 'sp_attack': 55, 'sp_defense': 65, 'speed': 60},
        'Ice': {'hp': 75, 'attack': 75, 'defense': 65, 'sp_attack': 80, 'sp_defense': 75, 'speed': 75},
        'Flying': {'hp': 70, 'attack': 75, 'defense': 60, 'sp_attack': 70, 'sp_defense': 65, 'speed': 90},
        'Poison': {'hp': 70, 'attack': 75, 'defense': 70, 'sp_attack': 75, 'sp_defense': 70, 'speed': 75}
    }

    profile = type_stat_profiles[pokemon_type]

    # Legendary boost
    multiplier = 1.35 if is_legendary else 1.0
    variance = 10 if not is_legendary else 15

    stats = {}
    for stat, mean in profile.items():
        value = np.random.normal(mean * multiplier, variance)
        stats[stat] = max(20, min(180, int(value)))  # Clamp between 20-180

    return stats

def generate_price(rarity: str, is_legendary: bool, total_stats: int) -> float:
    """Generate card price based on rarity, legendary status, and power."""

    base_prices = {
        'Common': 1.0,
        'Uncommon': 3.0,
        'Rare': 8.0,
        'Ultra Rare': 25.0,
        'Secret Rare': 100.0
    }

    base = base_prices[rarity]

    # Legendary multiplier
    if is_legendary:
        base *= 3.0

    # Stats influence (higher stats = higher price)
    stat_multiplier = 1 + (total_stats - 400) / 200

    price = base * stat_multiplier * np.random.uniform(0.7, 1.3)

    return round(max(0.5, price), 2)

def generate_dataset(n_samples: int = 800) -> pd.DataFrame:
    """Generate the complete Pokemon card dataset."""

    data = []

    for i in range(n_samples):
        # Select type
        poke_type = np.random.choice(TYPES, p=TYPE_PROBABILITIES)

        # Select rarity
        rarity = np.random.choice(RARITIES, p=RARITY_PROBABILITIES)

        # Legendary status (rare)
        is_legendary = np.random.random() < 0.05  # 5% legendary

        # Generation (1-9)
        generation = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     p=[0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08])

        # Generate name
        name_pool = TYPE_NAMES[poke_type]
        base_name = np.random.choice(name_pool)
        card_number = np.random.randint(1, 300)
        name = f"{base_name}-{card_number}"

        # Generate stats
        stats = generate_stats_by_type(poke_type, is_legendary)

        # Calculate total stats
        total_stats = sum(stats.values())

        # Generate price
        price = generate_price(rarity, is_legendary, total_stats)

        # Card ID
        card_id = f"PKM-{generation:02d}-{i:04d}"

        data.append({
            'card_id': card_id,
            'name': name,
            'type': poke_type,
            'hp': stats['hp'],
            'attack': stats['attack'],
            'defense': stats['defense'],
            'sp_attack': stats['sp_attack'],
            'sp_defense': stats['sp_defense'],
            'speed': stats['speed'],
            'generation': generation,
            'is_legendary': is_legendary,
            'rarity': rarity,
            'price_usd': price
        })

    df = pd.DataFrame(data)

    # Introduce some data quality issues for learning purposes
    # 1. Some missing HP values (2%)
    missing_hp_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[missing_hp_idx, 'hp'] = np.nan

    # 2. Some missing prices (3%)
    missing_price_idx = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[missing_price_idx, 'price_usd'] = np.nan

    # 3. Some duplicate card IDs (1% - data entry errors)
    duplicate_idx = np.random.choice(df.index[10:], size=int(len(df) * 0.01), replace=False)
    for idx in duplicate_idx:
        df.loc[idx, 'card_id'] = df.loc[idx - 1, 'card_id']

    # 4. Some outlier prices (data entry errors - extra zero)
    outlier_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx, 'price_usd'] = df.loc[outlier_idx, 'price_usd'] * 10

    # 5. Inconsistent type capitalization (2%)
    inconsistent_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[inconsistent_idx, 'type'] = df.loc[inconsistent_idx, 'type'].str.lower()

    return df

if __name__ == "__main__":
    print("Generating Pokemon card dataset...")
    df = generate_dataset(n_samples=800)

    # Save to CSV
    output_path = "/Users/benny.wong/whatnot/marimo-pokemon/data/pokemon_cards.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset generated: {len(df)} records")
    print(f"Saved to: {output_path}")
    print("\nDataset info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nType distribution:")
    print(df['type'].value_counts())
    print(f"\nLegendary Pokemon: {df['is_legendary'].sum()}")
    print(f"\nPrice range: ${df['price_usd'].min():.2f} - ${df['price_usd'].max():.2f}")
