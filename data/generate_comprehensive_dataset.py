"""
Generate a comprehensive, realistic Pokemon card dataset.

This creates a large-scale synthetic dataset with realistic distributions
based on actual Pokemon card characteristics and game mechanics.

Includes ~16,000+ cards covering all generations and expansions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("COMPREHENSIVE POKEMON CARD DATASET GENERATOR")
print("="*70)
print("\nGenerating realistic Pokemon card dataset with:")
print("  - 16,000+ unique cards")
print("  - All 18 Pokemon types")
print("  - 9 generations")
print("  - Realistic stat distributions")
print("  - Card prices based on rarity and stats")
print()

# Pokemon types with game-accurate distributions
TYPES = [
    'Water', 'Fire', 'Grass', 'Psychic', 'Electric',
    'Fighting', 'Dark', 'Dragon', 'Normal', 'Steel',
    'Fairy', 'Bug', 'Ghost', 'Rock', 'Ground', 'Ice', 'Flying', 'Poison'
]

TYPE_PROBABILITIES = np.array([
    0.13, 0.12, 0.11, 0.09, 0.09,
    0.08, 0.07, 0.04, 0.08, 0.04,
    0.05, 0.05, 0.04, 0.03, 0.03, 0.02, 0.04, 0.03
])
TYPE_PROBABILITIES = TYPE_PROBABILITIES / TYPE_PROBABILITIES.sum()

# Rarities based on actual card distributions
RARITIES = ['Common', 'Uncommon', 'Rare', 'Rare Holo', 'Rare Holo EX', 'Rare Holo GX', 'Rare Ultra', 'Secret Rare']
RARITY_PROBABILITIES = np.array([0.35, 0.30, 0.18, 0.10, 0.03, 0.02, 0.015, 0.005])
RARITY_PROBABILITIES = RARITY_PROBABILITIES / RARITY_PROBABILITIES.sum()

# Generation-based card counts (realistic to Pokemon TCG history)
GENERATION_COUNTS = {
    1: 2500,  # Base Set through Neo era (1999-2003)
    2: 2000,  # Ruby & Sapphire era (2003-2006)
    3: 2200,  # Diamond & Pearl era (2007-2010)
    4: 1800,  # Black & White era (2011-2013)
    5: 2100,  # XY era (2014-2016)
    6: 2000,  # Sun & Moon era (2017-2019)
    7: 1800,  # Sword & Shield era (2020-2022)
    8: 1400,  # Scarlet & Violet era (2023-2024)
    9: 200,   # Latest releases (2024+)
}

# Comprehensive Pokemon name database (realistic names)
POKEMON_NAMES = {
    'Water': ['Squirtle', 'Wartortle', 'Blastoise', 'Psyduck', 'Golduck', 'Poliwag', 'Poliwhirl', 'Poliwrath',
              'Tentacool', 'Tentacruel', 'Slowpoke', 'Slowbro', 'Magikarp', 'Gyarados', 'Lapras', 'Vaporeon',
              'Totodile', 'Croconaw', 'Feraligatr', 'Marill', 'Azumarill', 'Qwilfish', 'Corsola', 'Suicune',
              'Mudkip', 'Marshtomp', 'Swampert', 'Lotad', 'Lombre', 'Ludicolo', 'Wingull', 'Pelipper', 'Wailmer',
              'Wailord', 'Feebas', 'Milotic', 'Spheal', 'Sealeo', 'Walrein', 'Kyogre', 'Piplup', 'Prinplup',
              'Empoleon', 'Buizel', 'Floatzel', 'Shellos', 'Gastrodon', 'Finneon', 'Lumineon', 'Palkia'],
    'Fire': ['Charmander', 'Charmeleon', 'Charizard', 'Vulpix', 'Ninetales', 'Growlithe', 'Arcanine', 'Ponyta',
             'Rapidash', 'Magmar', 'Flareon', 'Moltres', 'Cyndaquil', 'Quilava', 'Typhlosion', 'Slugma', 'Magcargo',
             'Houndour', 'Houndoom', 'Entei', 'Torchic', 'Combusken', 'Blaziken', 'Torkoal', 'Numel', 'Camerupt',
             'Chimchar', 'Monferno', 'Infernape', 'Heatran', 'Tepig', 'Pignite', 'Emboar', 'Pansear', 'Simisear',
             'Darumaka', 'Darmanitan', 'Victini', 'Reshiram', 'Fennekin', 'Braixen', 'Delphox', 'Fletchinder',
             'Talonflame', 'Litten', 'Torracat', 'Incineroar', 'Scorbunny', 'Raboot', 'Cinderace'],
    'Grass': ['Bulbasaur', 'Ivysaur', 'Venusaur', 'Oddish', 'Gloom', 'Vileplume', 'Bellsprout', 'Weepinbell',
              'Victreebel', 'Exeggcute', 'Exeggutor', 'Tangela', 'Chikorita', 'Bayleef', 'Meganium', 'Bellossom',
              'Sunkern', 'Sunflora', 'Treecko', 'Grovyle', 'Sceptile', 'Lotad', 'Lombre', 'Ludicolo', 'Seedot',
              'Nuzleaf', 'Shiftry', 'Shroomish', 'Breloom', 'Roselia', 'Cacnea', 'Cacturne', 'Turtwig', 'Grotle',
              'Torterra', 'Budew', 'Roserade', 'Cherubi', 'Cherrim', 'Snover', 'Abomasnow', 'Shaymin', 'Snivy',
              'Servine', 'Serperior', 'Pansage', 'Simisage', 'Chespin', 'Quilladin', 'Chesnaught'],
    'Electric': ['Pikachu', 'Raichu', 'Magnemite', 'Magneton', 'Voltorb', 'Electrode', 'Electabuzz', 'Jolteon',
                 'Zapdos', 'Pichu', 'Mareep', 'Flaaffy', 'Ampharos', 'Chinchou', 'Lanturn', 'Elekid', 'Raikou',
                 'Electrike', 'Manectric', 'Plusle', 'Minun', 'Shinx', 'Luxio', 'Luxray', 'Pachirisu', 'Rotom',
                 'Blitzle', 'Zebstrika', 'Emolga', 'Joltik', 'Galvantula', 'Thundurus', 'Zekrom', 'Helioptile',
                 'Heliolisk', 'Dedenne', 'Charjabug', 'Vikavolt', 'Togedemaru', 'Zeraora', 'Yamper', 'Boltund'],
    'Psychic': ['Abra', 'Kadabra', 'Alakazam', 'Slowpoke', 'Slowbro', 'Drowzee', 'Hypno', 'Exeggcute', 'Exeggutor',
                'Starmie', 'Mr. Mime', 'Jynx', 'Mewtwo', 'Mew', 'Espeon', 'Natu', 'Xatu', 'Girafarig', 'Lugia',
                'Celebi', 'Ralts', 'Kirlia', 'Gardevoir', 'Meditite', 'Medicham', 'Spoink', 'Grumpig', 'Lunatone',
                'Solrock', 'Baltoy', 'Claydol', 'Chimecho', 'Latias', 'Latios', 'Jirachi', 'Deoxys', 'Uxie',
                'Mesprit', 'Azelf', 'Cresselia', 'Victini', 'Munna', 'Musharna', 'Gothita', 'Gothorita', 'Gothitelle'],
    'Fighting': ['Mankey', 'Primeape', 'Machop', 'Machoke', 'Machamp', 'Hitmonlee', 'Hitmonchan', 'Tyrogue',
                 'Hitmontop', 'Heracross', 'Makuhita', 'Hariyama', 'Meditite', 'Medicham', 'Riolu', 'Lucario',
                 'Croagunk', 'Toxicroak', 'Timburr', 'Gurdurr', 'Conkeldurr', 'Throh', 'Sawk', 'Mienfoo', 'Mienshao',
                 'Pancham', 'Pangoro', 'Hawlucha', 'Stufful', 'Bewear', 'Crabrawler', 'Crabominable', 'Clobbopus'],
    'Dark': ['Umbreon', 'Murkrow', 'Sneasel', 'Houndour', 'Houndoom', 'Tyranitar', 'Poochyena', 'Mightyena',
             'Nuzleaf', 'Shiftry', 'Sableye', 'Cacturne', 'Crawdaunt', 'Absol', 'Honchkrow', 'Spiritomb',
             'Darkrai', 'Purrloin', 'Liepard', 'Sandile', 'Krokorok', 'Krookodile', 'Zorua', 'Zoroark',
             'Vullaby', 'Mandibuzz', 'Pawniard', 'Bisharp', 'Deino', 'Zweilous', 'Hydreigon', 'Yveltal'],
    'Dragon': ['Dragonite', 'Dragonair', 'Dratini', 'Kingdra', 'Vibrava', 'Flygon', 'Altaria', 'Salamence',
               'Latias', 'Latios', 'Rayquaza', 'Gible', 'Gabite', 'Garchomp', 'Dialga', 'Palkia', 'Giratina',
               'Axew', 'Fraxure', 'Haxorus', 'Druddigon', 'Reshiram', 'Zekrom', 'Kyurem', 'Goomy', 'Sliggoo',
               'Goodra', 'Zygarde', 'Jangmo-o', 'Hakamo-o', 'Kommo-o', 'Regidrago', 'Dragapult'],
    'Normal': ['Pidgey', 'Pidgeotto', 'Pidgeot', 'Rattata', 'Raticate', 'Spearow', 'Fearow', 'Jigglypuff',
               'Wigglytuff', 'Meowth', 'Persian', 'Farfetchd', 'Doduo', 'Dodrio', 'Lickitung', 'Chansey',
               'Kangaskhan', 'Tauros', 'Ditto', 'Eevee', 'Porygon', 'Snorlax', 'Sentret', 'Furret', 'Hoothoot',
               'Noctowl', 'Aipom', 'Girafarig', 'Dunsparce', 'Teddiursa', 'Ursaring', 'Porygon2', 'Smeargle',
               'Miltank', 'Blissey', 'Zigzagoon', 'Linoone', 'Slakoth', 'Vigoroth', 'Slaking', 'Whismur'],
    'Steel': ['Magnemite', 'Magneton', 'Steelix', 'Scizor', 'Skarmory', 'Mawile', 'Aron', 'Lairon', 'Aggron',
              'Beldum', 'Metang', 'Metagross', 'Registeel', 'Shieldon', 'Bastiodon', 'Bronzor', 'Bronzong',
              'Lucario', 'Magnezone', 'Dialga', 'Heatran', 'Klink', 'Klang', 'Klinklang', 'Pawniard', 'Bisharp',
              'Honedge', 'Doublade', 'Aegislash', 'Togedemaru', 'Meltan', 'Melmetal', 'Corviknight', 'Duraludon'],
    'Fairy': ['Clefairy', 'Clefable', 'Jigglypuff', 'Wigglytuff', 'Mr. Mime', 'Cleffa', 'Igglybuff', 'Togepi',
              'Togetic', 'Azurill', 'Marill', 'Azumarill', 'Ralts', 'Kirlia', 'Gardevoir', 'Snubbull', 'Granbull',
              'Mawile', 'Mime Jr.', 'Togekiss', 'Spritzee', 'Aromatisse', 'Swirlix', 'Slurpuff', 'Sylveon',
              'Dedenne', 'Carbink', 'Klefki', 'Xerneas', 'Morelull', 'Shiinotic', 'Mimikyu', 'Tapu Koko'],
    'Bug': ['Caterpie', 'Metapod', 'Butterfree', 'Weedle', 'Kakuna', 'Beedrill', 'Paras', 'Parasect', 'Venonat',
            'Venomoth', 'Scyther', 'Pinsir', 'Ledyba', 'Ledian', 'Spinarak', 'Ariados', 'Yanma', 'Pineco',
            'Forretress', 'Heracross', 'Wurmple', 'Silcoon', 'Beautifly', 'Cascoon', 'Dustox', 'Nincada',
            'Ninjask', 'Shedinja', 'Volbeat', 'Illumise', 'Anorith', 'Armaldo', 'Kricketot', 'Kricketune'],
    'Ghost': ['Gastly', 'Haunter', 'Gengar', 'Misdreavus', 'Mismagius', 'Sableye', 'Shuppet', 'Banette', 'Duskull',
              'Dusclops', 'Dusknoir', 'Drifloon', 'Drifblim', 'Spiritomb', 'Giratina', 'Yamask', 'Cofagrigus',
              'Frillish', 'Jellicent', 'Litwick', 'Lampent', 'Chandelure', 'Golett', 'Golurk', 'Pumpkaboo',
              'Gourgeist', 'Hoopa', 'Sandygast', 'Palossand', 'Mimikyu', 'Dhelmise', 'Sinistea', 'Polteageist'],
    'Rock': ['Geodude', 'Graveler', 'Golem', 'Onix', 'Rhyhorn', 'Rhydon', 'Omanyte', 'Omastar', 'Kabuto',
             'Kabutops', 'Aerodactyl', 'Sudowoodo', 'Larvitar', 'Pupitar', 'Tyranitar', 'Nosepass', 'Aron',
             'Lairon', 'Aggron', 'Lileep', 'Cradily', 'Anorith', 'Armaldo', 'Regirock', 'Cranidos', 'Rampardos',
             'Shieldon', 'Bastiodon', 'Bonsly', 'Rhyperior', 'Roggenrola', 'Boldore', 'Gigalith', 'Dwebble'],
    'Ground': ['Sandshrew', 'Sandslash', 'Diglett', 'Dugtrio', 'Geodude', 'Graveler', 'Golem', 'Onix', 'Cubone',
               'Marowak', 'Rhyhorn', 'Rhydon', 'Wooper', 'Quagsire', 'Swinub', 'Piloswine', 'Phanpy', 'Donphan',
               'Larvitar', 'Pupitar', 'Trapinch', 'Vibrava', 'Flygon', 'Numel', 'Camerupt', 'Barboach', 'Whiscash',
               'Baltoy', 'Claydol', 'Groudon', 'Hippopotas', 'Hippowdon', 'Gible', 'Gabite', 'Garchomp'],
    'Ice': ['Dewgong', 'Cloyster', 'Jynx', 'Articuno', 'Swinub', 'Piloswine', 'Delibird', 'Sneasel', 'Swinub',
            'Snorunt', 'Glalie', 'Spheal', 'Sealeo', 'Walrein', 'Regice', 'Snover', 'Abomasnow', 'Weavile',
            'Glaceon', 'Mamoswine', 'Froslass', 'Vanillite', 'Vanillish', 'Vanilluxe', 'Cubchoo', 'Beartic',
            'Cryogonal', 'Bergmite', 'Avalugg', 'Alolan Sandshrew', 'Alolan Sandslash', 'Alolan Vulpix'],
    'Flying': ['Pidgey', 'Pidgeotto', 'Pidgeot', 'Zubat', 'Golbat', 'Farfetchd', 'Doduo', 'Dodrio', 'Aerodactyl',
               'Articuno', 'Zapdos', 'Moltres', 'Dragonite', 'Hoothoot', 'Noctowl', 'Crobat', 'Togetic', 'Xatu',
               'Yanma', 'Murkrow', 'Delibird', 'Skarmory', 'Lugia', 'Ho-Oh', 'Taillow', 'Swellow', 'Wingull',
               'Pelipper', 'Swablu', 'Altaria', 'Tropius', 'Rayquaza', 'Starly', 'Staravia', 'Staraptor'],
    'Poison': ['Ekans', 'Arbok', 'Nidoran', 'Nidorina', 'Nidoqueen', 'Nidorino', 'Nidoking', 'Zubat', 'Golbat',
               'Oddish', 'Gloom', 'Vileplume', 'Venonat', 'Venomoth', 'Bellsprout', 'Weepinbell', 'Victreebel',
               'Tentacool', 'Tentacruel', 'Grimer', 'Muk', 'Koffing', 'Weezing', 'Crobat', 'Spinarak', 'Ariados',
               'Qwilfish', 'Seviper', 'Gulpin', 'Swalot', 'Roselia', 'Stunky', 'Skuntank', 'Croagunk', 'Toxicroak'],
}

# Card sets by generation (realistic to Pokemon TCG)
CARD_SETS = {
    1: ['Base Set', 'Jungle', 'Fossil', 'Base Set 2', 'Team Rocket', 'Gym Heroes', 'Gym Challenge', 'Neo Genesis', 'Neo Discovery'],
    2: ['Neo Revelation', 'Neo Destiny', 'Legendary Collection', 'Expedition', 'Aquapolis', 'Skyridge'],
    3: ['Ruby & Sapphire', 'Sandstorm', 'Dragon', 'Team Magma vs Team Aqua', 'EX FireRed & LeafGreen', 'EX Emerald'],
    4: ['Diamond & Pearl', 'Mysterious Treasures', 'Secret Wonders', 'Great Encounters', 'Majestic Dawn', 'Legends Awakened'],
    5: ['XY Base', 'Flashfire', 'Furious Fists', 'Phantom Forces', 'Primal Clash', 'Roaring Skies', 'Ancient Origins'],
    6: ['Sun & Moon', 'Guardians Rising', 'Burning Shadows', 'Crimson Invasion', 'Ultra Prism', 'Forbidden Light'],
    7: ['Sword & Shield', 'Rebel Clash', 'Darkness Ablaze', 'Vivid Voltage', 'Battle Styles', 'Chilling Reign'],
    8: ['Scarlet & Violet', 'Paldea Evolved', 'Obsidian Flames', 'Paradox Rift', '151', 'Paldean Fates'],
    9: ['Temporal Forces', 'Twilight Masquerade', 'Shrouded Fable'],
}


def generate_cards_for_generation(generation: int, num_cards: int) -> list:
    """Generate cards for a specific generation."""
    cards = []

    for _ in range(num_cards):
        # Select type
        poke_type = np.random.choice(TYPES, p=TYPE_PROBABILITIES)

        # Select Pokemon name
        name = np.random.choice(POKEMON_NAMES[poke_type])

        # Add variant suffix sometimes (ex, V, GX, etc.)
        variant_chance = np.random.random()
        if variant_chance < 0.05:
            name = f"{name} EX"
        elif variant_chance < 0.08:
            name = f"{name} GX"
        elif variant_chance < 0.10:
            name = f"{name} V"

        # Rarity
        rarity = np.random.choice(RARITIES, p=RARITY_PROBABILITIES)

        # HP based on rarity and type
        base_hp = {
            'Common': 50, 'Uncommon': 70, 'Rare': 90,
            'Rare Holo': 100, 'Rare Holo EX': 170, 'Rare Holo GX': 180,
            'Rare Ultra': 200, 'Secret Rare': 220
        }[rarity]

        hp = int(base_hp + np.random.normal(0, 15))
        hp = max(30, min(340, hp))  # Clamp between realistic bounds

        # Stats based on type and rarity
        rarity_bonus = {'Common': 0, 'Uncommon': 10, 'Rare': 20,
                       'Rare Holo': 30, 'Rare Holo EX': 50, 'Rare Holo GX': 60,
                       'Rare Ultra': 70, 'Secret Rare': 80}[rarity]

        # Type-based stat modifiers
        type_mods = {
            'Fire': (1.2, 0.8, 0.7, 0.9, 0.8, 1.0),
            'Water': (0.9, 0.9, 1.1, 0.8, 1.1, 0.9),
            'Grass': (0.8, 0.9, 1.0, 1.1, 0.9, 0.9),
            'Electric': (0.8, 1.1, 0.7, 0.9, 0.8, 1.3),
            'Psychic': (0.7, 0.7, 0.9, 1.3, 1.2, 1.0),
            'Fighting': (0.9, 1.3, 1.1, 0.7, 0.9, 0.8),
            'Dark': (0.9, 1.1, 0.9, 1.0, 0.9, 1.1),
            'Dragon': (1.0, 1.2, 1.0, 1.2, 1.0, 1.1),
            'Normal': (0.9, 0.9, 0.9, 0.9, 0.9, 1.0),
            'Steel': (1.0, 0.9, 1.3, 0.8, 1.2, 0.7),
            'Fairy': (0.9, 0.7, 1.0, 1.1, 1.2, 1.0),
            'Bug': (0.7, 1.0, 0.9, 0.8, 0.9, 1.1),
            'Ghost': (0.7, 0.9, 0.8, 1.2, 1.0, 1.1),
            'Rock': (1.0, 1.1, 1.3, 0.7, 1.0, 0.6),
            'Ground': (1.0, 1.2, 1.1, 0.8, 0.9, 0.7),
            'Ice': (0.9, 0.9, 1.1, 1.0, 1.0, 0.8),
            'Flying': (0.8, 0.9, 0.7, 0.9, 0.8, 1.4),
            'Poison': (0.8, 1.0, 0.9, 1.0, 1.0, 0.9),
        }

        base_stat = 50
        mods = type_mods[poke_type]

        attack = int((base_stat + rarity_bonus) * mods[0] + np.random.normal(0, 10))
        defense = int((base_stat + rarity_bonus) * mods[1] + np.random.normal(0, 10))
        sp_attack = int((base_stat + rarity_bonus) * mods[2] + np.random.normal(0, 10))
        sp_defense = int((base_stat + rarity_bonus) * mods[3] + np.random.normal(0, 10))
        speed = int((base_stat + rarity_bonus) * mods[4] + np.random.normal(0, 10))

        # Clamp stats
        attack = max(10, min(200, attack))
        defense = max(10, min(200, defense))
        sp_attack = max(10, min(200, sp_attack))
        sp_defense = max(10, min(200, sp_defense))
        speed = max(10, min(200, speed))

        # Legendary status
        is_legendary = (hp >= 150 and rarity in ['Rare Holo EX', 'Rare Holo GX', 'Rare Ultra', 'Secret Rare']) or \
                      ('EX' in name or 'GX' in name or 'V' in name)

        # Set info
        card_set = np.random.choice(CARD_SETS[generation])

        # Price based on rarity, HP, and randomness
        rarity_base_price = {
            'Common': 0.25, 'Uncommon': 0.50, 'Rare': 1.50,
            'Rare Holo': 5.00, 'Rare Holo EX': 25.00, 'Rare Holo GX': 15.00,
            'Rare Ultra': 50.00, 'Secret Rare': 150.00
        }[rarity]

        # Price multiplier based on stats and legendary status
        stat_total = attack + defense + sp_attack + sp_defense + speed
        stat_multiplier = 1 + (stat_total - 250) / 500  # Higher stats = higher price
        legendary_multiplier = 2.0 if is_legendary else 1.0

        price = rarity_base_price * stat_multiplier * legendary_multiplier * np.random.lognormal(0, 0.5)
        price = round(price, 2)

        card = {
            'name': name,
            'type': poke_type,
            'hp': hp,
            'attack': attack,
            'defense': defense,
            'sp_attack': sp_attack,
            'sp_defense': sp_defense,
            'speed': speed,
            'generation': generation,
            'rarity': rarity,
            'is_legendary': is_legendary,
            'set_name': card_set,
            'price_usd': price,
        }

        cards.append(card)

    return cards


def main():
    """Generate comprehensive Pokemon card dataset."""
    all_cards = []

    print("Generating cards by generation:")
    for gen, count in GENERATION_COUNTS.items():
        print(f"  Generation {gen}: {count} cards...")
        cards = generate_cards_for_generation(gen, count)
        all_cards.extend(cards)

    # Create DataFrame
    df = pd.DataFrame(all_cards)

    print(f"\n✅ Generated {len(df)} total cards!")
    print("\nDataset Summary:")
    print(f"  Types: {df['type'].nunique()}")
    print(f"  Unique Pokemon: {df['name'].nunique()}")
    print(f"  Generations: {df['generation'].nunique()}")
    print(f"  Rarity levels: {df['rarity'].nunique()}")
    print(f"  Price range: ${df['price_usd'].min():.2f} - ${df['price_usd'].max():.2f}")
    print(f"  Median price: ${df['price_usd'].median():.2f}")

    # Save full dataset
    output_path = "data/pokemon_cards.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved full dataset: {output_path}")

    # Save sample for quick testing
    sample_df = df.sample(n=1000, random_state=42)
    sample_path = "data/pokemon_cards_sample_1000.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"✅ Saved sample dataset: {sample_path}")

    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE!")
    print("="*70)
    print(f"\nYou now have {len(df):,} Pokemon cards ready for ML training!")
    print("\nFiles created:")
    print(f"  1. {output_path} ({len(df):,} cards)")
    print(f"  2. {sample_path} (1,000 cards)")
    print("\nYou can now use these in the ML course!")


if __name__ == "__main__":
    main()
