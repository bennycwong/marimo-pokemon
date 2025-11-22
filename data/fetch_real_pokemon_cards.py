"""
Fetch ALL real Pokemon cards from the Pokemon TCG API.

This script fetches comprehensive data on all Pokemon cards ever released
and formats it for use in the ML engineering course.

API: https://pokemontcg.io/ (free, no auth required)
"""

import requests
import pandas as pd
import time
from typing import List, Dict, Any
import json

# Pokemon TCG API base URL
BASE_URL = "https://api.pokemontcg.io/v2"
HEADERS = {
    "User-Agent": "ML-Engineering-Course/1.0"
}

def fetch_all_cards(page_size: int = 250, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch all Pokemon cards from the API with pagination.

    Args:
        page_size: Number of cards to fetch per page (max 250)
        max_retries: Maximum number of retries per page

    Returns:
        List of card dictionaries
    """
    all_cards = []
    page = 1

    print("Fetching Pokemon cards from API...")
    print(f"API: {BASE_URL}/cards")
    print("This may take 5-10 minutes...")

    while True:
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                # Fetch page with increased timeout
                response = requests.get(
                    f"{BASE_URL}/cards",
                    headers=HEADERS,
                    params={
                        "page": page,
                        "pageSize": page_size
                    },
                    timeout=60  # Increased timeout
                )
                response.raise_for_status()

                data = response.json()
                cards = data.get("data", [])

                if not cards:
                    print(f"  No more cards on page {page}. Done!")
                    return all_cards

                all_cards.extend(cards)
                print(f"  ✅ Page {page}: {len(cards)} cards (Total: {len(all_cards)})")

                success = True
                page += 1

                # Rate limiting - be nice to the API
                time.sleep(1)

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  ⚠️  Timeout on page {page}, retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)
                else:
                    print(f"  ❌ Failed to fetch page {page} after {max_retries} retries. Stopping.")
                    return all_cards

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"  ⚠️  Error on page {page}: {e}. Retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)
                else:
                    print(f"  ❌ Failed to fetch page {page} after {max_retries} retries: {e}")
                    return all_cards

    print(f"\n✅ Fetched {len(all_cards)} total cards!")
    return all_cards


def extract_card_features(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant features from a card for ML purposes.

    Args:
        card: Raw card data from API

    Returns:
        Dictionary with cleaned features
    """
    # Basic info
    features = {
        "id": card.get("id"),
        "name": card.get("name"),
        "supertype": card.get("supertype"),  # Pokemon, Trainer, Energy
        "subtypes": ",".join(card.get("subtypes", [])),
    }

    # Only process Pokemon cards (not trainers/energy)
    if features["supertype"] == "Pokémon":
        # Types
        types = card.get("types", [])
        features["type"] = types[0] if types else None
        features["type_secondary"] = types[1] if len(types) > 1 else None

        # Stats (HP and attacks)
        features["hp"] = card.get("hp")

        # Get stats from attacks if they exist
        attacks = card.get("attacks", [])
        if attacks:
            # Average damage across all attacks
            damages = []
            for attack in attacks:
                damage_str = attack.get("damage", "0")
                # Clean damage string (remove +, ×, etc.)
                damage_str = ''.join(filter(str.isdigit, damage_str))
                if damage_str:
                    damages.append(int(damage_str))
            features["attack"] = int(sum(damages) / len(damages)) if damages else 0
        else:
            features["attack"] = 0

        # Weaknesses and resistances (use for defense-like stats)
        weaknesses = card.get("weaknesses", [])
        resistances = card.get("resistances", [])
        features["defense"] = 50 + len(resistances) * 10 - len(weaknesses) * 10

        # Special attack/defense (derived from attack count and conversion cost)
        features["sp_attack"] = len(attacks) * 15 if attacks else 0
        features["sp_defense"] = features["defense"] + 10

        # Speed (derived from retreat cost - lower cost = faster)
        retreat_cost = len(card.get("retreatCost", []))
        features["speed"] = max(10, 100 - (retreat_cost * 20))

        # Rarity
        features["rarity"] = card.get("rarity", "Common")

        # Set info
        card_set = card.get("set", {})
        features["set_name"] = card_set.get("name")
        features["set_series"] = card_set.get("series")
        features["set_release_date"] = card_set.get("releaseDate")

        # Map set series to generation (approximate)
        series = features["set_series"]
        generation_map = {
            "Base": 1,
            "Gym": 1,
            "Neo": 2,
            "Legendary Collection": 2,
            "E-Card": 3,
            "EX": 3,
            "Diamond & Pearl": 4,
            "Platinum": 4,
            "HeartGold & SoulSilver": 4,
            "Black & White": 5,
            "XY": 6,
            "Sun & Moon": 7,
            "Sword & Shield": 8,
            "Scarlet & Violet": 9,
        }
        features["generation"] = 5  # Default
        for key, gen in generation_map.items():
            if series and key in series:
                features["generation"] = gen
                break

        # Price (if available)
        tcgplayer = card.get("tcgplayer", {})
        prices = tcgplayer.get("prices", {})

        # Try to get a price from various sources
        price = None
        for price_type in ["holofoil", "normal", "reverseHolofoil", "1stEditionHolofoil"]:
            if price_type in prices and prices[price_type]:
                market_price = prices[price_type].get("market")
                if market_price:
                    price = float(market_price)
                    break

        features["price_usd"] = price

        # Legendary status (derived from rarity and HP)
        hp_val = int(features["hp"]) if features["hp"] and features["hp"].isdigit() else 0
        is_legendary = (
            features["rarity"] in ["Rare Holo", "Rare Holo EX", "Rare Holo GX", "Rare Holo V", "Rare Ultra"] or
            hp_val >= 120 or
            "GX" in card.get("name", "") or
            "EX" in card.get("name", "") or
            "V" in card.get("name", "")
        )
        features["is_legendary"] = is_legendary

        # Evolution stage
        features["evolvesFrom"] = card.get("evolvesFrom")
        features["stage"] = card.get("subtypes", [""])[0] if card.get("subtypes") else "Basic"

        # Number in set
        features["number"] = card.get("number")

        # Image URL
        images = card.get("images", {})
        features["image_url"] = images.get("small", "")

    return features


def process_cards(cards: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process raw card data into a clean DataFrame.

    Args:
        cards: List of raw card dictionaries

    Returns:
        Cleaned DataFrame
    """
    # Extract features from all cards
    print("\nProcessing cards...")
    processed = []

    for i, card in enumerate(cards):
        try:
            features = extract_card_features(card)
            # Only include Pokemon cards with required features
            if features.get("supertype") == "Pokémon" and features.get("hp"):
                processed.append(features)

            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(cards)} cards...")
        except Exception as e:
            # Skip cards that can't be processed
            continue

    df = pd.DataFrame(processed)

    print(f"\n✅ Processed {len(df)} Pokemon cards")

    # Clean up the data
    print("\nCleaning data...")

    # Convert HP to numeric
    df["hp"] = pd.to_numeric(df["hp"], errors="coerce")

    # Drop rows with missing critical features
    df = df.dropna(subset=["name", "type", "hp"])

    # Fill missing numeric values
    numeric_cols = ["attack", "defense", "sp_attack", "sp_defense", "speed"]
    for col in numeric_cols:
        df[col] = df[col].fillna(50)

    # Fill missing rarity
    df["rarity"] = df["rarity"].fillna("Common")

    # Convert booleans
    df["is_legendary"] = df["is_legendary"].astype(bool)

    print(f"✅ Cleaned dataset: {len(df)} cards")

    return df


def save_dataset(df: pd.DataFrame, output_path: str = "data/pokemon_cards.csv"):
    """
    Save the dataset to CSV.

    Args:
        df: DataFrame to save
        output_path: Path to save CSV
    """
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved dataset to: {output_path}")

    # Print summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total cards: {len(df)}")
    print(f"Unique Pokemon: {df['name'].nunique()}")
    print(f"Date range: {df['set_release_date'].min()} to {df['set_release_date'].max()}")
    print(f"\nTypes distribution:")
    print(df["type"].value_counts().head(10))
    print(f"\nRarity distribution:")
    print(df["rarity"].value_counts())
    print(f"\nGeneration distribution:")
    print(df["generation"].value_counts().sort_index())
    print(f"\nCards with prices: {df['price_usd'].notna().sum()} ({df['price_usd'].notna().sum()/len(df)*100:.1f}%)")
    if df['price_usd'].notna().sum() > 0:
        print(f"Price range: ${df['price_usd'].min():.2f} - ${df['price_usd'].max():.2f}")
        print(f"Median price: ${df['price_usd'].median():.2f}")
    print("="*70)


def main():
    """Main execution function."""
    print("="*70)
    print("POKEMON TCG API - COMPLETE CARD FETCHER")
    print("="*70)
    print("\nThis will fetch ALL Pokemon cards from the Pokemon TCG API.")
    print("Expected: 15,000+ cards (this may take a few minutes)")
    print()

    # Fetch all cards
    cards = fetch_all_cards(page_size=250)

    if not cards:
        print("❌ No cards fetched. Check your internet connection.")
        return

    # Process cards
    df = process_cards(cards)

    if df.empty:
        print("❌ No valid Pokemon cards after processing.")
        return

    # Save dataset
    save_dataset(df, output_path="data/pokemon_cards.csv")

    # Also save a sample for quick testing
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    save_dataset(sample_df, output_path="data/pokemon_cards_sample_1000.csv")

    print("\n✅ Dataset generation complete!")
    print(f"   Full dataset: data/pokemon_cards.csv ({len(df)} cards)")
    print(f"   Sample: data/pokemon_cards_sample_1000.csv (1000 cards)")
    print("\nYou can now use these datasets in the ML course!")


if __name__ == "__main__":
    main()
