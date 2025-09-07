# Replace the content of app/cli.py with this simpler version

import logging
from app.services import generate_and_save_lineup
from app.state_manager import state_manager
from app.formatting import build_text_report

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def view_lineup(game_mode: str):
    """Loads and displays the currently saved lineup."""
    print(f"\n--- Viewing Saved {game_mode.upper()} Lineup ---")
    lineup_data = state_manager.load_lineup(game_mode)
    if lineup_data:
        print(build_text_report(lineup_data))
    else:
        print(f"No lineup has been saved for '{game_mode}' mode yet.")

def generate_new_lineup(game_mode: str):
    """Triggers the generation of a new, AI-analyzed lineup."""
    print(f"\n--- Generating New AI-Powered {game_mode.upper()} Lineup ---")
    result = generate_and_save_lineup(game_mode)
    if result:
        print("Successfully generated and saved new lineup.")
        print(build_text_report(result))
    else:
        print("Failed to generate a lineup. Check logs for errors.")

def main():
    """Main function to run the interactive CLI."""
    print("--- FanDuel Lineup Manager (AI-Powered) ---")
    while True:
        print("\nAvailable Commands: [generate, view, quit]")
        command = input("Enter command: ").strip().lower()

        if command == "quit":
            break
        
        if command in ["generate", "view"]:
            while True:
                mode = input("Enter mode (league/h2h): ").strip().lower()
                if mode in ["league", "h2h"]:
                    break
                print("Invalid mode. Please enter 'league' or 'h2h'.")

            if command == "generate":
                generate_new_lineup(mode)
            elif command == "view":
                view_lineup(mode)
        else:
            print(f"Unknown command: '{command}'")

if __name__ == "__main__":
    main()
