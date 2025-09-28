# Changelog

## [v5.0.0] - 2025-09-28

### Highlights
- Monte Carlo integrated across player prep and lineup results.
- Contest-aware objective (GPP/Cash/Contrarian/Bestball) using floor/ceiling, boom/bust, variance.
- Async-safe sync wrapper: run simulations/optimization from CLI & web without event loop issues.
- Single-game (6-man) support with proper constraints.
- Lineup JSON export with MC insights to `data/lineups/`.

### Added
- Player MC metrics: `floor_10`, `ceiling_90`, `ceiling_95`, `boom_rate`, `bust_rate`, `variance`, `monte_carlo_analyzed`.
- Lineup MC metrics: `ceiling_90`, `ceiling_95`, `floor_10`, `floor_25`, `variance_score`, `sharpe_ratio`,
  `boom_probability`, `bust_probability`, `risk_level`, `monte_carlo_insights`.
- Contest-aware objective weighting (GPP/Cash/Contrarian/Bestball).
- Exporter: `data/lineups/lineups_<timestamp>.json`.

### Changed
- `optimizer.py`: integrated MC engine for player & lineup simulations; refined FanDuel roster constraints, stacking incentive, and team caps.

### Fixed
- Async issues around tournament wins and simulation by adding a safe sync runner.

### Notes
- If `monte_carlo_engine` is unavailable, optimizer gracefully falls back to non-MC objective.
- Optional AI ownership nudges via `ai_analyzer` are supported; failures are non-fatal.
