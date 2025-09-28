# 🏈 FanDuel NFL DFS Optimizer v2.1
**An AI-Enhanced Daily Fantasy Sports System for Tournament Winning**

> **Status: Active Development** - Built for a 12-person friends league with tournament-winning strategies

## 🎯 Project Mission

This system is designed to **WIN WEEKLY** against 11 friends in a season-long FanDuel tournament. Unlike generic DFS tools that optimize for mathematical perfection, this system focuses on:

- **Beating Human Psychology**: Leverage spots that casual players miss
- **Real Tournament Strategy**: Correlation, stacking, and ownership leverage
- **Weekly Automation**: From Wednesday prep to Sunday late-swap execution
- **AI-Enhanced Analysis**: Strategic insights for competitive advantage

## ⚡ Core Capabilities

### 🤖 Automated Data Pipeline
- **Real NFL Schedule Detection**: Automatically finds current week games via ESPN API
- **FanDuel Salary Integration**: Processes manually downloaded salary files with real FPPG data
- **Weather Intelligence**: Outdoor stadium conditions affecting game scripts
- **Injury Opportunity Detection**: Identifies backup players with increased value
- **Smart Player Filtering**: Conservative filtering preserves tournament-winning options

### 🧠 AI-Powered Strategic Analysis
- **Dual AI Integration**: OpenAI GPT-4o-mini + Anthropic Claude for comprehensive analysis
- **Contest Differentiation**: Actual strategic differences between GPP/Cash/Contrarian
- **Ownership Projection**: Predicts what your friends will do (not perfect ownership)
- **Leverage Identification**: Finds low-owned players with tournament upside
- **Cost Management**: $15/week budget with ROI tracking

### ⚙️ Advanced Optimization Engine
- **Exact FanDuel Format**: QB + 2RB + 3WR + 1TE + 1FLEX + 1DEF = 9 players
- **Contest-Specific Strategy**: Different algorithms for different contest types
- **Correlation Modeling**: QB-WR stacking, game stacks, bring-back strategies
- **Friends League Psychology**: Optimized for beating 11 casual players, not DraftKings pros

### 📅 NFL Weekly Cadence Automation
- **Wednesday 9 AM**: Baseline lineup construction + exposure planning
- **Thu-Sat**: Daily data refreshes + strategy refinements  
- **Sunday 11:30 AM**: Final early-slate preparation + inactive processing
- **Sunday 2:15 PM**: Lock started games + analyze early results + pivot late slate
- **Sunday 4:00 PM**: Final late-swap opportunities with leverage logic

## 📁 File Structure & Functions

### Core Application Files

**`main.py`** - Enhanced entry point with multiple operation modes
- `python main.py web` - Start web dashboard (recommended)
- `python main.py collect` - Data collection only
- `python main.py optimize` - Generate lineups only
- `python main.py scheduler` - Full automation mode
- `python main.py test` - System diagnostics

**`data_collector.py`** - FIXED data pipeline with smart filtering
- Real-time ESPN API integration for current week games
- Conservative player filtering (preserves tournament options)
- Injury opportunity detection applied BEFORE filtering
- Weather data for outdoor stadiums only
- Projection enhancement with real FanDuel FPPG data

**`optimizer.py`** - Enhanced DFS optimization with FRIENDS LEAGUE strategy
- Exact FanDuel position constraints (9 players total)
- Contest-specific algorithms that actually differ
- Friends league ownership psychology (not perfect projections)
- Correlation-aware stacking (QB+WR, game stacks, bring-backs)
- Lineup diversification for multiple entries

**`ai_analyzer.py`** - Dual AI strategic analysis system
- OpenAI GPT-4o-mini for slate analysis and leverage spots
- Anthropic Claude for alternative perspectives
- Cost tracking with $15/week budget management
- Ownership adjustments applied to optimization
- Fallback analysis when AI unavailable

**`scheduler.py`** - NFL weekly cadence automation
- Background data collection every 15-60 minutes
- Game-day aware scheduling (more frequent updates)
- Late-swap engine with game locking logic
- Performance tracking and ROI analysis

**`app.py`** - FastAPI web interface
- Real-time dashboard with lineup generation
- Contest type selection (GPP/Cash/Contrarian/Best Ball)
- Data freshness monitoring
- Exportable CSV files for FanDuel upload

### Data Processing Files

**`fanduel_salary_scraper.py`** - Manual CSV file processor
- Converts FanDuel download format to internal structure
- Preserves real FPPG data (not salary-based estimates)
- Conservative injury status handling
- Team and position normalization

**`injury_opportunity_detector.py`** - Value opportunity identification
- Identifies backup players with increased opportunity
- Conservative boost calculations (avoids false positives)
- Team-specific depth chart analysis
- Applied BEFORE general player filtering

**`enhanced_projections.py`** - Projection enhancement system
- FanDuel FPPG utilization (real data over estimates)
- Position-specific ceiling/floor calculations
- Contest-specific variance adjustments

### Configuration & Support

**`config.py`** - Enhanced system configuration
- NFL stadium data with indoor/outdoor classification
- Contest-specific optimization parameters
- Current week detection with multiple fallbacks
- API endpoints and rate limiting

**`requirements.txt`** - Python dependencies
- Core optimization: pandas, numpy, pulp
- Web framework: FastAPI, uvicorn
- AI integration: openai, anthropic
- Data sources: nfl-data-py, aiohttp
- Scheduling: apscheduler

## 🎮 Contest Type Strategies

### **Tournament/GPP** - High-Ceiling Plays
- **Target**: Beat 11 friends weekly with boom-or-bust lineups
- **Strategy**: Correlation stacking, low ownership leverage, ceiling optimization
- **Ownership**: Target 15-35% owned players (avoid super chalk)
- **Stacking**: QB+2WR from high-total games, bring-back strategies
- **Use Case**: Season-long tournament where you need weekly wins

### **Cash Game** - Consistent Value
- **Target**: Finish in top half consistently 
- **Strategy**: High-floor players, value optimization, minimal stacking
- **Ownership**: Exploit obvious value plays that friends miss (>3.5x)
- **Risk**: Low variance, weather-aware, injury-free lineups
- **Use Case**: Head-to-head or double-up contests

### **Contrarian** - Fade the Chalk
- **Target**: Differentiate from friends playing obvious lineups
- **Strategy**: Heavy chalk fades, unique stacking, narrative leverage
- **Ownership**: Target <20% owned skill players aggressively
- **Stacking**: Unconventional QB+TE, RB+Defense correlations
- **Use Case**: When you expect friends to play obvious "chalk" lineups

### **Best Ball/Single Game** - MVP Format
- **Target**: Highest raw scoring potential regardless of ownership
- **Strategy**: Pure ceiling optimization with MVP selection
- **Format**: 1 MVP (1.5x points) + 5 FLEX (any position)
- **Focus**: Game-specific correlation within single matchup
- **Use Case**: Single-game tournaments or showdown formats

## 🚀 Quick Start Guide

### 1. **Installation & Setup**
```bash
# Clone/download to your server
cd /home/brett/fanduel

# Install dependencies  
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (optional for basic functionality)
nano .env
```

### 2. **Data Preparation**
```bash
# Download FanDuel salary CSV manually from FanDuel
# Save as: data/fanduel_salaries_manual.csv
# Required columns: Id,Position,First Name,Last Name,FPPG,Salary,Game,Team,Opponent,Injury Indicator

# Test data collection
python main.py collect
```

### 3. **Generate Lineups**
```bash
# Web interface (recommended)
python main.py web
# Then visit: http://localhost:8020

# Command line generation
python main.py optimize

# Full automation
python main.py scheduler
```

### 4. **Contest-Specific Generation**
Through web interface or API calls:
- **GPP Lineups**: High ceiling, correlation stacking
- **Cash Lineups**: High floor, value plays  
- **Contrarian Lineups**: Low ownership fades
- **Single Game**: MVP + 5 FLEX format

## 🔄 System Lifecycle

### Data Collection Pipeline
```
ESPN API → Current Week Games → Team Filtering
     ↓
FanDuel CSV → Salary/FPPG Data → Position Validation
     ↓
Weather.gov → Outdoor Stadium Conditions → Game Impact
     ↓
Injury Analysis → Backup Opportunities → Value Detection
     ↓
CONSERVATIVE Filtering → Tournament Viable Players → Optimization Ready
```

### AI Enhancement Workflow
```
Player Pool + Weather + Vegas → AI Strategic Analysis
     ↓
Leverage Spots + Ownership Adjustments + Contest Strategy
     ↓  
Applied to Optimization Engine → Enhanced Player Values
     ↓
Lineup Generation → Contest-Specific Strategies → Export Ready
```

### Optimization Process
```
Filtered Players → Friends League Ownership Psychology
     ↓
Contest Type Selection → Strategy Application (GPP/Cash/Contrarian)
     ↓
Correlation Modeling → Stacking Logic → Position Constraints
     ↓
ILP Solver → Optimal Lineups → FanDuel Format Ordering
     ↓
CSV Export → Manual Upload to FanDuel → Tournament Victory
```

### Weekly Automation Cycle
```
Wednesday 9 AM → Baseline Build + AI Analysis + Exposure Planning
     ↓
Thu-Sat Daily → Data Refresh + Strategy Refinement
     ↓
Sunday 11:30 AM → Final Preparation + Inactive Processing
     ↓
Sunday 2:15 PM → Lock Started Games + Early Results Analysis
     ↓
Sunday 4:00 PM → Final Late Swaps + Leverage Pivots
```

## 🔧 Operation Modes

### **Web Dashboard** (Recommended)
```bash
python main.py web
```
- Interactive lineup generation
- Real-time data status monitoring
- Contest type selection
- CSV export for FanDuel upload
- AI analysis integration

### **Automated Scheduler**
```bash
python main.py scheduler
```
- Full NFL weekly cadence automation
- Background data collection
- Scheduled lineup optimization
- Late-swap automation
- Performance tracking

### **Data Collection Only**
```bash
python main.py collect
```
- Test data pipeline
- Verify current week detection
- Check player filtering logic
- Validate injury opportunities

### **Optimization Only**
```bash
python main.py optimize
```
- Generate sample lineups
- Test contest strategies
- Verify position constraints
- Export lineup files

### **System Diagnostics**
```bash
python main.py test
```
- Verify all imports
- Test NFL week detection
- Check configuration
- Validate dependencies

## 🎯 Key Features for Friends League

### **Psychological Advantage**
- **Conservative Ownership Projections**: 5-40% range for 12-person league
- **Value Spot Identification**: Finds obvious plays friends miss
- **Leverage Detection**: Low-owned players with tournament upside
- **Chalk Fade Logic**: Identifies when to fade popular plays

### **Strategic Differentiation** 
- **Contest-Specific Algorithms**: Actually different optimization for each contest type
- **Correlation Awareness**: Proper stacking vs individual player optimization
- **Weather Integration**: Outdoor games only (friends often ignore)
- **Injury Opportunities**: Backup players with increased roles

### **Operational Excellence**
- **Real Schedule Detection**: No hardcoded games or weeks
- **Conservative Filtering**: Preserves tournament-winning options
- **AI Budget Management**: $15/week maximum with ROI tracking
- **Export Integration**: Ready-to-upload CSV files

## ⚠️ Current Development Status

### ✅ **Working Components**
- Real-time ESPN API integration
- FanDuel salary processing with real FPPG
- Exact position constraint optimization
- Contest type differentiation
- Basic AI analysis integration
- Web dashboard interface

### 🔄 **In Development**
- Late-swap automation engine
- Advanced correlation modeling
- Historical performance tracking
- Mobile-responsive interface
- Enhanced AI prompt engineering

### 📋 **Planned Features**
- Machine learning projection models
- Advanced weather impact algorithms
- Real-time lineup adjustment
- Slack/Discord notifications
- Multi-week strategy optimization

## 🤝 Usage Philosophy

This system is built for **actual tournament winning**, not mathematical perfection. It focuses on:

1. **Beating Human Opponents** - Optimized for 12-person friend dynamics
2. **Weekly Wins Matter** - Tournament structure rewards consistent performance  
3. **Strategic Intelligence** - AI analysis for leverage and ownership spots
4. **Operational Excellence** - Automated execution of proven strategies
5. **Cost-Effective Enhancement** - $15/week AI budget with measurable ROI

The goal is sustainable competitive advantage in your specific league format, not generic DFS optimization.

## 🏆 Success Metrics

- **Weekly Win Rate**: Percentage of weeks finishing 1st in 12-person league
- **ROI Performance**: Return on entry fees vs friends
- **Leverage Success**: Low-owned players that outperform projections  
- **AI Cost Efficiency**: Strategic value per dollar of AI analysis
- **Automation Reliability**: Successful execution of weekly cadence

---

**Ready to dominate your friends league? Start with `python main.py web` and let the AI-enhanced optimization give you the competitive edge!**

For troubleshooting, check the logs in `logs/` directory or run `python main.py test` for system diagnostics.