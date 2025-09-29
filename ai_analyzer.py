"""
Complete Dual AI Integration for WINNING DFS Analysis
FIXED: Enhanced API key validation and better error handling
"""
import os
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime

# Import APIs with graceful fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available")
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic package not available")
    CLAUDE_AVAILABLE = False

class DualAIDFSAnalyzer:
    """Dual AI system for comprehensive DFS analysis with enhanced error handling"""

    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.weekly_budget = float(os.getenv('AI_WEEKLY_BUDGET', '15.0'))
        self.weekly_spend = 0.0
        self.cost_log = []

        # Enhanced OpenAI validation and initialization
        self.openai_client = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                # Validate API key format - FIXED for 164-character keys
                if self._validate_openai_key(self.openai_api_key):
                    self.openai_client = OpenAI(api_key=self.openai_api_key)
                    logger.info("✅ OpenAI client initialized successfully")
                else:
                    logger.error("❌ OpenAI API key format validation failed")
                    self.openai_client = None
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
                self.openai_client = None

        # Enhanced Claude validation and initialization
        self.claude_client = None
        if CLAUDE_AVAILABLE and self.claude_api_key:
            try:
                # Validate API key format
                if self._validate_claude_key(self.claude_api_key):
                    self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
                    logger.info("✅ Claude client initialized successfully")
                else:
                    logger.error("❌ Claude API key format validation failed")
                    self.claude_client = None
            except Exception as e:
                logger.error(f"❌ Claude initialization failed: {e}")
                self.claude_client = None

        # Log initialization status
        openai_status = "✅" if self.openai_client else "❌"
        claude_status = "✅" if self.claude_client else "❌"
        logger.info(f"AI Services: OpenAI {openai_status} | Claude {claude_status}")

    def _validate_openai_key(self, key: str) -> bool:
        """Validate OpenAI API key format - FIXED for project keys"""
        if not key:
            return False
        
        # OpenAI keys should start with 'sk-'
        if not key.startswith('sk-'):
            logger.error(f"OpenAI key should start with 'sk-', got: {key[:10]}...")
            return False
        
        # Accept both old format (~51 chars) and new project format (~164 chars)
        if len(key) < 40:
            logger.error(f"OpenAI key too short: {len(key)} chars (expected 40+)")
            return False
        
        if len(key) > 200:
            logger.error(f"OpenAI key too long: {len(key)} chars (expected <200)")
            return False
        
        # Check for truncation indicators
        if key.endswith('...') or '***' in key:
            logger.error("OpenAI key appears truncated")
            return False
            
        logger.info(f"✅ OpenAI key format valid: {len(key)} characters")
        return True

    def _validate_claude_key(self, key: str) -> bool:
        """Validate Claude API key format"""
        if not key:
            return False
        
        # Claude keys should start with 'sk-ant-'
        if not key.startswith('sk-ant-'):
            logger.error(f"Claude key should start with 'sk-ant-', got: {key[:10]}...")
            return False
        
        # Should be substantial length
        if len(key) < 40:
            logger.error(f"Claude key too short: {len(key)} chars (expected 40+)")
            return False
            
        logger.info(f"✅ Claude key format valid: {len(key)} characters")
        return True

    def analyze_slate_for_optimization(self, player_data: List[Dict],
                                       weather_data: Dict,
                                       vegas_data: Dict,
                                       contest_type: str = 'gpp') -> Dict[str, Any]:
        """Main analysis function using available AI services"""

        if not (self.openai_client or self.claude_client):
            logger.warning("No AI services available, using fallback analysis")
            return self._fallback_analysis(player_data, contest_type)

        try:
            # Prepare data for AI analysis
            analysis_data = self._prepare_slate_data(player_data, weather_data, vegas_data, contest_type)

            # Get insights from available AI services
            openai_insights = None
            claude_insights = None

            if self.openai_client:
                try:
                    openai_insights = self._get_openai_insights(analysis_data, contest_type)
                    logger.info("✅ OpenAI analysis completed successfully")
                except Exception as e:
                    logger.error(f"❌ OpenAI analysis failed: {e}")

            if self.claude_client:
                try:
                    claude_insights = self._get_claude_insights(analysis_data, contest_type)
                    logger.info("✅ Claude analysis completed successfully")
                except Exception as e:
                    logger.error(f"❌ Claude analysis failed: {e}")

            # Combine insights
            combined_analysis = self._synthesize_ai_insights(
                openai_insights, claude_insights, player_data, contest_type
            )

            # Track costs
            cost = 0.15 if (openai_insights and claude_insights) else 0.08
            self.weekly_spend += cost
            self._log_cost(f"ai_analysis_{contest_type}", cost)

            return combined_analysis

        except Exception as e:
            logger.error(f"AI analysis pipeline failed: {e}")
            return self._fallback_analysis(player_data, contest_type)

    def _prepare_slate_data(self, player_data: List[Dict], weather_data: Dict,
                           vegas_data: Dict, contest_type: str) -> Dict:
        """Prepare slate data for AI analysis"""
        
        # Basic slate summary
        return {
            'contest_type': contest_type,
            'slate_size': len(player_data),
            'avg_salary': sum(p.get('salary', 0) for p in player_data) / len(player_data) if player_data else 0,
            'top_players': {'QB': [], 'RB': [], 'WR': [], 'TE': []},
            'value_plays': [],
            'weather_impacts': [],
            'stack_candidates': []
        }

    def _get_openai_insights(self, data: Dict, contest_type: str) -> str:
        """Get strategic insights from OpenAI"""
        
        prompt = f"Analyze this NFL DFS slate for {contest_type} contest with {data['slate_size']} players."
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        return response.choices[0].message.content

    def _get_claude_insights(self, data: Dict, contest_type: str) -> str:
        """Get strategic insights from Claude"""
        
        prompt = f"Analyze this NFL DFS slate for {contest_type} contest with {data['slate_size']} players."
        
        response = self.claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

    def _synthesize_ai_insights(self, openai_insights: str, claude_insights: str,
                               player_data: List[Dict], contest_type: str) -> Dict[str, Any]:
        """Combine AI insights"""
        
        all_insights = ""
        sources_used = []

        if openai_insights:
            all_insights += f"OpenAI: {openai_insights}\n"
            sources_used.append("OpenAI")

        if claude_insights:
            all_insights += f"Claude: {claude_insights}\n"
            sources_used.append("Claude")

        return {
            'type': contest_type,
            'ai_strategy': all_insights or "No AI analysis available",
            'leverage_players': [],
            'avoid_players': [],
            'stack_teams': [],
            'ownership_adjustments': {},
            'ai_confidence': 0.8 if len(sources_used) == 2 else 0.6,
            'ai_enabled': True,
            'sources_used': sources_used,
            'analysis_quality': 'dual' if len(sources_used) == 2 else 'single'
        }

    def _fallback_analysis(self, players: List[Dict], contest_type: str) -> Dict[str, Any]:
        """Fallback when AI is unavailable"""
        
        return {
            'type': contest_type,
            'ai_strategy': f"Fallback {contest_type} analysis: Target value plays, avoid high ownership",
            'leverage_players': [],
            'avoid_players': [],
            'stack_teams': [],
            'ownership_adjustments': {},
            'ai_confidence': 0.4,
            'ai_enabled': False,
            'sources_used': ['fallback'],
            'analysis_quality': 'basic'
        }

    def _log_cost(self, analysis_type: str, cost: float):
        """Track AI costs"""
        self.cost_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': analysis_type,
            'cost': cost,
            'weekly_total': self.weekly_spend
        })

        logger.info(f"AI Cost: ${cost:.3f} for {analysis_type}")
        logger.info(f"Weekly total: ${self.weekly_spend:.3f} / ${self.weekly_budget:.2f}")

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return {
            'weekly_spend': self.weekly_spend,
            'weekly_budget': self.weekly_budget,
            'remaining_budget': self.weekly_budget - self.weekly_spend,
            'cost_log': self.cost_log[-10:] if self.cost_log else []
        }

# Backwards compatibility
WinningDFSAIAnalyzer = DualAIDFSAnalyzer
