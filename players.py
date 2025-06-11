import pdb
import os
import json
import time
import random
import anthropic
from typing import Dict, Any, List

class GameMove:
    """Represents a move in the game"""
    def __init__(self, player: str, action: str, data: Dict[str, Any] = None, timestamp: float = None):
        self.player = player
        self.action = action
        self.data = data or {}
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player": self.player,
            "action": self.action,
            "data": self.data,
            "timestamp": self.timestamp
        }

    def print(self):
        return f"{self.timestamp:.2f} - {self.player} performs action " \
            f"'{self.action}' with data {self.data}"


class Player:
    """Represents a player in the game"""
    def __init__(self, name: str, **kwargs):
        self.name = name

    def make_move(self, game: 'CompetitiveGame', player_name: str) -> 'GameMove':
        """Make a move in the game, to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement make_move method")

    @property
    def type(self) -> str:
        raise NotImplementedError("Subclasses must implement type property")


class HumanPlayer(Player):
    """Human player that interacts through console input"""

    def make_move(self, game: 'CompetitiveGame', player_name: str) -> 'GameMove':
        """Human player makes a move based on console input"""
        while True:
            try:
                move_input = input(f"{player_name}, enter your move ({game.get_move_description()}): \n").strip().split(',')
                print(f"Received input: {move_input}")
                return GameMove(player=player_name, action="place",
                                data={"position": move_input})
            except ValueError:
                print("Invalid input! Please ensure you use the correct move format.")

    @property
    def type(self) -> str:
        """Return player type"""
        return "human"


class ClaudePlayer(Player):
    """Claude AI player that makes strategic decisions"""

    def __init__(self, name: str = 'Claude', api_key: str = None,
                 model: str = "claude-sonnet-4-20250514", strategy: str = "competitive",
                 skill_level: str = 'intermediate', verbose=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )
        self.model = model
        self.personality = strategy  # competitive, friendly, aggressive, defensive
        self.skill_level = skill_level  # beginner, intermediate, expert
        self.verbose = verbose

    def make_move(self, game: 'CompetitiveGame', player_name: str) -> GameMove:
        """Claude analyzes the game and makes a strategic move"""

        # Get current game state from Claude's perspective
        game_rules = game.get_rules()
        game_state = game.get_player_perspective(player_name)
        valid_moves = game.get_valid_moves(player_name)
        game_description = game.get_game_description()

        # Create strategic prompt for Claude
        prompt = self._create_strategy_prompt(
            game_rules, game_description, game_state, valid_moves, player_name
        )

        if self.verbose:
            print(f"Claude's prompt:\n{prompt}\n")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse Claude's strategic decision
            decision = self._parse_claude_response(response.content[0].text, valid_moves)

            print(f"Claude's decision: {decision}")

            return GameMove(
                player=player_name,
                action=decision['action'],
                data=decision.get('data', {}),
            )

        except Exception as e:

            print(f"Error during Claude's move: {e}, falling back to random move")
            # Fallback to random valid move
            if valid_moves:
                move = random.choice(valid_moves)
                return GameMove(player=player_name, action=move['action'], data=move.get('data', {}))
            else:
                return GameMove(player=player_name, action="pass")

    def _create_strategy_prompt(self, game_rules: str, game_description: str,
                                game_state: Dict, valid_moves: List[Dict], player_name: str) -> str:
        """Create a strategic prompt for Claude"""
        return f"""
        You are playing a competitive game as player "{player_name}".

        GAME RULES:
        {game_rules}

        CURRENT GAME STATE:
        {game_description}

        DETAILED STATE:
        {json.dumps(game_state, indent=2)}

        YOUR VALID MOVES:
        {json.dumps(valid_moves, indent=2)}

        STRATEGY INSTRUCTIONS:
        1. Analyze the current position carefully
        2. Consider your winning chances and opponent threats
        3. Choose the move that gives you the best strategic advantage
        4. Think several moves ahead if possible
        5. Be {self.personality} in your approach
        6. Be aware of your skill level: {self.skill_level}

        Respond with your chosen move in this exact JSON format:
        {{
        "reasoning": "Brief explanation of your strategic thinking",
        "action": "the_action_name",
        "data": {{any additional data for the move}}
        }}

        Choose wisely - this is a competitive game and you want to win!
        """

    def _get_system_prompt(self) -> str:
        """Get system prompt based on Claude's personality and skill level"""
        personalities = {
            "competitive": "You are a highly competitive player who plays to win. Analyze every move strategically and always look for advantages.",
            "friendly": "You are a friendly but skilled player who enjoys good games. Play well but be sportsmanlike.",
            "aggressive": "You are an aggressive player who takes risks and applies pressure. Look for bold, attacking moves.",
            "defensive": "You are a careful, defensive player who minimizes risks and waits for opponent mistakes."
        }

        skill_levels = {
            "beginner": "You have basic understanding but may make some suboptimal moves.",
            "intermediate": "You have good strategic understanding and make solid moves.",
            "expert": "You are an expert player with deep strategic insight and excellent tactical awareness."
        }

        return f"""You are Claude, an AI playing competitive games.

        PERSONALITY: {personalities[self.personality]}
        SKILL LEVEL: {skill_levels[self.skill_level]}

        IMPORTANT: Always respond with valid JSON in the exact format requested.
        Your strategic analysis should be thorough but concise."""

    def _parse_claude_response(self, response: str, valid_moves: List[Dict]) -> Dict[str, Any]:
        """Parse Claude's response and validate the move"""
        print(f"Claude's response: {response}")
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                decision = json.loads(json_str)

                # Validate that the chosen action is valid
                valid_actions = [move['action'] for move in valid_moves]
                if decision.get('action') in valid_actions:
                    return decision

            # Fallback: choose first valid move
            if valid_moves:
                return {"action": valid_moves[0]['action'], "data": valid_moves[0].get('data', {})}

        except (json.JSONDecodeError, KeyError):
            pass

        # Final fallback
        return {"action": "pass", "data": {}}

    @property
    def type(self) -> str:
        """Return player type"""
        return "claude"


class RuleBasedSquaresPlayer(Player):
    """A rule-based player for the game of squares using strategic heuristics"""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.difficulty = kwargs.get('difficulty', 'medium')  # easy, medium, hard

    @property
    def type(self) -> str:
        return "rule_based"

    def make_move(self, game: 'CompetitiveGame', player_name: str) -> 'GameMove':
        """Make a strategic move based on game rules and heuristics"""
        available_moves = game.get_valid_moves(player_name)

        if not available_moves:
            raise ValueError("No available moves")

        # Strategy priority (highest to lowest):
        # 1. Complete a square if possible (take free points)
        # 2. Avoid giving opponent a square (don't create 3-sided squares)
        # 3. Create strategic positions for future moves
        # 4. Make safe moves that don't help opponent

        # Check for immediate square completions
        completing_moves = self._find_completing_moves(game, available_moves)
        if completing_moves:
            return self._select_best_completing_move(completing_moves)

        # Avoid dangerous moves that give opponent squares
        safe_moves = self._filter_dangerous_moves(game, available_moves)

        if not safe_moves:
            # If all moves are dangerous, pick the least harmful
            safe_moves = self._find_least_harmful_moves(game, available_moves)

        # Apply strategic move selection based on difficulty
        move = self._select_strategic_move(game, safe_moves)

        time.sleep(3)  # Simulate thinking time

        return GameMove(
            player=player_name,
            action=move['action'],
            data=move.get('data', {})
        )

    def _find_completing_moves(self, game, available_moves):
        """Find moves that complete a square immediately"""
        completing_moves = []

        for move in available_moves:
            # Simulate the move to see if it completes a square
            if self._move_completes_square(game, move):
                completing_moves.append(move)

        return completing_moves

    def _move_completes_square(self, game, move):
        """Check if a move completes one or more squares"""
        # This would need to be implemented based on the specific game representation
        # For now, assume game has a method to check this
        return game.move_completes_square(move) if hasattr(game, 'move_completes_square') else False

    def _select_best_completing_move(self, completing_moves):
        """Select the move that completes the most squares"""
        # If multiple moves complete squares, prefer the one completing more
        # For simplicity, return the first one
        return completing_moves[0]

    def _filter_dangerous_moves(self, game, available_moves):
        """Filter out moves that would give opponent a square on their next turn"""
        safe_moves = []

        for move in available_moves:
            if not self._move_creates_opportunity_for_opponent(game, move):
                safe_moves.append(move)

        return safe_moves if safe_moves else available_moves

    def _move_creates_opportunity_for_opponent(self, game, move):
        """Check if a move would allow opponent to complete a square"""
        # This would check if the move creates a 3-sided square
        # Implementation depends on game representation
        return game.move_creates_three_sided_square(move) if hasattr(game, 'move_creates_three_sided_square') else False

    def _find_least_harmful_moves(self, game, available_moves):
        """Find moves that give opponent the fewest squares"""
        move_scores = []

        for move in available_moves:
            # Calculate how many squares this move would give to opponent
            opponent_squares = self._count_opponent_opportunities(game, move)
            move_scores.append((move, opponent_squares))

        # Sort by least harmful (fewest opponent opportunities)
        move_scores.sort(key=lambda x: x[1])
        min_harm = move_scores[0][1]

        # Return all moves with minimum harm
        return [move for move, score in move_scores if score == min_harm]

    def _count_opponent_opportunities(self, game, move):
        """Count how many squares opponent could complete after this move"""
        # Implementation would depend on game representation
        return 0  # Placeholder

    def _select_strategic_move(self, game, safe_moves):
        """Select move based on strategic considerations and difficulty level"""
        if self.difficulty == 'easy':
            return self._select_random_move(safe_moves)
        elif self.difficulty == 'medium':
            return self._select_positional_move(game, safe_moves)
        else:  # hard
            return self._select_optimal_move(game, safe_moves)

    def _select_random_move(self, moves):
        """Select a random move from available moves"""
        import random
        return random.choice(moves)

    def _select_positional_move(self, game, moves):
        """Select move based on positional considerations"""
        # Prefer moves that:
        # - Are in the center of the board
        # - Don't create isolated lines
        # - Maintain board connectivity

        scored_moves = []
        for move in moves:
            score = self._evaluate_position(game, move)
            scored_moves.append((move, score))

        # Select move with highest positional score
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return scored_moves[0][0]

    def _select_optimal_move(self, game, moves):
        """Select the most optimal move using advanced analysis"""
        # Could implement minimax or more sophisticated analysis
        # For now, use positional evaluation
        return self._select_positional_move(game, moves)

    def _evaluate_position(self, game, move):
        """Evaluate the positional value of a move"""
        score = 0

        # Prefer central moves (assuming board has center)
        if hasattr(game, 'get_board_center'):
            center = game.get_board_center()
            distance_to_center = self._calculate_distance(move, center)
            score += max(0, 10 - distance_to_center)

        # Prefer moves that connect existing lines
        if hasattr(game, 'count_adjacent_lines'):
            adjacent_lines = game.count_adjacent_lines(move)
            score += adjacent_lines * 2

        # Avoid edge positions early in game
        if hasattr(game, 'is_edge_move'):
            if game.is_edge_move(move) and game.turn_count < 5:
                score -= 3

        return score

    def _calculate_distance(self, move1, move2):
        """Calculate distance between two moves/positions"""
        # This would depend on how moves/positions are represented
        # Placeholder implementation
        return 0
