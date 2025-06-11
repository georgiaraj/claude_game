import copy
import anthropic
import json
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import os

class GamePhase(Enum):
    SETUP = "setup"
    PLAYING = "playing"
    FINISHED = "finished"

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

class GameState:
    """Represents the current state of the game"""
    def __init__(self):
        self.phase = GamePhase.SETUP
        self.current_player = None
        self.players = {}
        self.board = {}
        self.scores = {}
        self.move_history = []
        self.turn_count = 0
        self.game_data = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "current_player": self.current_player,
            "players": {name: {"name": name, "type": p.type} for name, p in self.players.items()},
            "board": self.board,
            "scores": self.scores,
            "move_history": [move.to_dict() for move in self.move_history],
            "turn_count": self.turn_count,
            "game_data": self.game_data
        }

    def print(self):
        """Prints a human-readable representation of the game state"""
        description = f"Game Phase: {self.phase.value}\n"
        description += f"Current Player: {self.current_player}\n"
        description += "Players:\n"
        for name, player in self.players.items():
            description += f"  {name} ({player.type}) - Stats: {player.stats}\n"
        description += "Board State:\n"
        for position, value in self.board.items():
            description += f"  {position}: {value}\n"
        description += f"Scores: {self.scores}\n"
        description += f"Turn Count: {self.turn_count}\n"
        return description


class CompetitiveGame(ABC):
    """Base class for competitive games where Claude can play"""

    def __init__(self, game_config: Dict[str, Any]):
        # TODO tidy up
        self.config = game_config
        self.state = GameState()
        self.rules = game_config.get('rules', {})
        self.win_conditions = game_config.get('win_conditions', [])

    @abstractmethod
    def get_rules(self) -> Dict[str, Any]:
        """Get the rules of the game"""
        return self.rules

    @abstractmethod
    def setup_game(self, players: List[Player]) -> Dict[str, Any]:
        """Initialize the game with players"""
        pass

    @abstractmethod
    def get_move_description(self) -> str:
        """Get a human-readable description how to enter a valid move"""
        pass

    @abstractmethod
    def get_valid_moves(self, player: str) -> List[Dict[str, Any]]:
        """Get all valid moves for a player"""
        pass

    @abstractmethod
    def apply_move(self, move: GameMove) -> (Dict[str, Any], bool):
        """Apply a move and return the result and boolean indicating repeat move"""
        pass

    @abstractmethod
    def check_win_condition(self) -> Optional[Dict[str, Any]]:
        """Check if game is won and return winner info"""
        pass

    @abstractmethod
    def get_game_description(self) -> str:
        """Get human-readable description of current game state"""
        pass

    def get_player_perspective(self, player_name: str) -> Dict[str, Any]:
        """Get game state from specific player's perspective (may hide info)"""
        # Default: return full state (override for games with hidden information)
        return self.state.to_dict()


class HumanPlayer(Player):
    """Human player that interacts through console input"""

    def make_move(self, game: CompetitiveGame, player_name: str) -> GameMove:
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

    def make_move(self, game: CompetitiveGame, player_name: str) -> GameMove:
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


class GameSession:
    """Manages a competitive game session between players"""

    def __init__(self, game: CompetitiveGame):
        self.game = game
        self.players = {}
        self.move_log = []

    def add_player(self, name: str, player_type: str, claude_api_key: str = None,
                   strategy: str = "competitive", skill_level: str = "intermediate"):
        """Add a player to the game"""
        if player_type == "claude":
            claude_player = ClaudePlayer(name=name, claude_api_key=claude_api_key,
                                         strategy=strategy, skill_level=skill_level,
                                         verbose=False)
            self.players[name] = claude_player
        elif player_type == "human":
            human_player = HumanPlayer(name=name)
            self.players[name] = human_player
        else:
            raise ValueError(f"Unsupported player type: {player.type}")

    def start_game(self) -> Dict[str, Any]:
        """Start the competitive game"""
        players_list = list(self.players.values())
        setup_result = self.game.setup_game(players_list)

        return {
            "status": "game_started",
            "setup": setup_result,
            "current_state": self.game.state.to_dict()
        }

    def play_turn(self) -> List[Dict[str, Any]]:
        """Play one turn of the game, may include multiple moves"""
        current_player = self.game.state.current_player
        print(self.game.get_game_description())

        if not current_player:
            return {"error": "No current player set"}

        player = self.players[current_player]

        repeat = True

        results = []
        while repeat:

            # Get the move
            move = player.make_move(self.game, current_player)

            # Apply the move
            try:
                move_result, repeat = self.game.apply_move(move)
                print(self.game.get_game_description())

            except Exception as e:
                print(f"Error applying move: {e}")
                continue

            self.move_log.append(move)

            # Check win condition
            win_check = self.game.check_win_condition()
            if win_check:
                break

            results.append({
                "player": current_player,
                "move": move.to_dict(),
                "result": move_result,
                "current_state": self.game.state.to_dict()
            })

        if win_check:
            results[-1]["game_over"] = True
            results[-1]["winner"] = win_check

        self.change_player_turn()

        return results

    def change_player_turn(self):
        """Switch to the next player"""
        players = list(self.game.state.players.keys())
        current_idx = players.index(self.game.state.current_player)
        next_idx = (current_idx + 1) % len(players)
        self.game.state.current_player = players[next_idx]

# Example Game Implementation: Tic-Tac-Toe
class TicTacToe(CompetitiveGame):
    """Classic Tic-Tac-Toe implementation"""

    # Winning combinations
    lines = [
        # Rows
        ["0,0", "0,1", "0,2"], ["1,0", "1,1", "1,2"], ["2,0", "2,1", "2,2"],
        # Columns
        ["0,0", "1,0", "2,0"], ["0,1", "1,1", "2,1"], ["0,2", "1,2", "2,2"],
        # Diagonals
        ["0,0", "1,1", "2,2"], ["0,2", "1,1", "2,0"]
    ]

    def get_rules(self) -> str:
        """Get the rules of Tic-Tac-Toe"""
        return """
        Tic-Tac-Toe Rules:
        1. The game is played on a 3x3 grid.
        2. Players take turns placing their symbols (X or O) in empty cells.
        3. The first player to get 3 of their symbols in a row (horizontally, vertically, or diagonally) wins.
        4. If all cells are filled and no player has won, the game ends in a draw.
        """

    def setup_game(self, players: List[Player]) -> Dict[str, Any]:
        if len(players) != 2:
            raise ValueError("Tic-Tac-Toe requires exactly 2 players")

        self.state.phase = GamePhase.PLAYING
        self.state.players = {p.name: p for p in players}
        self.state.current_player = players[0].name

        # Initialize 3x3 board
        self.state.board = {f"{i},{j}": "" for i in range(3) for j in range(3)}
        self.state.scores = {p.name: 0 for p in players}

        # Assign symbols
        self.state.game_data = {
            players[0].name: "X",
            players[1].name: "O"
        }

        return {"message": f"Game started! {players[0].name} is X, {players[1].name} is O"}

    def get_move_description(self) -> str:
        """Get human-readable description how to enter a valid move"""
        return """
        To make a move, enter the row and column of the cell you want to place your symbol in.
        Use the format: row,col (e.g., 1,1 for the center cell).
        Valid positions are from 0,0 (top-left) to 2,2 (bottom-right).
        Example: '1,1' places your symbol in the center cell.
        """

    def get_valid_moves(self, player: str) -> List[Dict[str, Any]]:
        if self.state.phase != GamePhase.PLAYING:
            return []

        valid_moves = []
        for position, value in self.state.board.items():
            if value == "":
                row, col = map(int, position.split(","))
                valid_moves.append({
                    "action": "place",
                    "data": {"position": position, "row": row, "col": col}
                })

        return valid_moves

    def apply_move(self, move: GameMove) -> (Dict[str, Any], bool):
        if move.action != "place":
            raise ValueError("Invalid action for Tic-Tac-Toe")

        position = move.data.get("position")
        if not position or self.state.board.get(position) != "":
            raise ValueError(f"Invalid position: {position}. Selected position must be empty.")

        # Place the symbol
        symbol = self.state.game_data[move.player]
        self.state.board[position] = symbol
        self.state.move_history.append(move)
        self.state.turn_count += 1

        # Switch players
        players = list(self.state.players.keys())
        current_idx = players.index(self.state.current_player)
        self.state.current_player = players[(current_idx + 1) % 2]

        return ({"success": True, "placed": symbol, "position": position}, False)

    def check_win_condition(self) -> Optional[Dict[str, Any]]:
        # Check rows, columns, and diagonals
        board = self.state.board

        for line in self.lines:
            symbols = [board[pos] for pos in line]
            if symbols[0] != "" and all(s == symbols[0] for s in symbols):
                # Found winner
                winner_symbol = symbols[0]
                winner_name = None
                for player, symbol in self.state.game_data.items():
                    if symbol == winner_symbol:
                        winner_name = player
                        break

                self.state.phase = GamePhase.FINISHED
                self.state.scores[winner_name] = 1

                return {
                    "winner": winner_name,
                    "winning_line": line,
                    "method": "line"
                }

        # Check for draw
        if all(cell != "" for cell in board.values()):
            self.state.phase = GamePhase.FINISHED
            return {"winner": None, "method": "draw"}

        return None

    def get_game_description(self) -> str:
        board = self.state.board
        description = f"Current turn: {self.state.current_player}\n"
        description += f"Turn #{self.state.turn_count}\n\n"
        description += "Board:\n"

        for i in range(3):
            row = []
            for j in range(3):
                cell = board[f"{i},{j}"]
                row.append(cell if cell else ".")
            description += " | ".join(row) + "\n"
            if i < 2:
                description += "---------\n"

        description += "\nPossible winning lines:\n"
        for line in self.lines:
            description += " - " + ", ".join(line) + "\n"

        return description


class Squares(CompetitiveGame):

    """Squares game implementation"""
    def __init__(self, game_config: Dict[str, Any], num_squares: int = 4):
        super().__init__(game_config)
        self.num_squares = num_squares

    def _valid_position(self, position: Tuple[int, int, str]) -> bool:
        """Check if the position is valid for placing a square"""
        return position in self.state.board

    def _check_square_completion(self, row: int, col: int) -> bool:
        """Check if placing a line completes a square"""
        # Check if the square at (row, col) is completed
        try:
            return (self.state.boxes[(row, col)] is None and
                    self.state.board[(row, col, 'down')] == 1 and
                    self.state.board[(row, col, 'right')] == 1 and
                    self.state.board[(row + 1, col, 'right')] == 1 and
                    self.state.board[(row, col + 1, 'down')] == 1)
        except KeyError:
            raise ValueError(f"Invalid square position: ({row}, {col}). ")

    def get_rules(self) -> str:
        """Get the rules of the Squares game"""
        return f"""
        Squares Game Rules:
        1. The game is played on a {self.num_squares}x{self.num_squares} grid.
        2. Players take turns placing lines either down or right in empty cells.
        3. A square is completed when all four sides are filled.
        4. The player who completes a square scores a point and gets another turn.
        5. The game ends when all squares are completed.
        6. The player with the most completed squares wins.
        """

    def setup_game(self, players: List[Player]) -> Dict[str, Any]:
        if len(players) != 2:
            raise ValueError("Squares requires exactly 2 players")

        self.state.phase = GamePhase.PLAYING
        self.state.players = {p.name: p for p in players}
        self.state.current_player = players[0].name

        # Initialize the board and boxes
        self.state.board = {(i,j,k): 0 for i in range(self.num_squares)
                            for j in range(self.num_squares) for k in ['down', 'right']}
        # Add last row and column for squares
        self.state.board.update({(i,self.num_squares,'down'): 0 for i in range(self.num_squares)})
        self.state.board.update({(self.num_squares,j,'right'): 0 for j in range(self.num_squares)})
        self.state.boxes = {(i, j): None for i in range(self.num_squares)
                            for j in range(self.num_squares)}

        return {"message": "Game started!"}

    def get_move_description(self) -> str:
        """Get human-readable description how to enter a valid move"""
        return f"""
        To make a move, enter the position in the format: row,col,direction
        where 'direction' is either 'down' or 'right'.
        Example: '1,2,right' places a line going right from the point (1,2).
        Valid positions are from 0,0,('down'|'right') (top-left) to {self.num_squares},{self.num_squares-1},'right' or {self.num_squares-1},{self.num_squares},'down' (lines that will join to the bottom right.
        Note that the last row and column points will only accept 'right' and 'down' respectively.
        """

    def get_valid_moves(self, player: str) -> List[Dict[str, Any]]:
        if self.state.phase != GamePhase.PLAYING:
            return []

        valid_moves = []
        for position, value in self.state.board.items():
            if value == 0:
                row, col, direction = position
                valid_moves.append({
                    "action": "place",
                    "data": {"position": position, "row": row, "col": col, "direction": direction}
                })

        return valid_moves

    def apply_move(self, move: GameMove) -> (Dict[str, Any], bool):
        if move.action != "place":
            raise ValueError("Invalid action for Squares game")

        try:
            position = move.data.get("position")
            # Convert to int,int,str tuple
            position = (int(position[0]), int(position[1]), position[2])
        except ValueError:
            raise ValueError("Invalid position format. Use: row,col,direction")

        print(f"Applying move at position: {position}")

        try:
            if self.state.board[position] != 0:
                raise ValueError(f"Invalid position: {position}. Selected position must be empty "
                                 f" but position {position} already set {self.state.board['position']}.")
        except KeyError:
            raise ValueError(f"Invalid position: {position}. Position does not exist on the board.")

        # Place the value
        self.state.board[position] = 1

        repeat = False

        def _check_and_set_square_completion(row: int, col: int):
            """Check if placing a line completes a square and update state"""
            if self._check_square_completion(row, col):
                # Square completed
                player_name = self.state.current_player
                self.state.boxes[(row, col)] = player_name
                return True
            return False

        # Update scores if a square is completed
        # Check surrounding cells to see if a square is completed
        # Note we have already checked that the position is valid
        # So can not have 'right' or 'down' in the last col or row respectively
        row, col, direction = position
        if direction == 'down':
            # Check squares either side of the vertical line
            repeat = (col < self.num_squares and _check_and_set_square_completion(row, col)) or \
                (col > 0 and _check_and_set_square_completion(row, col-1))
        elif direction == 'right':
            # Check squares either side of the horizontal line
            repeat = (row < self.num_squares and _check_and_set_square_completion(row, col)) or \
                (row > 0 and _check_and_set_square_completion(row-1, col))

        return ({"success": True, "placed": 1, "position": position}, repeat)

    def check_win_condition(self) -> Optional[Dict[str, Any]]:
        # Check if all squares are completed
        if all(value is not None for value in self.state.boxes.values()):
            self.state.phase = GamePhase.FINISHED
            scores = {player: sum(1 for box in self.state.boxes.values() if box == player)
                      for player in self.state.players.keys()}
            winner = max(scores, key=scores.get)
            return {"winner": winner, "scores": scores, "method": "max_squares"}

        return None

    def get_game_description(self) -> str:
        board = self.state.board
        description = f"Current turn: {self.state.current_player}\n"
        description += f"Turn #{self.state.turn_count}\n\n"
        description += "Board:\n"

        # Display the board as a series of dots for empty cells
        # And lines for placed lines
        description += "   " + "    ".join([str(i) for i in range(self.num_squares + 1)]) + "\n"
        for i in range(self.num_squares + 1):
            description += f"{i}  "
            for j in range(self.num_squares + 1):
                description += "."
                if board.get((i, j, 'right'), 0) == 1:
                    description += " -- "
                else:
                    description += "    "
            description += "\n   "
            for j in range(self.num_squares + 1):
                if board.get((i, j, 'down'), 0) == 1:
                    description += "|    "
                else:
                    description += "     "
            description += "\n"

        description += "\nBoxes:\n"
        for (i, j), owner in self.state.boxes.items():
            description += f"Box ({i},{j}): {'Free' if owner is None else owner}\n"

        return description

    def get_player_perspective(self, player_name: str) -> Dict[str, Any]:
        """Get game state from specific player's perspective"""
        state = GameState()
        state.board = {f"{k[0]},{k[1]},{k[2]}": v for k, v in self.state.board.items()}
        state.boxes = {f"{k[0]},{k[1]}": v for k, v in self.state.boxes.items()}
        return state.to_dict()


games = {
    "tic_tac_toe": TicTacToe,
    "squares": Squares
}

def play_game(mode: str = "human", game_class: CompetitiveGame = TicTacToe):
    game = game_class({})
    session = GameSession(game)

    # Allow human input for player type
    strategy = input("Choose Claude's strategy (competitive, friendly, aggressive, defensive): ").strip().lower()
    if strategy not in ["competitive", "friendly", "aggressive", "defensive"]:
        print("Invalid strategy! Defaulting to 'competitive'.")
        strategy = "competitive"
    skill_level = input("Choose Claude's skill level (beginner, intermediate, expert): ").strip().lower()
    if skill_level not in ["beginner", "intermediate", "expert"]:
        print("Invalid skill level! Defaulting to 'expert'.")
        skill_level = "intermediate"

    # Add players
    if mode == "human":
        session.add_player('Human', 'human')
    elif mode == "auto":
        session.add_player('ClaudeOpp', player_type='claude', strategy=strategy, skill_level=skill_level)

    session.add_player(name='Claude', player_type='claude', strategy=strategy, skill_level=skill_level)

    session.start_game()
    players = list(session.players.values())

    print(f"=== {players[0].name} vs {players[1].name} ===")
    print("Enter moves as game.get_move_description().")

    while game.state.phase == GamePhase.PLAYING:
        current_player = game.state.current_player

        try:
            print(f"{current_player} is thinking...")
            result = session.play_turn()
        except Exception as e:
            print(f"Error during turn: {e}")
            continue

        if result[-1].get("game_over"):
            winner = result[-1].get("winner", {}).get("winner")
            if winner:
                print(f"\n🎉 {winner} wins!")
            else:
                print("\n🤝 It's a draw!")
            break


if __name__ == "__main__":
    print("Claude Competitive Game Framework")
    print("================================")
    print("\nAvailable modes:")
    print("1. Claude vs Claude (automatic)")
    print("2. Human vs Claude (interactive)")
    print("\nNote: Set ANTHROPIC_API_KEY environment variable for Claude players")

    choice = '2' #input("Choose mode (1 or 2): ").strip()

    if choice in ['1', '2']:
        mode = "human" if choice == '2' else "auto"
        game_name = 'squares' #input(f"Enter game name (choices: {', '.join(games.keys())}): ").strip().lower()
        game_class = games.get(game_name, TicTacToe)

        play_game(mode, game_class)
    else:
        print("Invalid choice! Exiting.")
        exit(1)
