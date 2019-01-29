from isolation import Isolation, DebugState
from sample_players import MinimaxPlayer
import datetime
from datetime import timedelta
import pickle
import random
import math
import json

class CustomPlayer(MinimaxPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        self.debug = True
        # self.alpha_beta_search(state)
        # self.mcts(state)
        # self.mcts(state, use_stm=True)
        # self.mcts(state, use_ltm=True)
        self.mcts(state, use_stm=True, use_ltm=True)

    # MiniMax with Alpha-Beta Pruning
    def alpha_beta_search(self, state):
        def min_value(state, depth, alpha, beta):
            if state.terminal_test():
                v = state.utility(self.player_id)
            elif depth <= 0:
                v = self.score(state)
            else:
                v = float("inf")
                for a in state.actions():
                    result, alpha, beta = max_value(state.result(a), depth - 1, alpha, beta)
                    v = min(v, result)
                    if v <= alpha:
                        break
                    beta = min(beta, v)
            return (v, alpha, beta)

        def max_value(state, depth, alpha, beta):
            if state.terminal_test():
                v = state.utility(self.player_id)
            elif depth <= 0:
                v = self.score(state)
            else:
                v = float("-inf")
                for a in state.actions():
                    result, alpha, beta = min_value(state.result(a), depth - 1, alpha, beta)
                    v = max(v, result)
                    if v >= beta:
                        break
                    alpha = max(alpha, v)
            return (v, alpha, beta)

        if state.ply_count < 2:
            random_move = random.choice(state.actions())
            self.queue.put(random_move)
        else:
            alpha = float("-inf")
            beta = float("inf")
            best_score = float("-inf")
            best_move = None
            available_actions = state.actions()
            if len(available_actions) == 1:
                self.queue.put(available_actions[0])
            else:
                # Apply iterative deepening in an attempt to make the full use of the time available.
                for depth in range(1, 5):
                    for a in available_actions:
                        v, alpha, beta = min_value(state.result(a), depth - 1, alpha, beta)
                        if v > best_score:
                            best_score = v
                            best_move = a

                    self.queue.put(best_move)

    def save_data(self, short_term=None, ltm_support=None, long_term=None):
        if not self.data:
            self.data = {}
        if short_term:
            if 'short_term' in self.data.keys():
                self.data['short_term'][self.player_id] = short_term
            else:
                self.data['short_term'] = {self.player_id: short_term}
        if ltm_support:
            if 'ltm_support' in self.data.keys():
                self.data['ltm_support'][self.player_id] = ltm_support
            else:
                self.data['ltm_support'] = {self.player_id: ltm_support}
        if long_term:
            self.data['long_term'] = long_term
        with open("data.pickle", 'wb') as f:
            pickle.dump(self.data, f)

    def load_data(self):
        self.log('Retrieving memory')
        try:
            with open("data.pickle", "rb") as f:
                self.data = pickle.load(f)
        except (IOError, TypeError) as e:
            self.log(str(e))
            self.data = None

    def mcts(self, state, use_stm=False, use_ltm=False):
        """MCTS (Monte Carlo Tree Search)

        Parameters
        ----------
        state : `isolation.Isolation`
            An instance of `isolation.Isolation` encoding the current state of the
        use_stm : `Boolean`
            A boolean value instructing the method to make use of short term memory persistence
            or not. The short term memory is used to persist the stats tree from one ply to the
            next. As long as the algorithm has sufficient time to execute, the persistence of the
            stats tree should allow the algorithm to accumulate additional knowledge of the tree
            with each additional ply.
        use_stm : `Boolean`
            A boolean value instructing the method to make use of long term memory persistence
            or not. The long term memory is used to persist statistics about games played to help
            refine which move is best for the opening move given whether the player is playing first
            or second and if playing second, where did the oponent place its piece.
        """
        self.log('************* PLY NUMBER: {} - PLAYER: {} *************'.format(state.ply_count, self.player_id))
        if self.debug: print(DebugState.from_state(state))

        current_location = state.locs[self.player_id]
        self.log('Current location: {}'.format(current_location))
        opponent_location = state.locs[self.player_id - 1]
        self.log('Oponent\'s location: {}'.format(opponent_location))

        # If the algorithm has so little time to execute a single iteration, make sure at least one
        # action has been recorded.
        # self.queue.put(random.choice(state.actions()))

        if use_stm or use_ltm:
            self.load_data();

        if use_stm:

            if self.data and self.data['short_term'] and self.player_id in self.data['short_term'].keys():
                self.log('STM Recovered')
                short_term_memory = self.data['short_term'][self.player_id]
                self.log('STM Ply: {}'.format(short_term_memory['ply']))
                self.log('STM at: {}'.format(short_term_memory['at']))
                self.log('STM oponent_at: {}'.format(short_term_memory['oponent_at']))
            else:
                self.log('STM Reset')
                short_term_memory = {
                    'ply': state.ply_count,
                    'at': current_location,
                    'oponent_at': opponent_location,
                    'stats_tree': Tree()
                }

        if use_ltm:
            if self.data:
                if self.player_id in self.data['ltm_support'].keys():
                    ltm_support = self.data['ltm_support'][self.player_id]
                else:
                    ltm_support = self.initialise_ltm_support(state)
                long_term_memory = self.data['long_term']
            else:
                # Initialise data for the first time as none is present.
                ltm_support = self.initialise_ltm_support(state)
                long_term_memory = self.initialise_ltm()
                self.save_data(ltm_support=ltm_support, long_term=long_term_memory)


            if state.ply_count < 2:
                # Reset LTM support as this is the first ply of a new game.
                ltm_support = self.initialise_ltm_support(state)

                if state.ply_count == 0:
                    opening_ltm = long_term_memory['first_moves']
                else:
                    opening_ltm = long_term_memory['second_moves'][opponent_location]
                    ltm_support['oponent_moves'].append(opponent_location)

                opening_move = self.opening_move(opening_ltm)
                self.save_data(ltm_support=ltm_support)
                self.queue.put(opening_move)

                return
            else:
                ltm_support['own_moves'].append(current_location)
                ltm_support['oponent_moves'].append(opponent_location)
                self.save_data(ltm_support=ltm_support)

        available_actions = state.actions()
        if len(available_actions) == 1:
            only_action = available_actions[0]
            self.queue.put(only_action)
            self.log('------------ ONLY ONE AVAILABLE ACTION: {} ------------'.format(only_action))

            # Unless the algorithm can learn from this ply, there is no need to continue
            if not use_stm and not use_ltm:
                return

        if use_stm and state.ply_count > 3 and short_term_memory['ply'] == state.ply_count - 2:
            self.log('Reconstructing Tree')
            previous_stats_tree = short_term_memory['stats_tree']
            if self.debug: print_tree(previous_stats_tree)
            self.log('Previous stats tree children: {}'.format(previous_stats_tree.children))

            if len(previous_stats_tree.children) == 0:
                self.log('---------- NO SHORT TERM MEMORY FOR PLY > 2 ----------')
                self.initialise_stats_tree(state)
            else:
                previous_location = short_term_memory['at']
                previous_oponent_location = short_term_memory['oponent_at']
                self.log('previous_location: {}'.format(previous_location))
                self.log('previous_oponent_location: {}'.format(previous_oponent_location))

                previous_action = current_location - previous_location
                oponents_action = opponent_location - previous_oponent_location
                self.log('Previous Action: {}'.format(previous_action))
                previous_action_node = [n for n in previous_stats_tree.children if n.action == previous_action]
                self.log('previous_action_node: {}'.format(previous_action_node))

                if len(previous_action_node) > 0:
                    oponent_nodes = previous_action_node[0].children
                    self.log('Oponent\'s options: {}'.format(oponent_nodes))
                    self.log('Oponent\'s Action: {}'.format(oponents_action))
                    self.stats_tree = [n for n in oponent_nodes if n.action == oponents_action][0]
                    self.stats_tree.action = 'root'
                    self.log('++++++++++ SHORT TERM MEMORY recovered from data ++++++++++')
                    num_matching_actions = 0
                    self.log('Immediate children: {}'.format(self.stats_tree.children))
                    self.log('Available actions: {}'.format(state.actions()))
                    for action in state.actions():
                        if len([n for n in self.stats_tree.children if n.action == action]) > 0:
                            num_matching_actions += 1
                        else:
                            self.stats_tree.add_child(Tree(action))
                    self.log('Number of actions matching immediate children: {}'.format(num_matching_actions))
                    self.log('Stats tree recovered from STM for this ply:')
                    if self.debug: print_tree(self.stats_tree)
                else:
                    self.log('---------- NO MATCHING DATA IN MEMEORY ----------')
                    self.initialise_stats_tree(state)
        else:
            if use_stm:
                self.log('---------- NO SHORT TERM MEMORY ----------')
            self.initialise_stats_tree(state)

        if use_stm:
            self.log('Assigning STM')
            short_term_memory['ply'] = state.ply_count
            short_term_memory['at'] = current_location
            short_term_memory['oponent_at'] = opponent_location
            short_term_memory['stats_tree'] = self.stats_tree

        evaluated_endgame = False
        if use_stm and self.stats_tree.fully_explored():
            self.log('STM Recovered stats tree is fully explored')
            self.save_data(short_term=short_term_memory)
            best_child = self.stats_tree.best_child()
            self.queue.put(best_child.action)
        else:
            # Keep iterating MCTS updating the best move until the game controller times out the
            # iteration or the tree is fully explored
            while any([not n.fully_explored() for n in self.stats_tree.children]):
                self.mcts_iteration(state)
                best_child = self.stats_tree.best_child()
                # self.log('MCTS Iteration best child: {}'.format(best_child))
                self.queue.put(best_child.action)

                if use_stm:
                    # self.log('Update the STM stats tree and save')
                    short_term_memory['stats_tree'] = self.stats_tree
                    self.save_data(short_term=short_term_memory)

                if use_ltm and not evaluated_endgame and len(best_child.children) == 0:
                    final_state = self.evaluate_end_of_game(state, [best_child.action])
                    if final_state:
                        self.update_ltm_with_end_of_game(final_state, ltm_support, long_term_memory)
                        evaluated_endgame = True

        if use_ltm and not evaluated_endgame:
            # When the stats_tree is fully explored determine whether this is the end of the game:
            actions = [best_child.action]
            if len(best_child.children) > 0:
                actions.append(best_child.worst_child().action)
            final_state = self.evaluate_end_of_game(state, actions)

            if final_state:
                self.update_ltm_with_end_of_game(final_state, ltm_support, long_term_memory)

    def initialise_stats_tree(self, state):
        self.stats_tree = Tree()
        for action in state.actions():
            self.stats_tree.add_child(Tree(action))

    def initialise_ltm_support(self, state):
        return {'initial_ply': state.ply_count, 'own_moves': [], 'oponent_moves': []}

    def initialise_ltm(self):
        board_positions = Isolation().actions()
        self.log('initialising LTM with board positions: {}'.format(board_positions))
        return {
            'first_moves': {p:{'wins': 0, 'games': 0} for p in board_positions},
            'second_moves': {op:{p:{'wins': 0, 'games': 0} for p in board_positions if p != op} for op in board_positions}
        }

    def opening_move(self, memory):
        self.log('Opening Move memory: {}'.format(memory))
        total_games = sum(memory[position]['games'] for position in memory)
        self.log('Opening Move total_games: {}'.format(total_games))
        def ucb1(experience):
            if experience['games'] == 0:
                return float("inf")
            return experience['wins']/experience['games'] + 2 * math.sqrt(math.log(total_games) /  experience['games'])

        return max(list(memory.keys()), key=lambda position: ucb1(memory[position]))

    def evaluate_end_of_game(self, state, actions):
        self.log('Evaluating end of game')
        # self.log('Actions to evaluate: {}'.format(actions))
        if state.terminal_test():
            self.log('State is terminal')
            return state
        if len(actions) == 0:
            self.log('This is not the end of the game.')
            return None
        next_action = actions.pop(0)
        # self.log('Next action to check: {}'.format(next_action))
        return self.evaluate_end_of_game(state.result(next_action), actions)

    def update_ltm_with_end_of_game(self, final_state, ltm_support, long_term_memory):
        will_win = final_state.utility(self.player_id) > 0
        first_move = ltm_support['own_moves'][0]
        if ltm_support['initial_ply'] == 0:
            opening = long_term_memory['first_moves'][first_move]
            opening['games'] += 1
            if will_win:
                opening['wins'] += 1
            long_term_memory['first_moves'][first_move] = opening
        else:
            oponents_first_move = ltm_support['oponent_moves'][0]
            opening = long_term_memory['second_moves'][oponents_first_move][first_move]
            opening['games'] += 1
            if will_win:
                opening['wins'] += 1
            long_term_memory['second_moves'][oponents_first_move][first_move] = opening
        self.save_data(long_term=long_term_memory)

    def mcts_iteration(self, state):
        leaf, state = self.selection(state)
        if not leaf: return
        child = self.expansion(leaf, state)
        won = self.simulation(child, state)
        self.back_propagation(child, won)

    def selection(self, state):
        # Selection: start from root R and select successive child nodes until a leaf node L is reached.
        # The root is the current game state and a leaf is any node from which no simulation (playout)
        # has yet been initiated.

        def choose_from_children(children):
            # non_terminal_children = [n for n in children if not n.terminal]
            leaf_nodes = [n for n in children if len(n.children) == 0 and not n.terminal]
            if len(leaf_nodes) > 0:
                return random.choice(leaf_nodes)
            explorabale_children = [n for n in children if not n.fully_explored()]
            if len(explorabale_children) == 0:
                return None
            return max(explorabale_children, key=lambda node: node.ucb1())

        original_actions = state.actions()
        leaf = None
        children = self.stats_tree.children
        level = 0
        while leaf == None:
            level += 1
            choice = choose_from_children(children)
            if choice:
                state = state.result(choice.action)
                if len(choice.children) == 0:
                    leaf = choice
                else:
                    children = choice.children
            else:
                self.log('Selection: TREE FULLY EXPLORED')
                return (leaf, state)
        return (leaf, state)

    def expansion(self, leaf, state):
        # Expansion: unless L ends the game decisively (e.g. win/loss/draw) for either player,
        # create one (or more) child nodes and choose node C from one of them. Child nodes are any
        # valid moves from the game position defined by L.
        child = leaf
        if state.terminal_test():
            leaf.terminal = True
        elif leaf.simulations > 0:
            for action in state.actions():
                leaf.add_child(Tree(action))
            child = random.choice(leaf.children)
        return child

    def simulation(self, child, state):
        # Simulation: complete one random playout from node C. This step is sometimes also called
        # playout or rollout. A playout may be as simple as choosing uniform random moves until the
        # game is decided (for example in chess, the game is won, lost, or drawn).

        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))

        return state.utility(self.player_id) > 0

    def back_propagation(self, child, won):
        # Backpropagation: use the result of the playout to update information in the nodes on the
        # path from C to R.
        node = child
        while node:
            node.simulations += 1
            if won:
                node.wins += 1
            node = node.parent

    def log(self, message):
        if self.debug: print(message)

# References
# - https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
# - AI 101: Monte Carlo Tree Search (https://www.youtube.com/watch?v=lhFXKNyA0QA)
# - Monte Carlo Tree Search (MCTS) Tutorial: https://www.youtube.com/watch?v=Fbs4lnGLS8M
# - https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
# - https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python-are-there-any-built-in-data-structures-in
# - https://www.cs.mcgill.ca/~vkules/bandits.pdf
# - http://mcts.ai/pubs/mcts-survey-master.pdf
# - Monte Carlo Tree Search https://www.youtube.com/watch?v=UXW2yZndl7U


# Modified version from https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python-are-there-any-built-in-data-structures-in
class Tree(object):
    def __init__(self, action='root', children=None):
        self.action = action
        self.wins = 0
        self.simulations = 0
        self.parent = None
        self.children = []
        self.terminal = False
        self._fully_explored = False
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        type = 'node'
        if self._fully_explored: type = 'FULLY EXPLORED'
        if self.terminal: type = 'TERMINAL'
        return "{} with action {}: {}/{}".format(type, self.action, self.wins, self.simulations)
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
        node.parent = self
    def fully_explored(self):
        if not self._fully_explored:
            self._fully_explored = self.terminal or (len(self.children) > 0 and all([n.fully_explored() for n in self.children]))
        return self._fully_explored
    def best_child(self):
        return max(self.children, key=lambda node: node.score())
    def worst_child(self):
        return min(self.children, key=lambda node: node.score())
    def score(self):
        if self.simulations == 0:
            return 0
        return self.wins / self.simulations
    def ucb1(self):
        """
        UCT: Upper Confidence Bound 1 applied to trees
        wi / ni + c * sqrt(ln(Ni)/ni)
        wi stands for the number of wins for the node considered after the i-th move
        ni stands for the number of simulations for the node considered after the i-th move
        Ni stands for the total number of simulations after the i-th move
        c is the exploration parameter—theoretically equal to √2; in practice usually chosen empirically
        """
        if self.simulations == 0:
            return float("inf")
        return self.score() + 2 * math.sqrt(math.log(self.parent.simulations) /  self.simulations)

# https://stackoverflow.com/questions/30893895/how-to-print-a-tree-in-python
def print_tree(current_node, indent="", last='updown'):

    nb_children = lambda node: sum(nb_children(child) for child in node.children) + 1
    size_branch = {child: nb_children(child) for child in current_node.children}

    """ Creation of balanced lists for "up" branch and "down" branch. """
    up = sorted(current_node.children, key=lambda node: nb_children(node))
    down = []
    while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
        down.append(up.pop())

    """ Printing of "up" branch. """
    for child in up:
        next_last = 'up' if up.index(child) is 0 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', " " * len(str(current_node)))
        print_tree(child, indent=next_indent, last=next_last)

    """ Printing of current node. """
    if last == 'up': start_shape = '┌'
    elif last == 'down': start_shape = '└'
    elif last == 'updown': start_shape = ' '
    else: start_shape = '├'

    if up: end_shape = '┤'
    elif down: end_shape = '┐'
    else: end_shape = ''

    print('{0}{1}{2}{3}'.format(indent, start_shape, str(current_node), end_shape))

    """ Printing of "down" branch. """
    for child in down:
        next_last = 'down' if down.index(child) is len(down) - 1 else ''
        next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', " " * len(str(current_node)))
        print_tree(child, indent=next_indent, last=next_last)
