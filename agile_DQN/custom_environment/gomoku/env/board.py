import random

class Board:
    def __init__(self):
        # internally self.board.squares holds a flat representation of tic tac toe board
        # where an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # where indexes are column wise order
        # 0 3 6
        # 1 4 7
        # 2 5 8

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        self.boardsize = 15**2
        self.length = 5
        self.squares = [0] * self.boardsize
        '''
        random_index1 = random.randint(0, self.boardsize - 1)
        random_index2 = random.randint(0, self.boardsize - 1)
        while random_index1 == random_index2:
            random_index2 = random.randint(0, self.boardsize - 1)
        self.squares[random_index1] = 1
        self.squares[random_index2] = 2'''

        # precommute possible winning combinations
        self.calculate_winners()

    def setup(self):
        self.calculate_winners()

    def play_turn(self, agent, pos):
        # if spot is empty
        if self.squares[pos] != 0:
            return
        if agent == 0:
            self.squares[pos] = 1
        elif agent == 1:
            self.squares[pos] = 2
        return

    def calculate_winners(self):
        winning_combinations = []
        indices = [x for x in range(0, self.boardsize)]

        # Vertical combinations
        for z in range(0,len(indices),15)[0:11]:
            for y in range(z,z+15):
                winning_combinations.append(tuple(indices[x] for x in range(y, y+15*5, 15)))

        # Horizontal combinations
        for j in range(0,11):
            for i in range(j,len(indices),15):
                winning_combinations.append(tuple(indices[i: i+5]))

        # Diagonal combinations
        # Negative diagonal
        for z in range(0,len(indices),15)[0:11]:
            for y in range(z+4,z+15):
                winning_combinations.append(tuple(indices[x] for x in range(y, y+15*4, 14)))

        # Positive diagonal
        for z in range(0,len(indices),15)[0:11]:
            for y in range(z,z+11):
                winning_combinations.append(tuple(indices[x] for x in range(y, y+15*5, 16)))

        self.winning_combinations = winning_combinations

    # returns:
    # -1 for no winner
    # 1 -- agent 0 wins
    # 2 -- agent 1 wins
    def check_for_winner(self):
        winner = -1
        for combination in self.winning_combinations:
            states = []
            for index in combination:
                states.append(self.squares[index])
            if all(x == 1 for x in states):
                winner = 1
            if all(x == 2 for x in states):
                winner = 2
        return winner

    def check_game_over(self):
        winner = self.check_for_winner()

        if winner == -1 and all(square in [1, 2] for square in self.squares):
            # tie
            return True
        elif winner in [1, 2]:
            return True
        else:
            return False

    def __str__(self):
        return str(self.squares)
