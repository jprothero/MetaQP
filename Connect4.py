import cv2
from cv2 import matchTemplate as cv2m
import numpy as np
from IPython.core.debugger import set_trace
import torch.functional as F

class Connect4:
    def __init__(self, rows=6, columns=7, datatype="uint8"):
        self.rows = rows
        self.columns = columns
        self.datatype = datatype
        self._create_win_patterns()
        self._create_legal_moves_pattern()
        self._create_valid_pattern()
        self._create_valid_pattern2()
        
    def _create_win_patterns(self):
        self.win_patterns = [i for i in range(4)]
        self.horizontal_win = self.win_patterns[0] = np.array([[1, 1, 1, 1]])
        
        vertical_win = np.array([1, 1, 1, 1])
        vertical_win = np.expand_dims(vertical_win, -1)
        self.vertical_win = self.win_patterns[1] = vertical_win
        
        self.left_diag_win = self.win_patterns[2] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.right_diag_win = self.win_patterns[3] = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ])
        
    def _create_legal_moves_pattern(self):
#         self.legal_move_pattern = np.array([1 for i in range(self.rows)])
        self.legal_move_pattern = np.array([0])
        self.legal_move_pattern = np.expand_dims(self.legal_move_pattern, -1)

    def _create_valid_pattern(self):
#         self.legal_move_pattern = np.array([1 for i in range(self.rows)])
        self.valid_pattern = np.array([1, 0])
        self.valid_pattern = np.expand_dims(self.valid_pattern, -1)

    def _create_valid_pattern2(self):
    #         self.legal_move_pattern = np.array([1 for i in range(self.rows)])
        self.valid_pattern2 = np.array([2])
        self.valid_pattern2 = np.expand_dims(self.valid_pattern2, -1)
        
    def calculate_reward(self, joint_states, idx):
        for k, state in enumerate(joint_states):
            match = cv2m(pattern.astype(self.datatype), state.astype(self.datatype), 
                            cv2.TM_SQDIFF)

            i, j = np.where(match==0)
            if len(i) != 0 or len(j) != 0:
                if k == 0:
                    return 1, True
                else:
                    return -1, True

        return 0, False
    
    def _create_actions(self):
        self.actions = [i for i in range(self.rows*self.columns)]

    def get_legal_actions(self, joint_states):
        assert len(joint_states) == 2
        board = joint_states[0] + joint_states[1]
        
        legal_moves = []
        
        for k in range(board.shape[1]):
            match = cv2m(self.legal_move_pattern.astype(self.datatype), 
                     board[:, k].astype(self.datatype), cv2.TM_SQDIFF)
            
            i, j = np.where(match==0)
            
            if len(i) != 0:
                legal_moves.append(np.max(i)*board.shape[1] + k)

        return legal_moves

    def transition_and_evaluate(self, state, action):
        orig_player = int(state[2][0])        

        idx = action
        
        i, j = np.unravel_index([idx], state.shape)
        i = i[0]
        j = j[0]
        
        state[i][j] = 1

        new_player = (orig_player+1)%2
        state[2] = new_player

        game_over = check_win(state, i, j)

        if game_over:
            result = 1
        else:
            legal_actions = self.get_legal_actions(state[:2])
            if len(legal_actions) == 0:
                result = 0
                game_over = True
            else:
                result = None

        return state, result, game_over

    def get_legal_mask(self, input_states, add_noise=False, deterministic=False):
            legal_moves_mask = np.copy(input_states.data.numpy())

            log_probas_list = []
            legal_moves_lists = []
            
            #legal_moves_mask will be (128,3 , 6, 7)
            #I only care about the first two states, so that's right
            for s, state in enumerate(legal_moves_mask[:, :2]):
                state_legal_moves_list = []
                for k in range(state.shape[1]):
                    match = cv2m(self.legal_move_pattern.astype(self.datatype), 
                                state[:, k].astype(self.datatype), cv2.TM_SQDIFF)
                    
                    i, j = np.where(match==0)
                    
                    if len(i) != 0:
                        #I think I need a k index
                        idx = np.max(i)*state.shape[1] + k
                        legal_moves_mask[s][k][idx] = 1
                        state_legal_moves_list.extend([idx])

                legal_moves_list.extend(state_legal_moves_list)
                legal_moves_mask[s][:2].flatten()[state_legal_moves_list] = 1

            return legal_moves_mask, legal_moves_list

#### Static testing functions
def test_transition():
    connect4 = Connect4()
    
    board = np.zeros(shape=(6, 7))
    
    for i in range(1000):
        action = np.random.randint(42)
        
        test = np.copy(board.flatten())
        test[action] = 1

        board = connect4.transition(board, action)
        
        assert (board.flatten() == test).all()

def test_legal_moves_finder():
    connect4 = Connect4()

    board = np.zeros(shape=(6, 7))
    res = connect4.get_legal_actions(board)
    assert len(res) == 7 and res[0] == 35
    board[1:, :] = 1
    res = connect4.get_legal_actions(board)
    assert len(res) == 7 and res[0] == 0
    board[0][0] = 1
    res = connect4.get_legal_actions(board)
    assert len(res) == 6

def test_win_checkers():
    board = np.zeros((6, 7))
    
    left_win = np.copy(board)
    right_win = np.copy(board)
    up_win = np.copy(board)
    down_win = np.copy(board)
    left_up_diag_win = np.copy(board)
    right_up_diag_win = np.copy(board)
    left_down_diag_win = np.copy(board)
    right_down_diag_win = np.copy(board)
    
    left_win[0][:4] = 1; left_win #0, 3
    assert(check_win(left_win, 0, 3))
    
    right_win[0][:4] = 1; right_win #0, 0
    assert(check_win(right_win, 0, 0))
    
    for i in range(4):
        up_win[i][0] = 1
    up_win #3, 0
    assert(check_win(up_win, 3, 0))
    
    for i in range(4):
        down_win[i][0] = 1
    down_win #0, 0
    assert(check_win(down_win, 0, 0))
    
    for i in range(4):
        left_up_diag_win[i, i] = 1
    left_up_diag_win #3, 3
    assert(check_win(left_up_diag_win, 3, 3))
    
    for i in range(4):
        left_down_diag_win[i, i] = 1
    left_down_diag_win #0, 0
    assert(check_win(left_down_diag_win, 0, 0))
    
    for i in range(4):
        right_up_diag_win[i, 3-i] = 1
    right_up_diag_win #3, 0
    assert(check_win(right_up_diag_win, 3, 0))
    
    for i in range(4):
        right_down_diag_win[i, 3-i] = 1
    right_down_diag_win #0, 3
    assert(check_win(right_down_diag_win, 0, 3))

def check_win(state, i, j):
    done = False
    done = check_left(state, i, j)
    if done:
        return done
    done = check_right(state, i, j)
    if done:
        return done
    done = check_up(state, i, j)
    if done:
        return done
    done = check_down(state, i, j)
    if done:
        return done
    done = check_diag_left_up(state, i, j)
    if done:
        return done
    done = check_diag_left_down(state, i, j)
    if done:
        return done
    done = check_diag_right_up(state, i, j)
    if done:
        return done
    done = check_diag_right_down(state, i, j)
    return done

def check_left(state, i, j):
    if j > 2:
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i][j-k] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False

def check_right(state, i, j):
    if j < (state.shape[1]-3):
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i][j+k] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False

def check_up(state, i, j):
    if i > 2:
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i-k][j] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False

def check_down(state, i, j):
    if i < (state.shape[0]-3):
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i+k][j] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False

def check_diag_left_up(state, i, j):
    if i > 2 or j > 2:
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i-k][j-k] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False

def check_diag_left_down(state, i, j):
    if i < (state.shape[0]-3) and j > 2:
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i+k][j-k] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False

def check_diag_right_up(state, i, j):
    if i < (state.shape[0]-3) and j < (state.shape[1]-3):
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i+k][j+k] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False

def check_diag_right_down(state, i, j):
    if i > 2 or j < (state.shape[1]-3):
        num_in_a_row = 0
        
        for k in range(1, 4):
            if state[i-k][j+k] == 0:
                break
            else:
                num_in_a_row += 1
            if num_in_a_row == 3:
                return True
            
    return False