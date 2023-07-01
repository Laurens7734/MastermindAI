import unittest
import MastermindEnv

class TestTheEnv(unittest.TestCase):
    def test_score_no_moves(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        self.assertEqual(env._get_reward(),500)
    
    def test_score_one_guess(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        for i in range(4):
            env.step(0)
        self.assertEqual(env._get_reward(),490)

    def test_score_two_guesses(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        for i in range(8):
            env.step(0)
        self.assertEqual(env._get_reward(),480)

    def test_state_insert(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        response = env.step(1)
        response = env.step(3)
        response = env.step(2)
        self.assertEqual(response.observation[0][0][0],1)
        self.assertEqual(response.observation[0][1][0],3)
        self.assertEqual(response.observation[0][2][0],2)

    def test_state_insert_next_line(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        for i in range(4):
            env.step(0)
        response = env.step(1)
        response = env.step(3)
        response = env.step(2)
        self.assertEqual(response.observation[1][0][0],1)
        self.assertEqual(response.observation[1][1][0],3)
        self.assertEqual(response.observation[1][2][0],2)

    def test_correct_guesses(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        env.code="abcd"
        result,_,_ = env.findCorrectGuesses("cdce")
        self.assertEqual(result,1)

    def test_remaining_chars(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        env.code="abcd"
        _,code,guess = env.findCorrectGuesses("cdce")
        self.assertIn('a',code)
        self.assertIn('b',code)
        self.assertIn('d',code)
        self.assertIn('c',guess)
        self.assertIn('d',guess)
        self.assertIn('e',guess)

    def test_wrong_position(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        env.code="abcd"
        result = env.findWrongPositions(['a', 'b', 'd'],['c', 'd', 'e'])
        self.assertEqual(result,1)

    def test_state_results(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        env.code="abcd"
        response = env.step(1)
        response = env.step(3)
        response = env.step(2)
        response = env.step(2)
        self.assertEqual(response.observation[0][0][1],1)
        self.assertEqual(response.observation[0][1][1],2)
        self.assertEqual(response.observation[0][2][1],2)
        self.assertEqual(response.observation[0][3][1],0)

    def test_score_on_succes(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        env.code="abcd"
        response = env.step(0)
        response = env.step(1)
        response = env.step(2)
        response = env.step(3)
        self.assertEqual(response.reward,490)

    def test_score_on_fail(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        env.code="abcd"
        for i in range(48):
            response = env.step(0)
        self.assertEqual(response.reward,-100)

    def test_score_on_succes_after_fail(self):
        env = MastermindEnv.MastermindEnv(6,12,4)
        env.code="abcd"
        for i in range(4):
            env.step(0)
        response = env.step(0)
        response = env.step(1)
        response = env.step(2)
        response = env.step(3)
        self.assertEqual(response.reward,480)

if __name__ == '__main__':
    unittest.main()