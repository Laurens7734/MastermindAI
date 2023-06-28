from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class MastermindEnv(py_environment.PyEnvironment):
    def __init__(self, amountOfLetters, amountOfTurns, amountOfPositions):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=amountOfLetters-1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(amountOfTurns,amountOfPositions,2), dtype=np.int32, minimum=0, name="observation"
        )

        self.amountOfLetters = amountOfLetters
        self.amountOfTurns = amountOfTurns
        self.amountOfPositions = amountOfPositions

        options = []
        start = ord('a')
        for i in range(amountOfLetters):
            options.append(chr(start+i))
        self.options = options

        self.reset()
        
        
    def action_spec(self):
        
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.generateCode(self.amountOfPositions)
        self.currentTurn = 0
        self.currentGuess = ""
        self._episode_ended = False
        self._state = [[[0 for _ in range(2)]for _ in range(self.amountOfPositions)]for _ in range(self.amountOfTurns)]
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if action >= self.amountOfLetters:
            raise ValueError(f"`action` must be lower than the amount of letters")

        self._state[self.currentTurn][len(self.currentGuess)][0] = action
        self.currentGuess += self.options[action]

        if len(self.currentGuess) == self.amountOfPositions:
            numberOfCorrect, remainingInCode, remainingInGuess = self.findCorrectGuesses(self.currentGuess)
            numberOfMisplaced = self.findWrongPositions(remainingInCode,remainingInGuess)
            self.setTurnResults(numberOfCorrect, numberOfMisplaced)
            self.currentTurn += 1
            self.currentGuess = ""
            if numberOfCorrect == self.amountOfPositions:
                self._episode_ended = True
                return ts.termination(np.array(self._state, dtype=np.int32), self._get_reward())
            if self.currentTurn == self.amountOfTurns:
                self._episode_ended = True
                return ts.termination(np.array(self._state, dtype=np.int32), -100)
        
        return ts.transition(np.array(self._state, dtype=np.int32), reward=0, discount=1.0)

    def _get_reward(self):
        return 500-(10*self.currentTurn)
    
    def setTurnResults(self, numberCorrect, numberInTheWrongPlace):
        for i in range(self.amountOfPositions):
            if numberCorrect > 0:
                self._state[self.currentTurn][i][1] = 1
                numberCorrect -= 1
            elif numberInTheWrongPlace > 0:
                self._state[self.currentTurn][i][1] = 2
                numberInTheWrongPlace -= 1
            
    
    def generateCode(self, codeLength):
        code = ""
        for i in range(codeLength):
            code += random.choice(self.options)
        self.code = code

    def findCorrectGuesses(self, guess):
        numberOfCorrect = 0
        unusedFromCode = []
        unusedFromQuess = []
        for i in range(self.amountOfPositions):
            if self.code[i] == guess[i]:
                numberOfCorrect +=1
            else:
                unusedFromCode.append(self.code[i])
                unusedFromQuess.append(guess[i])
        return (numberOfCorrect,unusedFromCode,unusedFromQuess)
    
    def findWrongPositions(self, remainingInCode,remainingInGuess):
        numberOfMatches = 0
        for letter in remainingInGuess:
            for i in range(len(remainingInCode)):
                if remainingInCode[i] == letter:
                    numberOfMatches += 1
                    del remainingInCode[i]
                    break
        return numberOfMatches