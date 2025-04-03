import unittest
import numpy as np

from solution import MultinomialLogReg, OrdinalLogReg

class HW2OwnTests(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[1,2,3],
                            [2,4,6],
                            [-3,-2,-1],
                            [-10,-5,-4]
        ])
    
        self.y_train = np.array([1, 1, 0, 0])
        self.X_test = np.array([[7,8,9]])

        
    def test_Multinom(self):
        reg = MultinomialLogReg()
        classifier = reg.build(self.X_train, self.y_train)
        probabilities = classifier.predict(self.X_test)
        print(probabilities)
        self.assertTrue(probabilities.shape == (1, 2))
        self.assertEqual(np.argmax(probabilities), 1)

    def test_Ordinal(self):
        reg = OrdinalLogReg()
        classifier = reg.build(self.X_train, self.y_train)
        probabilities = classifier.predict(self.X_test)
        self.assertTrue(probabilities.shape == (1, 2))
        self.assertEqual(np.argmax(probabilities), 1)

    


if __name__ == "__main__":
    unittest.main()
