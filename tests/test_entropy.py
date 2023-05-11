import unittest
import decision_trees.main as main

# Okay, I know this is done slightly incorrectly - ideally the class is more broad and I'd have a single test_entropy method
# to run the various cases, but whatever.

class TestEntropy(unittest.TestCase):
    
    def test_out_of_range_low(self):
        with self.assertRaises(ValueError):
            main.calculate_entropy(-1)
            
    def test_out_of_range_high(self):
        with self.assertRaises(ValueError):
            main.calculate_entropy(1.1)
        
    def test_zero_percent(self):
        self.assertEqual(main.calculate_entropy(0), 0)
        
    def test_100_percent(self):
        self.assertEqual(main.calculate_entropy(1), 0)
        
    def test_skewed_percent(self):
        self.assertAlmostEqual(main.calculate_entropy(5/6), 0.65, 2)
        
    def test_inverted_skewed_percent(self):
        self.assertAlmostEqual(main.calculate_entropy(5/6), main.calculate_entropy(1/6))
        
    def test_max_entropy(self):
        self.assertEqual(main.calculate_entropy(0.5), 1.0)
    

if __name__ == "__main__":
    unittest.main()
    