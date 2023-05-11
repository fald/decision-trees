import unittest
import decision_trees.main as main


class TestInfoGain(unittest.TestCase):
    
    def test_entropy(self):
        with self.assertRaises(ValueError):
            main.calculate_entropy(-0.1)
        with self.assertRaises(ValueError):
            main.calculate_entropy(1.1)
        self.assertEqual(main.calculate_entropy(0), 0)
        self.assertEqual(main.calculate_entropy(1), 0)
        self.assertAlmostEqual(main.calculate_entropy(5/6), 0.65, 2)
        self.assertAlmostEqual(main.calculate_entropy(5/6), main.calculate_entropy(1/6))        
        self.assertEqual(main.calculate_entropy(0.5), 1.0)
        
    def test_count_positives(self):
        self.assertEqual(main.count_positives(main.sample_data), 5)
        with self.assertRaises(TypeError):
            self.assertEqual(main.count_positives(main.sample_data, 0), 2) # Only want boolean vals as per the docstring
            
    def test_split_data(self):
        # Only testing for binary splits right now
        l, r = main.split_data(main.sample_data, 0)
        self.assertEqual(len(l), 5)
        self.assertEqual(len(l), len(r))
        l, r = main.split_data(main.sample_data, 1)
        self.assertEqual(len(l) +  len(r), 10)
        self.assertTrue(len(l) == 7 or len(r) == 7)
        
    def test_information_gain(self):
        self.assertAlmostEqual(main.information_gain(main.sample_data, 0), 0.28, 2)
    

if __name__ == "__main__":
    unittest.main()
    