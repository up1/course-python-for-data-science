import unittest
import fizzbuzz

class FizzBuzzTest(unittest.TestCase):
    def test_should_say_1_with_number_1(self):
        self.assertEqual(fizzbuzz.say(1), '1')

    def test_should_say_2_with_number_2(self):
        self.assertEqual(fizzbuzz.say(2), '2')

    def test_should_say_Fizz_with_number_3(self):
        self.assertEqual(fizzbuzz.say(3), 'Fizz')

    def test_should_say_Bizz_with_number_5(self):
        self.assertEqual(fizzbuzz.say(5), 'Buzz')

    def test_should_say_FizzBizz_with_number_15(self):
        self.assertEqual(fizzbuzz.say(15), 'FizzBuzz')

if __name__ == '__main__':
    unittest.main()
