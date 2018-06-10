import unittest
import hello

class HelloTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(hello.say_hi(),
        'Hello World')


if __name__ == '__main__':
    unittest.main()
