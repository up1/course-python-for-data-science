def divide_by_15(number):
    return number%15 == 0

def say(number):
    if divide_by_15(number):
        return 'FizzBuzz'
    if number%3 == 0:
        return 'Fizz'
    if number%5== 0:
        return 'Buzz'
    return str(number)
    # return "{0}".format(number)
