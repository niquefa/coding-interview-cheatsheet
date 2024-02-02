import sys, math, queue, itertools, collections, heapq, functools, string 
import random, re, json, datetime, bisect, array, pprint
from dataclasses import dataclass, field
from typing import List
def LLI():  return [list(map(int, l.split())) for l in sys.stdin.readlines().strip()]
def LI_():  return [int(x) - 1 for x in sys.stdin.readline().strip().split()]
def LF():   return [float(x) for x in sys.stdin.readline().strip().split()] 
def MI():   return map(int, sys.stdin.readline().strip().split())
def LI():   return list(map(int, sys.stdin.readline().strip().split()))
def LS():   return sys.stdin.readline().strip().split()
def I():    return int(sys.stdin.readline().strip())
def F():    return float(sys.stdin.readline().strip())
def S():    return sys.stdin.readline().strip()
def debug(*args):   print(*args, file=sys.stderr)

# python -m unittest discover -v

def math_utilities():
    n = 123.456789
    debug(abs(n)) # 123.456789
    debug(math.ceil(n)) # 124
    debug(math.floor(n)) # 123
    debug(math.trunc(n)) # 123
    debug(round(n)) # 123
    debug(round(n, 2)) # 123.46
    debug(round(n, -1)) # 120.0
    debug(round(n, -2)) # 100.0


def collections_utilities():
    deq = collections.deque([1, 2, 3, 4, 5])
    deq.appendleft(0)  # Insert at the beginning
    deq.append(6)      # Insert at the end
    debug('Deque:', deq) # ([0, 1, 2, 3, 4, 5, 6])
    debug('Popped from left:', deq.popleft()) # 0
    debug('Popped from right:', deq.pop()) # 6  
    cnt = collections.Counter('abracadabraaaabbb') 
    debug('Counter:', cnt) #Counter({'a': 8, 'b': 5, 'r': 2, 'c': 1, 'd': 1})
    debug('Most common:', cnt.most_common(3)) # [('a', 8), ('b', 5), ('r', 2)]
    for key, value in cnt.items():
        debug(f'Key: {key}({type(key)}), Value: {value}({type(value)})')     
        # Key: a(<class 'str'>), Value: 8(<class 'int'>)   
    ord_dict = collections.OrderedDict([('x', 1), ('b', 2), ('a', 3)])
    debug('OrderedDict:', ord_dict) # OrderedDict: OrderedDict([('x', 1), ('b', 2), ('a', 3)])
    for key, value in ord_dict.items():
        debug(f'Key: {key}({type(key)}), Value: {value}({type(value)})') 
        # Key: x(<class 'str'>), Value: 1(<class 'int'>)
    classic_dict = {'x': 1, 'b': 2, 'a': 3}
    debug('Classic dict:', classic_dict) # Classic dict: {'x': 1, 'b': 2, 'a': 3}
    for key, value in classic_dict.items():
        debug(f'Key: {key}({type(key)}), Value: {value}({type(value)})')
        # Key: x(<class 'str'>), Value: 1(<class 'int'>)
    classic_dict_from_counter = dict(cnt)
    classic_dict_from_ord_dict = dict(ord_dict)
    debug('Classic dict from counter:', classic_dict_from_counter) 
    # {'a': 8, 'b': 5, 'r': 2, 'c': 1, 'd': 1}
    debug('Classic dict from ordered dict:', classic_dict_from_ord_dict) # {'x': 1, 'b': 2, 'a': 3}
    c1 = collections.Counter(a=3, b=2)
    c2 = collections.Counter(a=1, b=4, c=1)
    c3, c4, c5, c6 = c1 + c2, c1 - c2, c1 & c2, c1 | c2
    debug(c3) # Counter({'b': 6, 'a': 4, 'c': 1})
    debug(c4) # Counter({'a': 2})
    debug(c5) # Counter({'a': 1, 'b': 2})
    debug(c6) # Counter({'b': 4, 'a': 3, 'c': 1})
    d = dict.fromkeys(['a', 'b', 'c'], 0)
    debug(type(d), d) # <class 'dict'> {'a': 0, 'b': 0, 'c': 0}


def priority_queue_example():
    pq = queue.PriorityQueue() # Min heap, first attends lowest priority
    pq.put((2, 'code'))
    pq.put((1, 'eat'))
    pq.put((3, 'sleep'))
    debug(pq) # <queue.PriorityQueue object at 0x10a96e090>
    while not pq.empty():
        next_item = pq.get()
        debug('Priority Queue item:', next_item)
        #Priority Queue item: (1, 'eat')
        #Priority Queue item: (2, 'code')
        #Priority Queue item: (3, 'sleep')


def heapq_priority_queue_example():
    heap = [] # Min heap, first attends lowest priority
    heapq.heappush(heap, (2, 'code'))
    heapq.heappush(heap, (1, 'eat'))
    heapq.heappush(heap, (3, 'sleep'))
    debug(heap) # [(1, 'eat'), (2, 'code'), (3, 'sleep')]
    while heap:
        priority, task = heapq.heappop(heap)
        debug(f'Priority: {priority}, Task: {task} len(heap): {len(heap)})')
        #Priority: 1, Task: eat len(heap): 2)
        #Priority: 2, Task: code len(heap): 1)
        #Priority: 3, Task: sleep len(heap): 0)


def string_utilities_example():
    example_string = "  Hello, World! Let's explore World Python string utilities.  "
    stripped_string = example_string.strip()
    debug(f'Stripped:_{stripped_string}_')
    split_string = stripped_string.split()
    debug(f'Split:_{split_string}_')
    starts_with_hello = stripped_string.startswith('Hello')
    debug(f'startsWith:_{starts_with_hello}_')
    count_o = stripped_string.count('o')
    debug('Count of "o":', count_o)
    try:
        index_world = stripped_string.index('World')
        debug('Index of "World":', index_world)
    except ValueError as e:
        debug('Error:', e)
    replaced_string = stripped_string.replace('World', 'there') #all occurrences
    debug('Replaced "World" with "there":', replaced_string)

    replaced_string = stripped_string.replace('World', 'there', 1)  # Replace only the first occurrence
    debug('Replaced "World" with "there":', replaced_string)

    upper_string = stripped_string.upper()
    lower_string = stripped_string.lower()
    debug('Upper case:', upper_string, 'Lower case:', lower_string)
    # find() like index(), returns -1 instead of raising an error if not found
    find_python = stripped_string.find('Python')
    debug('Find "Python":', find_python)
    # isdigit() checks if all characters in the string are digits
    debug('Is digit:', "ab123".isdigit(), "938745983745".isdigit())


def array_module_example():
    int_array = array.array('i', [1, 2, 3, 4, 5, 3, 4, 3])
    float_array = array.array('d', [1.0, 2.0, 3.0, 4.0, 5.0])
    debug('Integer array:', int_array, float_array)
    int_array.append(6)
    debug('Append 6:', int_array) #array('i', [1, 2, 3, 4, 5, 3, 4, 3, 6])
    int_array.extend([7, 8, 9]) 
    debug('Extend with [7, 8, 9]:', int_array)#array('i', [1, 2, 3, 4, 5, 3, 4, 3, 6, 7, 8, 9])
    int_array.remove(3) 
    debug('Remove 3:', int_array) # array('i', [1, 2, 4, 5, 3, 4, 3, 6, 7, 8, 9]) #ONLY FIRST OCCURRENCE
    debug('Element at index 2:', int_array[2])
    debug('Index of element 4:', int_array.index(4)) #FIRST OCCURRENCE
    debug('Buffer info:', int_array.buffer_info()) # Buffer info: (4328086576, 11)
    int_list = int_array.tolist()
    debug('Array to list:', int_list) # Array to list: [1, 2, 4, 5, 3, 4, 3, 6, 7, 8, 9]
    debug('Count occurrences of 4:', int_array.count(4)) # Count occurrences of 4: 2


def string_module_example():
    debug('Digits:', string.digits)  # '0123456789'
    debug('Ascii letters:', string.ascii_letters)  # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    debug('Punctuation:', string.punctuation)  # Punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    values = {'var': 'world'}
    template = string.Template("Hello, $var!")
    substituted = template.substitute(values)
    debug('Substituted:', substituted)  # 'Hello, world!'
    s = 'the quick brown fox jumps over the lazy dog'
    capitalized = string.capwords(s)
    debug('Capwords:', capitalized)  # 'The Quick Brown Fox Jumps Over The Lazy Dog'

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __str__(self):  return f"{self.name}({self.age})"
    def __repr__(self): return f"{self.age}({self.name})"
    def __eq__(self, other):    return self.age == other.age and self.name == other.name
    #Remove str() to be compared by int value
    def __lt__(self, other):    return str(self.age) < str(other.age) 
    def __hash__(self):    return hash(self.__str__())
    def __le__(self, other):    return self.age <= other.age
    def __ne__(self, other):    return not self.__eq__(other)
    def __gt__(self, other):    return self.age > other.age
    def __ge__(self, other):    return self.age >= other.age


def tda_example():
    a = [Person("X",5), Person("A",3), Person("B",4), Person("BB",44)]
    debug(a) # [5(X), 3(A), 4(B), 44(BB)]
    a.sort()
    # Sorted by age, logic in __lt__
    debug(a) # [3(A), 4(B), 44(BB), 5(X)]
    s = set(a)
    debug(s) # {5(X), 44(BB), 4(B), 3(A)}
    s.add(Person("X",5)) # Will not affect because Person("X",5) already in set
    s.add(Person("XXXX",444))
    debug(s) # {5(X), 3(A), 44(BB), 444(XXXX), 4(B)}
    m = {}
    m[Person("X",5)] = 4
    for person in s:
        m[person] = m[person] + 1 if person in m else 1
    debug(m) # {5(X): 5, 3(A): 1, 44(BB): 1, 444(XXXX): 1, 4(B): 1}


def sorting():
    names_with_scores = [('Alice', 9.5), ('Bob', 8.7), ('Charlie', 9.7), ('Diana', 9.5)]
    # Sort by the score in descending order, and then by name in ascending order
    sorted_list = sorted(names_with_scores, key=lambda x: (-x[1], x[0]))
    debug(sorted_list) # [('Charlie', 9.7), ('Alice', 9.5), ('Diana', 9.5), ('Bob', 8.7)]
    example_list = [5, 3, 2, 8, 1, 4]
    example_list.sort(key=lambda x: (x % 2, -x))
    # Sort by remainder when divided by 2 in ascending order, and then by value in descending order
    debug(example_list) # [8, 4, 2, 5, 3, 1]


def multidimensional_arrays():
    # m = [[0]*cols]*rows # WRONG!!!
    rows, cols = 3, 4
    m = [[0 for _ in range(cols)] for _ in range(rows)]
    debug(m)
    adj_list = m = [[] for _ in range(rows)]
    debug(adj_list) # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    depth, rows, cols = 2, 3, 4
    m3d = [[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
    debug(m3d)
    #[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]



def itertools_examples():
    cartesian_product = list(itertools.product([1, 2], ['a', 'b']))
    debug('Cartesian product:', cartesian_product) #  [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
    permutations = list(itertools.permutations([1, 2, 3]))
    debug('Permutations:', permutations) 
    # [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    combinations = list(itertools.combinations('ABC', 2))
    debug('Combinations:', combinations) # [('A', 'B'), ('A', 'C'), ('B', 'C')]
    # chain to connect multiple iterators
    chained = list(itertools.chain('ABC', 'DEF'))
    debug('Chained:', chained) # ['A', 'B', 'C', 'D', 'E', 'F']
    # cycle to cycle through an iterator indefinitely
    cycle = itertools.cycle('ABCD')
    debug('Cycle:', [next(cycle) for _ in range(15)])
    #['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C']
    # itertools.accumulate to get accumulated sums
    accumulated_sums = list(itertools.accumulate([1, 2, 3, 4, 5]))
    debug('Accumulated sums:', accumulated_sums) # Accumulated sums: [1, 3, 6, 10, 15]
    # itertools.groupby to group items by a certain function
    data = [('a', 1), ('b', 1), ('a', 2), ('b', 2)]
    lambda_sort_function = lambda x: x[0]
    data.sort(key=lambda_sort_function)
    #Must be sorted with the same function to groupby works fine
    grouped_data = {k: list(v) for k, v in itertools.groupby(data, lambda_sort_function)}
    debug('Grouped data:', grouped_data) 
    #  Grouped data: {'a': [('a', 1), ('a', 2)], 'b': [('b', 1), ('b', 2)]}
    # itertools.count to create an infinite iterator
    count = itertools.count(start=10, step=2)
    debug('Count start at 10 step 2:', [next(count) for _ in range(5)])
    # Count start at 10 step 2: [10, 12, 14, 16, 18]


def functools_examples():
    # functools.reduce to apply a function cumulatively to the items
    sum = functools.reduce(lambda a, b: a+b, [1, 2, 3, 4, 5])
    debug('Reduced Sum:', sum) # Reduced Sum: 15    
    # functools.partial to freeze some portion of function's arguments
    base_two = functools.partial(int, base=2)
    debug('Base 2 of 101010:', base_two('101010')) # Base 2 of 101010: 42
    
    # functools.lru_cache to cache the return value of a function
    @functools.lru_cache(maxsize=None)
    def fib(n):
        if n < 2:
            return n
        return fib(n-1) + fib(n-2)
    debug('Fibonacci with lru_cache:', fib(100)) 
    # Fibonacci with lru_cache: 354224848179261915075

    # functools.total_ordering to automatically generate ordering methods
    @functools.total_ordering
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        def __eq__(self, other):
            return self.age == other.age
        def __lt__(self, other):
            return self.age < other.age
    person1 = Person('Alice', 30)
    person2 = Person('Bob', 25)
    debug('Person1 < Person2:', person1 < person2) # Person1 < Person2: False
    debug('Person1 > Person2:', person1 > person2) # Person1 > Person2: True
    debug('Person1 == Person2:', person1 == person2) # Person1 == Person2: False
    debug('Person1 != Person2:', person1 != person2) # Person1 != Person2: True
    
    
def random_examples():
    # random.choice to select a random element from a list
    debug('Random choice:', random.choice(['apple', 'banana', 'cherry']))
    # random.randint to get a random integer within a range [1,10]
    debug('Random integer:', random.randint(1, 10))
    # random.shuffle to shuffle a list
    list_to_shuffle = list(range(5))
    random.shuffle(list_to_shuffle)
    debug('Shuffled list:', list_to_shuffle)
    # random.seed to seed the random number generator for reproducibility
    random.seed(10)
    debug('Random number with seed 10:', random.random())
    # random.sample to get a sample without replacement
    population = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    to_choose = 4 
    debug(f'Sample {to_choose} without replacement:', random.sample(population, to_choose))
    #Sample 4 without replacement: [14, 16, 2, 4]


def re_examples():
    text = "pre prefix ness in lazyness in predictable text,"
    text += "impressive presentation but not in press or pressureness"
    prefix_pattern = r"\bpre\w*"
    words_with_prefix = re.findall(prefix_pattern, text)
    debug('Words with "pre" prefix:', words_with_prefix)
    #['pre', 'prefix', 'predictable', 'presentation', 'press', 'pressureness']

    suffix_pattern = r"\w*ness\b"
    words_with_suffix = re.findall(suffix_pattern, text)
    debug('Words with "ness" suffix:', words_with_suffix)
    # Words with "ness" suffix: ['ness', 'lazyness', 'pressureness']

    substring_pattern = r"\w*pre\w*"
    words_with_substring = re.findall(substring_pattern, text)
    debug('Words containing "press" substring:', words_with_substring)
    #['pre', 'prefix', 'predictable', 'impressive', 'presentation', 'press', 'pressureness']

    # re.sub to replace occurrences of a pattern with a string
    debug('Substitute:', re.sub(r'pre', 'XX', text))
    #Substitute: XX XXfix ness in lazyness in XXdictable text,
    # imXXssive XXsentation but not in XXss or XXssureness

def json_examples():
    complex_data = {
        'name': 'John',
        'age': 30,
        'married': True,
        'children': [
            {'name': 'Alice', 'age': 5},
            {'name': 'Bob', 'age': 7}
        ],
        'pets': ['dog', 'cat'],
        'address': {
            'street': '123 Maple Street',
            'city': 'Faketown',
            'state': 'FS',
            'zip': '12345'
        }
    }
    debug(type(complex_data)) # <class 'dict'>
    # Serialize the Python object into a JSON formatted string
    json_string = json.dumps(complex_data, indent=2)
    debug(type(json_string)) # <class 'str'>
    debug('JSON string:', json_string)

    # Deserialize the JSON formatted string back into a Python object
    python_object = json.loads(json_string)
    debug(type(python_object)) # <class 'dict'>
    debug('Python object:', python_object)


def datetime_examples():
    # datetime.datetime.now to get the current date and time
    current_datetime = datetime.datetime.now()
    debug('Current datetime:', current_datetime) 
    #Current datetime: 2023-11-08 16:28:12.302351
    
    # datetime.timedelta to represent a duration
    one_day = datetime.timedelta(days=1)
    debug('One day later:', current_datetime + one_day) 
    #One day later: 2023-11-09 16:28:12.302351

    # Subtract a week from the current date and time
    one_week_ago = current_datetime - datetime.timedelta(weeks=1)
    debug('One week ago:', one_week_ago) # One week ago: 2023-11-01 16:28:12.302454

    # Add three hours and ten minutes to the current date and time
    three_hours_ten_minutes_from_now = \
        current_datetime + datetime.timedelta(hours=3, minutes=10)
    debug('Three hours and ten minutes from now:', three_hours_ten_minutes_from_now)
    #Three hours and ten minutes from now: 2023-11-08 19:38:12.302454

    # Find the day of the week
    day_of_week = current_datetime.weekday()
    debug('Day of the week:', day_of_week) # Day of the week: 2 (was Wednesday)

    # Format datetime as a string
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    debug('Formatted datetime:', formatted_datetime)
    # Formatted datetime: 2023-11-08 16:28:12

    # Parse a string to create a datetime object
    date_string = "2023-01-01 12:30:45"
    parsed_datetime = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    debug('Parsed datetime:', parsed_datetime) # Parsed datetime: 2023-01-01 12:30:45

    # Get the current UTC time
    current_utc_datetime = datetime.datetime.utcnow()
    debug('Current UTC datetime:', current_utc_datetime)
    #Current UTC datetime: 2023-11-08 21:28:12.309550

    # Replace the year, month, and day of the current datetime
    replaced_datetime = current_datetime.replace(year=2022, month=12, day=25)
    debug('Replaced datetime (Christmas 2022):', replaced_datetime)
    #Replaced datetime (Christmas 2022): 2022-12-25 16:28:12.302454

    # Calculate a future date by adding a timedelta to the current date
    two_months_later = current_datetime + datetime.timedelta(days=60)
    debug('Two months from now:', two_months_later)
    #Two months from now: 2024-01-07 16:28:12.302454

    # Calculate the time until New Year's Day of the next year
    next_year = current_datetime.year + 1
    new_years_day = datetime.datetime(next_year, 1, 1)
    time_until_new_year = new_years_day - current_datetime
    debug('Time until New Year:', time_until_new_year)
    #Time until New Year: 53 days, 7:31:47.697546


def bisect_examples():
    sorted_list = [1, 3, 4, 4, 5]
    left_position = bisect.bisect_left(sorted_list, 4) # Bisect left position (4): 2
    debug('Bisect left position (4):', left_position)

    right_position = bisect.bisect(sorted_list, 4)
    debug('Bisect right position (4):', right_position) # Bisect right position (4): 4

    bisect.insort_left(sorted_list, 2) # List after insort_left (2): [1, 2, 3, 4, 4, 5]
    debug('List after insort_left (2):', sorted_list)

    bisect.insort(sorted_list, 6) # List after insort (6): [1, 2, 3, 4, 4, 5, 6]
    debug('List after insort (6):', sorted_list)

    bisect.insort_left(sorted_list, 4) 
    #List after insort_left (4): [1, 2, 3, 4, 4, 4, 5, 6]
    debug('List after insort_left (4):', sorted_list)

    start_pos = bisect.bisect_left(sorted_list, 4)
    end_pos = bisect.bisect(sorted_list, 4)
    debug('Start position for range (4):', start_pos) #Start position for range (4): 3
    debug('End position for range (4):', end_pos) #End position for range (4): 6

def generate_id() -> int:
    return random.randint(1000, 9999)

@dataclass(frozen=True) #Frozen to make it in mutable and hashable
class Player:
    name: str
    team: str = "Free Agent"
    id: int = field(default_factory=generate_id)

    # def __post_init__(self): #this works if frozen=False
    #     self.name = self.name.title()  # Capitalize the player's name

def dataclass_uses():
    # Example usage:
    player = Player("john doe")
    debug(player)  # id will be a random number generated by generate_id
    #Player(name='john doe', team='Free Agent', id=8578)
    player1 = Player("John Doe", id=1234)
    player2 = Player("Jane Smith", id=5678)

    player_set = {player1, player2}
    player_dict = {player1: 'Goalkeeper', player2: 'Forward'}
    debug(player_set)
    #{Player(name='John Doe', team='Free Agent', id=1234),
    # Player(name='Jane Smith', team='Free Agent', id=5678)}
    debug(player_dict)
    #{Player(name='John Doe', team='Free Agent', id=1234): 'Goalkeeper', 
    # Player(name='Jane Smith', team='Free Agent', id=5678): 'Forward'}


def string_utilities_examples():
    debug("HELLO".lower() == "hello".lower())
    str_var = "    Hello world    "
    debug(f"_{str_var.strip()}_") # _Hello world_
    debug(f"_{str_var.lstrip()}_") # _Hello world    _
    debug(f"_{str_var.rstrip()}_") # _    Hello world_

    s = "the big brown fox over the lazy dog"
    l = list(s)
    debug(type(l), l)
    #<class 'list'> ['t', 'h', 'e', ' ', 'b', 'i', 'g', ' ', 
    # 'b', 'r', 'o', 'w', 'n', ' ', 'f', 'o', 'x', ' 
    # ', 'o', 'v', 'e', 'r', ' ', 't', 'h', 'e', ' ', 
    # 'l', 'a', 'z', 'y', ' ', 'd', 'o', 'g']

    s = "the big    brown fox        over the lazy dog"
    t = s.split()
    debug(type(t), t)
    #<class 'list'> ['the', 'big', 'brown', 'fox', 'over', 'the', 'lazy', 'dog']

    s = "the big XX   brown XfoX        ovXer the laXXXzy dog"
    t = s.split("X")
    debug(type(t), t)
    #<class 'list'> ['the big ', '', '   brown ', 'fo', '        ov',
    #  'er the la', '', '', 'zy dog']

    s = ' '.join(['Python', 'is', 'a', 'fun', 'programming', 'language'])
    m = f"_{s}_"
    debug(m) # _Python is a fun programming language_
    debug(type(m)) # <class 'str'>

    a = 12
    b = 12.34
    debug(type(a), type(b))
    sa = str(a)
    sb = str(b)
    debug(type(sa), type(sb))
    ia = int(sa)
    fb = float(sb)
    debug(type(ia), type(fb))

    a = [-3,2,3,4,1,1,2,3,4,1,2,3,4,4,3,-2,1]
    s = set(a)
    debug(type(s), s)
    l = list(s)
    debug(type(l), l)

    c = collections.Counter(a)
    debug(type(c), c) 
    # <class 'collections.Counter'> Counter({3: 4, 4: 4, 1: 4, 2: 3, -3: 1, -2: 1})
    for k in c:
        debug(f"{k} => {c[k]}", type(k), type(c[k]))

    for k, v in c.items():
        debug(f"{k} => {v}", type(k), type(v), type(c.items()))

    debug(type(c.keys()), c.keys())
    debug(type(c.values()), c.values())
    debug(set(c))  # same as c.keys()

    m = {}
    for k in a:
        m[k] = 1 if k not in m else m[k] + 1
    debug(type(m), m)
    for k, v in m.items():
        debug(f"{k} => {v}", type(m.items()))

    debug(type(m.keys()), m.keys())
    debug(type(m.values()), m.values())

    s = "_ str STR Str stR _"
    debug(s.upper())
    debug(s.lower())
    debug(s.title())

    t = "I like bananas and more bananas"
    x = t.replace("bananas", "apples")
    debug(x) # I like apples and more apples

    x = u'long Test long TESTstring TEST otherTEST'.replace('TEST', 'xxx', 1)
    debug(x) # long Test long xxxstring TEST otherTEST

    t = "0123456789abcdefghi"
    debug(t[3:10])#3456789

    t = "0123456789abcdefghi"
    index = t.find("xxx")
    debug(index) # -1
    index = t.find("89a")
    debug(index) #8

    t = "0123456789abcdefghi"
    debug(t.startswith("0123"))# True
    debug(t.endswith("fghi"))#True

    debug("{0:<10}".format("Guido"))  # 'Guido     '
    debug("{0:>10}".format("Guido"))  # '     Guido'
    debug("{0:^10}".format("Guido"))  # '  Guido   '

    line = "0123456789abcdefg"
    suffix = "cdefg"
    if line.endswith(suffix):
        line_new = line[:-len(suffix)]
        debug("Before:", line, "After:", line_new)
        # Before: 0123456789abcdefg After: 0123456789ab

    prefix = "01234"
    if line.startswith(prefix):
        line_new = line[len(prefix):]
        debug("Before:", line, "After:", line_new) 
        # Before: 0123456789abcdefg After: 56789abcdefg

    line = "0123456789abcdefg0123456789abcdefg"
    index = line.index("abcde")
    debug(index) # 10
    try:
        index = line.index("xxx")
    except ValueError:
        debug("substring not found")


def lambda_examples():
    add = lambda x, y: x + y
    debug(add(5, 3))  # Output: 8

    numbers = [1, 2, 3, 4, 5]
    squared = map(lambda x: x**2, numbers)
    debug(f"Type of squared ", type(squared))  # Output: <class 'map'>
    squared_list = list(squared)
    debug(squared_list)  # Output: [1, 4, 9, 16, 25]


def pprint_example():
    debug("A tuple with pprint:")
    tup = ('spam', ('eggs', ('lumberjack', ('knights', 
        ('ni', ('dead', ('parrot', ('fresh fruit',))))))))
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(tup)
    # ('spam', ('eggs', ('lumberjack', ('knights', ('ni', ('dead', (...)))))))

    debug("Now a dictionary:")
    some_dict = {'a':2, 'b':{'x':3, 'y':{'t1': 4, 't2':5}}}
    debug(json.dumps(some_dict, sort_keys=True, indent=4))
    #{
    #    "a": 2,
    #    "b": {
    #        "x": 3,
    #        "y": {
    #          "t1": 4,
    #           "t2": 5
    #        }
    #    }
    #}   
def flow_control():
    for index, char in enumerate("hello"):# Enumerate
        debug(f"Character {char} at position {index}")
    array = [1, 2, 3, 4, 5]    # For each
    for val in array:
        debug(val)
    for i in range(10):# For loop 0(inclusive) to n (exclusive)
        debug(i)
    a, b = 2, 8    # For loop a to b (inclusive both ends)
    for i in range(a, b+1):
        debug("[a,b]",i) # Prints 2, 3, 4, 5, 6, 7, 8
    for i in range(b, a-1, -1):# For loop b to a (decreasing, inclusive both ends)
        debug("[b,a]",i) # Prints 8, 7, 6, 5, 4, 3, 2
    for i in range(a, b+1, 2):# For loop from a to b with delta increases
        debug("[a,b,d=2]",i) # Prints 2, 4, 6, 8
    for i in range(b, a-1, -2):# For loop from b to a with delta decreases
        debug("[b,a,d=2]",i) # Prints 8, 6, 4, 2
    fruit = 'Banana'# Switch - case (Python uses if-elif-else)
    if fruit == 'Mango': 
        debug("fruit is Mango")
    elif fruit == "Grapes":
        debug("fruit is Grapes")
    elif fruit == "Banana":
        debug("fruit is Banana")
    else: 
        debug("fruit isn't Banana, Mango or Grapes")

def queue_examples():
    q = queue.Queue()
    q.put('item1')
    q.put('item2')
    q.put('item3')
    first_item = q.get() # Remove and return an item from the queue
    debug(f"First item removed from the queue: {first_item}. size: {q.qsize()}")
    debug(f"Is the queue empty? {q.empty()}")
    debug(f"Number of items in the queue: {q.qsize()}")

def stack_examples():
    stack = [] # Creating a stack using a list
    stack.append('item1')
    stack.append('item2')
    stack.append('item3')
    last_item = stack.pop()  # Remove and return the top item of the stack
    debug(f"Last item removed from the stack: {last_item}. Size: {len(stack)}")
    is_empty = not stack
    debug(f"Is the stack empty? {is_empty}")
    debug(f"Number of items in the stack: { len(stack)}")


def set_examples():
    my_set = set()
    my_set.add(4)
    debug(f"Set after adding an item: {my_set}")
    my_set.update([5, 6, 7])
    debug(f"Set after adding multiple items: {my_set}")
    
    # Removing an item from the set (will raise a KeyError if the item does not exist)
    my_set.discard(1)  # Discard is safer than remove, as it won't raise an error if the item is not found
    debug(f"Set after removing an item: {my_set}")
    is_present = 4 in my_set
    debug(f"Is 4 in the set? {is_present}")
    size = len(my_set)
    debug(f"Number of items in the set: {size}")
    another_set = {4, 5, 6, 7, 8}
    union = my_set | another_set
    debug(f"Union of the two sets: {union}")
    intersection = my_set & another_set
    debug(f"Intersection of the two sets: {intersection}")
    difference = my_set - another_set
    debug(f"Difference of the two sets: {difference}")
    symmetric_difference = my_set ^ another_set
    debug(f"Symmetric difference of the two sets: {symmetric_difference}")
    is_subset = {4, 5} <= my_set
    debug(f"Is { {4, 5} } a subset of {my_set}? {is_subset}")
    is_superset = my_set >= {4, 5}
    debug(f"Is {my_set} a superset of { {4, 5} }? {is_superset}")
    


def test_read():
    test_count = I()
    for test_case in range(test_count):
        debug("Test " + str(test_case + 1) + ":")
        R, C = MI()
        debug(R,C)
        print(R,C)
        grid = []
        for _ in range(R):
            grid.append(list(LS()))

        for row in grid:
            debug(row)
            print(row)
            for word in row:
                debug(word)
                print(word)
        debug("End case ", test_case + 1, "\n")
        print("End case ", test_case + 1, "\n")


def main():
    #To test reading un comment next line:
    test_read()
    debug("\nmath_utilities")
    math_utilities()
    debug("\ncollections_utilities")
    collections_utilities()
    debug("\npriority_queue_example")
    priority_queue_example()
    debug("\nheapq_priority_queue_example")
    heapq_priority_queue_example()
    debug("\nstring_utilities_example")
    string_utilities_example()
    debug("\narray_module_example")
    array_module_example()
    debug("\nstring_module_example")
    string_module_example()
    debug("\ntda_example")
    tda_example()
    debug("\nsorting")
    sorting()
    debug("\nmultidimensional_arrays")
    multidimensional_arrays()
    debug("\nitertools_examples")
    itertools_examples()
    debug("\nfunctools_examples")
    functools_examples()
    debug("\nrandom_examples")
    random_examples()
    debug("\nre_examples")
    re_examples()
    debug("\njson_examples")
    json_examples()
    debug("\ndatetime_examples")
    datetime_examples()
    debug("\nbisect_examples")
    bisect_examples()
    debug("\ndataclass_uses")
    dataclass_uses()
    debug("\nstring_utilities_examples")
    string_utilities_examples()
    debug("\nlambda_examples")
    lambda_examples()
    debug("\npprint_example")
    pprint_example()
    debug("\nflow_control")
    flow_control()
    debug("\nqueue_example")
    queue_examples()
    debug("\nstack_example")
    stack_examples()
    debug("\nset_examples")
    set_examples()


if __name__ == "__main__":
    main()