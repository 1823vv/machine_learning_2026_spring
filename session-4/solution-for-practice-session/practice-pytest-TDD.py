# python_exercises.py


# 1️⃣ List Exercise: Find all duplicates in a list
def find_duplicates(lst):
    """
    Given a list of elements, return a list of all elements that appear more than once.

    Task:
    - Iterate through the list
    - Keep track of elements you've seen
    - Identify duplicates
    - Return duplicates as a list (order doesn't matter)

    Example:
    >>> find_duplicates([1,2,2,3,4,4,5])
    [2,4]
    """
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)


# 2️⃣ Set Exercise: Return elements only in the first set
def difference_set(a, b):
    """
    Given two sets a and b, return a new set containing only the elements
    that are in the first set (a) but not in the second set (b).

    Task:
    - Use set operations to find the difference
    - Return the resulting set

    Example:
    >>> difference_set({1,2,3}, {2,3,4})
    {1}
    """
    return a - b


# 3️⃣ Tuple Exercise: Return a tuple with all elements squared
def square_tuple(tpl):
    """
    Given a tuple of numbers, return a new tuple where each element
    is squared.

    Task:
    - Iterate through the tuple
    - Square each element
    - Return a new tuple with the squared values

    Example:
    >>> square_tuple((1,2,3))
    (1,4,9)
    """
    return tuple(x**2 for x in tpl)


# 4️⃣ Dictionary Exercise: Merge two dictionaries and sum values of common keys
def merge_dicts(d1, d2):
    """
    Given two dictionaries with numeric values, return a new dictionary
    that merges them. If a key exists in both dictionaries, sum their values.

    Task:
    - Copy the first dictionary to avoid modifying it
    - Iterate through the second dictionary
    - If a key exists in both, add the values
    - Otherwise, add the new key-value pair

    Example:
    >>> merge_dicts({'a':1,'b':2}, {'b':3,'c':4})
    {'a':1,'b':5,'c':4}
    """
    merged = d1.copy()
    for k, v in d2.items():
        merged[k] = merged.get(k, 0) + v
    return merged


# 5️⃣ OOP Exercise: Simple ToDo list
class ToDo:
    """
    A simple ToDo list class for managing tasks.

    Tasks:
    - Store tasks in a list
    - Add new tasks
    - Remove existing tasks
    - Return a copy of the current tasks

    Example:
    >>> todo = ToDo()
    >>> todo.add_task("Buy milk")
    >>> todo.list_tasks()
    ["Buy milk"]
    >>> todo.remove_task("Buy milk")
    >>> todo.list_tasks()
    []
    """

    def __init__(self):
        """Initialize an empty list of tasks."""
        self.tasks = []

    def add_task(self, task):
        """
        Add a task (string) to the task list.

        Task:
        - Append the task to self.tasks
        """
        self.tasks.append(task)

    def remove_task(self, task):
        """
        Remove a task from the task list if it exists.

        Task:
        - Check if the task exists in self.tasks
        - Remove it if found
        """
        if task in self.tasks:
            self.tasks.remove(task)

    def list_tasks(self):
        """
        Return a copy of all tasks currently in the list.

        Task:
        - Return a shallow copy to avoid external modification
        """
        return self.tasks.copy()


# 6️⃣ Function Exercise: Flatten nested list (one level)
def flatten_list_once(lst):
    """
    Given a list that may contain nested lists (one level deep),
    return a new list with all elements flattened into a single list.

    Task:
    - Iterate through the input list
    - If the element is a list, extend the result with its elements
    - Otherwise, append the element itself
    - Return the flattened list

    Example:
    >>> flatten_list_once([[1,2],[3,4],5])
    [1,2,3,4,5]
    """
    return [
        item
        for sublist in lst
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]

