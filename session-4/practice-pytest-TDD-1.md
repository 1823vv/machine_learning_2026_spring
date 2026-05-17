# PyTest & TDD

## Why PyTest & TDD Matter

In this session, you will:

1. Solve **Python exercises** (Lists, Sets, Tuples, Dictionaries, Functions, OOP).
2. Write **PyTest test cases** for each exercise.
3. Run tests locally.
4. Set up **CI/CD** with GitHub Actions for automated testing.

---

## Step 1: Prepare the Repository

1. Create a folder `python_practice/`.
2. Inside, create `requirements.txt`:

```text
pytest
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 2: Implement Python Exercises

Create a file `python_exercises.py` with **longer, simple exercises**:

```python
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
    ______YOUR_CODE_HERE_________
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
    return ______YOUR_CODE_HERE_________


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
    return ______YOUR_CODE_HERE_________


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
    ______YOUR_CODE_HERE_________
    return ______YOUR_CODE_HERE_________


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
        ______YOUR_CODE_HERE_________

    def remove_task(self, task):
        """
        Remove a task from the task list if it exists.
        
        Task:
        - Check if the task exists in self.tasks
        - Remove it if found
        """
        ______YOUR_CODE_HERE_________

    def list_tasks(self):
        """
        Return a copy of all tasks currently in the list.
        
        Task:
        - Return a shallow copy to avoid external modification
        """
        ______YOUR_CODE_HERE_________


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
    return ______YOUR_CODE_HERE_________
```

---

## Step 3: Write Test Cases Using PyTest

Create `tests/test_all.py`:

```python
# tests/test_all.py
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from python_exercises import (
    find_duplicates,
    difference_set,
    square_tuple,
    merge_dicts,
    ToDo,
    flatten_list_once
)

# List Exercise
def test_find_duplicates():
    assert set(find_duplicates([1,2,2,3,4,4,5])) == {2,4}
    assert find_duplicates([1,2,3]) == []

# Set Exercise
def test_difference_set():
    assert difference_set({1,2,3},{2,3,4}) == {1}
    assert difference_set({1},{1}) == set()

# Tuple Exercise
def test_square_tuple():
    assert square_tuple((1,2,3)) == (1,4,9)
    assert square_tuple(()) == ()

# Dictionary Exercise
def test_merge_dicts():
    d1 = {'a':1,'b':2}
    d2 = {'b':3,'c':4}
    assert merge_dicts(d1,d2) == {'a':1,'b':5,'c':4}

# OOP Exercise
def test_todo_class():
    todo = ToDo()
    todo.add_task("task1")
    todo.add_task("task2")
    assert todo.list_tasks() == ["task1","task2"]
    todo.remove_task("task1")
    assert todo.list_tasks() == ["task2"]

# Flatten list exercise
def test_flatten_list_once():
    assert flatten_list_once([[1,2],[3,4],5]) == [1,2,3,4,5]
    assert flatten_list_once([]) == []
```

---

## Step 4: Run PyTest Locally

```bash
pytest tests/test_all.py -v
```

or simply,

```bash
pytest
```



* ✅ All tests pass → your code is correct.
* ❌ Any failing test → debug code or tests.

---

## Step 5: Set Up GitHub Actions for CI/CD

Create `.github/workflows/test.yml`:

```yaml
name: Test Python Exercises

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          python -m pytest -v
```

---

## Step 6: Push Your Work and Verify CI/CD

```bash
git add .
git commit -m "Python basic exercises with PyTest and CI/CD"
git push origin main
```

1. Go to your GitHub repository.
2. Click **Actions**.
3. Observe the workflow:

   * ✅ Green check: all tests passed
   * ❌ Red X: some tests failed (click for details)

---

## ✅ Key Takeaways

* TDD and PyTest are **useful for any Python project**, even simple exercises.
* Writing tests ensures **your code works for multiple inputs and edge cases**.
* CI/CD allows **automated testing on every push**, making your code reliable.
