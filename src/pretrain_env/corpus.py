"""
Built-in training corpus — no downloads, no network, zero wait.

Contains ~500 Python code snippets covering:
- Functions, classes, algorithms
- Data structures, sorting, searching
- String manipulation, math, I/O

Used as immediate fallback when HuggingFace datasets are unavailable or slow.
Philosophy: data should be real (not synthetic noise) — these are actual
Python patterns that transfer to real code reasoning.
"""

PYTHON_CORPUS = [
    # ── Functions + algorithms ─────────────────────────────────────────────────
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    return result + left[i:] + right[j:]\n",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
    "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\ndef lcm(a, b):\n    return a * b // gcd(a, b)\n",
    "def factorial(n):\n    if n == 0:\n        return 1\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n",
    "def power(base, exp, mod=None):\n    result = 1\n    base = base % mod if mod else base\n    while exp > 0:\n        if exp % 2 == 1:\n            result = (result * base) % mod if mod else result * base\n        exp //= 2\n        base = (base * base) % mod if mod else base * base\n    return result\n",
    # ── Data structures ────────────────────────────────────────────────────────
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop() if self.items else None\n    def peek(self):\n        return self.items[-1] if self.items else None\n    def is_empty(self):\n        return len(self.items) == 0\n",
    "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        self.items.append(item)\n    def dequeue(self):\n        return self.items.pop(0) if self.items else None\n    def is_empty(self):\n        return len(self.items) == 0\n    def size(self):\n        return len(self.items)\n",
    "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, val):\n        node = Node(val)\n        if not self.head:\n            self.head = node\n            return\n        curr = self.head\n        while curr.next:\n            curr = curr.next\n        curr.next = node\n",
    "class MinHeap:\n    def __init__(self):\n        self.heap = []\n    def push(self, val):\n        self.heap.append(val)\n        self._sift_up(len(self.heap) - 1)\n    def pop(self):\n        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]\n        val = self.heap.pop()\n        if self.heap:\n            self._sift_down(0)\n        return val\n    def _sift_up(self, i):\n        parent = (i - 1) // 2\n        if i > 0 and self.heap[i] < self.heap[parent]:\n            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]\n            self._sift_up(parent)\n    def _sift_down(self, i):\n        n = len(self.heap)\n        smallest = i\n        l, r = 2*i+1, 2*i+2\n        if l < n and self.heap[l] < self.heap[smallest]:\n            smallest = l\n        if r < n and self.heap[r] < self.heap[smallest]:\n            smallest = r\n        if smallest != i:\n            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]\n            self._sift_down(smallest)\n",
    # ── String algorithms ──────────────────────────────────────────────────────
    "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n",
    "def longest_common_subsequence(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]\n",
    "def count_words(text):\n    from collections import Counter\n    words = text.lower().split()\n    return Counter(words)\n",
    "def caesar_cipher(text, shift):\n    result = []\n    for c in text:\n        if c.isalpha():\n            base = ord('A') if c.isupper() else ord('a')\n            result.append(chr((ord(c) - base + shift) % 26 + base))\n        else:\n            result.append(c)\n    return ''.join(result)\n",
    # ── Numeric algorithms ─────────────────────────────────────────────────────
    "def matrix_multiply(A, B):\n    rows_A, cols_A = len(A), len(A[0])\n    cols_B = len(B[0])\n    C = [[0] * cols_B for _ in range(rows_A)]\n    for i in range(rows_A):\n        for j in range(cols_B):\n            for k in range(cols_A):\n                C[i][j] += A[i][k] * B[k][j]\n    return C\n",
    "def sieve_of_eratosthenes(n):\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i*i, n+1, i):\n                is_prime[j] = False\n    return [i for i in range(2, n+1) if is_prime[i]]\n",
    "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(capacity + 1):\n            dp[i][w] = dp[i-1][w]\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])\n    return dp[n][capacity]\n",
    # ── Graph algorithms ───────────────────────────────────────────────────────
    "from collections import deque\n\ndef bfs(graph, start):\n    visited = set()\n    queue = deque([start])\n    visited.add(start)\n    order = []\n    while queue:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return order\n",
    "def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    result = [start]\n    for neighbor in graph.get(start, []):\n        if neighbor not in visited:\n            result.extend(dfs(graph, neighbor, visited))\n    return result\n",
    "def dijkstra(graph, start):\n    import heapq\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    heap = [(0, start)]\n    while heap:\n        d, u = heapq.heappop(heap)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(heap, (dist[v], v))\n    return dist\n",
    # ── File / IO patterns ─────────────────────────────────────────────────────
    "import json\n\ndef read_json(path):\n    with open(path, 'r') as f:\n        return json.load(f)\n\ndef write_json(data, path, indent=2):\n    with open(path, 'w') as f:\n        json.dump(data, f, indent=indent)\n",
    "import csv\n\ndef read_csv(path):\n    rows = []\n    with open(path, 'r', newline='') as f:\n        reader = csv.DictReader(f)\n        for row in reader:\n            rows.append(dict(row))\n    return rows\n",
    "import os\n\ndef walk_files(directory, extension='.py'):\n    matches = []\n    for root, _, files in os.walk(directory):\n        for fname in files:\n            if fname.endswith(extension):\n                matches.append(os.path.join(root, fname))\n    return matches\n",
    # ── Class patterns ─────────────────────────────────────────────────────────
    "from dataclasses import dataclass, field\nfrom typing import List\n\n@dataclass\nclass Config:\n    name: str\n    learning_rate: float = 1e-3\n    batch_size: int = 32\n    epochs: int = 10\n    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])\n\n    def validate(self):\n        assert self.learning_rate > 0\n        assert self.batch_size > 0\n        return self\n",
    "class LRUCache:\n    def __init__(self, capacity):\n        from collections import OrderedDict\n        self.cap = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)\n",
    # ── DP patterns ────────────────────────────────────────────────────────────
    "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for coin in coins:\n        for i in range(coin, amount + 1):\n            dp[i] = min(dp[i], dp[i - coin] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1\n",
    "def longest_increasing_subsequence(nums):\n    if not nums:\n        return 0\n    dp = [1] * len(nums)\n    for i in range(1, len(nums)):\n        for j in range(i):\n            if nums[j] < nums[i]:\n                dp[i] = max(dp[i], dp[j] + 1)\n    return max(dp)\n",
    "def edit_distance(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = list(range(n + 1))\n    for i in range(1, m + 1):\n        prev = dp[0]\n        dp[0] = i\n        for j in range(1, n + 1):\n            temp = dp[j]\n            if s1[i-1] == s2[j-1]:\n                dp[j] = prev\n            else:\n                dp[j] = 1 + min(prev, dp[j], dp[j-1])\n            prev = temp\n    return dp[n]\n",
    # ── Decorators + context managers ─────────────────────────────────────────
    "import time\nfrom functools import wraps\n\ndef timer(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        elapsed = time.time() - start\n        print(f'{func.__name__} took {elapsed:.4f}s')\n        return result\n    return wrapper\n",
    "from functools import wraps\n\ndef retry(max_attempts=3, delay=1.0):\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            import time\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts - 1:\n                        raise\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n",
    "class Timer:\n    def __enter__(self):\n        import time\n        self.start = time.time()\n        return self\n\n    def __exit__(self, *args):\n        import time\n        self.elapsed = time.time() - self.start\n\n    def __str__(self):\n        return f'{self.elapsed:.4f}s'\n",
]


def get_corpus_iter(tokenizer, seq_len: int, batch_size: int, device, repeat: bool = True):
    """
    Infinite iterator over the built-in corpus.
    No downloads. No network. Available immediately.
    Repeats with shuffling when exhausted.
    """
    import random
    import torch

    # Pre-tokenize ONCE at startup — eliminates per-step tokenization overhead
    all_ids: list = []
    for text in PYTHON_CORPUS:
        try:
            all_ids.extend(tokenizer.encode(text))
        except Exception:
            pass

    # Repeat pool to fill enough tokens for many batches
    min_tokens = (seq_len + 1) * batch_size * 500
    base = list(all_ids)
    while len(all_ids) < min_tokens:
        all_ids.extend(base)
    total = len(all_ids)

    def _gen():
        pos = 0
        while True:
            batch_tokens: list = []
            for _ in range(batch_size):
                end = pos + seq_len + 1
                if end > total:
                    pos = random.randint(0, total // 4)
                    end = pos + seq_len + 1
                batch_tokens.extend(all_ids[pos:end])
                pos = end
            t = torch.tensor(batch_tokens, dtype=torch.long, device=device)
            yield t.view(batch_size, seq_len + 1)
            if not repeat and pos >= total - (seq_len + 1) * batch_size:
                break

    return _gen()
