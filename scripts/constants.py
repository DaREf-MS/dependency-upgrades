MAIN_GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"
TEMP_GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"
TEMP2_GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"

GITHUB_TOKENS = [
    MAIN_GITHUB_TOKEN,
    TEMP_GITHUB_TOKEN,
    TEMP2_GITHUB_TOKEN
]

SECURITY_KEYWORDS = [
    ########## FROM this paper https://arxiv.org/pdf/2307.11853
    'attack', 'bypass', 'cve', 'dos', 'exploit', 'injection',
    'leakage', 'malicious', 'overflow', 'smuggling', 'unauthorized',
    'underflow', 'vulnerability', 'access control', 'open redirect', 'race condition',
    'denial of service', 'out of bound', 'dot dot slash',
    ##### From this paper https://arxiv.org/pdf/2105.14565
    'use after free', 'double free', 'divide by zero', 'illegal', 'disclosure',
    'improper', 'unexpected', 'sanity check', 'uninitialize', 'fail', 'null pointer dereference',
    'null function pointer', 'crash', 'corrupt', 'deadlock', 'fuzz', 'verify',
    'undefined behavior', 'exposure', 'remote code execution', 'osvdb', 'redos',
    'NVD', 'clickjack', 'man-in-the-middle', 'hijack', 'advisory', 'insecure', 'cross-origin',
    'infinite loop', 'authentication', 'brute force', 'crack', 'credential', 'hack', 'harden',
    'lockout', 'password', 'proof of concept', 'poison', 'privilege', 'spoof', 'compromise',
    'out of array', 'exhaust', 'off-by-one', 'privesc', 'bugzilla', 'constant time', 'mishandle',
    'underflow', 'violation', 'recursion', 'snprintf', 'guard', 'protect',
    
    #######
    'cross site', 'request forgery', 'csrf', 'xsrf', 'forged', 'security', 'vulnerable', 'backdoor',
    'threat', 'breach', 'violate', 'blacklist', 'overrun'
]

BUG_KEYWORDS = [
    'fix', 'bug', 'repair', 'correct', 'prevent', 'issue', 
    'problem', 'error', 'exception', 'typo', 'failure'
]

TEST_KEYWORDS = ['test', 'e2e']

DEPRECATE_KEYWORDS = ['deprecate', 'delete', 'remove', 'disable', 'obsolete', 'downgrade', 'drop support']

PERFORMANCE_KEYWORDS = ["optimize", "optimization", "performance", "perf", "speed", "latency", "throughput"]