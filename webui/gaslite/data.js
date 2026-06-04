/* Gaslite — demo dataset: contract, side-by-side diff rows, optimization
   reasons, per-function stats and the gas/cost simulation model.
   Exposed as window.GL. All numbers are illustrative but internally consistent. */
(function () {
  // ---- optimization reasons (hover-a-line "why") ----
  const REASONS = {
    calldata: {
      tag: "memory → calldata",
      title: "Read array args straight from calldata",
      body: "External functions can read array arguments directly from calldata instead of copying them into memory first. Saves the copy plus 16 gas per byte.",
      gas: "~2,100 + 16/byte",
    },
    external: {
      tag: "public → external",
      title: "Mark single-entry functions external",
      body: "Functions never called internally don't need the public dispatch path that copies arguments to memory. external skips it.",
      gas: "~200–400 / call",
    },
    cacheLen: {
      tag: "cache array length",
      title: "Hoist users.length onto the stack",
      body: "Re-reading users.length every iteration costs a calldataload each time. Cache it once before the loop and compare against the stack value.",
      gas: "~100 / iteration",
    },
    customError: {
      tag: "require string → custom error",
      title: "Custom errors over revert strings",
      body: "A revert string is stored and ABI-encoded at runtime. A custom error is just a 4-byte selector — cheaper to deploy and to revert.",
      gas: "~50 runtime · ~250 deploy",
    },
    unchecked: {
      tag: "unchecked { ++i }",
      title: "Skip overflow checks on the counter",
      body: "The loop counter is bounded by len and can't overflow, so the Solidity 0.8 overflow guard is dead weight. Wrap the increment in unchecked.",
      gas: "~30–40 / iteration",
    },
    packing: {
      tag: "struct packing",
      title: "Pack the Stake struct into one slot",
      body: "uint128 + uint64 + bool fit inside a single 32-byte storage slot instead of spanning three. The first write becomes 1 SSTORE instead of 3.",
      gas: "~40,000 first write",
    },
    immutable: {
      tag: "immutable",
      title: "Store token in code, not storage",
      body: "token is set once in the constructor and never changes. Marking it immutable bakes it into the bytecode and removes an SLOAD on every read.",
      gas: "~2,100 / read",
    },
  };

  // ---- side-by-side diff rows ----
  // kind: ctx (unchanged) | chg (left removed, right added) | add (right only) | del (left only) | gap
  const R = (kind, l, r, why) => ({ kind, l, r, why: why || null });
  const ROWS = [
    R("ctx", "contract RewardPool {", "contract RewardPool {"),
    R("ctx", "", ""),
    R("chg",
      "    struct Stake { uint256 amount; uint256 since; bool active; }",
      "    struct Stake { uint128 amount; uint64 since; bool active; }",
      "packing"),
    R("ctx", "", ""),
    R("chg",
      "    address public token;",
      "    address public immutable token;",
      "immutable"),
    R("gap", "12 unchanged lines", "12 unchanged lines"),
    R("chg",
      "    function distribute(address[] memory users) public {",
      "    function distribute(address[] calldata users) external {",
      "calldata"),
    R("add", null, "        uint256 len = users.length;", "cacheLen"),
    R("chg",
      "        for (uint256 i = 0; i < users.length; i++) {",
      "        for (uint256 i; i < len;) {",
      "cacheLen"),
    R("ctx",
      "            uint256 reward = rewards[users[i]];",
      "            uint256 reward = rewards[users[i]];"),
    R("chg",
      '            require(reward > 0, "no reward");',
      "            if (reward == 0) revert NoReward();",
      "customError"),
    R("ctx",
      "            balances[users[i]] += reward;",
      "            balances[users[i]] += reward;"),
    R("add", null, "            unchecked { ++i; }", "unchecked"),
    R("ctx", "        }", "        }"),
    R("ctx", "    }", "    }"),
    R("gap", "8 unchanged lines", "8 unchanged lines"),
    R("chg",
      "    function setRewardRate(uint256 r) public {",
      "    function setRewardRate(uint256 r) external {",
      "external"),
    R("chg",
      '        require(msg.sender == owner, "not owner");',
      "        if (msg.sender != owner) revert NotOwner();",
      "customError"),
    R("ctx", "        rewardRate = r;", "        rewardRate = r;"),
    R("ctx", "    }", "    }"),
    R("ctx", "}", "}"),
  ];

  // ---- per-function stats (for the per-call selector) ----
  const FUNCS = [
    { name: "distribute", note: "10 recipients", before: 64200, after: 39900 },
    { name: "stake",      note: "first deposit", before: 88400, after: 60200 },
    { name: "claim",      note: "single claim",  before: 41800, after: 27500 },
    { name: "setRewardRate", note: "owner call", before: 28900, after: 23100 },
  ];

  // techniques applied (for the summary chip / list)
  const TECHNIQUES = [
    "calldata", "immutable", "packing", "customError", "unchecked", "cacheLen", "external",
  ];

  // ---- gas + cost simulation model ----
  // cumulative(n) = deploy + n * perRun
  const MODEL = {
    deploy: { before: 2418900, after: 1498200 },
    perRun: { before: 64200, after: 39900 }, // representative tx (distribute, 10 users)
    gasPriceGwei: 0.02,   // Mantle is cheap — honest default
    mntUsd: 0.62,
  };
  // saved gas over n runs (n runs of the representative tx, plus one deploy)
  MODEL.cumBefore = (n) => MODEL.deploy.before + n * MODEL.perRun.before;
  MODEL.cumAfter = (n) => MODEL.deploy.after + n * MODEL.perRun.after;
  MODEL.savedGas = (n) => MODEL.cumBefore(n) - MODEL.cumAfter(n);
  MODEL.savedPct = (n) => MODEL.savedGas(n) / MODEL.cumBefore(n);
  // gas → MNT → USD
  MODEL.gasToMnt = (g) => g * MODEL.gasPriceGwei * 1e-9;
  MODEL.gasToUsd = (g) => MODEL.gasToMnt(g) * MODEL.mntUsd;

  // simulation presets (run counts)
  const PRESETS = [
    { label: "Deploy", runs: 0 },
    { label: "10", runs: 10 },
    { label: "100", runs: 100 },
    { label: "1K", runs: 1000 },
    { label: "10K", runs: 10000 },
    { label: "100K", runs: 100000 },
    { label: "1M", runs: 1000000 },
  ];

  // ---- full contract sources (fed to the Monaco diff editor) ----
  // Coherent RewardPool.sol so Monaco can compute a real diff and collapse
  // unchanged regions itself, instead of the hand-collapsed ROWS above.
  const ORIGINAL_SRC = `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract RewardPool {
    struct Stake { uint256 amount; uint256 since; bool active; }

    address public token;
    address public owner;
    uint256 public rewardRate;
    uint256 public totalStaked;

    mapping(address => uint256) public rewards;
    mapping(address => uint256) public balances;
    mapping(address => Stake) public stakes;

    event Distributed(uint256 count);
    event Staked(address indexed user, uint256 amount);

    constructor(address _token) {
        token = _token;
        owner = msg.sender;
    }

    function stake(uint256 amount) public {
        require(amount > 0, "zero amount");
        balances[msg.sender] += amount;
        stakes[msg.sender] = Stake(amount, block.timestamp, true);
        totalStaked += amount;
        emit Staked(msg.sender, amount);
    }

    function distribute(address[] memory users) public {
        for (uint256 i = 0; i < users.length; i++) {
            uint256 reward = rewards[users[i]];
            require(reward > 0, "no reward");
            balances[users[i]] += reward;
        }
        emit Distributed(users.length);
    }

    function setRewardRate(uint256 r) public {
        require(msg.sender == owner, "not owner");
        rewardRate = r;
    }
}
`;

  const OPTIMIZED_SRC = `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract RewardPool {
    struct Stake { uint128 amount; uint64 since; bool active; }

    address public immutable token;
    address public owner;
    uint256 public rewardRate;
    uint256 public totalStaked;

    mapping(address => uint256) public rewards;
    mapping(address => uint256) public balances;
    mapping(address => Stake) public stakes;

    error NoReward();
    error NotOwner();

    event Distributed(uint256 count);
    event Staked(address indexed user, uint256 amount);

    constructor(address _token) {
        token = _token;
        owner = msg.sender;
    }

    function stake(uint256 amount) public {
        require(amount > 0, "zero amount");
        balances[msg.sender] += amount;
        stakes[msg.sender] = Stake(uint128(amount), uint64(block.timestamp), true);
        totalStaked += amount;
        emit Staked(msg.sender, amount);
    }

    function distribute(address[] calldata users) external {
        uint256 len = users.length;
        for (uint256 i; i < len;) {
            uint256 reward = rewards[users[i]];
            if (reward == 0) revert NoReward();
            balances[users[i]] += reward;
            unchecked { ++i; }
        }
        emit Distributed(len);
    }

    function setRewardRate(uint256 r) external {
        if (msg.sender != owner) revert NotOwner();
        rewardRate = r;
    }
}
`;

  // Optimization anchors: each `find` is a unique substring on a CHANGED line
  // in OPTIMIZED_SRC. The Monaco component locates the line, draws an accent
  // bar in the gutter and shows the matching REASONS entry on hover.
  const OPTIMIZATIONS = [
    { reason: "packing",     find: "struct Stake { uint128" },
    { reason: "immutable",   find: "address public immutable token;" },
    { reason: "calldata",    find: "address[] calldata users) external" },
    { reason: "cacheLen",    find: "uint256 len = users.length;" },
    { reason: "customError", find: "if (reward == 0) revert NoReward();" },
    { reason: "unchecked",   find: "unchecked { ++i; }" },
    { reason: "customError", find: "if (msg.sender != owner) revert NotOwner();" },
    { reason: "external",    find: "function setRewardRate(uint256 r) external" },
  ];

  window.GL = { REASONS, ROWS, FUNCS, TECHNIQUES, MODEL, PRESETS, ORIGINAL_SRC, OPTIMIZED_SRC, OPTIMIZATIONS };
})();
