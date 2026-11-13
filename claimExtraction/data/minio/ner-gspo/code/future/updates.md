# Implement VIGOR + Pilot-Commit for the existing GRPO trainer

## Goal

Modify the existing GRPO/RLVR trainer so rollout generation is allocated adaptively instead of using a fixed number of rollouts for every prompt.

Implement the ideas from these two 2026 papers:

1. **VIGOR**
   - Heyang Jiang, Henry Liu, Baharan Mirzasoleiman
   - "Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning"
   - arXiv:2607.22002v1, July 2026
   - https://arxiv.org/abs/2607.22002

2. **Pilot-Commit**
   - Woojeong Kim, Ziyi Yang, Jing Nathan Yan, Jialu Liu
   - "Spend Your Rollouts Where It Counts: Rollout Allocation for Group-Based RL Post-Training"
   - arXiv:2605.26606v1, May 2026
   - https://arxiv.org/abs/2605.26606

Do NOT blindly merge the papers into one algorithm. Preserve their distinct mechanics and expose them as configurable strategies:

- `allocation_strategy="grpo"`
- `allocation_strategy="vigor"`
- `allocation_strategy="pilot_commit"`
- optionally later: `allocation_strategy="vigor_pilot_commit"` as an experimental hybrid

The first implementation target is exact VIGOR. Pilot-Commit should be implemented separately so results can be compared fairly.

---

# Part 1 — Understand the GRPO baseline before touching it

For prompt x, sample G responses:

    y_1, ..., y_G ~ pi_theta(. | x)

Compute scalar rewards:

    r_i = R(x, y_i)

The standard normalized group advantage is:

    A_i = (r_i - r_bar) / (sigma_r + delta)

where:

    r_bar = (1/G) * sum_i r_i

and:

    sigma_r = sqrt((1/G) * sum_i (r_i - r_bar)^2)

The PPO/GRPO objective is approximately:

    L_GRPO(theta)
      = E[
          (1/G) sum_i
          min(
              rho_i(theta) * A_i,
              clip(rho_i(theta), 1-eps, 1+eps) * A_i
          )
        ]
        - beta * KL(pi_theta || pi_ref)

with:

    rho_i(theta)
      = pi_theta(y_i | x) / pi_theta_old(y_i | x)

The allocation algorithms MUST NOT silently change this objective.

They should primarily change WHICH and HOW MANY rollouts are generated.

---

# Part 2 — Why reward variance is the allocation signal

The VIGOR paper proves the key fact under binary RLVR rewards.

Assume:

    r_i in {+1, -1}

and the KL term is removed.

Let:

    S = sum_i |A_i|

Then:

    || grad_theta L_GRPO || <= (M/G) * S

if:

    || grad_theta rho_i(theta) || <= M

For binary rewards, if p is the fraction of positive samples:

    mean = 2p - 1

    sigma^2 = 4p(1-p)

The positive and negative normalized advantages are:

    A_+ = 2(1-p) / sigma

    A_- = -2p / sigma

Therefore:

    S
      = G[
          p |A_+|
          + (1-p) |A_-|
        ]

      = G * sigma

Hence:

    || grad_theta L_GRPO || <= M * sigma

So larger within-group reward standard deviation means potentially larger GRPO learning signal.

For binary {0,1} rewards instead:

    Var(r) = p(1-p)

and the maximum occurs at:

    p = 0.5

Therefore the useful regime is where the model is uncertain:

    not solved
    not hopeless
    but producing a mixture of success/failure trajectories

This is the fundamental reason to allocate more rollout budget to high-variance prompts.

Important:
Do not interpret "high variance" as "hardest prompt" in a generic difficulty sense.

The useful criterion is:

    high CURRENT reward variance under the CURRENT policy

because prompt informativeness changes as the model learns.

---

# Part 3 — Implement exact VIGOR

## 3.1 Core idea

Standard GRPO:

    every prompt -> fixed G rollouts

VIGOR:

    start with a small rollout count for every prompt
    -> estimate reward variance
    -> keep only the highest-variance prompts
    -> give those prompts more rollouts
    -> repeat
    -> use the cumulative retained rollout sets for the GRPO update

The VIGOR paper explicitly defines:

    active set A_1 = B

For every prompt x:

    Y_x^(0) = empty

    m_x^(1) = m_0

At refinement round t:

    generate m_x^(t) NEW rollouts

    Y_x^(t)
      = Y_x^(t-1) union new_rollouts

Then:

    u_x^(t)
      = Var(
          { R(x,y) : y in Y_x^(t) }
        )

If t < T:

    A_(t+1)
      = top-alpha fraction of A_t
        ranked by u_x^(t)

For retained prompts:

    m_x^(t+1)
      = gamma * m_x^(t)

Prompts removed from the active set receive NO more rollouts in this training step.

At the end:

    use the accumulated rollout sets of the retained prompts
    in the normal GRPO update.

Reference:
VIGOR Sections 4.1 and 4.2, Algorithm 1.

---

# Part 4 — Important VIGOR detail: cumulative rollouts

This is critical.

Do NOT do this:

    round 1 -> 2 rollouts
    select prompt
    throw those 2 rollouts away
    round 2 -> 4 new rollouts
    train only on the 4 new rollouts

That is not VIGOR.

Do this:

    Y_x = previous_rollouts + new_rollouts

So with:

    m0 = 2
    gamma = 2
    T = 4

a surviving prompt gets:

    round 1: +2
    round 2: +4
    round 3: +8
    round 4: +16

and its final training set contains:

    2 + 4 + 8 + 16 = 30 rollouts

The paper's Algorithm 1 explicitly says the rollout sets are merged and that the GRPO update uses the retained rollout sets.

However, note the fair-compute experimental configuration in the paper:

    T = 4
    m0 = 2
    gamma = 2
    alpha = 0.5

while the GRPO baseline uses 8 rollouts/prompt.

The implementation therefore needs a BUDGET ACCOUNTING layer rather than simply allowing every survivor to reach 30 rollouts.

---

# Part 5 — Correct budget accounting

Let:

    B = number of prompts in the sampling batch

Let:

    m0 = initial rollout count

Let:

    alpha = fraction retained after each refinement

Let:

    gamma = rollout expansion factor

Let:

    T = number of refinement rounds

At round t, approximately:

    |A_t| = B * alpha^(t-1)

and per-prompt NEW rollout count is:

    m_t = m0 * gamma^(t-1)

Therefore expected NEW rollouts generated at round t are:

    C_t
      = B * alpha^(t-1) * m0 * gamma^(t-1)

      = B*m0*(alpha*gamma)^(t-1)

Total expected rollout generation:

    C_total
      = B*m0 * sum_{t=0}^{T-1} (alpha*gamma)^t

If:

    alpha * gamma = 1

then:

    C_total = B * m0 * T

This is the budget-conserving regime emphasized by VIGOR's theorem.

For the paper's configuration:

    alpha = 0.5
    gamma = 2
    alpha*gamma = 1

Therefore:

    C_total = B * 2 * 4
            = 8B

which matches the standard GRPO baseline of:

    G = 8 rollouts/prompt

This is extremely important.

Do NOT hard-code "gamma=2 means 2x the compute."

The shrinking active set compensates for the expanding per-prompt rollout allocation.

---

# Part 6 — Recommended exact VIGOR defaults

Use these defaults first because they are the paper's experimental configuration:

```python
vigor_m0 = 2
vigor_T = 4
vigor_gamma = 2.0
vigor_alpha = 0.5
````

This gives:

```
round 1:
    100% prompts
    2 NEW rollouts each

round 2:
    50% prompts
    4 NEW rollouts each

round 3:
    25% prompts
    8 NEW rollouts each

round 4:
    12.5% prompts
    16 NEW rollouts each
```

Approximate total:

```
2B + 2B + 2B + 2B = 8B
```

So the nominal rollout generation budget matches GRPO with G=8.

The exact active-set counts must use integer rounding, and the implementation must account for this.

---

# Part 7 — Integer rounding and budget-safe selection

Do not use floating-point fractions of a batch.

Implement:

```
k_t = max(1, ceil(alpha * |A_t|))
```

or a clearly documented deterministic rounding rule.

But because integer rounding can cause actual rollout cost to differ from the theoretical budget, maintain a real rollout-budget counter:

```
budget_used
budget_total
```

Before every generation request:

```
requested = number_of_prompts * new_rollouts_per_prompt
```

Check:

```
budget_used + requested <= budget_total
```

If not, shrink the final request.

Never silently exceed the configured rollout budget.

For the fair-comparison mode:

```
budget_total = B * G_baseline
```

Example:

```
B = 128
G_baseline = 8
```

then:

```
budget_total = 1024 rollouts/step
```

The theoretical VIGOR schedule should consume approximately this amount.

---

# Part 8 — Variance estimator

For every active prompt x after each refinement:

```
rewards_x = [r_1, ..., r_n]
```

Compute:

```
mean_x = (1/n) * sum_i r_i

var_x = (1/n) * sum_i (r_i - mean_x)^2
```

For binary 0/1 rewards:

```
var_x = p_hat_x * (1 - p_hat_x)
```

where:

```
p_hat_x = mean_x
```

Use the population variance for the allocation signal because the paper defines the utility as reward variance over the rollout set.

Do not use an unrelated batch-global variance.

The variance is PER PROMPT.

---

# Part 9 — Selection rule

At each refinement:

```
rank prompts in A_t by descending u_x
```

where:

```
u_x = var_x
```

Keep only:

```
top alpha fraction
```

Important:
Selection is relative to the CURRENT active set, not always the entire original batch.

Example with alpha=0.5:

```
A1 = 128 prompts

A2 = top 64 from A1

A3 = top 32 from A2

A4 = top 16 from A3
```

Do NOT select:

```
top 64 from the original 128 every round
```

That would be a different algorithm.

---

# Part 10 — What happens to filtered prompts?

A prompt removed from A_t is NOT trained on additional rollouts.

But its existing rollouts remain valid.

For example:

```
Prompt X:
    round 1 -> 2 rollouts
    low variance
    filtered
```

Its 2 existing rollouts are simply excluded from additional generation.

Whether its round-1 rollouts are included in the final GRPO minibatch depends on whether the training implementation expects all prompts or selected prompts.

The VIGOR paper says the GRPO update uses "all retained rollouts", so operationally:

```
only prompts surviving the progressive selection
contribute to the final GRPO update of that step
```

This needs to be implemented explicitly.

Do not accidentally train on low-variance prompts as if VIGOR had not filtered them.

---

# Part 11 — Very important on-policy issue

All VIGOR rollouts within one training step should be generated from the SAME frozen policy snapshot:

```
pi_theta_step
```

Do NOT perform optimizer updates between refinement rounds.

Correct:

```
freeze policy snapshot

round 1 generate
round 1 reward
select

round 2 generate from SAME policy snapshot
round 2 reward
select

round 3 generate from SAME policy snapshot
...

finalize rollout groups

perform GRPO optimizer update
```

Incorrect:

```
round 1 generate
optimizer.step()

round 2 generate from changed policy

optimizer.step()
```

That would make the rollout group internally off-policy and would invalidate the intended GRPO bookkeeping.

---

# Part 12 — Store old-policy log probabilities for ALL rollouts

Each generated response needs at least:

```python
Rollout(
    prompt_id,
    input_ids,
    output_ids,
    reward,
    old_logprobs,
    attention_mask,
)
```

For VIGOR, because rollouts are generated at different refinement rounds, they still all come from the same policy snapshot for that training step.

Therefore:

```
old_logprobs
```

must correspond to that same policy snapshot.

Do not recompute old log probabilities after the optimizer step.

---

# Part 13 — Advantages after VIGOR selection

For every prompt that survives to training:

```
R_x = [r_1, ..., r_Gx]
```

where G_x is the total number of accumulated rollouts for that prompt.

Compute:

```
mu_x = mean(R_x)

sigma_x = std(R_x)
```

and:

```
A_i = (r_i - mu_x) / (sigma_x + delta)
```

using the ENTIRE accumulated group for that prompt.

Do not normalize every refinement round separately.

This is one of the most important implementation details.

Example:

```
round 1 rewards:
    [1, 0]

round 2 rewards:
    [1, 1, 0, 0]
```

final group:

```
[1, 0, 1, 1, 0, 0]
```

Compute ONE mean/std over those six rewards.

Then calculate GRPO advantages for all six responses.

---

# Part 14 — Zero-variance handling

If:

```
sigma_x = 0
```

the prompt contains no reward variation.

For example:

```
[0,0]
[1,1]
[0,0,0,0]
[1,1,1,1]
```

In standard GRPO this gives zero relative advantage.

Implement:

```python
if std <= eps:
    advantages = torch.zeros_like(rewards)
```

Do not allow NaNs.

VIGOR should normally reduce the chance that such prompts consume the entire budget, but zero variance can still occur.

---

# Part 15 — VIGOR and non-binary rewards

The VIGOR theorem is explicitly derived for binary rewards.

However, the algorithm itself defines:

```
utility = empirical reward variance
```

So support both:

```
reward_dtype=float
```

and:

```
variance-based selection
```

For binary RLVR:

```
utility = p(1-p)
```

For continuous rewards:

```
utility = empirical Var[r]
```

but clearly mark that the strongest theoretical guarantee is for the binary RLVR setting.

Do not convert arbitrary continuous rewards into binary values merely for VIGOR.

---

# Part 16 — VIGOR pseudocode

Implement something structurally equivalent to:

```python
def vigor_allocate(batch, policy_snapshot, total_budget):
    active = list(batch)

    rollout_sets = {
        prompt.id: []
        for prompt in batch
    }

    budget_used = 0

    m = vigor_m0

    for t in range(vigor_T):
        if len(active) == 0:
            break

        # Generate NEW rollouts only.
        requested = len(active) * m

        if budget_used + requested > total_budget:
            requested_budget = total_budget - budget_used
            active, m = fit_to_remaining_budget(active, m, requested_budget)

        new_rollouts = generate_rollouts(
            prompts=active,
            num_per_prompt=m,
            policy=policy_snapshot,
        )

        budget_used += count(new_rollouts)

        # Merge into cumulative prompt-specific groups.
        for rollout in new_rollouts:
            rollout_sets[rollout.prompt_id].append(rollout)

        # Compute variance using ALL accumulated rewards.
        utility = {}

        for prompt in active:
            rewards = [
                r.reward
                for r in rollout_sets[prompt.id]
            ]

            utility[prompt.id] = population_variance(rewards)

        if t == vigor_T - 1:
            break

        # Keep only top-alpha active prompts.
        keep_count = max(
            1,
            ceil(vigor_alpha * len(active))
        )

        active = top_k(
            active,
            keep_count,
            key=lambda p: utility[p.id],
        )

        # Increase NEW rollout count for survivors.
        m = int(round(m * vigor_gamma))

    return {
        "rollouts": rollout_sets,
        "budget_used": budget_used,
        "selected_prompts": active,
    }
```

Refine this implementation so the actual rollout budget is exactly respected.

---

# Part 17 — Do NOT use greedy "highest variance prompt" allocation

A previous naive proposal would repeatedly do:

```
select highest variance prompt
+2 rollouts
recompute
repeat
```

Do NOT implement that as the primary VIGOR algorithm.

That is not the paper's Algorithm 1.

The actual VIGOR algorithm uses:

```
active-set reduction by top-alpha
+
multiplicative rollout expansion by gamma
+
cumulative rollout groups
+
fixed refinement rounds
```

The distinction matters.

---

# Part 18 — Implement Pilot-Commit as a separate strategy

Pilot-Commit uses a different mechanism.

Its purpose is:

```
cheaply estimate prompt informativeness
-> discard low-signal prompts
-> spend the remaining rollout budget on selected prompts
```

It has TWO stages:

```
PILOT
COMMIT
```

---

# Part 19 — Pilot stage math

Let:

```
B_g = sampling batch size
```

For each prompt q:

```
sample n_pilot rollouts

o_j ~ pi_theta(. | q)
```

For binary reward:

```
r_j in {0,1}
```

Estimate success rate:

```
p_hat(q)
  = (1 / n_pilot) * sum_j r_j
```

For binary reward:

```
Var[r | q]
  = p(q) * (1-p(q))
```

The pilot therefore estimates informativeness from p_hat.

Select:

```
B' =
  { q :
    p_lower <= p_hat(q) <= p_upper
  }
```

The paper uses ASYMMETRIC thresholds.

The main experimental defaults are:

```
p_lower = 0.125
p_upper = 0.75
```

This is deliberate.

Do not simply use:

```
|p - 0.5| < epsilon
```

The paper argues that directional learning dynamics make asymmetric filtering useful.

A prompt with a tiny but nonzero success probability may be approaching the informative regime, so the lower boundary should be permissive.

---

# Part 20 — Why Pilot-Commit does not simply use p=0.5

Mathematically:

```
p(1-p)
```

is maximized at:

```
p = 0.5
```

But the paper does not simply retain only prompts close to 0.5.

Reason:

The model evolves.

A prompt can move:

```
unsolved -> partially solved -> highly informative -> solved
```

A very hard prompt with:

```
p ~= 0.05
```

may become:

```
p ~= 0.30
```

after training.

If we immediately discard every prompt below 0.5, we may throw away useful emerging prompts.

Hence Pilot-Commit uses an asymmetric lower threshold.

---

# Part 21 — Commit stage

For selected prompt q:

```
generate n_commit NEW rollouts
```

The training group is:

```
pilot rollouts UNION commit rollouts
```

That means the effective training group size is:

```
n_pilot + n_commit
```

The paper explicitly trains on the union.

Example:

```
n_pilot = 16
n_commit = 48
```

effective group:

```
64 rollouts
```

---

# Part 22 — Pilot-Commit sampling batch vs training batch

This is another critical implementation distinction.

Define:

```
b_t = training batch size

b_g = sampling batch size
```

Pilot-Commit uses:

```
b_g = s * b_t
```

with the paper's default:

```
s = 3
```

Therefore:

```
b_g = 3 b_t
```

Example:

```
b_t = 128

b_g = 384
```

Pilot samples:

```
384 * n_pilot
```

rollouts.

Then only selected prompts are committed into:

```
128 training prompts
```

This means Pilot-Commit can inspect more prompts than it ultimately trains on.

---

# Part 23 — Pilot-Commit sampling cost

For baseline GRPO with n rollouts per training prompt:

```
C_GRPO = b_t * n
```

For DAPO:

```
C_DAPO = b_g * n
```

Pilot-Commit:

```
C_PC
  = b_g * n_pilot
    + b_t * n_commit
```

The paper matches:

```
n = n_pilot + n_commit
```

in its main comparisons.

Examples from the paper:

```
n = 128:
    n_pilot = 32
    n_commit = 96

n = 64:
    n_pilot = 16
    n_commit = 48

n = 16:
    n_pilot = 8
    n_commit = 8
```

The 16/48 split is the main default for the 64-rollout setting.

---

# Part 24 — Pilot-Commit selection when too many prompts survive

After pilot filtering:

```
|B'| may be > b_t
```

Only:

```
b_t
```

prompts can enter the training batch.

Therefore select/cap the retained set.

The paper uses a replay buffer for excess survivors.

Implement:

```python
if len(survivors) > training_batch_size:
    selected = choose_training_subset(survivors)
    replay_buffer.add(excess_survivors)
```

The exact policy for choosing among survivors should be deterministic and configurable.

At minimum support:

```
random
highest_variance
oldest_first
```

but default to the policy described by the paper rather than inventing a new scoring rule.

---

# Part 25 — Pilot-Commit replay buffer

The replay buffer is important.

Suppose:

```
pilot produces 170 survivors
```

but:

```
training batch = 128
```

Then:

```
128 -> current training batch
42 -> replay buffer
```

At a later step, if the pilot stage has too few survivors:

```
draw from replay buffer
```

to fill the training batch.

But replay prompts must not become arbitrarily stale.

Define:

```
age = current_step - pilot_step
```

Evict prompts when:

```
age > d
```

The paper's default:

```
d = 4 steps
```

This limits off-policy staleness.

---

# Part 26 — Pilot-Commit binding optimization

A naive implementation does:

```
step t:
    pilot
    commit
    train
```

This creates two sequential rollout phases.

The paper instead recommends binding:

```
pilot at step t-1
->
commit at step t
```

so the inference workload can be overlapped.

This creates a small policy lag.

Implement this as an optimization after the simple version works.

Architecture:

```text
Step t-1:
    pilot on prompts for t

Step t:
    train using commit for those prompts
    simultaneously run pilot for t+1
```

Maintain metadata:

```python
PilotCandidate(
    prompt_id,
    pilot_rollouts,
    pilot_rewards,
    pilot_success_rate,
    pilot_policy_step,
)
```

At commit time:

```python
policy_age = current_step - pilot_policy_step
```

Reject/replay candidates whose:

```
policy_age > d
```

if strict freshness is enabled.

For the paper's default:

```
d = 4
```

---

# Part 27 — Solved-prompt eviction

Pilot-Commit introduces a second dataset-level mechanism:

```
solved prompts should eventually stop entering the pilot pool
```

Default:

```
p_solve = 1.0
```

So if:

```
p_hat(q) >= p_solve
```

the prompt may be evicted from the future pilot pool.

With p_solve=1.0 this means:

```
ALL pilot responses were correct
```

The authors find this is conservative and robust.

Do NOT use aggressive permanent eviction by default.

Expose:

```python
p_solve = 1.0
```

The paper's ablation found p_solve values 0.75, 0.875, and 1.0 had similar final accuracy, but 1.0 minimizes premature eviction risk.

---

# Part 28 — Why solved eviction matters

Without eviction, the pilot sampler repeatedly spends rollouts confirming that already-solved prompts remain solved.

The paper reports:

```
PC with eviction:
    fewer rollouts/step

PC without eviction:
    significantly more refill sampling
```

In their DeepMath experiment:

```
no eviction:
    35.8% of steps required a second sampling round
    33,379 mean rollouts/step

eviction:
    4.5% of steps required a second round
    25,676 mean rollouts/step
```

about:

```
23% sampling savings
```

with essentially unchanged peak accuracy.

Implement this mechanism because it is not just cosmetic.

---

# Part 29 — Important difference between VIGOR and Pilot-Commit

VIGOR:

```
online
progressive
active set shrinks inside ONE training step
repeated variance estimation
cumulative rollouts
no extra prompt pool required
```

Pilot-Commit:

```
pilot estimates informativeness
then commits additional rollouts only for survivors
can inspect a larger sampling batch
can use replay buffer
can evict solved prompts
can pipeline pilot and commit across steps
```

Do not collapse these into one implementation class with tangled logic.

Use a common interface:

```python
class RolloutAllocator:
    def allocate(...):
        ...
```

and separate implementations:

```python
class FixedGRPOAllocator(RolloutAllocator):
    ...

class VIGORAllocator(RolloutAllocator):
    ...

class PilotCommitAllocator(RolloutAllocator):
    ...
```

---

# Part 30 — Proposed internal data model

Implement a prompt-level state object:

```python
@dataclass
class PromptRolloutState:
    prompt_id: str

    # All generated rollouts for this training step.
    rollouts: list[Rollout]

    # Reward statistics.
    reward_mean: float
    reward_variance: float
    reward_std: float

    # VIGOR bookkeeping.
    active: bool
    refinement_round: int
    next_rollout_count: int

    # Pilot-Commit bookkeeping.
    pilot_success_rate: float | None
    pilot_step: int | None
    evicted: bool

    # Replay bookkeeping.
    replay_age: int | None
```

Do not duplicate the same rollout tensors across structures.

Store references/indices where possible.

Long-CoT training makes memory a real concern.

---

# Part 31 — Do not store unnecessary generated text twice

For each rollout, keep:

```python
prompt_id
input_ids
response_ids
reward
old_logprobs
attention_mask
```

The allocator should operate mostly on:

```python
prompt_id
reward
```

and only hand the selected rollouts back to the trainer.

This prevents the allocator from becoming a GPU-memory bottleneck.

---

# Part 32 — GPU scheduling

Rollout generation is usually the major wall-clock component in this setting.

The VIGOR paper reports rollout generation taking at least about half of total training time in its analyzed configuration, and observes rollout generation scaling roughly with rollout count once sufficiently batched.

Therefore the allocator should produce BATCHED generation requests.

Bad:

```python
for prompt in active:
    generate(prompt, n=m)
```

Good:

```python
generate_batch(
    prompts=active_prompts,
    num_samples=m,
)
```

Use the existing high-throughput inference engine.

Do not turn VIGOR into hundreds of tiny inference requests.

---

# Part 33 — Asynchronous refinement scheduling

For performance, use:

```
reward computation
-> variance computation
-> active-set selection
```

on CPU or lightweight device operations while the inference engine is prepared for the next request.

Do not synchronize the GPU unnecessarily between every prompt.

The intended refinement boundary is:

```
generate batch
-> collect rewards
-> rank prompt IDs
-> generate next batch
```

not:

```
generate one prompt
-> compute variance
-> generate another prompt
```

---

# Part 34 — Fair experiment protocol

When comparing:

```
GRPO
VIGOR
Pilot-Commit
```

do NOT compare only:

```
accuracy after N optimizer steps
```

because rollout costs differ.

The papers primarily evaluate:

```
accuracy vs cumulative rollout count
```

This is the correct efficiency metric.

Log:

```text
optimizer_step
cumulative_rollouts
rollouts_this_step
selected_prompts
mean_group_size
mean_reward_variance
reward_variance_percentiles
tokens_generated
rollout_wall_time
update_wall_time
total_wall_time
validation_accuracy
```

Plot:

```
validation accuracy
vs
cumulative rollouts
```

and separately:

```
validation accuracy
vs
wall-clock time
```

---

# Part 35 — VIGOR metrics you MUST log

For each training step:

```text
VIGOR:
    active_count_r1
    active_count_r2
    ...
    active_count_rT

    new_rollouts_r1
    new_rollouts_r2
    ...
    new_rollouts_rT

    mean_variance_r1
    mean_variance_r2
    ...
    mean_variance_rT

    p50_variance
    p90_variance
    p95_variance
    p99_variance

    budget_used
    budget_target

    final_selected_prompt_count

    mean_final_group_size
    max_final_group_size
```

Also verify:

```text
budget_used <= budget_target
```

for every step.

---

# Part 36 — Pilot-Commit metrics you MUST log

```text
PC:
    sampling_batch_size

    pilot_rollouts
    pilot_survivors

    fraction_too_easy
    fraction_too_hard
    fraction_selected

    commit_rollouts

    replay_buffer_size
    replay_buffer_added
    replay_buffer_used
    replay_buffer_evicted

    solved_prompt_evictions

    second_sampling_round_rate

    rollout_wall_time
```

This will make it possible to diagnose whether the speedup is actually coming from better allocation.

---

# Part 37 — VIGOR ablations to implement

Implement configuration switches for:

### A. No selection

```text
alpha = 1
```

This should approximately collapse toward fixed rollout allocation.

### B. Variance selection

The actual VIGOR criterion:

```text
u_x = Var(rewards_x)
```

### C. Random selection

Same active-set sizes as VIGOR but random selection.

This is essential to verify that variance ranking, rather than merely reducing rollout count, causes the improvement.

### D. Mean-reward selection

Rank by:

```text
mean(reward)
```

This should test whether reward level itself explains the gain.

### E. Fixed budget vs compute-skew budget

Check:

```text
alpha * gamma = 1
```

versus:

```text
alpha * gamma > 1
```

---

# Part 38 — VIGOR theoretical parameters

The paper's convergence analysis assumes the upper tail of prompt reward variance follows a Pareto distribution:

```
f(u) = k * u_min^k / u^(k+1)
```

for:

```
u >= u_min
k > 1
```

In the budget-conserving case:

```
alpha * gamma = 1
```

the paper gives:

```
tau_GRPO / tau_VIGOR
  =
  [
    ((alpha^(-1/k))^T - 1)
    /
    (T * (alpha^(-1/k) - 1))
  ]^(1/3)
```

In the compute-skew case:

```
alpha * gamma > 1
```

the leading scaling is:

```
tau_GRPO / tau_VIGOR
  =
  Theta(
    (alpha^(-1/k))^(T/3)
  )
```

Do NOT use these equations to dynamically set hyperparameters during training.

They are theoretical convergence-time analysis.

Use them as justification for:

```
more refinement rounds
+
heavier-tailed variance distributions
=>
larger potential efficiency gains
```

But the actual implementation should obey the explicit rollout budget.

---

# Part 39 — Recommended initial VIGOR implementation

Use:

```yaml
allocation_strategy: vigor

m0: 2
T: 4
gamma: 2.0
alpha: 0.5

baseline_group_size: 8

budget_mode: match_baseline
```

For each training step:

```text
B prompts
↓
2 rollouts/prompt
↓
compute variance
↓
keep top 50%
↓
4 NEW rollouts/survivor
↓
merge with old rollouts
↓
recompute variance
↓
keep top 50%
↓
8 NEW rollouts/survivor
↓
merge
↓
keep top 50%
↓
16 NEW rollouts/survivor
↓
merge
↓
GRPO update on selected cumulative groups
```

The theoretical budget is:

```
2B + 2B + 2B + 2B
= 8B
```

matching baseline GRPO with 8 samples/prompt.

---

# Part 40 — Recommended initial Pilot-Commit configuration

Use the paper's main 64-rollout setting:

```yaml
allocation_strategy: pilot_commit

training_batch_size: 128
sampling_batch_multiplier: 3

n_pilot: 16
n_commit: 48

p_lower: 0.125
p_upper: 0.75

p_solve: 1.0

max_policy_delay: 4
```

Then:

```
b_g = 3 * 128 = 384
```

Pilot cost:

```
384 * 16
= 6144 rollouts
```

Commit cost:

```
128 * 48
= 6144 rollouts
```

Total:

```
12288 rollouts/step
```

The paper uses this 16/48 split as its main configuration for n=64.

---

# Part 41 — Do not accidentally compare different compute

For baseline GRPO:

```text
128 prompts
64 rollouts/prompt

= 8192 rollouts/step
```

Pilot-Commit:

```text
384 pilot prompts
16 pilot rollouts
+
128 selected prompts
48 commit rollouts

= 6144 + 6144
= 12288 rollouts/step
```

So Pilot-Commit's per-step sampling cost is HIGHER than baseline GRPO in the paper's setup.

That is intentional.

The relevant comparison is:

```
cumulative rollouts needed to reach target accuracy
```

not:

```
equal rollouts per optimizer step
```

Do not report a fake "per-step speedup."

---

# Part 42 — Measure both rollout efficiency and wall-clock efficiency

Because Pilot-Commit performs additional pilot sampling, it may have:

```
more rollouts/step
```

while still needing:

```
fewer cumulative rollouts to reach target accuracy
```

The paper's Qwen3-14B wall-clock measurement showed:

```
GRPO total step time = 248 s
Pilot-Commit = 299 s
```

while DAPO = 453 s

So Pilot-Commit is NOT necessarily faster per optimizer step.

Its advantage is:

```
fewer cumulative rollouts to reach the target.
```

The trainer should report BOTH:

```
rollout efficiency
wall-clock efficiency
```

---

# Part 43 — Hybrid VIGOR + Pilot-Commit

Do this ONLY after both standalone implementations are correct.

Possible hybrid design:

```
stage 1:
    Pilot-Commit filters a large sampling batch

stage 2:
    VIGOR progressively allocates the commit budget
    among the surviving prompts
```

But DO NOT claim this is the exact algorithm from either paper.

It is a new hybrid.

Label it:

```text
experimental_vigor_pilot_commit
```

and keep the standalone baselines untouched.

For a first implementation, prioritize:

```
GRPO
VIGOR
Pilot-Commit
```

before implementing the hybrid.

---

# Part 44 — Correct implementation order

Implement in this order:

## Step 1

Refactor rollout generation behind a common interface:

```python
generate_rollouts(
    prompts,
    num_samples,
    policy_snapshot,
)
```

## Step 2

Implement the current fixed GRPO allocator behind the same interface.

## Step 3

Add prompt-level rollout state.

## Step 4

Implement VIGOR exactly.

## Step 5

Add budget accounting and assertions.

## Step 6

Add variance-selection diagnostics.

## Step 7

Validate VIGOR against toy deterministic rewards.

## Step 8

Implement Pilot-Commit separately.

## Step 9

Add replay buffer and solved-prompt eviction.

## Step 10

Add pilot/commit pipelining.

## Step 11

Only then experiment with a VIGOR + Pilot-Commit hybrid.

---

# Part 45 — Unit tests

Create deterministic unit tests.

## Test 1: all rewards equal

Input:

```text
[1,1,1,1]
```

Expected:

```text
variance = 0
```

## Test 2: maximum binary variance

Input:

```text
[1,0,1,0]
```

Expected:

```text
p = 0.5
variance = 0.25   # for 0/1 rewards
```

## Test 3: VIGOR ranking

Prompt A:

```text
[1,0]
```

Prompt B:

```text
[1,1]
```

Expected:

```text
A ranks above B
```

## Test 4: cumulative groups

After two VIGOR rounds:

```text
len(rollouts[prompt])
=
m0 + gamma*m0
```

for retained prompts.

## Test 5: budget conservation

For:

```text
alpha = 0.5
gamma = 2
m0 = 2
T = 4
```

verify actual total requested rollouts approximately equals:

```text
8 * B
```

subject only to integer rounding.

## Test 6: no optimizer update between VIGOR refinement rounds

Assert all refinement rollouts use the same policy version.

## Test 7: final advantage normalization

Verify advantages are computed from the ENTIRE cumulative group, not separately per refinement round.

## Test 8: Pilot-Commit thresholds

Given:

```text
p_hat = 0.0
```

expect:

```text
defer
```

Given:

```text
p_hat = 0.10
```

expect:

```text
defer
```

Given:

```text
p_hat = 0.50
```

expect:

```text
commit
```

Given:

```text
p_hat = 0.80
```

expect:

```text
skip as too easy
```

Given:

```text
p_hat = 1.0
```

expect:

```text
solved -> eligible for eviction
```

---

# Part 46 — Numerical details

Use:

```python
eps = 1e-8
```

or the trainer's existing normalization epsilon.

Use float32 or float64 for:

```
reward mean
reward variance
success rate
```

Do not calculate ranking in bfloat16.

For binary rewards, this is unnecessary anyway.

Tie handling:

```
deterministic tie-break by prompt_id or original batch index
```

This is important for reproducible multi-seed experiments.

---

# Part 47 — Distributed training requirements

The allocator must work across multiple GPUs/nodes.

Prompt selection happens on global prompt groups, not per-GPU independently.

Correct:

```text
GPU 0 sees prompt utilities
GPU 1 sees prompt utilities
GPU 2 sees prompt utilities
...
all-reduce/gather utilities
global top-alpha selection
broadcast selected prompt IDs
```

Incorrect:

```text
GPU 0 -> local top 50%
GPU 1 -> local top 50%
...
```

because that changes the global allocation budget and biases selection.

Use globally consistent:

```
active prompt IDs
rollout counts
reward statistics
```

---

# Part 48 — Failure handling

If reward computation fails:

```
mark rollout invalid
```

If all rollouts for a prompt are invalid:

```
do not let variance become NaN
```

If a prompt has fewer than 2 valid rewards:

```
use utility = 0
unless the codebase has a better explicitly justified estimator
```

Do not allow a malformed prompt to consume the entire refinement budget.

---

# Part 49 — Logging example

At every VIGOR step log something like:

```text
step=120
strategy=vigor

budget_target=1024
budget_used=1024

round=1
active=128
new_rollouts=256
p50_var=0.18
p90_var=0.25

round=2
active=64
new_rollouts=256
p50_var=0.21
p90_var=0.25

round=3
active=32
new_rollouts=256

round=4
active=16
new_rollouts=256

final_group_size:
    mean=30
    max=30

optimizer_update_groups=16
```

The exact counts will depend on integer selection and your chosen budget handling.

---

# Part 50 — What success looks like

After implementation, verify the following qualitative behavior:

### Standard GRPO

Nearly every prompt gets the same rollout count.

### VIGOR

The distribution of rollout counts becomes highly skewed:

```text
many prompts:
    very few rollouts

fewer prompts:
    many rollouts
```

and those high-rollout prompts should have higher measured reward variance.

### Pilot-Commit

The trainer should:

```text
sample many prompts cheaply
↓
discard too-easy and too-hard prompts
↓
commit heavily to informative prompts
↓
reuse pilot rollouts in training
```

and the replay buffer should keep useful pilot survivors from being wasted.

---

# Part 51 — Paper-grounded defaults

Use these as the initial configuration, not arbitrary tuning:

## VIGOR

```yaml
m0: 2
T: 4
gamma: 2
alpha: 0.5
```

The paper explicitly reports these values in its main experiments.

## Pilot-Commit

```yaml
sampling_batch_multiplier: 3

p_lower: 0.125
p_upper: 0.75

p_solve: 1.0
max_policy_delay: 4

# main 64-rollout experiment
n_pilot: 16
n_commit: 48
```

These values come directly from the Pilot-Commit experimental setup.

---

# Part 52 — Final implementation constraint

Do not rewrite the GRPO optimizer.

The architectural goal is:

```text
                  +----------------------+
dataset ----------> rollout allocator   |
                  |                      |
                  | GRPO                 |
                  | VIGOR                |
                  | Pilot-Commit         |
                  +----------+-----------+
                             |
                             v
                     selected rollouts
                             |
                             v
                     existing GRPO loss
                             |
                             v
                         optimizer
```

The allocator decides:

```
which prompts
how many rollouts
when to stop generating
```

The existing trainer remains responsible for:

```
log probabilities
advantages
clipping
KL if configured
minibatching
backpropagation
optimizer steps
```

Do not mix allocation policy with PPO/GRPO loss code unless necessary.

---

# Part 53 — Deliverables

Implement:

1. `VIGORAllocator`

2. `PilotCommitAllocator`

3. configuration objects

4. unit tests

5. distributed-safe selection

6. rollout-budget accounting

7. replay buffer for Pilot-Commit

8. solved-prompt eviction

9. pilot/commit policy-age tracking

10. training metrics

11. an experiment script comparing:

    GRPO
    VIGOR
    Pilot-Commit

12. plots:

    accuracy vs cumulative rollouts
    accuracy vs wall-clock time
    rollouts/step
    reward variance distribution
    VIGOR active-set size by refinement round

13. a short README explaining:

    algorithm
    equations
    configuration
    how budget matching works
    how to reproduce the paper-style experiments

Before modifying the training code, inspect the existing rollout, reward, advantage, distributed sampling, and optimizer-update paths and adapt the implementation to those abstractions instead of creating a second parallel training stack.

```

### The two most important implementation corrections

**VIGOR's budget conservation is the subtle part.** With the paper's parameters \(m_0=2,\alpha=0.5,\gamma=2,T=4\), the shrinking active set exactly offsets the doubling rollout count because \(\alpha\gamma=1\): each refinement consumes approximately \(2B\) new rollouts, giving \(8B\) total—the same as an 8-sample GRPO baseline. :contentReference[oaicite:1]{index=1}

**Pilot-Commit is not just VIGOR with thresholds.** It uses an oversized pilot sampling batch, filters by asymmetric success-rate thresholds, commits additional rollouts to selected prompts, trains on the union of pilot+commit rollouts, and uses replay/eviction to handle the changing prompt population. The paper's main 64-rollout setup uses \(b_g=3b_t\), \(n_\text{pilot}=16\), \(n_\text{commit}=48\), \(p_\text{lower}=0.125\), \(p_\text{upper}=0.75\), \(p_\text{solve}=1\), and maximum policy delay \(d=4\). :contentReference[oaicite:2]{index=2}

The papers report VIGOR reaching target performance with up to **2.3× fewer rollouts on math** and **1.49× fewer on coding**; Pilot-Commit reports **1.5–1.9× fewer rollouts than GRPO** in its ample-budget settings. :contentReference[oaicite:3]{index=3}
```

[1]: https://arxiv.org/html/2607.22002v1 "Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning"
