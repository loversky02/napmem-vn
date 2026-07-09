from napmem.reward import napmem_reward


def test_paper_reward_table():
    assert napmem_reward(False, True, True) == -1.0
    assert napmem_reward(True, True, True) == 1.0
    assert napmem_reward(True, True, False) == 0.0
    assert napmem_reward(True, False, True) == -0.5
    assert napmem_reward(True, False, False) == -1.0


def test_usage_bonus_ablation_removes_tool_spam_incentive():
    assert napmem_reward(True, True, False, use_usage_bonus=False) == 1.0
    assert napmem_reward(True, False, True, use_usage_bonus=False) == -1.0
