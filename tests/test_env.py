import pytest

from src.env import Env_level

LEVEL_1 = "levels/level_1.txt"


class TestEnvLevel:
    def test_load_level(self):
        env = Env_level(LEVEL_1)
        assert env.grid is not None
        assert env.height > 0
        assert env.width > 0

    def test_invalid_file_path(self):
        with pytest.raises(ValueError):
            Env_level("")

    def test_nonexistent_file(self):
        with pytest.raises(ValueError):
            Env_level("nonexistent.txt")

    def test_reset_returns_state(self):
        env = Env_level(LEVEL_1)
        state = env.reset()
        assert state is not None
        assert len(state) == 7

    def test_step_returns_tuple(self):
        env = Env_level(LEVEL_1)
        state, reward, done = env.step("U")
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)

    def test_step_invalid_action(self):
        env = Env_level(LEVEL_1)
        with pytest.raises(ValueError):
            env.step("X")

    def test_possible_actions(self):
        env = Env_level(LEVEL_1)
        assert set(env.possible_actions) == {"U", "D", "L", "R"}

    def test_rewards_defined(self):
        env = Env_level(LEVEL_1)
        for key in ("wall", "key", "chest", "lava", "out_bounds", "action", "break_a_wall"):
            assert key in env.rewards


class TestAgent:
    def test_get_missing_key(self):
        from src.agent import get
        assert get({}, ("a", "b")) == 0

    def test_get_existing_key(self):
        from src.agent import get
        assert get({("a",): 42}, ("a",)) == 42

    def test_epsilon_greedy_random(self):
        from src.agent import epsilonGreedy
        env = Env_level(LEVEL_1)
        Q = {}
        # epsilon=1.0 always picks randomly - should not raise
        action = epsilonGreedy(1.0, Q, env.state, env.possible_actions)
        assert action in env.possible_actions

    def test_max_action(self):
        from src.agent import maxAction
        env = Env_level(LEVEL_1)
        Q = {}
        action = maxAction(Q, env.state, env.possible_actions)
        assert action in env.possible_actions
